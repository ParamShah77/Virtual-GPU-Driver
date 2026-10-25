#include "sim.hpp"

#include <sys/mman.h>

#include <cerrno>
#include <chrono>
#include <cstring>

extern "C" {
#include "vrm_core.h"
}

namespace vrm {

// ---- SimGpu ---------------------------------------------------------------

SimGpu::SimGpu(const Config &cfg) : cfg_(cfg)
{
    void *p = ::mmap(nullptr, cfg.vram_bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED)
        throw Error("mmap vram", -errno);
    vram_ = static_cast<uint8_t *>(p);

    gpu_.reset(new vrm_gpu);
    int r = vrm_gpu_init(gpu_.get(), vram_, cfg.vram_bytes);
    if (r) {
        ::munmap(vram_, cfg.vram_bytes);
        throw Error("vrm_gpu_init", r);
    }
    for (uint32_t e = 0; e < VGPU_NUM_ENGINES; e++)
        threads_[e] = std::thread([this, e] { engine_loop(e); });
}

SimGpu::~SimGpu()
{
    {
        std::lock_guard<std::mutex> g(mu_);
        stop_ = true;
        seq_++;
    }
    cv_.notify_all();
    for (auto &t : threads_)
        t.join();
    vrm_gpu_fini(gpu_.get());
    ::munmap(vram_, cfg_.vram_bytes);
}

uint64_t SimGpu::seq()
{
    std::lock_guard<std::mutex> g(mu_);
    return seq_;
}

void SimGpu::signal()
{
    {
        std::lock_guard<std::mutex> g(mu_);
        seq_++;
    }
    cv_.notify_all();
}

bool SimGpu::wait_event(uint64_t seen, int64_t timeout_us)
{
    std::unique_lock<std::mutex> g(mu_);
    auto changed = [&] { return seq_ != seen; };
    if (timeout_us < 0) {
        cv_.wait(g, changed);
        return true;
    }
    return cv_.wait_for(g, std::chrono::microseconds(timeout_us), changed);
}

void SimGpu::engine_loop(uint32_t e)
{
    // Linux timers can't sleep for a microsecond; batch the modelled cost.
    constexpr uint64_t kPaceChunkNs = 50000;
    uint64_t owed = 0;
    uint64_t idle_since = 0;

    auto settle = [&] {
        if (cfg_.pace && owed) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(owed));
        }
        owed = 0;
    };

    for (;;) {
        uint64_t seen;
        {
            std::lock_guard<std::mutex> g(mu_);
            if (stop_)
                return;
            seen = seq_;
        }
        vrm_step st;
        vrm_engine_step(gpu_.get(), e, &st);
        if (st.signal)
            signal();
        if (st.kind == VRM_STEP_RAN) {
            idle_since = 0;
            owed += st.sleep_ns;
            if (owed >= kPaceChunkNs)
                settle();
            continue;
        }
        settle();
        // Out of work: poll briefly (clients submit without doorbells while
        // we're awake), then arm need_doorbell and sleep.
        uint64_t now = vrm_now_ns();
        if (!idle_since)
            idle_since = now;
        if (now - idle_since < cfg_.poll_ns) {
            std::this_thread::yield();
            continue;
        }
        if (vrm_engine_arm_sleep(gpu_.get(), e))
            continue;
        wait_event(seen, st.kind == VRM_STEP_IDLE ? -1 : 200);
        vrm_engine_disarm(gpu_.get(), e);
        idle_since = 0;
    }
}

// ---- Backend over the core -------------------------------------------------

namespace {

class SimBackend : public Backend {
public:
    explicit SimBackend(std::shared_ptr<SimGpu> gpu) : gpu_(std::move(gpu))
    {
        int r = vrm_ctx_create(gpu_->core(), &ctx_);
        if (r)
            throw Error("vrm_ctx_create", r);
    }
    ~SimBackend() override { vrm_ctx_destroy(gpu_->core(), ctx_); }

    int get_info(vgpu_info *o) override {
        vrm_gpu_get_info(gpu_->core(), o);
        return 0;
    }
    int alloc(vgpu_alloc *a) override {
        int r = vrm_mem_alloc(gpu_->core(), ctx_, a->size, a->flags, &a->handle,
                              &a->va, &a->alloc_bytes);
        a->mmap_offset = (uint64_t)a->handle << VGPU_PAGE_SHIFT;
        return r;
    }
    int free(uint32_t h) override { return vrm_mem_free(gpu_->core(), ctx_, h); }
    int map(uint32_t handle, size_t, void **out) override {
        u64 off, bytes;
        int r = vrm_mem_map_get(gpu_->core(), ctx_, handle, &off, &bytes);
        if (!r)
            *out = gpu_->vram() + off;
        return r;
    }
    int unmap(uint32_t handle, void *, size_t) override {
        vrm_mem_map_put(gpu_->core(), ctx_, handle);
        return 0;
    }
    int chan_create(vgpu_chan_create *c) override {
        int r = vrm_chan_create(gpu_->core(), ctx_, c->engine, c->priority,
                                c->ring_va, c->ring_entries, &c->chan_id,
                                &c->ctrl_handle, &c->ctrl_va);
        c->ctrl_mmap_offset = (uint64_t)c->ctrl_handle << VGPU_PAGE_SHIFT;
        return r;
    }
    int chan_destroy(uint32_t id) override {
        return vrm_chan_destroy(gpu_->core(), ctx_, id);
    }
    int doorbell(uint32_t id) override {
        int r = vrm_chan_doorbell(gpu_->core(), ctx_, id);
        if (!r)
            gpu_->signal();
        return r;
    }
    int wait_sem(vgpu_wait_sem *w) override {
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(w->timeout_ms);
        for (;;) {
            uint64_t seen = gpu_->seq();
            int ready = 0;
            uint32_t obs = 0;
            int r = vrm_sem_poll(gpu_->core(), ctx_, w->va, w->value, &obs, &ready);
            w->observed = obs;
            if (r)
                return r;
            if (ready)
                return 0;
            auto left = std::chrono::duration_cast<std::chrono::microseconds>(
                deadline - std::chrono::steady_clock::now());
            if (left.count() <= 0)
                return -ETIMEDOUT;
            gpu_->wait_event(seen, left.count());
        }
    }
    int chan_stats(vgpu_chan_stats *s) override {
        return vrm_chan_stats(gpu_->core(), ctx_, s);
    }
    int get_trace(vgpu_trace_req *r, vgpu_trace_ev *buf) override {
        vrm_trace_read(gpu_->core(), r->from_seq, buf, r->max_events,
                       &r->n_events, &r->next_seq, &r->dropped);
        return 0;
    }

private:
    std::shared_ptr<SimGpu> gpu_;
    vrm_ctx *ctx_ = nullptr;
};

}  // namespace

Device open_sim(std::shared_ptr<SimGpu> gpu)
{
    return Device(std::make_shared<SimBackend>(std::move(gpu)));
}

}  // namespace vrm
