#include "vrm.hpp"

#include <fcntl.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>

namespace vrm {

static void check(int r, const char *what)
{
    if (r < 0)
        throw Error(what, r);
}

// ---- Kernel backend: thin ioctl/mmap wrapper over /dev/vgpu0 -------------

namespace {

class KernelBackend : public Backend {
public:
    explicit KernelBackend(const std::string &path)
    {
        fd_ = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
        if (fd_ < 0)
            throw Error("open " + path, -errno);
    }
    ~KernelBackend() override { if (fd_ >= 0) ::close(fd_); }

    int get_info(vgpu_info *o) override { return ioc(VGPU_IOC_GET_INFO, o); }
    int alloc(vgpu_alloc *a) override { return ioc(VGPU_IOC_ALLOC, a); }
    int free(uint32_t h) override {
        vgpu_free f{h, 0};
        return ioc(VGPU_IOC_FREE, &f);
    }
    int map(uint32_t handle, size_t bytes, void **out) override {
        void *p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
                         (off_t)handle << VGPU_PAGE_SHIFT);
        if (p == MAP_FAILED)
            return -errno;
        *out = p;
        return 0;
    }
    int unmap(uint32_t, void *p, size_t bytes) override {
        return ::munmap(p, bytes) < 0 ? -errno : 0;
    }
    int chan_create(vgpu_chan_create *c) override { return ioc(VGPU_IOC_CHAN_CREATE, c); }
    int chan_destroy(uint32_t id) override {
        vgpu_chan_id c{id, 0};
        return ioc(VGPU_IOC_CHAN_DESTROY, &c);
    }
    int doorbell(uint32_t id) override {
        vgpu_chan_id c{id, 0};
        return ioc(VGPU_IOC_DOORBELL, &c);
    }
    int wait_sem(vgpu_wait_sem *w) override { return ioc(VGPU_IOC_WAIT_SEM, w); }
    int chan_stats(vgpu_chan_stats *s) override { return ioc(VGPU_IOC_CHAN_STATS, s); }
    int get_trace(vgpu_trace_req *r, vgpu_trace_ev *buf) override {
        r->buf = reinterpret_cast<uint64_t>(buf);
        return ioc(VGPU_IOC_GET_TRACE, r);
    }

private:
    int ioc(unsigned long req, void *arg) {
        int r;
        do {
            r = ::ioctl(fd_, req, arg);
        } while (r < 0 && errno == EINTR);
        return r < 0 ? -errno : 0;
    }
    int fd_ = -1;
};

}  // namespace

Device Device::open_kernel(const std::string &path)
{
    return Device(std::make_shared<KernelBackend>(path));
}

// ---- Device ---------------------------------------------------------------

vgpu_info Device::info() const
{
    vgpu_info i{};
    check(be_->get_info(&i), "GET_INFO");
    return i;
}

Buffer Device::alloc(size_t bytes, uint32_t flags)
{
    vgpu_alloc a{};
    a.size = bytes;
    a.flags = flags;
    check(be_->alloc(&a), "ALLOC");

    Buffer b;
    b.be_ = be_;
    b.handle_ = a.handle;
    b.va_ = a.va;
    b.bytes_ = a.alloc_bytes;
    int r = be_->map(a.handle, b.bytes_, &b.ptr_);
    if (r < 0) {
        be_->free(a.handle);
        b.be_.reset();
        throw Error("mmap", r);
    }
    return b;
}

Channel Device::channel(uint32_t engine, uint32_t prio, uint32_t ring_entries)
{
    Channel ch;
    ch.be_ = be_;
    ch.entries_ = ring_entries;
    ch.ring_ = alloc((size_t)ring_entries * sizeof(vgpu_cmd));

    vgpu_chan_create c{};
    c.engine = engine;
    c.priority = prio;
    c.ring_va = ch.ring_.va();
    c.ring_entries = ring_entries;
    check(be_->chan_create(&c), "CHAN_CREATE");
    ch.id_ = c.chan_id;

    // The control page is a kernel-owned VRAM object; map it by handle.
    void *p = nullptr;
    int r = be_->map(c.ctrl_handle, VGPU_PAGE_SIZE, &p);
    if (r < 0) {
        be_->chan_destroy(c.chan_id);
        ch.id_ = 0;
        throw Error("mmap ctrl", r);
    }
    ch.ctrl_buf_.be_ = be_;
    ch.ctrl_buf_.handle_ = c.ctrl_handle;
    ch.ctrl_buf_.ptr_ = p;
    ch.ctrl_buf_.bytes_ = VGPU_PAGE_SIZE;
    ch.ctrl_buf_.va_ = c.ctrl_va;
    ch.ctrl_buf_.free_on_reset_ = false;   // kernel-owned object
    ch.ctrl_ = static_cast<vgpu_chan_ctrl *>(p);
    return ch;
}

bool Device::wait_sem(uint64_t va, uint32_t value, uint32_t timeout_ms)
{
    vgpu_wait_sem w{};
    w.va = va;
    w.value = value;
    w.timeout_ms = timeout_ms;
    int r = be_->wait_sem(&w);
    if (r == -ETIMEDOUT)
        return false;
    check(r, "WAIT_SEM");
    return true;
}

std::vector<vgpu_trace_ev> Device::read_trace(uint64_t *cursor, uint64_t *dropped,
                                              uint32_t max_events)
{
    std::vector<vgpu_trace_ev> ev(max_events);
    vgpu_trace_req req{};
    req.max_events = max_events;
    req.from_seq = *cursor;
    check(be_->get_trace(&req, ev.data()), "GET_TRACE");
    ev.resize(req.n_events);
    *cursor = req.next_seq;
    if (dropped)
        *dropped = req.dropped;
    return ev;
}

// ---- Buffer ---------------------------------------------------------------

Buffer &Buffer::operator=(Buffer &&o) noexcept
{
    if (this != &o) {
        reset();
        be_ = std::move(o.be_);
        ptr_ = o.ptr_;
        va_ = o.va_;
        bytes_ = o.bytes_;
        handle_ = o.handle_;
        free_on_reset_ = o.free_on_reset_;
        o.ptr_ = nullptr;
        o.handle_ = 0;
        o.free_on_reset_ = true;
    }
    return *this;
}

void Buffer::reset()
{
    if (!be_)
        return;
    if (ptr_)
        be_->unmap(handle_, ptr_, bytes_);
    if (handle_ && free_on_reset_)
        be_->free(handle_);
    be_.reset();
    ptr_ = nullptr;
    handle_ = 0;
    free_on_reset_ = true;
}

// ---- Channel --------------------------------------------------------------

Channel &Channel::operator=(Channel &&o) noexcept
{
    if (this != &o) {
        reset();
        be_ = std::move(o.be_);
        ring_ = std::move(o.ring_);
        ctrl_buf_ = std::move(o.ctrl_buf_);
        ctrl_ = o.ctrl_;
        id_ = o.id_;
        entries_ = o.entries_;
        local_put_ = o.local_put_;
        o.id_ = 0;
        o.ctrl_ = nullptr;
    }
    return *this;
}

void Channel::reset()
{
    if (!be_ || !id_)
        return;
    ctrl_buf_.reset();          // unmap only; CHAN_DESTROY frees the page
    be_->chan_destroy(id_);
    id_ = 0;
    ctrl_ = nullptr;
    ring_.reset();
    be_.reset();
}

void Channel::push(const vgpu_cmd &cmd)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
        uint32_t get = __atomic_load_n(&ctrl_->get, __ATOMIC_ACQUIRE);
        if (local_put_ - get < entries_)
            break;
        if (__atomic_load_n(&ctrl_->error, __ATOMIC_ACQUIRE))
            throw Error("channel faulted", -EIO);
        if (std::chrono::steady_clock::now() > deadline)
            throw Error("ring full", -ETIMEDOUT);
        // Ring full: make sure the engine has been told, then yield.
        submit();
        sched_yield();
    }
    auto *slots = ring_.as<vgpu_cmd>();
    slots[local_put_ & (entries_ - 1)] = cmd;
    local_put_++;
}

Channel &Channel::raw(const vgpu_cmd &cmd) { push(cmd); return *this; }

Channel &Channel::nop()
{
    vgpu_cmd c{};
    c.op = VGPU_OP_NOP;
    return raw(c);
}

Channel &Channel::memset(uint64_t dst, uint64_t len, uint8_t v)
{
    vgpu_cmd c{};
    c.op = VGPU_OP_MEMSET;
    c.arg0 = dst;
    c.arg1 = len;
    c.arg2 = v;
    return raw(c);
}

Channel &Channel::memcpy(uint64_t dst, uint64_t src, uint64_t len)
{
    vgpu_cmd c{};
    c.op = VGPU_OP_MEMCPY;
    c.arg0 = dst;
    c.arg1 = src;
    c.arg2 = len;
    return raw(c);
}

Channel &Channel::vec_add(uint64_t dst, uint64_t a, uint64_t b, uint32_t n)
{
    vgpu_cmd c{};
    c.op = VGPU_OP_VEC_ADD;
    c.count = n;
    c.arg0 = dst;
    c.arg1 = a;
    c.arg2 = b;
    return raw(c);
}

Channel &Channel::vec_mul(uint64_t dst, uint64_t a, uint64_t b, uint32_t n)
{
    vgpu_cmd c{};
    c.op = VGPU_OP_VEC_MUL;
    c.count = n;
    c.arg0 = dst;
    c.arg1 = a;
    c.arg2 = b;
    return raw(c);
}

Channel &Channel::fence(uint64_t sem_va, uint32_t value)
{
    vgpu_cmd c{};
    c.op = VGPU_OP_FENCE;
    c.arg0 = sem_va;
    c.arg1 = value;
    return raw(c);
}

Channel &Channel::wait_sem(uint64_t sem_va, uint32_t value)
{
    vgpu_cmd c{};
    c.op = VGPU_OP_WAIT_SEM;
    c.arg0 = sem_va;
    c.arg1 = value;
    return raw(c);
}

void Channel::submit()
{
    __atomic_store_n(&ctrl_->put, local_put_, __ATOMIC_RELEASE);
    int r = be_->doorbell(id_);
    if (r < 0 && r != -EIO)
        throw Error("DOORBELL", r);
}

bool Channel::idle() const
{
    return __atomic_load_n(&ctrl_->get, __ATOMIC_ACQUIRE) == local_put_;
}

uint32_t Channel::error() const
{
    return __atomic_load_n(&ctrl_->error, __ATOMIC_ACQUIRE);
}

void Channel::wait_idle(uint32_t timeout_ms)
{
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);
    while (!idle()) {
        if (error())
            throw Error("channel faulted (vgpu_fault " + std::to_string(error()) + ")",
                        -EIO);
        if (std::chrono::steady_clock::now() > deadline)
            throw Error("wait_idle", -ETIMEDOUT);
        sched_yield();
    }
}

vgpu_chan_stats Channel::stats() const
{
    vgpu_chan_stats s{};
    s.chan_id = id_;
    check(be_->chan_stats(&s), "CHAN_STATS");
    return s;
}

}  // namespace vrm
