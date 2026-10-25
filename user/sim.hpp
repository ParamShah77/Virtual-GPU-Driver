// In-process simulator: runs the same resource-manager core as vgpu.ko, with
// one thread per engine standing in for the GPU's front end. Lets the test
// suite, benchmarks and tooling run without the kernel module.
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "vrm.hpp"

struct vrm_gpu;

namespace vrm {

class SimGpu {
public:
    struct Config {
        uint64_t vram_bytes = 64ull << 20;
        bool pace = false;   // sleep for modelled command cost (wall-clock timing)
        uint64_t poll_ns = 20000;  // spin this long after running dry before sleeping
    };

    explicit SimGpu(const Config &cfg);
    SimGpu() : SimGpu(Config{}) {}
    ~SimGpu();
    SimGpu(const SimGpu &) = delete;
    SimGpu &operator=(const SimGpu &) = delete;

    vrm_gpu *core() { return gpu_.get(); }
    uint8_t *vram() { return vram_; }

    // Event counter shared by engines and CPU waiters (doorbell, fence, fault).
    uint64_t seq();
    void signal();
    // Waits until seq() != seen. timeout_us < 0 waits forever. False on timeout.
    bool wait_event(uint64_t seen, int64_t timeout_us);

private:
    void engine_loop(uint32_t e);

    Config cfg_;
    std::unique_ptr<vrm_gpu> gpu_;
    uint8_t *vram_ = nullptr;
    bool stop_ = false;
    std::mutex mu_;
    std::condition_variable cv_;
    uint64_t seq_ = 0;
    std::thread threads_[VGPU_NUM_ENGINES];
};

// Create a client context (a "process") on a simulated GPU.
Device open_sim(std::shared_ptr<SimGpu> gpu);

}  // namespace vrm
