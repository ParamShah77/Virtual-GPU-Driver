// libvrm: user-mode runtime for the virtual GPU.
//
//   Device  -> one client context (one open fd on /dev/vgpu0, or one context
//              on a simulated GPU)
//   Buffer  -> VRAM allocation mapped into this process and into the context's
//              GPU virtual address space
//   Channel -> a command ring bound to an engine; commands are written
//              straight into mapped memory, published with a release store to
//              'put', then a doorbell.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vgpu_uapi.h"

namespace vrm {

class Error : public std::runtime_error {
public:
    Error(const std::string &what, int err)
        : std::runtime_error(what + ": " + std::to_string(err)), err_(err) {}
    int code() const { return err_; }          // negative errno
private:
    int err_;
};

// Transport to a GPU. Every method returns 0 or a negative errno. The kernel
// backend maps these 1:1 onto ioctl/mmap; the simulator calls the core directly.
class Backend {
public:
    virtual ~Backend() = default;
    virtual int get_info(vgpu_info *out) = 0;
    virtual int alloc(vgpu_alloc *a) = 0;
    virtual int free(uint32_t handle) = 0;
    virtual int map(uint32_t handle, size_t bytes, void **out) = 0;
    virtual int unmap(uint32_t handle, void *p, size_t bytes) = 0;
    virtual int chan_create(vgpu_chan_create *c) = 0;
    virtual int chan_destroy(uint32_t chan_id) = 0;
    virtual int doorbell(uint32_t chan_id) = 0;
    virtual int wait_sem(vgpu_wait_sem *w) = 0;
    virtual int chan_stats(vgpu_chan_stats *s) = 0;
    virtual int get_trace(vgpu_trace_req *req, vgpu_trace_ev *buf) = 0;
};

class Device;

class Buffer {
public:
    Buffer() = default;
    Buffer(Buffer &&o) noexcept { *this = std::move(o); }
    Buffer &operator=(Buffer &&o) noexcept;
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    ~Buffer() { reset(); }

    void *data() const { return ptr_; }
    template <class T> T *as() const { return static_cast<T *>(ptr_); }
    uint64_t va() const { return va_; }
    uint64_t va_at(size_t off) const { return va_ + off; }
    size_t size() const { return bytes_; }       // page-rounded
    uint32_t handle() const { return handle_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    void reset();

private:
    friend class Device;
    friend class Channel;
    std::shared_ptr<Backend> be_;
    void *ptr_ = nullptr;
    uint64_t va_ = 0;
    size_t bytes_ = 0;
    uint32_t handle_ = 0;
    bool free_on_reset_ = true;
};

class Channel {
public:
    Channel(Channel &&o) noexcept { *this = std::move(o); }
    Channel &operator=(Channel &&o) noexcept;
    Channel(const Channel &) = delete;
    Channel &operator=(const Channel &) = delete;
    ~Channel() { reset(); }

    uint32_t id() const { return id_; }

    // Command builders. They only fill ring entries; submit() publishes them.
    Channel &nop();
    Channel &memset(uint64_t dst, uint64_t len, uint8_t value);
    Channel &memcpy(uint64_t dst, uint64_t src, uint64_t len);
    Channel &vec_add(uint64_t dst, uint64_t a, uint64_t b, uint32_t n);
    Channel &vec_mul(uint64_t dst, uint64_t a, uint64_t b, uint32_t n);
    Channel &fence(uint64_t sem_va, uint32_t value);
    Channel &wait_sem(uint64_t sem_va, uint32_t value);
    Channel &raw(const vgpu_cmd &cmd);

    // Publish put; ring the doorbell only if the engine asked for one.
    void submit();
    // Benchmark/debug: always take the ioctl path, like a classic doorbell.
    void set_always_doorbell(bool on) { always_doorbell_ = on; }
    bool idle() const;                              // get == put
    void wait_idle(uint32_t timeout_ms = 5000);     // throws on timeout/fault
    uint32_t error() const;                         // enum vgpu_fault
    vgpu_chan_stats stats() const;
    void reset();

private:
    friend class Device;
    Channel() = default;
    void push(const vgpu_cmd &cmd);

    std::shared_ptr<Backend> be_;
    Buffer ring_;
    Buffer ctrl_buf_;
    vgpu_chan_ctrl *ctrl_ = nullptr;
    uint32_t id_ = 0;
    uint32_t entries_ = 0;
    uint32_t local_put_ = 0;
    bool always_doorbell_ = false;
};

class Device {
public:
    explicit Device(std::shared_ptr<Backend> be) : be_(std::move(be)) {}
    static Device open_kernel(const std::string &path = "/dev/vgpu0");

    vgpu_info info() const;
    Buffer alloc(size_t bytes, uint32_t flags = 0);
    Channel channel(uint32_t engine, uint32_t prio = VGPU_PRIO_NORMAL,
                    uint32_t ring_entries = 256);

    // Block until *va >= value (wrapping compare). false on timeout.
    bool wait_sem(uint64_t va, uint32_t value, uint32_t timeout_ms = 5000);

    // Incremental trace read; *cursor is advanced.
    std::vector<vgpu_trace_ev> read_trace(uint64_t *cursor,
                                          uint64_t *dropped = nullptr,
                                          uint32_t max_events = 4096);

    std::shared_ptr<Backend> backend() const { return be_; }

private:
    std::shared_ptr<Backend> be_;
};

}  // namespace vrm
