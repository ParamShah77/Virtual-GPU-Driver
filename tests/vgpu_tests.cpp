// Backend-agnostic test suite: the same tests run against the in-process
// simulator and against the real kernel module (/dev/vgpu0).
//
//   vgpu_tests [--backend sim|kernel] [--dev /dev/vgpu0] [substring-filter]

#include <atomic>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "sim.hpp"
#include "vrm.hpp"

using namespace vrm;

// ---- mini test framework ---------------------------------------------------

static int g_failed_checks = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "    CHECK failed: %s (%s:%d)\n", #cond,      \
                         __FILE__, __LINE__);                                  \
            g_failed_checks++;                                                 \
        }                                                                      \
    } while (0)

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond))                                                           \
            throw std::runtime_error(std::string("REQUIRE failed: ") + #cond + \
                                     " (" __FILE__ ":" + std::to_string(__LINE__) + ")"); \
    } while (0)

// Expect an Error with the given negative errno.
#define CHECK_ERR(expr, errno_val)                                             \
    do {                                                                       \
        bool threw_ = false;                                                   \
        try { (void)(expr); } catch (const Error &e_) {                        \
            threw_ = true;                                                     \
            if (e_.code() != (errno_val)) {                                    \
                std::fprintf(stderr, "    wrong errno %d (want %d) at %s:%d\n", \
                             e_.code(), (int)(errno_val), __FILE__, __LINE__); \
                g_failed_checks++;                                             \
            }                                                                  \
        }                                                                      \
        if (!threw_) {                                                         \
            std::fprintf(stderr, "    expected error not thrown: %s (%s:%d)\n", \
                         #expr, __FILE__, __LINE__);                           \
            g_failed_checks++;                                                 \
        }                                                                      \
    } while (0)

struct Env {
    std::string name;
    std::function<Device()> new_device;   // a fresh client context
};

struct TestCase {
    const char *name;
    std::function<void(Env &)> fn;
};

static std::vector<TestCase> &registry()
{
    static std::vector<TestCase> r;
    return r;
}

struct Reg {
    Reg(const char *n, std::function<void(Env &)> f) { registry().push_back({n, std::move(f)}); }
};
#define TEST(name) \
    static void name(Env &env); \
    static Reg reg_##name(#name, name); \
    static void name(Env &env)

// ---- helpers -------------------------------------------------------------

static const uint32_t COPY = VGPU_ENGINE_COPY;
static const uint32_t COMPUTE = VGPU_ENGINE_COMPUTE;

// Append a fence, publish, and wait for the CPU-visible semaphore.
static void sync(Device &d, Channel &ch, Buffer &sem, uint32_t value)
{
    ch.fence(sem.va(), value).submit();
    REQUIRE(d.wait_sem(sem.va(), value, 10000));
}

static void fill_pattern(void *p, size_t n, uint32_t seed)
{
    auto *b = static_cast<uint8_t *>(p);
    for (size_t i = 0; i < n; i++)
        b[i] = (uint8_t)((i * 131 + seed) ^ (i >> 8));
}

static bool all_bytes(const void *p, size_t n, uint8_t v)
{
    auto *b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; i++)
        if (b[i] != v)
            return false;
    return true;
}

// ---- memory ----------------------------------------------------------------

TEST(alloc_basic)
{
    Device d = env.new_device();
    vgpu_info i0 = d.info();
    Buffer a = d.alloc(10000);
    CHECK(a.size() == 3 * VGPU_PAGE_SIZE);
    CHECK(all_bytes(a.data(), a.size(), 0));
    vgpu_info i1 = d.info();
    CHECK(i0.vram_free_bytes - i1.vram_free_bytes == a.size());

    Buffer b = d.alloc(4096);
    // Allocations are separated by an unmapped guard page.
    bool apart = b.va() >= a.va() + a.size() + VGPU_PAGE_SIZE ||
                 b.va() + b.size() + VGPU_PAGE_SIZE <= a.va();
    CHECK(apart);

    a.reset();
    b.reset();
    CHECK(d.info().vram_free_bytes == i0.vram_free_bytes);
}

TEST(alloc_exhaustion_and_fragmentation)
{
    Device d = env.new_device();
    vgpu_info i0 = d.info();
    const size_t MiB = 1 << 20;
    size_t n = i0.vram_free_bytes / MiB;
    REQUIRE(n >= 8);

    CHECK_ERR(d.alloc(i0.vram_bytes + 1), -EINVAL);
    CHECK_ERR(d.alloc(0), -EINVAL);

    std::vector<Buffer> bufs;
    for (size_t i = 0; i < n; i++)
        bufs.push_back(d.alloc(MiB));
    CHECK_ERR(d.alloc(MiB), -ENOMEM);

    // Free every other megabyte: lots of free memory, but no 2 MiB hole.
    for (size_t i = 0; i < n; i += 2)
        bufs[i].reset();
    vgpu_info frag = d.info();
    CHECK(frag.vram_free_bytes >= (n / 2) * MiB);
    CHECK(frag.vram_largest_free_bytes == MiB);
    CHECK_ERR(d.alloc(2 * MiB), -ENOMEM);

    bufs.clear();
    Buffer big = d.alloc(2 * MiB);   // coalesced again
    CHECK(big);
}

TEST(mapped_buffer_cannot_be_freed)
{
    Device d = env.new_device();
    Buffer b = d.alloc(4096);
    CHECK(d.backend()->free(b.handle()) == -EBUSY);
}

TEST(freed_memory_is_scrubbed)
{
    Device a = env.new_device();
    Device b = env.new_device();
    const size_t sz = 1 << 20;
    {
        Buffer s = a.alloc(sz);
        std::memset(s.data(), 0x5A, sz);
    }
    Buffer t = b.alloc(sz);
    CHECK(all_bytes(t.data(), sz, 0));
}

TEST(context_teardown_returns_vram)
{
    Device probe = env.new_device();
    uint64_t before = probe.info().vram_free_bytes;
    {
        Device d = env.new_device();
        Buffer b = d.alloc(2 << 20);
        Channel c = d.channel(COPY);
        c.nop().submit();
        c.wait_idle();
    }
    CHECK(probe.info().vram_free_bytes == before);
}

// ---- engines -------------------------------------------------------------

TEST(copy_engine_memset_memcpy)
{
    Device d = env.new_device();
    const size_t N = 1 << 20;
    Buffer src = d.alloc(N), dst = d.alloc(N), sem = d.alloc(4096);
    fill_pattern(src.data(), N, 7);

    Channel ch = d.channel(COPY);
    ch.memcpy(dst.va(), src.va(), N).memset(dst.va() + 4096, 4096, 0xAB);
    sync(d, ch, sem, 1);

    auto *s = src.as<uint8_t>();
    auto *t = dst.as<uint8_t>();
    CHECK(std::memcmp(t, s, 4096) == 0);
    CHECK(all_bytes(t + 4096, 4096, 0xAB));
    CHECK(std::memcmp(t + 8192, s + 8192, N - 8192) == 0);
}

TEST(compute_engine_vector_ops)
{
    Device d = env.new_device();
    const uint32_t n = 100000;
    Buffer a = d.alloc(n * 4), b = d.alloc(n * 4), sum = d.alloc(n * 4),
           prod = d.alloc(n * 4), sem = d.alloc(4096);
    for (uint32_t i = 0; i < n; i++) {
        a.as<uint32_t>()[i] = i;
        b.as<uint32_t>()[i] = 3 * i + 1;
    }
    Channel ch = d.channel(COMPUTE);
    ch.vec_add(sum.va(), a.va(), b.va(), n).vec_mul(prod.va(), a.va(), b.va(), n);
    sync(d, ch, sem, 1);

    bool ok = true;
    for (uint32_t i = 0; i < n; i++) {
        ok &= sum.as<uint32_t>()[i] == i + (3 * i + 1);
        ok &= prod.as<uint32_t>()[i] == i * (3 * i + 1);
    }
    CHECK(ok);
}

TEST(ring_wraparound_and_backpressure)
{
    Device d = env.new_device();
    Channel ch = d.channel(COPY, VGPU_PRIO_NORMAL, 8);   // tiny ring
    for (int i = 0; i < 1000; i++) {
        ch.nop();
        if (i % 5 == 4)
            ch.submit();
    }
    ch.submit();
    ch.wait_idle();
    CHECK(ch.stats().cmds_executed == 1000);
    CHECK(ch.error() == 0);
}

TEST(cross_channel_semaphore_dependency)
{
    Device d = env.new_device();
    const size_t N = 4096;
    Buffer buf = d.alloc(N), out = d.alloc(N), gate = d.alloc(4096),
           done = d.alloc(4096);
    Channel producer = d.channel(COPY);
    Channel consumer = d.channel(COMPUTE);

    // The consumer parks on the gate; nothing may run until the producer fences.
    consumer.wait_sem(gate.va(), 1)
        .vec_add(out.va(), buf.va(), buf.va(), N / 4)
        .fence(done.va(), 1)
        .submit();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(!consumer.idle());
    CHECK(all_bytes(out.data(), N, 0));
    CHECK(!d.wait_sem(done.va(), 1, 20));

    producer.memset(buf.va(), N, 0x01).fence(gate.va(), 1).submit();
    REQUIRE(d.wait_sem(done.va(), 1, 10000));
    for (size_t i = 0; i < N / 4; i++)
        REQUIRE(out.as<uint32_t>()[i] == 0x02020202u);
}

TEST(cpu_can_release_a_gate)
{
    Device d = env.new_device();
    Buffer gate = d.alloc(4096), buf = d.alloc(4096), done = d.alloc(4096);
    Channel ch = d.channel(COPY);
    ch.wait_sem(gate.va(), 5).memset(buf.va(), 4096, 9).fence(done.va(), 1).submit();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(all_bytes(buf.data(), 4096, 0));
    __atomic_store_n(gate.as<uint32_t>(), 5u, __ATOMIC_RELEASE);
    CHECK(d.wait_sem(done.va(), 1, 5000));
    CHECK(all_bytes(buf.data(), 4096, 9));
}

// ---- faults and isolation ----------------------------------------------------

static void expect_fault(Device &d, Channel &ch, uint32_t code)
{
    ch.submit();
    try {
        ch.wait_idle(5000);
    } catch (const Error &e) {
        CHECK(e.code() == -EIO);
    }
    CHECK(ch.error() == code);
    CHECK(ch.stats().state == 2);
    (void)d;
}

TEST(fault_unmapped_address)
{
    Device d = env.new_device();
    Channel ch = d.channel(COPY);
    ch.memset(0x7777000, 64, 1);
    expect_fault(d, ch, VGPU_FAULT_UNMAPPED);
}

TEST(fault_guard_page_overrun)
{
    Device d = env.new_device();
    Buffer b = d.alloc(4096);
    Channel ch = d.channel(COPY);
    ch.memset(b.va() + 4000, 200, 1);       // runs into the guard page
    expect_fault(d, ch, VGPU_FAULT_UNMAPPED);
    CHECK(all_bytes(b.data(), 4096, 0));
}

TEST(fault_write_to_readonly_mapping)
{
    Device d = env.new_device();
    Buffer ro = d.alloc(4096, VGPU_MEM_GPU_READONLY);
    Buffer rw = d.alloc(4096);
    Channel ch = d.channel(COPY);
    ch.memcpy(rw.va(), ro.va(), 4096);       // reading it is fine
    ch.memset(ro.va(), 4096, 1);             // writing it is not
    expect_fault(d, ch, VGPU_FAULT_READONLY);
    CHECK(ch.stats().cmds_executed == 1);
}

TEST(fault_wrong_engine_and_bad_op)
{
    Device d = env.new_device();
    Buffer b = d.alloc(4096);
    {
        Channel ch = d.channel(COMPUTE);
        ch.memset(b.va(), 64, 1);
        expect_fault(d, ch, VGPU_FAULT_WRONG_ENGINE);
    }
    {
        Channel ch = d.channel(COPY);
        vgpu_cmd c{};
        c.op = 99;
        ch.raw(c);
        expect_fault(d, ch, VGPU_FAULT_BAD_OP);
    }
    {
        Channel ch = d.channel(COMPUTE);
        ch.vec_add(b.va() + 2, b.va(), b.va(), 4);   // misaligned
        expect_fault(d, ch, VGPU_FAULT_BAD_ARG);
    }
}

TEST(fault_does_not_wedge_other_channels)
{
    Device d = env.new_device();
    Device other = env.new_device();
    Buffer sem = d.alloc(4096);
    Buffer obuf = other.alloc(4096), osem = other.alloc(4096);

    Channel bad = d.channel(COPY);
    bad.memset(0x7777000, 64, 1).submit();

    // A healthy client sharing the engine keeps running, repeatedly.
    Channel og = other.channel(COPY);
    for (uint8_t v = 1; v <= 20; v++) {
        og.memset(obuf.va(), 4096, v);
        sync(other, og, osem, v);
        CHECK(all_bytes(obuf.data(), 4096, v));
    }

    // A CPU wait that can never complete reports the faulted channel.
    CHECK_ERR(d.wait_sem(sem.va(), 99, 2000), -EIO);
}

TEST(context_isolation)
{
    Device a = env.new_device();
    Device b = env.new_device();
    Buffer pad1 = a.alloc(1 << 20), pad2 = a.alloc(1 << 20);
    Buffer secret = a.alloc(4096);
    std::memset(secret.data(), 0x5A, 4096);

    // b has nothing mapped at a's secret address.
    Buffer mine = b.alloc(4096);
    Channel ch = b.channel(COPY);
    ch.memcpy(mine.va(), secret.va(), 4096);
    expect_fault(b, ch, VGPU_FAULT_UNMAPPED);
    CHECK(all_bytes(mine.data(), 4096, 0));
}

// ---- scheduler ---------------------------------------------------------------

struct Share { int a; int b; int n; };

// Two clients, one engine, identical streams of 256 KiB memsets released at the
// same moment. Returns how many commands each had executed when the first one
// finished, reconstructed from the trace (deterministic: slices are measured
// in modelled time, not wall time).
static Share contend(Env &env, uint32_t prio_a, uint32_t prio_b, int n)
{
    Device da = env.new_device(), db = env.new_device();
    Buffer ba = da.alloc(256 << 10), bb = db.alloc(256 << 10);
    Buffer ga = da.alloc(4096), gb = db.alloc(4096);
    Channel ca = da.channel(COPY, prio_a, 2048);
    Channel cb = db.channel(COPY, prio_b, 2048);
    ca.wait_sem(ga.va(), 1);
    cb.wait_sem(gb.va(), 1);
    for (int i = 0; i < n; i++) {
        ca.memset(ba.va(), 256 << 10, 1);
        cb.memset(bb.va(), 256 << 10, 1);
    }
    uint64_t cursor = 0;
    da.read_trace(&cursor);                 // discard setup noise
    ca.submit();
    cb.submit();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    __atomic_store_n(ga.as<uint32_t>(), 1u, __ATOMIC_RELEASE);
    __atomic_store_n(gb.as<uint32_t>(), 1u, __ATOMIC_RELEASE);
    ca.wait_idle(30000);
    cb.wait_idle(30000);

    uint64_t dropped = 0;
    auto ev = da.read_trace(&cursor, &dropped, 4096);
    REQUIRE(dropped == 0);
    Share s{0, 0, n};
    uint32_t ida = ca.id(), idb = cb.id();
    // +1 for the WAIT_SEM command each channel executes first.
    for (auto &e : ev) {
        if (e.type != VGPU_EV_CMD)
            continue;
        if (e.chan == ida) s.a++;
        if (e.chan == idb) s.b++;
        if (s.a >= n + 1 || s.b >= n + 1)
            break;
    }
    return s;
}

TEST(scheduler_equal_priority_shares_engine)
{
    Share s = contend(env, VGPU_PRIO_NORMAL, VGPU_PRIO_NORMAL, 1200);
    std::fprintf(stderr, "    equal prio: a=%d b=%d\n", s.a, s.b);
    int lo = std::min(s.a, s.b), hi = std::max(s.a, s.b);
    CHECK(hi >= s.n);
    CHECK(lo >= s.n - 2 * 62);      // at most ~2 slices behind
}

TEST(scheduler_priority_scales_timeslice)
{
    Share s = contend(env, VGPU_PRIO_HIGH, VGPU_PRIO_LOW, 1200);
    std::fprintf(stderr, "    high=%d low=%d\n", s.a, s.b);
    CHECK(s.a >= s.n);
    // HIGH slices are 4x LOW slices, so LOW gets ~1/4 of HIGH's throughput.
    CHECK(s.b > s.n / 5 && s.b < s.n / 3);
}

TEST(trace_records_scheduling_events)
{
    Device d = env.new_device();
    uint64_t cursor = 0;
    d.read_trace(&cursor);
    Channel ch = d.channel(COPY);
    ch.nop().nop().submit();
    ch.wait_idle();
    auto ev = d.read_trace(&cursor);
    int doorbells = 0, cmds = 0, begins = 0;
    for (auto &e : ev) {
        doorbells += e.type == VGPU_EV_DOORBELL;
        cmds += e.type == VGPU_EV_CMD;
        begins += e.type == VGPU_EV_SLICE_BEGIN;
    }
    CHECK(doorbells >= 1);
    CHECK(cmds == 2);
    CHECK(begins >= 1);
    uint64_t c2 = cursor;
    CHECK(d.read_trace(&c2).empty());
}

// ---- concurrency ---------------------------------------------------------------

TEST(concurrent_clients_stress)
{
    Device probe = env.new_device();
    uint64_t free0 = probe.info().vram_free_bytes;
    std::atomic<int> failures{0};

    auto worker = [&](int tid) {
        try {
            Device d = env.new_device();
            Channel cp = d.channel(COPY, tid % 3);
            Channel cc = d.channel(COMPUTE, (tid + 1) % 3);
            Buffer sem = d.alloc(4096);
            std::mt19937 rng(tid * 7919 + 1);
            uint32_t fence = 0;
            for (int it = 0; it < 150; it++) {
                size_t sz = 4096 * (1 + rng() % 64);
                Buffer a = d.alloc(sz), b = d.alloc(sz), c = d.alloc(sz);
                fill_pattern(a.data(), sz, it + tid);
                cp.memcpy(b.va(), a.va(), sz);
                sync(d, cp, sem, ++fence);
                if (std::memcmp(a.data(), b.data(), sz) != 0)
                    failures++;
                uint32_t n = (uint32_t)(sz / 4);
                cc.vec_add(c.va(), a.va(), b.va(), n);
                sync(d, cc, sem, ++fence);
                for (uint32_t i = 0; i < n; i += 97)
                    if (c.as<uint32_t>()[i] != 2 * a.as<uint32_t>()[i])
                        failures++;
            }
        } catch (const std::exception &e) {
            std::fprintf(stderr, "    worker %d: %s\n", tid, e.what());
            failures++;
        }
    };
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; t++)
        ts.emplace_back(worker, t);
    for (auto &t : ts)
        t.join();
    CHECK(failures == 0);
    CHECK(probe.info().vram_free_bytes == free0);
}

// ---- driver -------------------------------------------------------------------

int main(int argc, char **argv)
{
    std::string backend = "sim", dev = "/dev/vgpu0", filter;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) backend = argv[++i];
        else if (a == "--dev" && i + 1 < argc) dev = argv[++i];
        else filter = a;
    }

    int ran = 0, failed = 0;
    for (auto &t : registry()) {
        if (!filter.empty() && std::string(t.name).find(filter) == std::string::npos)
            continue;
        std::fprintf(stderr, "[ RUN  ] %s\n", t.name);
        g_failed_checks = 0;
        bool ok = true;
        try {
            Env env;
            std::shared_ptr<SimGpu> gpu;
            if (backend == "sim") {
                gpu = std::make_shared<SimGpu>();
                env.name = "sim";
                env.new_device = [gpu] { return open_sim(gpu); };
            } else {
                env.name = "kernel";
                env.new_device = [dev] { return Device::open_kernel(dev); };
            }
            t.fn(env);
        } catch (const std::exception &e) {
            std::fprintf(stderr, "    exception: %s\n", e.what());
            ok = false;
        }
        ok = ok && g_failed_checks == 0;
        std::fprintf(stderr, "[ %s ] %s\n", ok ? " OK " : "FAIL", t.name);
        ran++;
        failed += !ok;
    }
    std::fprintf(stderr, "\n%d tests, %d failed (backend: %s)\n", ran, failed,
                 backend.c_str());
    return failed ? 1 : 0;
}
