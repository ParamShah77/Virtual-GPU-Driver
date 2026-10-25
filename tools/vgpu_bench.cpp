// Benchmarks and scheduler experiments.
//   vgpu_bench [--backend sim|kernel] [--dev PATH] [--trace out.csv] <scenario>
// scenarios: latency | throughput | fairness
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "sim.hpp"
#include "trace_csv.hpp"
#include "vrm.hpp"

using namespace vrm;
using Clock = std::chrono::steady_clock;

static double us_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

static void latency(const std::function<Device()> &mk)
{
    Device d = mk();
    Buffer sem = d.alloc(4096);
    Channel ch = d.channel(VGPU_ENGINE_COPY);
    const int N = 3000;
    std::vector<double> us;
    for (int i = 1; i <= N; i++) {
        auto t0 = Clock::now();
        ch.fence(sem.va(), i).submit();
        while ((int32_t)(__atomic_load_n(sem.as<uint32_t>(), __ATOMIC_ACQUIRE) - i) < 0)
            ;
        us.push_back(us_since(t0));
    }
    std::sort(us.begin(), us.end());
    std::printf("submit->fence round trip, %d samples (us): p50=%.1f p90=%.1f "
                "p99=%.1f max=%.1f\n",
                N, us[N / 2], us[N * 9 / 10], us[N * 99 / 100], us.back());
}

static void throughput(const std::function<Device()> &mk)
{
    Device d = mk();
    const size_t chunk = 8 << 20;
    Buffer a = d.alloc(chunk), b = d.alloc(chunk), sem = d.alloc(4096);
    Channel ch = d.channel(VGPU_ENGINE_COPY, VGPU_PRIO_NORMAL, 64);
    const int iters = 200;
    vgpu_info i0 = d.info();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++)
        ch.memcpy(b.va(), a.va(), chunk);
    ch.fence(sem.va(), 1).submit();
    d.wait_sem(sem.va(), 1, 60000);
    double sec = us_since(t0) / 1e6;
    vgpu_info i1 = d.info();
    double gb = (double)chunk * iters / 1e9;
    double modelled_s = (i1.eng[0].busy_ns - i0.eng[0].busy_ns) / 1e9;
    std::printf("memcpy %d x %zu MiB: wall %.3f s (%.1f GB/s); modelled engine "
                "time %.3f s (%.1f GB/s)\n",
                iters, chunk >> 20, sec, gb / sec, modelled_s, gb / modelled_s);
}

// Three clients (LOW/NORMAL/HIGH) contend for the copy engine with identical
// work. Shares are computed from the trace up to the moment the first client
// finishes, in modelled time.
static void fairness(const std::function<Device()> &mk, const std::string &trace_path)
{
    const int n = 800;
    const uint32_t prios[3] = {VGPU_PRIO_LOW, VGPU_PRIO_NORMAL, VGPU_PRIO_HIGH};
    const char *names[3] = {"low", "normal", "high"};

    std::vector<Device> devs;
    std::vector<Buffer> bufs, gates;
    std::vector<Channel> chans;
    for (int i = 0; i < 3; i++) {
        devs.push_back(mk());
        bufs.push_back(devs[i].alloc(256 << 10));
        gates.push_back(devs[i].alloc(4096));
        chans.push_back(devs[i].channel(VGPU_ENGINE_COPY, prios[i], 1024));
    }
    uint64_t cursor = 0;
    devs[0].read_trace(&cursor);
    for (int i = 0; i < 3; i++) {
        chans[i].wait_sem(gates[i].va(), 1);
        for (int k = 0; k < n; k++)
            chans[i].memset(bufs[i].va(), 256 << 10, 1);
        chans[i].submit();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    for (int i = 0; i < 3; i++)
        __atomic_store_n(gates[i].as<uint32_t>(), 1u, __ATOMIC_RELEASE);
    for (auto &c : chans)
        c.wait_idle(60000);

    uint64_t dropped = 0;
    auto ev = devs[0].read_trace(&cursor, &dropped, 4096);
    if (!trace_path.empty()) {
        FILE *f = std::fopen(trace_path.c_str(), "w");
        if (f) {
            std::fprintf(f, "%s\n", trace_csv_header());
            for (auto &e : ev)
                trace_csv_row(f, e);
            std::fclose(f);
            std::printf("wrote %zu trace events to %s\n", ev.size(), trace_path.c_str());
        }
    }

    uint64_t cmds[3] = {0, 0, 0}, ns[3] = {0, 0, 0}, total = 0;
    for (auto &e : ev) {
        if (e.type != VGPU_EV_CMD)
            continue;
        for (int i = 0; i < 3; i++)
            if (e.chan == chans[i].id()) {
                cmds[i]++;
                ns[i] += e.arg2;
                total += e.arg2;
            }
        if (cmds[0] > (uint64_t)n || cmds[1] > (uint64_t)n || cmds[2] > (uint64_t)n)
            break;
    }
    std::printf("\nshare of copy-engine time until the first client finished:\n");
    std::printf("  %-8s %8s %12s %8s\n", "prio", "cmds", "modelled ms", "share");
    for (int i = 0; i < 3; i++)
        std::printf("  %-8s %8llu %12.2f %7.1f%%\n", names[i],
                    (unsigned long long)cmds[i], ns[i] / 1e6,
                    total ? 100.0 * ns[i] / total : 0.0);
    if (dropped)
        std::printf("note: %llu trace events were dropped\n", (unsigned long long)dropped);
}

int main(int argc, char **argv)
{
    std::string backend = "sim", dev = "/dev/vgpu0", trace, scenario;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) backend = argv[++i];
        else if (a == "--dev" && i + 1 < argc) dev = argv[++i];
        else if (a == "--trace" && i + 1 < argc) trace = argv[++i];
        else scenario = a;
    }
    if (scenario.empty()) {
        std::fprintf(stderr,
                     "usage: %s [--backend sim|kernel] [--dev PATH] [--trace out.csv] "
                     "latency|throughput|fairness\n", argv[0]);
        return 2;
    }
    try {
        std::shared_ptr<SimGpu> gpu;
        std::function<Device()> mk;
        if (backend == "sim") {
            gpu = std::make_shared<SimGpu>();
            mk = [gpu] { return open_sim(gpu); };
        } else {
            mk = [dev] { return Device::open_kernel(dev); };
        }
        if (scenario == "latency") latency(mk);
        else if (scenario == "throughput") throughput(mk);
        else if (scenario == "fairness") fairness(mk, trace);
        else {
            std::fprintf(stderr, "unknown scenario '%s'\n", scenario.c_str());
            return 2;
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
