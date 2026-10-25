// Dump the GPU scheduling trace from /dev/vgpu0 as CSV.
//   vgpu_trace [--dev /dev/vgpu0] [--follow]  > trace.csv
// Plot with tools/plot_trace.py.
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#include "trace_csv.hpp"
#include "vrm.hpp"

int main(int argc, char **argv)
{
    std::string dev = "/dev/vgpu0";
    bool follow = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--dev" && i + 1 < argc) dev = argv[++i];
        else if (a == "--follow") follow = true;
        else {
            std::fprintf(stderr, "usage: %s [--dev PATH] [--follow]\n", argv[0]);
            return 2;
        }
    }
    try {
        vrm::Device d = vrm::Device::open_kernel(dev);
        uint64_t cursor = 0, dropped = 0;
        std::puts(vrm::trace_csv_header());
        do {
            auto ev = d.read_trace(&cursor, &dropped);
            for (auto &e : ev)
                vrm::trace_csv_row(stdout, e);
            std::fflush(stdout);
            if (dropped)
                std::fprintf(stderr, "warning: %llu events dropped\n",
                             (unsigned long long)dropped);
            if (follow)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } while (follow);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
