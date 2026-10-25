#!/usr/bin/env bash
# Load vgpu.ko, run the full test suite against it, run benchmarks, show /proc,
# unload. Run as root inside WSL:  bash scripts/demo.sh   (after wsl-build.sh kmod user)
set -euo pipefail

B="${VGPU_BUILD_DIR:-$HOME/vgpu-build}"
cd "$B"

if lsmod | grep -q '^vgpu '; then rmmod vgpu; fi
insmod kmod/vgpu.ko "$@"
trap 'rmmod vgpu 2>/dev/null || true' EXIT

echo "== dmesg"; dmesg | grep vgpu | tail -3
echo; echo "== kernel-backend tests"
./build/vgpu_tests --backend kernel
echo; echo "== latency";    ./build/vgpu_bench --backend kernel latency
echo; echo "== throughput"; ./build/vgpu_bench --backend kernel throughput
echo; echo "== fairness";   ./build/vgpu_bench --backend kernel --trace "$B/fairness.csv" fairness
echo; echo "== /proc/vgpu"; cat /proc/vgpu
echo; echo "== trace summary"; python3 tools/plot_trace.py "$B/fairness.csv" -o "$B/gantt.png" || true
