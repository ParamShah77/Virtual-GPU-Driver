#!/usr/bin/env bash
# Sync the project from the Windows drive into the WSL ext4 filesystem (fast,
# and kernel-module builds don't like drvfs) and build there.
#
#   scripts/wsl-build.sh [user|test|asan|tsan|kmod|all]
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${VGPU_BUILD_DIR:-$HOME/vgpu-build}"
KSRC="${KSRC:-/opt/wsl-kernel/src}"
what="${1:-all}"

mkdir -p "$DST"
rsync -rlp --checksum --no-times --delete --exclude 'build*' --exclude '*.ko' --exclude '.tmp_versions' \
      --exclude '*.o' --exclude '*.cmd' "$SRC/" "$DST/"
cd "$DST"

build_user() {
  cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
  cmake --build build -j"$(nproc)" 2>&1 | grep -E 'warning|error|Error' || true
  cmake --build build -j"$(nproc)" >/dev/null
  echo "user-space build OK: $DST/build"
}

build_san() {
  local name="$1" flags="$2"
  cmake -S . -B "build-$name" -DCMAKE_BUILD_TYPE=Debug -DSANITIZE="$flags" >/dev/null
  cmake --build "build-$name" -j"$(nproc)" --target vgpu_tests >/dev/null
  # TSan needs ASLR off on kernels with high mmap entropy (per-process, no sysctl).
  setarch "$(uname -m)" -R "./build-$name/vgpu_tests" --backend sim
}

case "$what" in
  user) build_user ;;
  test) build_user; ./build/vgpu_tests --backend sim ;;
  asan) build_san asan address,undefined ;;
  tsan) build_san tsan thread ;;
  kmod) make -C kmod KSRC="$KSRC" ;;
  all)  build_user; ./build/vgpu_tests --backend sim ;;
  *) echo "unknown target $what"; exit 2 ;;
esac
