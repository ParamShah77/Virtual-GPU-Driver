#!/usr/bin/env bash
# One-time setup of a WSL2 Ubuntu environment for building vgpu.ko against the
# *running* WSL2 kernel. Run as root inside WSL:   bash scripts/wsl-setup.sh
#
# WSL2 ships no kernel headers, so we clone Microsoft's kernel source at the tag
# matching `uname -r`, reuse the running kernel's config, and build vmlinux once
# to produce Module.symvers (the kernel has CONFIG_MODVERSIONS=y, so modules
# must carry symbol CRCs to load). This takes a while the first time.
set -euo pipefail

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  build-essential cmake git rsync python3 python3-pip python3-matplotlib \
  flex bison bc libssl-dev libelf-dev dwarves pahole

REL="$(uname -r)"                      # e.g. 6.6.87.2-microsoft-standard-WSL2
VER="${REL%%-*}"                       # 6.6.87.2
KROOT=/opt/wsl-kernel
mkdir -p "$KROOT"

if [ ! -d "$KROOT/src/.git" ]; then
  git clone --depth 1 --branch "linux-msft-wsl-$VER" \
      https://github.com/microsoft/WSL2-Linux-Kernel.git "$KROOT/src"
fi

cd "$KROOT/src"
zcat /proc/config.gz > .config
: > .scmversion                        # no "+" suffix, so vermagic matches
make olddefconfig >/dev/null
make -j"$(nproc)" LOCALVERSION= modules_prepare
if [ ! -f Module.symvers ]; then
  make -j"$(nproc)" LOCALVERSION= vmlinux
  # `make vmlinux` writes the exported-symbol CRCs to vmlinux.symvers;
  # external module builds look for Module.symvers.
  cp vmlinux.symvers Module.symvers
fi
grep UTS_RELEASE include/generated/utsrelease.h
echo "kernel tree ready: $KROOT/src (KSRC)"
