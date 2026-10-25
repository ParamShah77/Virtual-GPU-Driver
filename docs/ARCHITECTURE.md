# vgpu architecture

vgpu is a virtual GPU driver stack: a Linux kernel module that implements a
GPU-style **resource manager** (memory, address spaces, channels, scheduling)
for a simulated device, and a user-mode runtime that talks to it through the
same interfaces a real driver stack uses (ioctl, mmap, ring buffers, doorbells,
semaphores).

```
  application / tests / benchmarks
        │  C++ API: Device, Buffer, Channel
  ┌─────▼──────────────────────────────┐
  │ libvrm (user/)                      │   user-mode driver
  │   Backend interface                 │
  └───┬───────────────────────────┬────┘
      │ KernelBackend             │ SimBackend
      │ ioctl + mmap              │ direct calls
  ┌───▼───────────────┐      ┌────▼────────────────────┐
  │ kmod/vgpu_main.c  │      │ user/sim.cpp            │   OS interface layers
  │ misc dev, ioctl,  │      │ engine threads, events  │
  │ mmap, kthreads,   │      │                         │
  │ /proc/vgpu        │      │                         │
  └───┬───────────────┘      └────┬────────────────────┘
      └───────────┬────────────────┘
        ┌─────────▼──────────────────────────┐
        │ core/  (OS-agnostic RM core, C)    │   compiled into BOTH
        │  vrm_mem.c   VRAM + page tables    │
        │  vrm_exec.c  channels, executor,   │
        │              scheduler             │
        │  vrm_misc.c  trace, introspection  │
        └────────────────────────────────────┘
```

## Why the core is compiled twice

Production GPU drivers are split the same way: an OS-agnostic core (the
"resource manager") plus a thin per-OS interface layer. Here `core/` has no
dependency on Linux: everything OS-specific (locks, allocation, time, atomics)
is behind `core/vrm_os.h`, which has a `__KERNEL__` and a user-space branch.
The same source runs

* inside `vgpu.ko`, driving real kernel threads, `vmalloc`'d VRAM and `mmap`, and
* in-process as a simulator, so the whole stack can be unit-tested, fuzzed and
  run under ASan/UBSan/TSan without loading a module.

The test suite is written once against the `Device` API and runs against either
backend (`vgpu_tests --backend sim|kernel`), so the simulator doubles as a
reference implementation for the kernel path.

## The device model

* **VRAM**: a fixed pool (64 MiB default) managed as 4 KiB pages with a bitmap,
  first-fit contiguous allocation. Pages are zeroed when handed out, so freed
  memory never leaks between clients.
* **Contexts**: one per `open()` of `/dev/vgpu0`. Each has its own 32-bit GPU
  virtual address space backed by a two-level page table (10+10+12 bits).
  PTEs carry valid / writable bits; objects are separated by an unmapped guard
  page, so a linear overrun faults instead of corrupting a neighbour.
* **Memory objects**: allocated through `VGPU_IOC_ALLOC`, which returns a GPU VA
  and an mmap handle. Mapping a buffer pins it (`mmap_refs`); freeing a mapped
  buffer fails with `EBUSY`, so a client can never keep a CPU mapping of pages
  that have been recycled to someone else.
* **Engines**: two, a *copy* engine (MEMSET, MEMCPY) and a *compute* engine
  (VEC_ADD, VEC_MUL). Each is driven by its own kernel thread
  (`vgpu-eng0/1`) in the module, its own thread in the simulator.
* **Channels**: a channel is a command ring (an ordinary client allocation of
  32-byte commands) plus a control page (`get`, `put`, `error`) that the kernel
  allocates in VRAM and the client maps. The client writes commands, publishes
  `put` with a release store, and rings the doorbell (an ioctl here; an MMIO
  write on hardware). The engine consumes with an acquire load of `put` and a
  release store of `get`; there are no locks on the submission path.
* **Synchronisation**: `FENCE` writes a value to a semaphore in memory;
  `WAIT_SEM` parks a channel until a semaphore reaches a value (wrapping
  compare). This gives cross-channel and cross-engine dependencies and CPU
  waits (`VGPU_IOC_WAIT_SEM`). The CPU can also release a `WAIT_SEM` by writing
  the semaphore directly; blocked engines re-poll every 200 µs for that case.
* **Faults**: every GPU access is translated through the page tables per
  command. A bad address, a write to a read-only mapping, an access spanning
  non-contiguous pages, a wrong-engine command, a bad opcode or a corrupt
  `put` moves the channel to the ERROR state, records the code in the control
  page, appends to a fault log and wakes waiters. Other channels, including
  ones on the same engine, are unaffected.

## Scheduler

Per engine, round-robin over runnable channels (a channel is runnable if it
has queued work and its head command is not a `WAIT_SEM` that would block).
A scheduled channel runs until its **timeslice** is used up or it runs dry;
slice length depends on priority (low 1 ms, normal 2 ms, high 4 ms), so under
contention throughput is roughly 1 : 2 : 4.

Slices are measured in the engine's *virtual clock*: the sum of each command's
modelled cost (1 µs + bytes/8 for copies, 1 µs + 4 ns/element for compute).
This makes scheduling deterministic and identical in the kernel and the
simulator, independent of timer jitter, and it is what the fairness tests
assert on. With `pace=1` the engine thread additionally sleeps for the modelled
cost so wall-clock latency and bandwidth look like a slower device.

Preemption is at command boundaries only. A real GPU preempts mid-kernel; this
model does not, so a single huge command can overrun its slice (commands are
capped at 16 MiB to bound this).

## Concurrency and locking

* One mutex (`gpu->lock`) protects all RM state. Every public core function
  takes it; the OS layer never touches core state directly.
* Engines take the lock **per command** and sleep for the modelled cost outside
  it. Waking is event based: a single sequence counter bumped on doorbell, fence
  and fault, with waiters sampling the counter *before* re-checking their
  condition, so no wakeup is lost.
* Fences and faults are reported out of `vrm_engine_step()` as a `signal` flag
  and delivered by the OS layer after the lock is dropped, avoiding a
  lock-order inversion between the core lock and the OS layer's event lock.
* Client-visible ring memory is shared without locks, with the acquire/release
  protocol above.

## Observability

* `/proc/vgpu`: VRAM usage, per-engine busy time and context switches,
  contexts, per-channel state / get / put / commands / busy time / slices, and
  the recent fault log.
* `VGPU_IOC_GET_TRACE`: a 4096-entry ring of typed events (slice begin/end,
  command, doorbell, fault) with a sequence cursor so tools can read
  incrementally and detect drops. `tools/plot_trace.py` turns it into a Gantt
  chart and share summary.

## Known simplifications

* Doorbell is an ioctl, not a mapped register page.
* No DMA / PCI / interrupts; the "device" is the engine threads.
* Contiguous VRAM allocation only, so the allocator can fragment (the tests
  show this on purpose).
* One lock for everything; fine at this scale, the first thing to split for
  more engines.
* Commands execute atomically with respect to the scheduler (see Preemption).
