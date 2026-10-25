/* SPDX-License-Identifier: GPL-2.0 */
/*
 * vrm core: the OS-agnostic "resource manager" of the virtual GPU.
 *
 *  - VRAM page allocator
 *  - per-context GPU virtual address spaces (2-level page tables)
 *  - channels (command rings) and a command executor ("the hardware")
 *  - per-engine timeslice scheduler
 *
 * Locking: every public function takes gpu->lock itself. The OS layer owns
 * threads, blocking and wakeups; it drives engines by calling vrm_engine_step()
 * in a loop, sleeping for the returned modelled cost between steps.
 */
#ifndef VRM_CORE_H
#define VRM_CORE_H

#include "vrm_os.h"
#include "vgpu_uapi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VRM_MAX_CTX        16
#define VRM_MAX_CHAN       32
#define VRM_TRACE_ENTRIES  4096u   /* power of two */
#define VRM_FAULT_LOG      16

#define VRM_VA_BASE        0x00100000ull
#define VRM_VA_LIMIT       0x100000000ull   /* 32-bit GPU VA space */

#define VRM_PTE_VALID      0x1u
#define VRM_PTE_WRITE      0x2u

/* Default timeslices (virtual ns), indexed by VGPU_PRIO_*. */
#define VRM_SLICE_LOW_NS     1000000ull
#define VRM_SLICE_NORMAL_NS  2000000ull
#define VRM_SLICE_HIGH_NS    4000000ull

struct vrm_pt {
	u32 pte[1024];            /* [31:12] VRAM page number, [1:0] flags */
};

struct vrm_obj {
	struct vrm_obj *next;     /* ctx list, sorted by va */
	u32 handle;
	u32 first_page;           /* first VRAM page */
	u32 npages;
	u32 flags;
	u32 mmap_refs;            /* client mappings; blocks free */
	int internal;             /* owned by a channel, not user-freeable */
	u64 va;
};

struct vrm_ctx {
	int in_use;
	u32 id;
	struct vrm_pt **pgd;      /* 1024 entries */
	struct vrm_obj *objs;
	u32 next_handle;
	u32 nchan;
	u64 mem_bytes;
};

enum vrm_chan_state {
	VRM_CHAN_FREE   = 0,
	VRM_CHAN_ACTIVE = 1,
	VRM_CHAN_ERROR  = 2
};

struct vrm_channel {
	enum vrm_chan_state state;
	u32 id;
	struct vrm_ctx *ctx;
	u32 engine;
	u32 prio;
	u64 ring_va;
	u32 ring_entries;
	struct vrm_obj *ctrl_obj;
	struct vgpu_chan_ctrl *ctrl;   /* points into VRAM */
	u64 progress;              /* bytes/elements done of the head command */
	u64 cmds_executed;
	u64 busy_ns;
	u64 slices;
};

struct vrm_engine {
	struct vrm_channel *cur;
	u64 vclock_ns;            /* virtual time: sum of modelled costs */
	u64 slice_end_ns;         /* in vclock */
	u64 slice_start_ns;
	u32 last_idx;
	u64 busy_ns;
	u64 cmds;
	u64 switches;
};

struct vrm_fault_rec {
	u64 ts_ns;
	u32 chan;
	u32 code;
	u64 va;
	u32 op;
	u32 pad;
};

struct vrm_gpu {
	vrm_lock_t lock;

	u8 *vram;
	u64 vram_bytes;
	u32 vram_pages;
	u32 free_pages;
	u32 *bitmap;

	struct vrm_ctx ctx[VRM_MAX_CTX];
	struct vrm_channel chan[VRM_MAX_CHAN];
	struct vrm_engine eng[VGPU_NUM_ENGINES];

	u64 timeslice_ns[VGPU_NUM_PRIOS];

	struct vgpu_trace_ev trace[VRM_TRACE_ENTRIES];
	u64 trace_seq;

	struct vrm_fault_rec faults[VRM_FAULT_LOG];
	u64 fault_seq;

	u64 doorbells;
};

enum vrm_step_kind {
	VRM_STEP_RAN     = 0,   /* executed a command; sleep_ns of modelled work */
	VRM_STEP_IDLE    = 1,   /* nothing queued: wait for a doorbell           */
	VRM_STEP_BLOCKED = 2    /* work queued but waiting on a semaphore        */
};

struct vrm_step {
	enum vrm_step_kind kind;
	u64 sleep_ns;
	int signal;             /* a fence/fault happened: wake waiters + engines */
};

/* ---- lifecycle (vrm_mem.c) ---- */
int  vrm_gpu_init(struct vrm_gpu *g, void *vram, u64 vram_bytes);
void vrm_gpu_fini(struct vrm_gpu *g);

int  vrm_ctx_create(struct vrm_gpu *g, struct vrm_ctx **out);
void vrm_ctx_destroy(struct vrm_gpu *g, struct vrm_ctx *ctx);

/* ---- memory (vrm_mem.c) ---- */
int vrm_mem_alloc(struct vrm_gpu *g, struct vrm_ctx *ctx, u64 bytes, u32 flags,
		  u32 *handle, u64 *va, u64 *alloc_bytes);
int vrm_mem_free(struct vrm_gpu *g, struct vrm_ctx *ctx, u32 handle);
/* mmap support: pin the object and return its VRAM byte offset. */
int vrm_mem_map_get(struct vrm_gpu *g, struct vrm_ctx *ctx, u32 handle,
		    u64 *vram_off, u64 *bytes);
void vrm_mem_map_put(struct vrm_gpu *g, struct vrm_ctx *ctx, u32 handle);

/* ---- channels / scheduler (vrm_exec.c) ---- */
int  vrm_chan_create(struct vrm_gpu *g, struct vrm_ctx *ctx, u32 engine,
		     u32 prio, u64 ring_va, u32 ring_entries, u32 *chan_id,
		     u32 *ctrl_handle, u64 *ctrl_va);
int  vrm_chan_destroy(struct vrm_gpu *g, struct vrm_ctx *ctx, u32 chan_id);
int  vrm_chan_doorbell(struct vrm_gpu *g, struct vrm_ctx *ctx, u32 chan_id);
int  vrm_chan_stats(struct vrm_gpu *g, struct vrm_ctx *ctx,
		    struct vgpu_chan_stats *st);
void vrm_engine_step(struct vrm_gpu *g, u32 engine, struct vrm_step *st);

/*
 * Doorbell-less submission. Before an engine thread sleeps it calls
 * vrm_engine_arm_sleep(): this sets need_doorbell in every control page on the
 * engine, issues a full barrier, then re-checks for work. A non-zero return
 * means work arrived and the engine must not sleep. After waking it calls
 * vrm_engine_disarm(). Paired with the client's put-store / barrier /
 * need_doorbell-load, at least one side always sees the other.
 */
int  vrm_engine_arm_sleep(struct vrm_gpu *g, u32 engine);
void vrm_engine_disarm(struct vrm_gpu *g, u32 engine);

/* Semaphore poll for CPU waiters: *ready set when *va >= value (wrapping).
 * Returns -EIO if the context has a faulted channel, -EFAULT if va unmapped. */
int  vrm_sem_poll(struct vrm_gpu *g, struct vrm_ctx *ctx, u64 va, u32 value,
		  u32 *observed, int *ready);

/* ---- introspection (vrm_misc.c) ---- */
void vrm_gpu_get_info(struct vrm_gpu *g, struct vgpu_info *out);
void vrm_trace_read(struct vrm_gpu *g, u64 from_seq, struct vgpu_trace_ev *out,
		    u32 max, u32 *n_out, u64 *next_seq, u64 *dropped);

/* ---- internal helpers shared between core files ---- */
void vrm_trace(struct vrm_gpu *g, u32 type, u32 chan, u32 engine, u32 arg,
	       u64 arg2);
void vrm_chan_free_locked(struct vrm_gpu *g, struct vrm_channel *ch);
int  vrm_mem_alloc_locked(struct vrm_gpu *g, struct vrm_ctx *ctx, u64 bytes,
			  u32 flags, int internal, struct vrm_obj **out);
void vrm_mem_free_locked(struct vrm_gpu *g, struct vrm_ctx *ctx,
			 struct vrm_obj *obj);
struct vrm_obj *vrm_obj_find(struct vrm_ctx *ctx, u32 handle);
u32  vrm_vram_largest_free(struct vrm_gpu *g);
/* Translate a GPU VA range to a host pointer; returns an enum vgpu_fault. */
u32  vrm_xlate(struct vrm_gpu *g, struct vrm_ctx *ctx, u64 va, u64 len,
	       int write, void **out);

#ifdef __cplusplus
}
#endif

#endif /* VRM_CORE_H */
