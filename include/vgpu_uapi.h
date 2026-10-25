/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * vgpu UAPI: the contract between user-mode (libvrm) and the vgpu kernel
 * module. Shared with the OS-agnostic core, which executes the command stream
 * defined here.
 */
#ifndef VGPU_UAPI_H
#define VGPU_UAPI_H

#include <linux/types.h>
#include <linux/ioctl.h>

#define VGPU_PAGE_SHIFT   12
#define VGPU_PAGE_SIZE    4096u

/* Engines. The copy engine runs MEMSET/MEMCPY, the compute engine runs VEC_*. */
#define VGPU_ENGINE_COPY     0
#define VGPU_ENGINE_COMPUTE  1
#define VGPU_NUM_ENGINES     2

/* Channel priority: selects the scheduler timeslice. */
#define VGPU_PRIO_LOW     0
#define VGPU_PRIO_NORMAL  1
#define VGPU_PRIO_HIGH    2
#define VGPU_NUM_PRIOS    3

/* Memory object flags. */
#define VGPU_MEM_GPU_READONLY  (1u << 0)  /* GPU writes to this object fault */

/* ---- Command stream --------------------------------------------------- */

enum vgpu_op {
	VGPU_OP_NOP       = 0,
	VGPU_OP_MEMSET    = 1, /* arg0=dst va, arg1=len, arg2=fill byte         */
	VGPU_OP_MEMCPY    = 2, /* arg0=dst va, arg1=src va, arg2=len            */
	VGPU_OP_VEC_ADD   = 3, /* count u32 elems: arg0=dst, arg1=a, arg2=b     */
	VGPU_OP_VEC_MUL   = 4, /* same operands as VEC_ADD                      */
	VGPU_OP_FENCE     = 5, /* arg0=semaphore va, arg1=value written        */
	VGPU_OP_WAIT_SEM  = 6, /* block channel until *arg0 >= arg1 (wrapping)  */
	VGPU_OP_MAX
};

/* One ring entry: 32 bytes. */
struct vgpu_cmd {
	__u32 op;
	__u32 count;
	__u64 arg0;
	__u64 arg1;
	__u64 arg2;
};

/*
 * Channel control block, one page of VRAM mapped into the client. The client
 * owns 'put' (index of next free ring entry), the engine owns 'get'. Both are
 * free-running counters; the ring slot is (counter & (entries - 1)).
 */
struct vgpu_chan_ctrl {
	__u32 get;
	__u8  pad0[60];
	__u32 put;
	__u8  pad1[60];
	__u32 error;   /* enum vgpu_fault; non-zero once the channel has faulted */
	__u8  pad2[60];
	/*
	 * Set by the engine before it sleeps. After publishing 'put' the client
	 * issues a full barrier and rings the doorbell ioctl only if this is
	 * set; while the engine is awake, submission is pure memory writes.
	 */
	__u32 need_doorbell;
};

enum vgpu_fault {
	VGPU_FAULT_NONE         = 0,
	VGPU_FAULT_UNMAPPED     = 1, /* GPU VA has no valid PTE               */
	VGPU_FAULT_READONLY     = 2, /* write to a GPU-read-only mapping      */
	VGPU_FAULT_DISCONTIG    = 3, /* access spans non-contiguous backing   */
	VGPU_FAULT_BAD_OP       = 4,
	VGPU_FAULT_BAD_ARG      = 5,
	VGPU_FAULT_WRONG_ENGINE = 6,
	VGPU_FAULT_RING         = 7  /* put/get corrupted                     */
};

/* ---- Tracing ---------------------------------------------------------- */

enum vgpu_ev_type {
	VGPU_EV_SLICE_BEGIN = 1, /* chan scheduled onto engine; arg=prio      */
	VGPU_EV_SLICE_END   = 2, /* arg2 = virtual ns consumed in this slice  */
	VGPU_EV_CMD         = 3, /* arg=op, arg2=modelled cost in ns          */
	VGPU_EV_DOORBELL    = 4,
	VGPU_EV_FAULT       = 5, /* arg=enum vgpu_fault                       */
	VGPU_EV_BLOCKED     = 6, /* channel parked on WAIT_SEM                */
	VGPU_EV_CHUNK       = 7  /* partial command; arg=op, arg2=cost in ns  */
};

struct vgpu_trace_ev {
	__u64 ts_ns;
	__u32 type;
	__u32 chan;
	__u32 engine;
	__u32 arg;
	__u64 arg2;
};

/* ---- ioctl interface -------------------------------------------------- */

struct vgpu_info {
	__u64 vram_bytes;
	__u64 vram_free_bytes;
	__u64 vram_largest_free_bytes;
	__u32 page_size;
	__u32 n_engines;
	__u32 max_channels;
	__u32 n_contexts;
	__u64 timeslice_ns[VGPU_NUM_PRIOS];
	struct {
		__u64 busy_ns;
		__u64 cmds;
		__u64 switches;
	} eng[VGPU_NUM_ENGINES];
	__u64 now_ns;
	__u64 doorbells;     /* doorbell ioctls received since load */
};

struct vgpu_alloc {
	__u64 size;          /* in: bytes (rounded up to pages)             */
	__u32 flags;         /* in: VGPU_MEM_*                              */
	__u32 handle;        /* out                                         */
	__u64 va;            /* out: GPU virtual address                    */
	__u64 alloc_bytes;   /* out: page-rounded size                      */
	__u64 mmap_offset;   /* out: offset to pass to mmap() on the fd     */
};

struct vgpu_free {
	__u32 handle;
	__u32 pad;
};

struct vgpu_chan_create {
	__u32 engine;        /* in */
	__u32 priority;      /* in */
	__u64 ring_va;       /* in: GPU VA of ring (a normal allocation)    */
	__u32 ring_entries;  /* in: power of two                            */
	__u32 chan_id;       /* out */
	__u32 ctrl_handle;   /* out: mmap this to reach struct vgpu_chan_ctrl */
	__u32 pad;
	__u64 ctrl_va;       /* out */
	__u64 ctrl_mmap_offset; /* out */
};

struct vgpu_chan_id {
	__u32 chan_id;
	__u32 pad;
};

struct vgpu_wait_sem {
	__u64 va;
	__u32 value;
	__u32 timeout_ms;
	__u32 observed;      /* out */
	__u32 pad;
};

struct vgpu_chan_stats {
	__u32 chan_id;       /* in */
	__u32 state;         /* out: 1=active 2=error */
	__u32 error;
	__u32 engine;
	__u32 priority;
	__u32 get;
	__u32 put;
	__u32 pad;
	__u64 cmds_executed;
	__u64 busy_ns;
	__u64 slices;
};

struct vgpu_trace_req {
	__u64 buf;           /* in: user pointer to struct vgpu_trace_ev[]  */
	__u32 max_events;    /* in */
	__u32 n_events;      /* out */
	__u64 from_seq;      /* in: first event sequence number wanted      */
	__u64 next_seq;      /* out: pass as from_seq next time             */
	__u64 dropped;       /* out: events lost to ring overwrite          */
};

#define VGPU_IOC_MAGIC 'V'
#define VGPU_IOC_GET_INFO      _IOR (VGPU_IOC_MAGIC, 0, struct vgpu_info)
#define VGPU_IOC_ALLOC         _IOWR(VGPU_IOC_MAGIC, 1, struct vgpu_alloc)
#define VGPU_IOC_FREE          _IOW (VGPU_IOC_MAGIC, 2, struct vgpu_free)
#define VGPU_IOC_CHAN_CREATE   _IOWR(VGPU_IOC_MAGIC, 3, struct vgpu_chan_create)
#define VGPU_IOC_CHAN_DESTROY  _IOW (VGPU_IOC_MAGIC, 4, struct vgpu_chan_id)
#define VGPU_IOC_DOORBELL      _IOW (VGPU_IOC_MAGIC, 5, struct vgpu_chan_id)
#define VGPU_IOC_WAIT_SEM      _IOWR(VGPU_IOC_MAGIC, 6, struct vgpu_wait_sem)
#define VGPU_IOC_CHAN_STATS    _IOWR(VGPU_IOC_MAGIC, 7, struct vgpu_chan_stats)
#define VGPU_IOC_GET_TRACE     _IOWR(VGPU_IOC_MAGIC, 8, struct vgpu_trace_req)

#endif /* VGPU_UAPI_H */
