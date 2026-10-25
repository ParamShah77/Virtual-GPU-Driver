/* SPDX-License-Identifier: GPL-2.0 */
/*
 * OS abstraction for the resource-manager core. The core is written once and
 * compiled twice: into the kernel module (vgpu.ko) and into a user-space
 * simulator. Everything OS specific the core needs is in this header.
 */
#ifndef VRM_OS_H
#define VRM_OS_H

#ifdef __KERNEL__

#include <linux/types.h>
#include <linux/errno.h>
#include <linux/string.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/ktime.h>
#include <linux/printk.h>
#include <asm/barrier.h>

typedef struct mutex vrm_lock_t;

static inline void vrm_lock_init(vrm_lock_t *l)    { mutex_init(l); }
static inline void vrm_lock_destroy(vrm_lock_t *l) { mutex_destroy(l); }
static inline void vrm_lock(vrm_lock_t *l)         { mutex_lock(l); }
static inline void vrm_unlock(vrm_lock_t *l)       { mutex_unlock(l); }

static inline void *vrm_zalloc(size_t n) { return kzalloc(n, GFP_KERNEL); }
static inline void vrm_free(void *p)     { kfree(p); }
static inline u64 vrm_now_ns(void)       { return ktime_get_ns(); }

#define vrm_load_acquire32(p)       smp_load_acquire(p)
#define vrm_store_release32(p, v)   smp_store_release(p, v)
#define vrm_mb()                    smp_mb()
#define vrm_log(...)                pr_debug("vgpu: " __VA_ARGS__)

#else /* user space */

#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>

#include <linux/types.h>

/* Same definitions as the kernel's: u64 is __u64 (unsigned long long), so
 * pointers to UAPI fields convert without casts. */
typedef __u8  u8;
typedef __u16 u16;
typedef __u32 u32;
typedef __u64 u64;
typedef __s32 s32;

typedef pthread_mutex_t vrm_lock_t;

static inline void vrm_lock_init(vrm_lock_t *l)    { pthread_mutex_init(l, NULL); }
static inline void vrm_lock_destroy(vrm_lock_t *l) { pthread_mutex_destroy(l); }
static inline void vrm_lock(vrm_lock_t *l)         { pthread_mutex_lock(l); }
static inline void vrm_unlock(vrm_lock_t *l)       { pthread_mutex_unlock(l); }

static inline void *vrm_zalloc(size_t n) { return calloc(1, n); }
static inline void vrm_free(void *p)     { free(p); }

static inline u64 vrm_now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (u64)ts.tv_sec * 1000000000ull + (u64)ts.tv_nsec;
}

#define vrm_load_acquire32(p)       __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define vrm_store_release32(p, v)   __atomic_store_n((p), (v), __ATOMIC_RELEASE)
#define vrm_mb()                    __atomic_thread_fence(__ATOMIC_SEQ_CST)
#define vrm_log(...)                ((void)0)

#endif /* __KERNEL__ */

#endif /* VRM_OS_H */
