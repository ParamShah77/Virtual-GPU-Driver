// SPDX-License-Identifier: GPL-2.0
/*
 * vgpu: virtual GPU kernel driver.
 *
 * This file is the Linux "OS interface layer" around the OS-agnostic resource
 * manager core in ../core: the misc device, ioctl/mmap entry points, per-engine
 * kernel threads that stand in for the GPU front end, and a /proc view.
 */
#include <linux/delay.h>
#include <linux/fs.h>
#include <linux/kthread.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/vmalloc.h>
#include <linux/wait.h>

#include "vrm_core.h"

#define DRV_NAME "vgpu"

static unsigned long vram_mb = 64;
module_param(vram_mb, ulong, 0444);
MODULE_PARM_DESC(vram_mb, "Size of the virtual VRAM pool in MiB (default 64)");

static bool pace;
module_param(pace, bool, 0644);
MODULE_PARM_DESC(pace, "Sleep for the modelled cost of each command (default off)");

struct vgpu_dev {
	struct vrm_gpu *gpu;
	void *vram;
	struct task_struct *eng_thread[VGPU_NUM_ENGINES];
	/* One event counter for engines and CPU waiters (doorbell/fence/fault). */
	wait_queue_head_t wq;
	atomic64_t seq;
};

static struct vgpu_dev vdev;

struct vgpu_file {
	struct vrm_ctx *ctx;
};

static void vgpu_signal(void)
{
	atomic64_inc(&vdev.seq);
	wake_up_all(&vdev.wq);
}

/* ---- engine threads -------------------------------------------------------- */

#define PACE_CHUNK_NS 50000ull

static void pace_settle(u64 *owed)
{
	if (pace && *owed >= 1000) {
		unsigned long us = *owed / 1000;

		usleep_range(us, us + us / 8 + 10);
	}
	*owed = 0;
}

static int vgpu_engine_thread(void *data)
{
	u32 e = (unsigned long)data;
	u64 owed = 0;

	while (!kthread_should_stop()) {
		struct vrm_step st;
		s64 seen = atomic64_read(&vdev.seq);

		vrm_engine_step(vdev.gpu, e, &st);
		if (st.signal)
			vgpu_signal();

		switch (st.kind) {
		case VRM_STEP_RAN:
			owed += st.sleep_ns;
			if (owed >= PACE_CHUNK_NS)
				pace_settle(&owed);
			cond_resched();
			break;
		case VRM_STEP_IDLE:
			pace_settle(&owed);
			/* Interruptible, so an idle engine doesn't count as load
			 * or trip the hung-task detector. */
			wait_event_interruptible(vdev.wq,
						 atomic64_read(&vdev.seq) != seen ||
						 kthread_should_stop());
			break;
		case VRM_STEP_BLOCKED:
			pace_settle(&owed);
			/* Semaphores can also be released by the CPU with no wakeup. */
			wait_event_interruptible_hrtimeout(vdev.wq,
					atomic64_read(&vdev.seq) != seen ||
					kthread_should_stop(),
					ns_to_ktime(200 * NSEC_PER_USEC));
			break;
		}
	}
	return 0;
}

/* ---- mmap -------------------------------------------------------------------- */

static void vgpu_vma_open(struct vm_area_struct *vma)
{
	struct vgpu_file *vf = vma->vm_file->private_data;
	u64 off, bytes;

	vrm_mem_map_get(vdev.gpu, vf->ctx, (u32)(unsigned long)vma->vm_private_data,
			&off, &bytes);
}

static void vgpu_vma_close(struct vm_area_struct *vma)
{
	struct vgpu_file *vf = vma->vm_file->private_data;

	vrm_mem_map_put(vdev.gpu, vf->ctx, (u32)(unsigned long)vma->vm_private_data);
}

static const struct vm_operations_struct vgpu_vm_ops = {
	.open = vgpu_vma_open,
	.close = vgpu_vma_close,
};

/* mmap offset is (object handle << PAGE_SHIFT). */
static int vgpu_mmap(struct file *f, struct vm_area_struct *vma)
{
	struct vgpu_file *vf = f->private_data;
	u32 handle = vma->vm_pgoff;
	unsigned long size = vma->vm_end - vma->vm_start;
	u64 off, bytes;
	int ret;

	if (!(vma->vm_flags & VM_SHARED))
		return -EINVAL;
	ret = vrm_mem_map_get(vdev.gpu, vf->ctx, handle, &off, &bytes);
	if (ret)
		return ret;
	if (size > bytes) {
		vrm_mem_map_put(vdev.gpu, vf->ctx, handle);
		return -EINVAL;
	}
	vm_flags_set(vma, VM_DONTEXPAND | VM_DONTDUMP | VM_DONTCOPY);
	vma->vm_private_data = (void *)(unsigned long)handle;
	/* Maps the whole VMA, starting 'off' bytes into the VRAM vmalloc area. */
	ret = remap_vmalloc_range(vma, vdev.vram, off >> PAGE_SHIFT);
	if (ret) {
		vrm_mem_map_put(vdev.gpu, vf->ctx, handle);
		return ret;
	}
	/* Only now: close() must run exactly once per successful mapping. */
	vma->vm_ops = &vgpu_vm_ops;
	return 0;
}

/* ---- file operations --------------------------------------------------------- */

static int vgpu_open(struct inode *inode, struct file *f)
{
	struct vgpu_file *vf = kzalloc(sizeof(*vf), GFP_KERNEL);
	int ret;

	if (!vf)
		return -ENOMEM;
	ret = vrm_ctx_create(vdev.gpu, &vf->ctx);
	if (ret) {
		kfree(vf);
		return ret;
	}
	f->private_data = vf;
	return 0;
}

static int vgpu_release(struct inode *inode, struct file *f)
{
	struct vgpu_file *vf = f->private_data;

	vrm_ctx_destroy(vdev.gpu, vf->ctx);
	kfree(vf);
	vgpu_signal();
	return 0;
}

static long ioctl_wait_sem(struct vgpu_file *vf, struct vgpu_wait_sem *w)
{
	long remaining = msecs_to_jiffies(w->timeout_ms);
	long ret;

	for (;;) {
		s64 seen = atomic64_read(&vdev.seq);
		int ready = 0;

		ret = vrm_sem_poll(vdev.gpu, vf->ctx, w->va, w->value,
				   &w->observed, &ready);
		if (ret)
			return ret;
		if (ready)
			return 0;
		if (remaining <= 0)
			return -ETIMEDOUT;
		remaining = wait_event_interruptible_timeout(vdev.wq,
				atomic64_read(&vdev.seq) != seen, remaining);
		if (remaining < 0)
			return -EINTR;
		/* remaining == 0 means timed out; the loop re-checks once more. */
	}
}

static long ioctl_get_trace(struct vgpu_trace_req *r)
{
	struct vgpu_trace_ev *buf;
	u32 max = min_t(u32, r->max_events, VRM_TRACE_ENTRIES);
	long ret = 0;

	if (!max)
		return -EINVAL;
	buf = kvmalloc_array(max, sizeof(*buf), GFP_KERNEL);
	if (!buf)
		return -ENOMEM;
	vrm_trace_read(vdev.gpu, r->from_seq, buf, max, &r->n_events,
		       &r->next_seq, &r->dropped);
	if (copy_to_user(u64_to_user_ptr(r->buf), buf,
			 r->n_events * sizeof(*buf)))
		ret = -EFAULT;
	kvfree(buf);
	return ret;
}

#define ARG_FROM_USER(type, var)                                                  \
	type var;                                                          \
	if (copy_from_user(&var, uarg, sizeof(var)))                       \
		return -EFAULT

static long vgpu_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	struct vgpu_file *vf = f->private_data;
	void __user *uarg = (void __user *)arg;
	long ret;

	switch (cmd) {
	case VGPU_IOC_GET_INFO: {
		struct vgpu_info info;

		vrm_gpu_get_info(vdev.gpu, &info);
		return copy_to_user(uarg, &info, sizeof(info)) ? -EFAULT : 0;
	}
	case VGPU_IOC_ALLOC: {
		ARG_FROM_USER(struct vgpu_alloc, a);

		ret = vrm_mem_alloc(vdev.gpu, vf->ctx, a.size, a.flags, &a.handle,
				    &a.va, &a.alloc_bytes);
		if (ret)
			return ret;
		a.mmap_offset = (u64)a.handle << VGPU_PAGE_SHIFT;
		return copy_to_user(uarg, &a, sizeof(a)) ? -EFAULT : 0;
	}
	case VGPU_IOC_FREE: {
		ARG_FROM_USER(struct vgpu_free, fr);

		return vrm_mem_free(vdev.gpu, vf->ctx, fr.handle);
	}
	case VGPU_IOC_CHAN_CREATE: {
		ARG_FROM_USER(struct vgpu_chan_create, c);

		ret = vrm_chan_create(vdev.gpu, vf->ctx, c.engine, c.priority,
				      c.ring_va, c.ring_entries, &c.chan_id,
				      &c.ctrl_handle, &c.ctrl_va);
		if (ret)
			return ret;
		c.ctrl_mmap_offset = (u64)c.ctrl_handle << VGPU_PAGE_SHIFT;
		return copy_to_user(uarg, &c, sizeof(c)) ? -EFAULT : 0;
	}
	case VGPU_IOC_CHAN_DESTROY: {
		ARG_FROM_USER(struct vgpu_chan_id, id);

		ret = vrm_chan_destroy(vdev.gpu, vf->ctx, id.chan_id);
		if (!ret)
			vgpu_signal();
		return ret;
	}
	case VGPU_IOC_DOORBELL: {
		ARG_FROM_USER(struct vgpu_chan_id, id);

		ret = vrm_chan_doorbell(vdev.gpu, vf->ctx, id.chan_id);
		if (!ret)
			vgpu_signal();
		return ret;
	}
	case VGPU_IOC_WAIT_SEM: {
		ARG_FROM_USER(struct vgpu_wait_sem, w);

		ret = ioctl_wait_sem(vf, &w);
		if (copy_to_user(uarg, &w, sizeof(w)))
			return -EFAULT;
		return ret;
	}
	case VGPU_IOC_CHAN_STATS: {
		ARG_FROM_USER(struct vgpu_chan_stats, s);

		ret = vrm_chan_stats(vdev.gpu, vf->ctx, &s);
		if (ret)
			return ret;
		return copy_to_user(uarg, &s, sizeof(s)) ? -EFAULT : 0;
	}
	case VGPU_IOC_GET_TRACE: {
		ARG_FROM_USER(struct vgpu_trace_req, r);

		ret = ioctl_get_trace(&r);
		if (ret)
			return ret;
		return copy_to_user(uarg, &r, sizeof(r)) ? -EFAULT : 0;
	}
	}
	return -ENOTTY;
}

static const struct file_operations vgpu_fops = {
	.owner = THIS_MODULE,
	.open = vgpu_open,
	.release = vgpu_release,
	.unlocked_ioctl = vgpu_ioctl,
	.mmap = vgpu_mmap,
	.llseek = no_llseek,
};

static struct miscdevice vgpu_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "vgpu0",
	.fops = &vgpu_fops,
	.mode = 0666,
};

/* ---- /proc/vgpu ---------------------------------------------------------------- */

static const char *chan_state_name(enum vrm_chan_state s)
{
	return s == VRM_CHAN_ACTIVE ? "active" : s == VRM_CHAN_ERROR ? "ERROR" : "free";
}

static int vgpu_proc_show(struct seq_file *m, void *v)
{
	struct vrm_gpu *g = vdev.gpu;
	static const char *const prio_name[] = { "low", "normal", "high" };
	static const char *const eng_name[] = { "copy", "compute" };
	u32 i;

	vrm_lock(&g->lock);
	seq_printf(m, "vram: %llu KiB total, %llu KiB free, largest free run %u KiB\n",
		   g->vram_bytes >> 10, (u64)g->free_pages << (VGPU_PAGE_SHIFT - 10),
		   vrm_vram_largest_free(g) << (VGPU_PAGE_SHIFT - 10));
	seq_printf(m, "pace: %s, timeslices(us): low=%llu normal=%llu high=%llu\n",
		   pace ? "on" : "off", g->timeslice_ns[0] / 1000,
		   g->timeslice_ns[1] / 1000, g->timeslice_ns[2] / 1000);

	seq_puts(m, "\nengines:\n");
	for (i = 0; i < VGPU_NUM_ENGINES; i++)
		seq_printf(m, "  %-8s cmds=%llu busy=%llu us switches=%llu running=%u\n",
			   eng_name[i], g->eng[i].cmds, g->eng[i].busy_ns / 1000,
			   g->eng[i].switches,
			   g->eng[i].cur ? g->eng[i].cur->id : 0);

	seq_puts(m, "\ncontexts:\n");
	for (i = 0; i < VRM_MAX_CTX; i++)
		if (g->ctx[i].in_use)
			seq_printf(m, "  ctx %-2u mem=%llu KiB channels=%u\n",
				   g->ctx[i].id, g->ctx[i].mem_bytes >> 10,
				   g->ctx[i].nchan);

	seq_puts(m, "\nchannels:\n");
	for (i = 0; i < VRM_MAX_CHAN; i++) {
		struct vrm_channel *c = &g->chan[i];

		if (c->state == VRM_CHAN_FREE)
			continue;
		seq_printf(m, "  chan %-2u ctx=%u %-7s %-6s %-6s get=%u put=%u cmds=%llu busy=%llu us slices=%llu err=%u\n",
			   c->id, c->ctx->id, eng_name[c->engine], prio_name[c->prio],
			   chan_state_name(c->state), c->ctrl->get, c->ctrl->put,
			   c->cmds_executed, c->busy_ns / 1000, c->slices,
			   c->ctrl->error);
	}

	seq_puts(m, "\nrecent faults (newest first):\n");
	for (i = 0; i < VRM_FAULT_LOG && i < g->fault_seq; i++) {
		struct vrm_fault_rec *f =
			&g->faults[(g->fault_seq - 1 - i) % VRM_FAULT_LOG];

		seq_printf(m, "  chan %u fault=%u op=%u va=0x%llx\n", f->chan, f->code,
			   f->op, f->va);
	}
	vrm_unlock(&g->lock);
	return 0;
}

/* ---- module -------------------------------------------------------------------- */

static int __init vgpu_init(void)
{
	u64 vram_bytes = (u64)vram_mb << 20;
	int ret, i;

	if (vram_mb < 1 || vram_mb > 1024)
		return -EINVAL;

	init_waitqueue_head(&vdev.wq);
	atomic64_set(&vdev.seq, 0);

	vdev.vram = vmalloc_user(vram_bytes);
	vdev.gpu = vzalloc(sizeof(*vdev.gpu));
	if (!vdev.vram || !vdev.gpu) {
		ret = -ENOMEM;
		goto err_free;
	}
	ret = vrm_gpu_init(vdev.gpu, vdev.vram, vram_bytes);
	if (ret)
		goto err_free;

	for (i = 0; i < VGPU_NUM_ENGINES; i++) {
		vdev.eng_thread[i] = kthread_run(vgpu_engine_thread,
						 (void *)(unsigned long)i,
						 "vgpu-eng%d", i);
		if (IS_ERR(vdev.eng_thread[i])) {
			ret = PTR_ERR(vdev.eng_thread[i]);
			vdev.eng_thread[i] = NULL;
			goto err_threads;
		}
	}
	ret = misc_register(&vgpu_misc);
	if (ret)
		goto err_threads;
	proc_create_single("vgpu", 0444, NULL, vgpu_proc_show);
	pr_info(DRV_NAME ": /dev/vgpu0 ready, %lu MiB VRAM, %u engines\n", vram_mb,
		VGPU_NUM_ENGINES);
	return 0;

err_threads:
	for (i = 0; i < VGPU_NUM_ENGINES; i++)
		if (vdev.eng_thread[i]) {
			kthread_stop(vdev.eng_thread[i]);
		}
	vrm_gpu_fini(vdev.gpu);
err_free:
	vfree(vdev.gpu);
	vfree(vdev.vram);
	return ret;
}

static void __exit vgpu_exit(void)
{
	int i;

	remove_proc_entry("vgpu", NULL);
	misc_deregister(&vgpu_misc);
	for (i = 0; i < VGPU_NUM_ENGINES; i++) {
		/* Make sure a sleeping engine re-checks kthread_should_stop(). */
		kthread_stop(vdev.eng_thread[i]);
	}
	vrm_gpu_fini(vdev.gpu);
	vfree(vdev.gpu);
	vfree(vdev.vram);
}

module_init(vgpu_init);
module_exit(vgpu_exit);

MODULE_DESCRIPTION("Virtual GPU: resource manager, channels and scheduler");
MODULE_LICENSE("GPL");
