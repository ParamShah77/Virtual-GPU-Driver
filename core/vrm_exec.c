// SPDX-License-Identifier: GPL-2.0
/*
 * Channels, the command executor ("the hardware") and the per-engine
 * timeslice scheduler.
 *
 * Scheduling policy: round-robin over runnable channels bound to an engine;
 * each channel runs for a priority-dependent timeslice measured in the engine's
 * *virtual* clock (the sum of modelled command costs), so behaviour is
 * deterministic and identical in the kernel and user-space builds. Preemption
 * happens only at command boundaries.
 */
#include "vrm_core.h"

#define CMD_BASE_NS        1000ull
#define MAX_CMD_BYTES      (16ull << 20)
#define MAX_VEC_ELEMS      (4u << 20)
#define MAX_RING_ENTRIES   65536u

typedef char vrm_assert_cmd_size[(sizeof(struct vgpu_cmd) == 32) ? 1 : -1];
typedef char vrm_assert_ctrl_size[(sizeof(struct vgpu_chan_ctrl) <= VGPU_PAGE_SIZE) ? 1 : -1];

/* ---- helpers ---------------------------------------------------------- */

static struct vrm_channel *chan_by_id(struct vrm_gpu *g, struct vrm_ctx *ctx,
				      u32 id)
{
	struct vrm_channel *ch;

	if (!id || id > VRM_MAX_CHAN)
		return NULL;
	ch = &g->chan[id - 1];
	if (ch->state == VRM_CHAN_FREE || ch->ctx != ctx)
		return NULL;
	return ch;
}

static u32 fetch_cmd(struct vrm_gpu *g, struct vrm_channel *ch, u32 get,
		     struct vgpu_cmd *cmd, u64 *fault_va)
{
	u32 slot = get & (ch->ring_entries - 1);
	u64 va = ch->ring_va + (u64)slot * sizeof(*cmd);
	void *p;
	u32 code;

	code = vrm_xlate(g, ch->ctx, va, sizeof(*cmd), 0, &p);
	if (code) {
		*fault_va = va;
		return code;
	}
	memcpy(cmd, p, sizeof(*cmd));
	return VGPU_FAULT_NONE;
}

/* Does the channel have a command it can execute right now? */
static int chan_ready(struct vrm_gpu *g, struct vrm_channel *ch)
{
	struct vgpu_cmd cmd;
	u64 fva;
	u32 get = ch->ctrl->get;
	u32 put = vrm_load_acquire32(&ch->ctrl->put);
	void *p;

	if (ch->state != VRM_CHAN_ACTIVE || get == put)
		return 0;
	if (fetch_cmd(g, ch, get, &cmd, &fva))
		return 1;               /* let the executor report the fault */
	if (cmd.op == VGPU_OP_WAIT_SEM) {
		if ((cmd.arg0 & 3) ||
		    vrm_xlate(g, ch->ctx, cmd.arg0, 4, 0, &p))
			return 1;
		return (s32)(vrm_load_acquire32((u32 *)p) - (u32)cmd.arg1) >= 0;
	}
	return 1;
}

static int chan_has_work(struct vrm_channel *ch)
{
	return ch->state == VRM_CHAN_ACTIVE &&
	       ch->ctrl->get != vrm_load_acquire32(&ch->ctrl->put);
}

static void chan_fault(struct vrm_gpu *g, struct vrm_channel *ch, u32 code,
		       u64 va, u32 op)
{
	struct vrm_fault_rec *f = &g->faults[g->fault_seq % VRM_FAULT_LOG];

	ch->state = VRM_CHAN_ERROR;
	vrm_store_release32(&ch->ctrl->error, code);
	f->ts_ns = vrm_now_ns();
	f->chan = ch->id;
	f->code = code;
	f->va = va;
	f->op = op;
	g->fault_seq++;
	vrm_trace(g, VGPU_EV_FAULT, ch->id, ch->engine, code, va);
}

/* ---- command execution ------------------------------------------------ */

enum exec_result { EXEC_RAN, EXEC_IDLE, EXEC_BLOCKED, EXEC_FAULT };

static enum exec_result exec_one(struct vrm_gpu *g, struct vrm_channel *ch,
				 u64 *cost_ns, int *signal)
{
	struct vgpu_cmd cmd;
	struct vrm_ctx *ctx = ch->ctx;
	u32 get = ch->ctrl->get;
	u32 put = vrm_load_acquire32(&ch->ctrl->put);
	u32 code, i;
	u64 fva = 0, cost = CMD_BASE_NS;
	void *dst = NULL, *a = NULL, *b = NULL;

	memset(&cmd, 0, sizeof(cmd));
	if (get == put)
		return EXEC_IDLE;
	if (put - get > ch->ring_entries) {
		chan_fault(g, ch, VGPU_FAULT_RING, 0, 0);
		*signal = 1;
		return EXEC_FAULT;
	}
	code = fetch_cmd(g, ch, get, &cmd, &fva);
	if (code)
		goto fault;

	switch (cmd.op) {
	case VGPU_OP_NOP:
		break;

	case VGPU_OP_MEMSET:
		if (ch->engine != VGPU_ENGINE_COPY) {
			code = VGPU_FAULT_WRONG_ENGINE;
			goto fault;
		}
		fva = cmd.arg0;
		if (!cmd.arg1 || cmd.arg1 > MAX_CMD_BYTES) {
			code = VGPU_FAULT_BAD_ARG;
			goto fault;
		}
		code = vrm_xlate(g, ctx, cmd.arg0, cmd.arg1, 1, &dst);
		if (code)
			goto fault;
		memset(dst, (int)(cmd.arg2 & 0xff), cmd.arg1);
		cost += cmd.arg1 >> 3;          /* 8 bytes / ns */
		break;

	case VGPU_OP_MEMCPY:
		if (ch->engine != VGPU_ENGINE_COPY) {
			code = VGPU_FAULT_WRONG_ENGINE;
			goto fault;
		}
		fva = cmd.arg0;
		if (!cmd.arg2 || cmd.arg2 > MAX_CMD_BYTES) {
			code = VGPU_FAULT_BAD_ARG;
			goto fault;
		}
		code = vrm_xlate(g, ctx, cmd.arg0, cmd.arg2, 1, &dst);
		if (code)
			goto fault;
		fva = cmd.arg1;
		code = vrm_xlate(g, ctx, cmd.arg1, cmd.arg2, 0, &a);
		if (code)
			goto fault;
		memmove(dst, a, cmd.arg2);
		cost += cmd.arg2 >> 3;
		break;

	case VGPU_OP_VEC_ADD:
	case VGPU_OP_VEC_MUL: {
		u32 *d, *x, *y;
		u64 bytes = (u64)cmd.count * 4;

		if (ch->engine != VGPU_ENGINE_COMPUTE) {
			code = VGPU_FAULT_WRONG_ENGINE;
			goto fault;
		}
		fva = cmd.arg0;
		if (!cmd.count || cmd.count > MAX_VEC_ELEMS ||
		    ((cmd.arg0 | cmd.arg1 | cmd.arg2) & 3)) {
			code = VGPU_FAULT_BAD_ARG;
			goto fault;
		}
		code = vrm_xlate(g, ctx, cmd.arg0, bytes, 1, &dst);
		if (code)
			goto fault;
		fva = cmd.arg1;
		code = vrm_xlate(g, ctx, cmd.arg1, bytes, 0, &a);
		if (code)
			goto fault;
		fva = cmd.arg2;
		code = vrm_xlate(g, ctx, cmd.arg2, bytes, 0, &b);
		if (code)
			goto fault;
		d = dst;
		x = a;
		y = b;
		if (cmd.op == VGPU_OP_VEC_ADD)
			for (i = 0; i < cmd.count; i++)
				d[i] = x[i] + y[i];
		else
			for (i = 0; i < cmd.count; i++)
				d[i] = x[i] * y[i];
		cost += (u64)cmd.count << 2;    /* 4 ns / element */
		break;
	}

	case VGPU_OP_FENCE:
		fva = cmd.arg0;
		if (cmd.arg0 & 3) {
			code = VGPU_FAULT_BAD_ARG;
			goto fault;
		}
		code = vrm_xlate(g, ctx, cmd.arg0, 4, 1, &dst);
		if (code)
			goto fault;
		vrm_store_release32((u32 *)dst, (u32)cmd.arg1);
		*signal = 1;
		break;

	case VGPU_OP_WAIT_SEM:
		fva = cmd.arg0;
		if (cmd.arg0 & 3) {
			code = VGPU_FAULT_BAD_ARG;
			goto fault;
		}
		code = vrm_xlate(g, ctx, cmd.arg0, 4, 0, &a);
		if (code)
			goto fault;
		if ((s32)(vrm_load_acquire32((u32 *)a) - (u32)cmd.arg1) < 0) {
			vrm_trace(g, VGPU_EV_BLOCKED, ch->id, ch->engine, 0, 0);
			return EXEC_BLOCKED;
		}
		break;

	default:
		code = VGPU_FAULT_BAD_OP;
		goto fault;
	}

	vrm_trace(g, VGPU_EV_CMD, ch->id, ch->engine, cmd.op, cost);
	vrm_store_release32(&ch->ctrl->get, get + 1);
	ch->cmds_executed++;
	ch->busy_ns += cost;
	*cost_ns = cost;
	return EXEC_RAN;

fault:
	chan_fault(g, ch, code, fva, cmd.op);
	*signal = 1;
	return EXEC_FAULT;
}

/* ---- scheduler -------------------------------------------------------- */

static void end_slice(struct vrm_gpu *g, struct vrm_engine *eng, u32 e)
{
	if (!eng->cur)
		return;
	vrm_trace(g, VGPU_EV_SLICE_END, eng->cur->id, e, 0,
		  eng->vclock_ns - eng->slice_start_ns);
	eng->cur = NULL;
}

static struct vrm_channel *pick_next(struct vrm_gpu *g, struct vrm_engine *eng,
				     u32 e)
{
	u32 i;

	for (i = 1; i <= VRM_MAX_CHAN; i++) {
		u32 idx = (eng->last_idx + i) % VRM_MAX_CHAN;
		struct vrm_channel *ch = &g->chan[idx];

		if (ch->state != VRM_CHAN_ACTIVE || ch->engine != e)
			continue;
		if (!chan_ready(g, ch))
			continue;
		eng->last_idx = idx;
		return ch;
	}
	return NULL;
}

void vrm_engine_step(struct vrm_gpu *g, u32 e, struct vrm_step *st)
{
	struct vrm_engine *eng = &g->eng[e];
	struct vrm_channel *ch;
	u64 cost = 0;
	int signal = 0;
	enum exec_result r;

	st->kind = VRM_STEP_IDLE;
	st->sleep_ns = 0;
	st->signal = 0;

	vrm_lock(&g->lock);

	ch = eng->cur;
	if (ch && (eng->vclock_ns >= eng->slice_end_ns || !chan_ready(g, ch))) {
		end_slice(g, eng, e);
		ch = NULL;
	}
	if (!ch) {
		ch = pick_next(g, eng, e);
		if (!ch) {
			u32 i;

			for (i = 0; i < VRM_MAX_CHAN; i++)
				if (g->chan[i].engine == e &&
				    chan_has_work(&g->chan[i]))
					st->kind = VRM_STEP_BLOCKED;
			goto out;
		}
		eng->cur = ch;
		eng->slice_start_ns = eng->vclock_ns;
		eng->slice_end_ns = eng->vclock_ns + g->timeslice_ns[ch->prio];
		eng->switches++;
		ch->slices++;
		vrm_trace(g, VGPU_EV_SLICE_BEGIN, ch->id, e, ch->prio, 0);
	}

	r = exec_one(g, ch, &cost, &signal);
	if (r == EXEC_RAN) {
		eng->vclock_ns += cost;
		eng->busy_ns += cost;
		eng->cmds++;
		st->kind = VRM_STEP_RAN;
		st->sleep_ns = cost;
	} else {
		end_slice(g, eng, e);
		st->kind = (r == EXEC_BLOCKED) ? VRM_STEP_BLOCKED : VRM_STEP_IDLE;
		if (r == EXEC_FAULT)
			st->kind = VRM_STEP_RAN;   /* more work may be queued */
	}
out:
	st->signal = signal;
	vrm_unlock(&g->lock);
}

/* ---- channel management ----------------------------------------------- */

void vrm_chan_free_locked(struct vrm_gpu *g, struct vrm_channel *ch)
{
	struct vrm_engine *eng = &g->eng[ch->engine];

	if (eng->cur == ch)
		end_slice(g, eng, ch->engine);
	if (ch->ctrl_obj) {
		ch->ctrl_obj->mmap_refs = 0;
		ch->ctrl_obj->internal = 0;
		vrm_mem_free_locked(g, ch->ctx, ch->ctrl_obj);
	}
	if (ch->ctx->nchan)
		ch->ctx->nchan--;
	memset(ch, 0, sizeof(*ch));
}

int vrm_chan_create(struct vrm_gpu *g, struct vrm_ctx *ctx, u32 engine,
		    u32 prio, u64 ring_va, u32 ring_entries, u32 *chan_id,
		    u32 *ctrl_handle, u64 *ctrl_va)
{
	struct vrm_channel *ch = NULL;
	struct vrm_obj *ctrl;
	void *p;
	u32 i;
	int ret;

	if (engine >= VGPU_NUM_ENGINES || prio >= VGPU_NUM_PRIOS ||
	    ring_entries < 2 || ring_entries > MAX_RING_ENTRIES ||
	    (ring_entries & (ring_entries - 1)) || (ring_va & 31))
		return -EINVAL;

	vrm_lock(&g->lock);
	if (vrm_xlate(g, ctx, ring_va, (u64)ring_entries * sizeof(struct vgpu_cmd),
		      0, &p)) {
		ret = -EFAULT;
		goto out;
	}
	for (i = 0; i < VRM_MAX_CHAN; i++) {
		if (g->chan[i].state == VRM_CHAN_FREE) {
			ch = &g->chan[i];
			break;
		}
	}
	if (!ch) {
		ret = -EBUSY;
		goto out;
	}
	ret = vrm_mem_alloc_locked(g, ctx, VGPU_PAGE_SIZE, 0, 1, &ctrl);
	if (ret)
		goto out;

	memset(ch, 0, sizeof(*ch));
	ch->state = VRM_CHAN_ACTIVE;
	ch->id = i + 1;
	ch->ctx = ctx;
	ch->engine = engine;
	ch->prio = prio;
	ch->ring_va = ring_va;
	ch->ring_entries = ring_entries;
	ch->ctrl_obj = ctrl;
	ch->ctrl = (struct vgpu_chan_ctrl *)(g->vram +
			((u64)ctrl->first_page << VGPU_PAGE_SHIFT));
	ctx->nchan++;

	*chan_id = ch->id;
	*ctrl_handle = ctrl->handle;
	*ctrl_va = ctrl->va;
out:
	vrm_unlock(&g->lock);
	return ret;
}

int vrm_chan_destroy(struct vrm_gpu *g, struct vrm_ctx *ctx, u32 chan_id)
{
	struct vrm_channel *ch;
	int ret = 0;

	vrm_lock(&g->lock);
	ch = chan_by_id(g, ctx, chan_id);
	if (!ch)
		ret = -ENOENT;
	else if (ch->ctrl_obj->mmap_refs)
		ret = -EBUSY;       /* unmap the control page first */
	else
		vrm_chan_free_locked(g, ch);
	vrm_unlock(&g->lock);
	return ret;
}

int vrm_chan_doorbell(struct vrm_gpu *g, struct vrm_ctx *ctx, u32 chan_id)
{
	struct vrm_channel *ch;
	int ret = 0;

	vrm_lock(&g->lock);
	ch = chan_by_id(g, ctx, chan_id);
	if (!ch)
		ret = -ENOENT;
	else if (ch->state == VRM_CHAN_ERROR)
		ret = -EIO;
	else
		vrm_trace(g, VGPU_EV_DOORBELL, ch->id, ch->engine, 0, 0);
	vrm_unlock(&g->lock);
	return ret;
}

int vrm_chan_stats(struct vrm_gpu *g, struct vrm_ctx *ctx,
		   struct vgpu_chan_stats *st)
{
	struct vrm_channel *ch;
	int ret = 0;

	vrm_lock(&g->lock);
	ch = chan_by_id(g, ctx, st->chan_id);
	if (!ch) {
		ret = -ENOENT;
	} else {
		st->state = ch->state;
		st->error = ch->ctrl->error;
		st->engine = ch->engine;
		st->priority = ch->prio;
		st->get = ch->ctrl->get;
		st->put = ch->ctrl->put;
		st->cmds_executed = ch->cmds_executed;
		st->busy_ns = ch->busy_ns;
		st->slices = ch->slices;
	}
	vrm_unlock(&g->lock);
	return ret;
}

int vrm_sem_poll(struct vrm_gpu *g, struct vrm_ctx *ctx, u64 va, u32 value,
		 u32 *observed, int *ready)
{
	void *p;
	u32 i, v;
	int ret = 0;

	if (va & 3)
		return -EINVAL;
	vrm_lock(&g->lock);
	if (vrm_xlate(g, ctx, va, 4, 0, &p)) {
		ret = -EFAULT;
		goto out;
	}
	v = vrm_load_acquire32((u32 *)p);
	*observed = v;
	*ready = (s32)(v - value) >= 0;
	if (!*ready) {
		for (i = 0; i < VRM_MAX_CHAN; i++)
			if (g->chan[i].state == VRM_CHAN_ERROR &&
			    g->chan[i].ctx == ctx)
				ret = -EIO;
	}
out:
	vrm_unlock(&g->lock);
	return ret;
}
