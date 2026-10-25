// SPDX-License-Identifier: GPL-2.0
/* Tracing and introspection. */
#include "vrm_core.h"

/* Caller holds g->lock. */
void vrm_trace(struct vrm_gpu *g, u32 type, u32 chan, u32 engine, u32 arg,
	       u64 arg2)
{
	struct vgpu_trace_ev *ev =
		&g->trace[g->trace_seq & (VRM_TRACE_ENTRIES - 1)];

	ev->ts_ns = vrm_now_ns();
	ev->type = type;
	ev->chan = chan;
	ev->engine = engine;
	ev->arg = arg;
	ev->arg2 = arg2;
	g->trace_seq++;
}

void vrm_trace_read(struct vrm_gpu *g, u64 from_seq, struct vgpu_trace_ev *out,
		    u32 max, u32 *n_out, u64 *next_seq, u64 *dropped)
{
	u64 oldest, seq;
	u32 n = 0;

	vrm_lock(&g->lock);
	oldest = g->trace_seq > VRM_TRACE_ENTRIES ?
		 g->trace_seq - VRM_TRACE_ENTRIES : 0;
	*dropped = 0;
	if (from_seq < oldest) {
		*dropped = oldest - from_seq;
		from_seq = oldest;
	}
	for (seq = from_seq; seq < g->trace_seq && n < max; seq++)
		out[n++] = g->trace[seq & (VRM_TRACE_ENTRIES - 1)];
	*n_out = n;
	*next_seq = seq;
	vrm_unlock(&g->lock);
}

void vrm_gpu_get_info(struct vrm_gpu *g, struct vgpu_info *out)
{
	u32 i, nctx = 0;

	memset(out, 0, sizeof(*out));
	vrm_lock(&g->lock);
	out->vram_bytes = g->vram_bytes;
	out->vram_free_bytes = (u64)g->free_pages << VGPU_PAGE_SHIFT;
	out->vram_largest_free_bytes =
		(u64)vrm_vram_largest_free(g) << VGPU_PAGE_SHIFT;
	out->page_size = VGPU_PAGE_SIZE;
	out->n_engines = VGPU_NUM_ENGINES;
	out->max_channels = VRM_MAX_CHAN;
	for (i = 0; i < VRM_MAX_CTX; i++)
		nctx += g->ctx[i].in_use ? 1 : 0;
	out->n_contexts = nctx;
	for (i = 0; i < VGPU_NUM_PRIOS; i++)
		out->timeslice_ns[i] = g->timeslice_ns[i];
	for (i = 0; i < VGPU_NUM_ENGINES; i++) {
		out->eng[i].busy_ns = g->eng[i].busy_ns;
		out->eng[i].cmds = g->eng[i].cmds;
		out->eng[i].switches = g->eng[i].switches;
	}
	out->now_ns = vrm_now_ns();
	out->doorbells = g->doorbells;
	vrm_unlock(&g->lock);
}
