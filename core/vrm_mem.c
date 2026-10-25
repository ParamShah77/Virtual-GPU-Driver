// SPDX-License-Identifier: GPL-2.0
/* VRAM allocator, per-context page tables and GPU VA allocation. */
#include "vrm_core.h"

#define PAGE_BYTES  ((u64)VGPU_PAGE_SIZE)

static inline int bm_test(const struct vrm_gpu *g, u32 i)
{
	return (g->bitmap[i >> 5] >> (i & 31)) & 1u;
}

static inline void bm_set(struct vrm_gpu *g, u32 i)
{
	g->bitmap[i >> 5] |= 1u << (i & 31);
}

static inline void bm_clear(struct vrm_gpu *g, u32 i)
{
	g->bitmap[i >> 5] &= ~(1u << (i & 31));
}

/* ---- lifecycle -------------------------------------------------------- */

int vrm_gpu_init(struct vrm_gpu *g, void *vram, u64 vram_bytes)
{
	/* PTEs hold a 20-bit page number, so at most 4 GiB of VRAM. */
	if (!vram || vram_bytes < 16 * PAGE_BYTES ||
	    (vram_bytes & (PAGE_BYTES - 1)) || vram_bytes > (1ull << 32))
		return -EINVAL;

	memset(g, 0, sizeof(*g));
	g->vram = vram;
	g->vram_bytes = vram_bytes;
	g->vram_pages = (u32)(vram_bytes >> VGPU_PAGE_SHIFT);
	g->free_pages = g->vram_pages;
	g->bitmap = vrm_zalloc(((g->vram_pages + 31) / 32) * sizeof(u32));
	if (!g->bitmap)
		return -ENOMEM;

	g->timeslice_ns[VGPU_PRIO_LOW]    = VRM_SLICE_LOW_NS;
	g->timeslice_ns[VGPU_PRIO_NORMAL] = VRM_SLICE_NORMAL_NS;
	g->timeslice_ns[VGPU_PRIO_HIGH]   = VRM_SLICE_HIGH_NS;
	vrm_lock_init(&g->lock);
	return 0;
}

void vrm_gpu_fini(struct vrm_gpu *g)
{
	u32 i;

	for (i = 0; i < VRM_MAX_CTX; i++)
		if (g->ctx[i].in_use)
			vrm_ctx_destroy(g, &g->ctx[i]);
	vrm_free(g->bitmap);
	g->bitmap = NULL;
	vrm_lock_destroy(&g->lock);
}

/* ---- page tables ------------------------------------------------------ */

static int vm_map(struct vrm_ctx *ctx, u64 va, u32 first_page, u32 npages,
		  u32 flags)
{
	u32 i;

	for (i = 0; i < npages; i++) {
		u32 p = (u32)(va >> VGPU_PAGE_SHIFT) + i;
		struct vrm_pt *pt = ctx->pgd[p >> 10];
		u32 pte = (first_page + i) << VGPU_PAGE_SHIFT;

		if (!pt) {
			pt = vrm_zalloc(sizeof(*pt));
			if (!pt)
				return -ENOMEM;
			ctx->pgd[p >> 10] = pt;
		}
		pte |= VRM_PTE_VALID;
		if (!(flags & VGPU_MEM_GPU_READONLY))
			pte |= VRM_PTE_WRITE;
		pt->pte[p & 1023] = pte;
	}
	return 0;
}

static void vm_unmap(struct vrm_ctx *ctx, u64 va, u32 npages)
{
	u32 i;

	for (i = 0; i < npages; i++) {
		u32 p = (u32)(va >> VGPU_PAGE_SHIFT) + i;
		struct vrm_pt *pt = ctx->pgd[p >> 10];

		if (pt)
			pt->pte[p & 1023] = 0;
	}
}

u32 vrm_xlate(struct vrm_gpu *g, struct vrm_ctx *ctx, u64 va, u64 len,
	      int write, void **out)
{
	u32 first, last, p, ph0 = 0;

	if (!len)
		return VGPU_FAULT_BAD_ARG;
	if (va < VRM_VA_BASE || va + len < va || va + len > VRM_VA_LIMIT)
		return VGPU_FAULT_UNMAPPED;

	first = (u32)(va >> VGPU_PAGE_SHIFT);
	last = (u32)((va + len - 1) >> VGPU_PAGE_SHIFT);
	for (p = first; p <= last; p++) {
		struct vrm_pt *pt = ctx->pgd[p >> 10];
		u32 pte, ph;

		if (!pt)
			return VGPU_FAULT_UNMAPPED;
		pte = pt->pte[p & 1023];
		if (!(pte & VRM_PTE_VALID))
			return VGPU_FAULT_UNMAPPED;
		if (write && !(pte & VRM_PTE_WRITE))
			return VGPU_FAULT_READONLY;
		ph = pte >> VGPU_PAGE_SHIFT;
		if (p == first)
			ph0 = ph;
		else if (ph != ph0 + (p - first))
			return VGPU_FAULT_DISCONTIG;
	}
	*out = g->vram + ((u64)ph0 << VGPU_PAGE_SHIFT) + (va & (PAGE_BYTES - 1));
	return VGPU_FAULT_NONE;
}

/* ---- VRAM page allocator (first fit over a bitmap) --------------------- */

static int vram_alloc_run(struct vrm_gpu *g, u32 npages, u32 *first)
{
	u32 i, run = 0, start = 0;

	if (npages > g->free_pages)
		return -ENOMEM;
	for (i = 0; i < g->vram_pages; i++) {
		if (bm_test(g, i)) {
			run = 0;
			continue;
		}
		if (run == 0)
			start = i;
		if (++run == npages) {
			*first = start;
			return 0;
		}
	}
	return -ENOMEM;
}

static void vram_take(struct vrm_gpu *g, u32 first, u32 npages)
{
	u32 i;

	for (i = 0; i < npages; i++)
		bm_set(g, first + i);
	g->free_pages -= npages;
	/* Never leak another client's data through a recycled page. */
	memset(g->vram + ((u64)first << VGPU_PAGE_SHIFT), 0,
	       (u64)npages << VGPU_PAGE_SHIFT);
}

static void vram_release(struct vrm_gpu *g, u32 first, u32 npages)
{
	u32 i;

	for (i = 0; i < npages; i++)
		bm_clear(g, first + i);
	g->free_pages += npages;
}

u32 vrm_vram_largest_free(struct vrm_gpu *g)
{
	u32 i, run = 0, best = 0;

	for (i = 0; i < g->vram_pages; i++) {
		if (bm_test(g, i)) {
			run = 0;
			continue;
		}
		if (++run > best)
			best = run;
	}
	return best;
}

/* ---- GPU VA allocator: first fit between sorted objects, 1 guard page --- */

static int va_find(struct vrm_ctx *ctx, u32 npages, u64 *va,
		   struct vrm_obj ***link)
{
	struct vrm_obj **pp = &ctx->objs;
	u64 cur = VRM_VA_BASE;
	u64 need = (u64)npages * PAGE_BYTES;

	while (*pp) {
		struct vrm_obj *o = *pp;

		if (o->va >= cur + need + PAGE_BYTES)
			break;
		cur = o->va + (u64)o->npages * PAGE_BYTES + PAGE_BYTES;
		pp = &o->next;
	}
	if (cur + need + PAGE_BYTES > VRM_VA_LIMIT)
		return -ENOMEM;
	*va = cur;
	*link = pp;
	return 0;
}

/* ---- contexts --------------------------------------------------------- */

int vrm_ctx_create(struct vrm_gpu *g, struct vrm_ctx **out)
{
	u32 i;
	int ret = -EBUSY;

	vrm_lock(&g->lock);
	for (i = 0; i < VRM_MAX_CTX; i++) {
		struct vrm_ctx *c = &g->ctx[i];

		if (c->in_use)
			continue;
		c->pgd = vrm_zalloc(1024 * sizeof(*c->pgd));
		if (!c->pgd) {
			ret = -ENOMEM;
			break;
		}
		c->in_use = 1;
		c->id = i + 1;
		c->next_handle = 1;
		*out = c;
		ret = 0;
		break;
	}
	vrm_unlock(&g->lock);
	return ret;
}

void vrm_ctx_destroy(struct vrm_gpu *g, struct vrm_ctx *ctx)
{
	u32 i;

	vrm_lock(&g->lock);
	for (i = 0; i < VRM_MAX_CHAN; i++)
		if (g->chan[i].state != VRM_CHAN_FREE && g->chan[i].ctx == ctx)
			vrm_chan_free_locked(g, &g->chan[i]);
	while (ctx->objs) {
		struct vrm_obj *o = ctx->objs;

		ctx->objs = o->next;
		vram_release(g, o->first_page, o->npages);
		vrm_free(o);
	}
	for (i = 0; i < 1024; i++)
		vrm_free(ctx->pgd[i]);
	vrm_free(ctx->pgd);
	memset(ctx, 0, sizeof(*ctx));
	vrm_unlock(&g->lock);
}

/* ---- memory objects --------------------------------------------------- */

struct vrm_obj *vrm_obj_find(struct vrm_ctx *ctx, u32 handle)
{
	struct vrm_obj *o;

	for (o = ctx->objs; o; o = o->next)
		if (o->handle == handle)
			return o;
	return NULL;
}

int vrm_mem_alloc_locked(struct vrm_gpu *g, struct vrm_ctx *ctx, u64 bytes,
			 u32 flags, int internal, struct vrm_obj **out)
{
	struct vrm_obj *o, **link;
	u32 npages, first;
	u64 va;
	int ret;

	if (!bytes || bytes > g->vram_bytes ||
	    (flags & ~VGPU_MEM_GPU_READONLY))
		return -EINVAL;
	npages = (u32)((bytes + PAGE_BYTES - 1) >> VGPU_PAGE_SHIFT);

	ret = vram_alloc_run(g, npages, &first);
	if (ret)
		return ret;
	ret = va_find(ctx, npages, &va, &link);
	if (ret)
		return ret;
	o = vrm_zalloc(sizeof(*o));
	if (!o)
		return -ENOMEM;
	ret = vm_map(ctx, va, first, npages, flags);
	if (ret) {
		vm_unmap(ctx, va, npages);
		vrm_free(o);
		return ret;
	}
	vram_take(g, first, npages);

	o->handle = ctx->next_handle++;
	o->first_page = first;
	o->npages = npages;
	o->flags = flags;
	o->internal = internal;
	o->va = va;
	o->next = *link;
	*link = o;
	ctx->mem_bytes += (u64)npages * PAGE_BYTES;
	*out = o;
	return 0;
}

void vrm_mem_free_locked(struct vrm_gpu *g, struct vrm_ctx *ctx,
			 struct vrm_obj *obj)
{
	struct vrm_obj **pp;

	for (pp = &ctx->objs; *pp; pp = &(*pp)->next) {
		if (*pp != obj)
			continue;
		*pp = obj->next;
		break;
	}
	vm_unmap(ctx, obj->va, obj->npages);
	vram_release(g, obj->first_page, obj->npages);
	ctx->mem_bytes -= (u64)obj->npages * PAGE_BYTES;
	vrm_free(obj);
}

int vrm_mem_alloc(struct vrm_gpu *g, struct vrm_ctx *ctx, u64 bytes, u32 flags,
		  u32 *handle, u64 *va, u64 *alloc_bytes)
{
	struct vrm_obj *o;
	int ret;

	vrm_lock(&g->lock);
	ret = vrm_mem_alloc_locked(g, ctx, bytes, flags, 0, &o);
	if (!ret) {
		*handle = o->handle;
		*va = o->va;
		*alloc_bytes = (u64)o->npages * PAGE_BYTES;
	}
	vrm_unlock(&g->lock);
	return ret;
}

int vrm_mem_free(struct vrm_gpu *g, struct vrm_ctx *ctx, u32 handle)
{
	struct vrm_obj *o;
	int ret = 0;

	vrm_lock(&g->lock);
	o = vrm_obj_find(ctx, handle);
	if (!o || o->internal)
		ret = -ENOENT;
	else if (o->mmap_refs)
		ret = -EBUSY;
	else
		vrm_mem_free_locked(g, ctx, o);
	vrm_unlock(&g->lock);
	return ret;
}

int vrm_mem_map_get(struct vrm_gpu *g, struct vrm_ctx *ctx, u32 handle,
		    u64 *vram_off, u64 *bytes)
{
	struct vrm_obj *o;
	int ret = 0;

	vrm_lock(&g->lock);
	o = vrm_obj_find(ctx, handle);
	if (!o) {
		ret = -ENOENT;
	} else {
		o->mmap_refs++;
		*vram_off = (u64)o->first_page << VGPU_PAGE_SHIFT;
		*bytes = (u64)o->npages * PAGE_BYTES;
	}
	vrm_unlock(&g->lock);
	return ret;
}

void vrm_mem_map_put(struct vrm_gpu *g, struct vrm_ctx *ctx, u32 handle)
{
	struct vrm_obj *o;

	vrm_lock(&g->lock);
	o = vrm_obj_find(ctx, handle);
	if (o && o->mmap_refs)
		o->mmap_refs--;
	vrm_unlock(&g->lock);
}
