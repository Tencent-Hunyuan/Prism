
from typing import Optional, Tuple

import torch

# NOTE: `bsa_interface` pulls in Triton. We import its symbols lazily (inside the
# functions that launch kernels) so that the pure-PyTorch geometry helpers in
# this module remain importable / unit-testable on machines without Triton.
from .dynamic_block_shape import (
    ALL_CANDIDATE_SHAPES,
    ZONE_SIZE,
    select_zone_shapes,
)

_TILE = 64  # kernel tile size (Section 8.7: unified BLOCK_N=64)


# =====================================================================
# Per-shape local permutation precompute (constant; cached per device).
# =====================================================================
def _precompute_local_orders(device):
    """For each candidate shape, the [512] permutation mapping a rearranged
    within-zone position -> local flat coord (lt*64 + lh*8 + lw).

    Also returns, per shape, (product, tiles_per_block). All cached.
    """
    cache = _precompute_local_orders._cache
    if device in cache:
        return cache[device]

    zs = ZONE_SIZE
    n = zs * zs * zs  # 512
    lf = torch.arange(n, device=device)
    lt = lf // (zs * zs)
    lh = (lf // zs) % zs
    lw = lf % zs

    orders = []
    prods = []
    tpb = []
    for (bt, bh, bw) in ALL_CANDIDATE_SHAPES:
        bsz = bt * bh * bw
        Nh, Nw = zs // bh, zs // bw
        nt, it = lt // bt, lt % bt
        nh, ih = lh // bh, lh % bh
        nw, iw = lw // bw, lw % bw
        blk = (nt * Nh + nh) * Nw + nw                 # block index within zone
        off = (it * bh + ih) * bw + iw                 # offset within block
        rearr_pos = blk * bsz + off                    # [512]: local_flat -> rearr pos
        order = torch.empty(n, dtype=torch.long, device=device)
        order[rearr_pos] = lf                          # invert -> rearr pos -> local_flat
        orders.append(order)
        prods.append(bsz)
        tpb.append(bsz // _TILE)

    local_orders = torch.stack(orders, dim=0)                                   # [C, 512]
    products = torch.tensor(prods, dtype=torch.long, device=device)            # [C]
    tiles_per_block = torch.tensor(tpb, dtype=torch.long, device=device)       # [C]
    cache[device] = (local_orders, products, tiles_per_block)
    return cache[device]


_precompute_local_orders._cache = {}


# =====================================================================
# Pure BSA Triton wrapper (forward + backward through the kernel only).
# Mirrors bias_rectification._bsa_kernel_only but lives here to keep the
# dynamic path fully isolated. Selection / pooling stay in PyTorch outside.
# =====================================================================
class _dyn_bsa_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, block_indices, block_indices_lens,
                chunk_size_q, chunk_size_k, sparsity, kv_valid_mask):
        from .bsa_interface import attn_fwd_bsa_varlen_triton
        o, lse = attn_fwd_bsa_varlen_triton(
            q, k, v, sm_scale, block_indices, block_indices_lens,
            chunk_size_q, chunk_size_k, sparsity, kv_valid_mask=kv_valid_mask,
        )
        ctx.save_for_backward(q, k, v, o, lse, block_indices, block_indices_lens)
        ctx.sm_scale = sm_scale
        ctx.chunk_size_q = chunk_size_q
        ctx.chunk_size_k = chunk_size_k
        ctx.sparsity = sparsity
        ctx.kv_valid_mask = kv_valid_mask
        return o

    @staticmethod
    def backward(ctx, do):
        from .bsa_interface import attn_bwd_bsa_varlen_triton
        q, k, v, o, lse, block_indices, block_indices_lens = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        attn_bwd_bsa_varlen_triton(
            do.contiguous(), q, k, v, o, dq, dk, dv,
            ctx.sm_scale, lse, block_indices, block_indices_lens,
            ctx.chunk_size_q, ctx.chunk_size_k, ctx.sparsity,
            kv_valid_mask=ctx.kv_valid_mask,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


# =====================================================================
# Geometry helpers (no grad).
# =====================================================================
def _build_rearrange_index(shape_ids, ZT, ZH, ZW, H, W, device):
    """Return src_idx[S], inv_idx[S] for the per-zone anisotropic rearrange.

    src_idx[rearr_pos] = original_thw_index ; inv_idx = its inverse permutation.
    Both contiguous int64.
    """
    zs = ZONE_SIZE
    Nz = ZT * ZH * ZW
    local_orders, _, _ = _precompute_local_orders(device)
    lo = local_orders[shape_ids]                       # [Nz, 512] local flat per rearr pos
    lt = lo // (zs * zs)
    lh = (lo // zs) % zs
    lw = lo % zs

    zidx = torch.arange(Nz, device=device)
    zt = zidx // (ZH * ZW)
    zh = (zidx // ZW) % ZH
    zw = zidx % ZW
    gt = zt[:, None] * zs + lt
    gh = zh[:, None] * zs + lh
    gw = zw[:, None] * zs + lw
    global_thw = (gt * H + gh) * W + gw                # [Nz, 512]
    src_idx = global_thw.reshape(-1).contiguous()      # [S]
    # src_idx is a permutation, so its inverse is an O(S) scatter (cheaper than
    # an O(S log S) argsort): inv_idx[src_idx[r]] = r.
    S = src_idx.shape[0]
    inv_idx = torch.empty_like(src_idx)
    inv_idx[src_idx] = torch.arange(S, device=device)
    return src_idx, inv_idx


def _build_block_tile_maps(shape_ids, ZT, ZH, ZW, device):
    """Logical-block <-> tile bookkeeping for the rearranged sequence.

    Returns:
      block_id_of_tile [N_tiles]   : logical-block id of each 64-tile
      kblock_tiles     [N_b+1, 2]  : the (up-to-2) tile ids of each logical block
                                     (row N_b is the sentinel = N_tiles, used for
                                     invalid/sentinel selection slots)
      N_b (int), N_tiles (int)
    """
    Nz = ZT * ZH * ZW
    _, products, tiles_per_block = _precompute_local_orders(device)
    tpb = tiles_per_block[shape_ids]                   # [Nz] in {1,2}
    nbz = (512 // products[shape_ids])                 # blocks per zone [Nz] in {8,4}
    N_tiles = Nz * (512 // _TILE)                      # = Nz * 8
    N_b = int(nbz.sum().item())

    # exclusive cumsum of blocks-per-zone -> per-zone block offset
    block_offset = torch.zeros(Nz, dtype=torch.long, device=device)
    if Nz > 1:
        block_offset[1:] = torch.cumsum(nbz, dim=0)[:-1]

    tile_g = torch.arange(N_tiles, device=device)
    z_of_tile = tile_g // 8
    tl_local = tile_g % 8
    tpb_t = tpb[z_of_tile]
    block_local = tl_local // tpb_t
    pos_in_block = tl_local % tpb_t
    block_id_of_tile = block_offset[z_of_tile] + block_local   # [N_tiles]

    kblock_tiles = torch.full((N_b + 1, 2), N_tiles, dtype=torch.long, device=device)
    kblock_tiles[block_id_of_tile, pos_in_block] = tile_g
    return block_id_of_tile, kblock_tiles, N_b, N_tiles


def _logical_block_means(x_tile, tile_counts, block_id_of_tile, N_b):
    """Aggregate masked tile-means -> logical-block masked means.

    block_mean = sum(tile_mean * tile_valid_count) / sum(tile_valid_count),
    correctly averaging only valid tokens across the 1-or-2 tiles of a block.

    x_tile: [B,H,N_tiles,D]; tile_counts: [N_tiles]; returns ([B,H,N_b,D], block_count[N_b]).
    """
    B, Hn, N_tiles, D = x_tile.shape
    w = x_tile * tile_counts.view(1, 1, N_tiles, 1).to(x_tile.dtype)
    block_sum = x_tile.new_zeros(B, Hn, N_b, D)
    block_sum.index_add_(2, block_id_of_tile, w)
    block_count = tile_counts.new_zeros(N_b)
    block_count.index_add_(0, block_id_of_tile, tile_counts.to(block_count.dtype))
    block_mean = block_sum / block_count.clamp(min=1.0).view(1, 1, N_b, 1).to(x_tile.dtype)
    return block_mean, block_count


# =====================================================================
# Public entry point.
# =====================================================================
def flash_attn_bsa_3d_dynamic(
    q: torch.Tensor,           # [B, H, S, D] THW order (already padded to mult of ZONE_SIZE)
    k: torch.Tensor,
    v: torch.Tensor,
    grid_padded: Tuple[int, int, int],
    *,
    method: str,               # "ivpq" or "penalty"
    sparsity: Optional[float] = 0.9375,
    cdf_threshold: Optional[float] = None,
    audio_token_norms: Optional[torch.Tensor] = None,  # [S] THW order, or None
    valid_mask: Optional[torch.Tensor] = None,         # [S] bool THW order, or None
    lambda_a: float = 0.5,
    tau_128: float = 0.15,
    lambda_128: float = 1.0,
    shape_ids: Optional[torch.Tensor] = None,
    return_shape_ids: bool = False,
) -> torch.Tensor:
    """Dynamic-block-shape BSA for video self-attention. Returns [B,H,S,D] THW order.

    shape_ids: if given, skip the (non-deterministic) shape selection and use it
        verbatim — used to keep the checkpoint recompute identical to the forward.
    return_shape_ids: if True, also return the shape_ids used (for caching).
    """
    from .bsa_interface import (
        mean_pooling_compression,
        masked_mean_pooling_compression,
        cal_score,
        get_select_indices_topk_from_score,
        get_select_indices_cdf_from_score,
        get_select_indices_cdf_topk_from_score,
    )
    B, Hn, S, D = q.shape
    T, H, W = grid_padded
    assert T * H * W == S
    assert T % ZONE_SIZE == 0 and H % ZONE_SIZE == 0 and W % ZONE_SIZE == 0
    device = q.device
    sm_scale = 1.0 / (D ** 0.5)
    ZT, ZH, ZW = T // ZONE_SIZE, H // ZONE_SIZE, W // ZONE_SIZE

    # ---- (1) per-zone shape decision (head-agnostic, SP-consistent) ----
    with torch.no_grad():
        if shape_ids is None:
            v_headavg = v[0].mean(dim=0).view(T, H, W, D)          # [T,H,W,D]
            shape_ids = select_zone_shapes(
                v_headavg, audio_token_norms, grid_padded, method,
                lambda_a=lambda_a, tau_128=tau_128, lambda_128=lambda_128,
                valid_mask_thw=valid_mask, sp_reduce=True,
            )                                                       # [Nz] int64

        # ---- (2) build the rearrange + tile/block bookkeeping ----
        src_idx, inv_idx = _build_rearrange_index(shape_ids, ZT, ZH, ZW, H, W, device)
        block_id_of_tile, kblock_tiles, N_b, N_tiles = _build_block_tile_maps(
            shape_ids, ZT, ZH, ZW, device
        )

    # ---- (3) rearrange q/k/v into the tile-contiguous layout (differentiable) ----
    q_re = q.index_select(2, src_idx).contiguous()
    k_re = k.index_select(2, src_idx).contiguous()
    v_re = v.index_select(2, src_idx).contiguous()
    valid_re = valid_mask[src_idx].contiguous() if valid_mask is not None else None

    # ---- (4) logical-block scoring + selection (no grad — selection only) ----
    with torch.no_grad():
        if valid_re is not None:
            q_tile = masked_mean_pooling_compression(q_re, _TILE, valid_re)
            k_tile = masked_mean_pooling_compression(k_re, _TILE, valid_re)
            tile_counts = valid_re.view(N_tiles, _TILE).sum(dim=-1).float()
        else:
            q_tile = mean_pooling_compression(q_re, _TILE)
            k_tile = mean_pooling_compression(k_re, _TILE)
            tile_counts = torch.full((N_tiles,), float(_TILE), device=device)

        q_cmp, _ = _logical_block_means(q_tile, tile_counts, block_id_of_tile, N_b)
        k_cmp, block_count = _logical_block_means(k_tile, tile_counts, block_id_of_tile, N_b)
        block_valid = block_count > 0                          # [N_b]
        del q_tile, k_tile

        has_invalid = not bool(block_valid.all())
        invalid_mask = ~block_valid.view(1, 1, 1, -1) if has_invalid else None
        del block_valid, block_count

        sel_blocks_list = []
        for _h in range(Hn):
            score_h = cal_score(q_cmp[:, _h:_h+1], k_cmp[:, _h:_h+1])
            if invalid_mask is not None:
                score_h = score_h.masked_fill(invalid_mask, float('-inf'))
            if sparsity is not None and cdf_threshold is None:
                sel_h, _ = get_select_indices_topk_from_score(score_h, sparsity)
            elif sparsity is None and cdf_threshold is not None:
                sel_h, _ = get_select_indices_cdf_from_score(score_h, cdf_threshold, sm_scale)
            elif sparsity is not None and cdf_threshold is not None:
                sel_h, _ = get_select_indices_cdf_topk_from_score(score_h, sparsity, cdf_threshold, sm_scale)
            else:
                raise ValueError("Either sparsity or cdf_threshold must be provided")
            sel_blocks_list.append(sel_h)
            del score_h, sel_h
        del q_cmp, k_cmp, invalid_mask

        sel_blocks = torch.cat(sel_blocks_list, dim=1)
        del sel_blocks_list
        # sel_blocks values in [0, N_b]; N_b is the sentinel (cdf/varlen padding).
        sel_blocks = sel_blocks.clamp(max=N_b)                 # safety (int64 index)

        # ---- (5) expand selected logical blocks -> tile-level indices ----
        kblock_tiles = kblock_tiles.to(torch.int32)
        t0 = kblock_tiles[:, 0][sel_blocks]                    # [B,H,N_bq,r] int32
        t1 = kblock_tiles[:, 1][sel_blocks]
        del sel_blocks, kblock_tiles
        tile_sel = torch.cat([t0, t1], dim=-1)                 # [B,H,N_bq,2r] int32; values in [0, N_tiles]
        del t0, t1

        if valid_re is not None:
            tile_is_valid = tile_counts > 0                    # [N_tiles] bool
            in_range = tile_sel < N_tiles
            sel_valid = in_range & tile_is_valid[tile_sel.clamp(max=N_tiles - 1)]
            tile_sel = torch.where(sel_valid, tile_sel, torch.full_like(tile_sel, N_tiles))
        del tile_counts

        tile_sel, _ = torch.sort(tile_sel, dim=-1)             # int32 sort; valid first, sentinel(N_tiles) at tail
        tile_lens_block = (tile_sel < N_tiles).sum(dim=-1).to(torch.int32)

        # map q-block rows -> q-tile rows (a 128 q-block shares its row across its 2 tiles).
        # tile_sel is already int32, so index_select yields the int32 block indices directly.
        tile_block_indices = tile_sel.index_select(2, block_id_of_tile).contiguous()
        tile_block_indices_lens = tile_lens_block.index_select(2, block_id_of_tile).contiguous()
        del tile_sel, tile_lens_block, block_id_of_tile

    # ---- (6) BSA Triton kernel at chunk=64 (autograd through q_re/k_re/v_re) ----
    o_re = _dyn_bsa_kernel.apply(
        q_re, k_re, v_re, sm_scale,
        tile_block_indices, tile_block_indices_lens,
        _TILE, _TILE, sparsity, valid_re,
    )

    # ---- (7) scatter back to original THW order ----
    out = o_re.index_select(2, inv_idx).contiguous()
    if return_shape_ids:
        return out, shape_ids
    return out


__all__ = ["flash_attn_bsa_3d_dynamic"]
