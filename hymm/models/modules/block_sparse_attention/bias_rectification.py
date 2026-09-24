

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .bsa_interface import (
    mean_pooling_compression,
    masked_mean_pooling_compression,
    cal_score,
    get_select_indices_topk_from_score,
    get_select_indices_cdf_from_score,
    get_select_indices_cdf_topk_from_score,
    create_mask_from_indices_varlen,
    attn_fwd_bsa_varlen_triton,
    attn_bwd_bsa_varlen_triton,
)


# =====================================================================
# Pure BSA Triton wrapper — only this is an autograd Function.
# Everything else (pooling, scoring, correction) lives outside in
# autograd-friendly PyTorch ops so gradients flow correctly.
# =====================================================================
class _bsa_kernel_only(torch.autograd.Function):
    """Forward + backward through `attn_fwd_bsa_varlen_triton` only.

    Returns (o, lse_natural). lse is the log-sum-exp of the *attended* keys
    in natural log units (= LN2 * the kernel's log2 lse).
    """

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, block_indices, block_indices_lens,
                chunk_size_q, chunk_size_k, sparsity, kv_valid_mask):
        o, lse_natural = attn_fwd_bsa_varlen_triton(
            q, k, v, sm_scale, block_indices, block_indices_lens,
            chunk_size_q, chunk_size_k, sparsity,
            kv_valid_mask=kv_valid_mask,
        )
        ctx.save_for_backward(q, k, v, o, lse_natural, block_indices, block_indices_lens)
        ctx.sm_scale = sm_scale
        ctx.chunk_size_q = chunk_size_q
        ctx.chunk_size_k = chunk_size_k
        ctx.sparsity = sparsity
        ctx.kv_valid_mask = kv_valid_mask
        return o, lse_natural

    @staticmethod
    def backward(ctx, do, dlse_unused):
        q, k, v, o, lse, block_indices, block_indices_lens = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do = do.contiguous()
        attn_bwd_bsa_varlen_triton(
            do, q, k, v, o, dq, dk, dv,
            ctx.sm_scale, lse,
            block_indices, block_indices_lens,
            ctx.chunk_size_q, ctx.chunk_size_k, ctx.sparsity,
            kv_valid_mask=ctx.kv_valid_mask,
        )
        # 10 inputs: q, k, v, sm_scale, block_indices, block_indices_lens,
        #            chunk_size_q, chunk_size_k, sparsity, kv_valid_mask
        return dq, dk, dv, None, None, None, None, None, None, None


def _bsa_kernel(q, k, v, sm_scale, block_indices, block_indices_lens,
                chunk_size_q, chunk_size_k, sparsity, kv_valid_mask):
    """Wrapper that exposes (o, lse) but always detaches lse.

    `lse` is an attention normalisation statistic used downstream as a
    coefficient in the Taylor merge weights. We treat it as a fixed
    quantity (FlashAttention convention) and explicitly detach it so the
    autograd engine does not try to backpropagate through a path we
    intentionally do not implement in `_bsa_kernel_only.backward`.
    """
    o, lse = _bsa_kernel_only.apply(
        q, k, v, sm_scale, block_indices, block_indices_lens,
        chunk_size_q, chunk_size_k, sparsity, kv_valid_mask,
    )
    return o, lse.detach()


# =====================================================================
# Shared helpers (block-level pooling, selection, validity)
# =====================================================================
def _pool_qk(q, k, chunk_size_q, chunk_size_k, kv_valid_mask, q_valid_mask):
    """Pool Q and K to block representatives, masking padded tokens.

    Self-attn case (Q/K share grid): both use kv_valid_mask.
    Cross-attn case: each side uses its own valid_mask.

    All ops are autograd-tracked (gradients to q / k flow through pooling).
    """
    if q_valid_mask is not None:
        q_cmp = masked_mean_pooling_compression(q, chunk_size_q, q_valid_mask)
    elif kv_valid_mask is not None and q.shape[2] == k.shape[2]:
        # self-attention: Q shares K's valid mask
        q_cmp = masked_mean_pooling_compression(q, chunk_size_q, kv_valid_mask)
    else:
        q_cmp = mean_pooling_compression(q, chunk_size_q)
    if kv_valid_mask is not None:
        k_cmp = masked_mean_pooling_compression(k, chunk_size_k, kv_valid_mask)
    else:
        k_cmp = mean_pooling_compression(k, chunk_size_k)
    return q_cmp, k_cmp


def _pool_v(v, chunk_size_k, kv_valid_mask):
    if kv_valid_mask is not None:
        return masked_mean_pooling_compression(v, chunk_size_k, kv_valid_mask)
    return mean_pooling_compression(v, chunk_size_k)


def _select_blocks(selection_score, sparsity, cdf_threshold, sm_scale):
    """topk / top-p / hybrid selection (non-differentiable)."""
    if sparsity is not None and cdf_threshold is None:
        return get_select_indices_topk_from_score(selection_score, sparsity)
    if sparsity is None and cdf_threshold is not None:
        return get_select_indices_cdf_from_score(selection_score, cdf_threshold, sm_scale)
    if sparsity is not None and cdf_threshold is not None:
        return get_select_indices_cdf_topk_from_score(selection_score, sparsity, cdf_threshold, sm_scale)
    raise ValueError("Either sparsity or cdf_threshold must be provided")


def _compute_block_valid_mask(kv_valid_mask: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """A K block is valid if any token within it is valid."""
    S = kv_valid_mask.shape[0]
    num_blocks = S // chunk_size
    return kv_valid_mask.view(num_blocks, chunk_size).any(dim=-1)


def _compute_block_valid_token_counts(kv_valid_mask: torch.Tensor, chunk_size: int) -> torch.Tensor:
    S = kv_valid_mask.shape[0]
    num_blocks = S // chunk_size
    return kv_valid_mask.view(num_blocks, chunk_size).sum(dim=-1).float()


def _masked_softmax_along_last(score: torch.Tensor, sm_scale: float,
                               block_valid: Optional[torch.Tensor]) -> torch.Tensor:
    """softmax(score * sm_scale, dim=-1) with -inf at padded K blocks."""
    if block_valid is None:
        return torch.softmax(score * sm_scale, dim=-1)
    masked = score.masked_fill(~block_valid.view(1, 1, 1, -1), float('-inf'))
    return torch.softmax(masked * sm_scale, dim=-1)


def _masked_var_along_last(probs: torch.Tensor,
                           block_valid: Optional[torch.Tensor]) -> torch.Tensor:
    """Variance of softmax distribution restricted to valid K blocks.

    Without masking, padded blocks contribute zero (since A_pool=0 there) but
    inflate the denominator → biases variance downward → biases the LIVEditor
    flat/sharp split. We compute Var[p] only over valid entries.
    """
    if block_valid is None:
        return probs.var(dim=-1, unbiased=False)
    bv = block_valid.view(1, 1, 1, -1).to(probs.dtype)
    n_valid = bv.sum(dim=-1).clamp(min=1.0)
    p_masked = probs * bv
    mean = p_masked.sum(dim=-1, keepdim=True) / n_valid
    var = ((p_masked - mean) ** 2 * bv).sum(dim=-1) / n_valid
    return var


def _global_quantile_across_sp(values: torch.Tensor, q: float) -> torch.Tensor:

    # Imported at call scope but NOT guarded by try/except: swallowing an import
    # error here would make one rank skip the all_gather below while its peers
    # still enter it, which is an unrecoverable NCCL hang rather than a crash.
    from hymm.utils.parallel_states import nccl_info, get_sequence_parallel_state

    if (not get_sequence_parallel_state()) or nccl_info.sp_size <= 1:
        return torch.quantile(values.float().reshape(-1), q)

    sp_group = nccl_info.sp_group
    flat = values.float().reshape(-1).contiguous()

    # All ranks in SP have the same flat shape because Q/K/V are split by HEAD,
    # not by token. So we can use a tensor all-gather.
    world_size = nccl_info.sp_size
    gather_list = [torch.empty_like(flat) for _ in range(world_size)]
    dist.all_gather(gather_list, flat, group=sp_group)
    global_flat = torch.cat(gather_list, dim=0)
    return torch.quantile(global_flat, q)


def _taylor_merge_pytorch(
    q: torch.Tensor,
    k_cmp: torch.Tensor,
    v_cmp: torch.Tensor,
    o_spa: torch.Tensor,
    lse_spa: torch.Tensor,
    nonsel_mask: torch.Tensor,
    is_flat: torch.Tensor,
    block_token_counts: torch.Tensor,
    chunk_size_q: int,
    sm_scale: float,
) -> torch.Tensor:

    B, H, S_q, D = o_spa.shape
    N_q, N_k = nonsel_mask.shape[-2], nonsel_mask.shape[-1]

    nonsel_per_tok = (
        nonsel_mask.unsqueeze(3)
        .expand(B, H, N_q, chunk_size_q, N_k)
        .contiguous()
        .reshape(B, H, S_q, N_k)
    )

    raw_scores = torch.matmul(q, k_cmp.transpose(-1, -2)) * sm_scale  # [B, H, S_q, N_k]
    log_counts = torch.log(block_token_counts.float().clamp(min=1.0)).view(1, 1, 1, N_k)
    scores_full = raw_scores + log_counts.to(raw_scores.dtype)

    # ---- LSE: -inf-masked path; logsumexp handles all-(-inf) rows safely ----
    scores_inf = scores_full.masked_fill(~nonsel_per_tok, float('-inf'))
    lse_taylor = torch.logsumexp(scores_inf, dim=-1)             # [B, H, S_q]
    has_taylor = torch.isfinite(lse_taylor)                      # bool

    # ---- Softmax: use a finite sentinel to avoid NaN; gate rows w/o nonsel ----
    LARGE_NEG = -1.0e30
    scores_safe = scores_full.masked_fill(~nonsel_per_tok, LARGE_NEG)
    weights = torch.softmax(scores_safe, dim=-1)                 # finite everywhere
    # For rows with NO valid nonsel, every entry is LARGE_NEG → softmax = uniform
    # → zero them out so o_taylor = 0 there.
    weights = weights * has_taylor.unsqueeze(-1).to(weights.dtype)
    o_taylor = torch.matmul(weights, v_cmp)

    # ---- Stable lse merge ----
    lse_max = torch.maximum(lse_spa, lse_taylor)
    finite = torch.isfinite(lse_max)

    # diff is undefined when both lses are -inf (rare; lse_spa is always finite
    # in practice because BSA selects ≥1 block). Replace those NaNs by 0 inside
    # the discarded `where` branch so backward never sees NaN.
    diff = torch.minimum(lse_spa, lse_taylor) - lse_max
    diff = torch.where(finite, diff, torch.zeros_like(diff))
    lse_total = torch.where(finite, lse_max + torch.log1p(torch.exp(diff)), lse_max)

    w_spa = torch.where(finite, torch.exp(lse_spa - lse_total), torch.ones_like(lse_total))
    w_taylor = torch.where(
        finite & has_taylor,
        torch.exp(lse_taylor - lse_total),
        torch.zeros_like(lse_total),
    )

    merged = w_spa.unsqueeze(-1) * o_spa + w_taylor.unsqueeze(-1) * o_taylor

    # Sharp queries (block-level boolean) keep o_spa unchanged.
    flat_tok = (
        is_flat.unsqueeze(-1)
        .expand(B, H, N_q, chunk_size_q)
        .contiguous()
        .reshape(B, H, S_q)
    )
    return torch.where(flat_tok.unsqueeze(-1), merged, o_spa).to(o_spa.dtype).contiguous()


def bsa_taylor_sparse_attn(
    q: torch.Tensor,                  # [B, H, S_q, D] in 3d-block layout
    k: torch.Tensor,                  # [B, H, S_k, D]
    v: torch.Tensor,                  # [B, H, S_k, D]
    chunk_size_q: int,
    chunk_size_k: int,
    sparsity: Optional[float],
    cdf_threshold: Optional[float],
    sm_scale: float,
    *,
    alpha_f: float = 0.5,
    kv_valid_mask: Optional[torch.Tensor] = None,
    q_valid_mask: Optional[torch.Tensor] = None,
    score_modifier: Optional[torch.Tensor] = None,
    use_triton_inference: bool = True,
) -> torch.Tensor:
    # --- (a) pool Q and K (autograd-tracked) ---
    q_cmp, k_cmp = _pool_qk(q, k, chunk_size_q, chunk_size_k, kv_valid_mask, q_valid_mask)

    # --- (b) block scoring + selection (selection itself is non-differentiable) ---
    raw_score = cal_score(q_cmp, k_cmp)  # [B, H, N_q, N_k]
    with torch.no_grad():
        sel_score = raw_score * score_modifier if score_modifier is not None else raw_score
        block_indices, block_indices_lens = _select_blocks(sel_score, sparsity, cdf_threshold, sm_scale)

    # --- (c) BSA Triton kernel; this is the only autograd Function call ---
    o_spa, lse_spa = _bsa_kernel(
        q, k, v, sm_scale, block_indices, block_indices_lens,
        chunk_size_q, chunk_size_k, sparsity, kv_valid_mask,
    )

    # --- (d) Taylor side products (block_valid, A_pool, M_n, is_flat) ---
    block_valid = _compute_block_valid_mask(kv_valid_mask, chunk_size_k) if kv_valid_mask is not None else None

    with torch.no_grad():
        sel_mask = create_mask_from_indices_varlen(block_indices, raw_score.shape[-1])
        nonsel_mask = ~sel_mask
        if block_valid is not None:
            nonsel_mask = nonsel_mask & block_valid.view(1, 1, 1, -1)

        # Block-level true probability distribution (no score_modifier, no scale boost)
        A_pool_for_class = _masked_softmax_along_last(raw_score.detach(), sm_scale, block_valid)
        M_n = _masked_var_along_last(A_pool_for_class, block_valid)  # [B, H, N_q]
        tau_M = _global_quantile_across_sp(M_n, alpha_f)
        is_flat = M_n < tau_M

        if kv_valid_mask is not None:
            btc = _compute_block_valid_token_counts(kv_valid_mask, chunk_size_k)
        else:
            btc = torch.full(
                (k.shape[2] // chunk_size_k,), float(chunk_size_k),
                device=q.device, dtype=torch.float32,
            )

    # --- (e) v_cmp (autograd-tracked: gradients to V flow through this) ---
    v_cmp = _pool_v(v, chunk_size_k, kv_valid_mask)

    # --- (f) merge (autograd-tracked) ---
    use_triton = (
        use_triton_inference
        and not torch.is_grad_enabled()
        and q.is_cuda
    )
    if use_triton:
        try:
            from .bias_rectification_triton import taylor_merge_triton
            return taylor_merge_triton(
                q, k_cmp, v_cmp, o_spa, lse_spa, nonsel_mask, is_flat, btc,
                chunk_size_q, sm_scale,
            )
        except Exception as e:
            # Fall through to PyTorch path on any kernel error.
            print(f"[bias_rectification] taylor_merge_triton fallback: {e}")

    return _taylor_merge_pytorch(
        q, k_cmp, v_cmp, o_spa, lse_spa, nonsel_mask, is_flat, btc,
        chunk_size_q, sm_scale,
    )

def _rectified_correction_pytorch(
    o_spa: torch.Tensor,
    R_n: torch.Tensor,
    A_pool: torch.Tensor,
    rectify_mask: torch.Tensor,
    v_cmp: torch.Tensor,
    chunk_size_q: int,
) -> torch.Tensor:
    """O' = R_n * O_spa + (A_pool * rectify_mask) @ V_cmp, autograd-tracked.

    Memory-contiguous via expand+contiguous+reshape: each broadcast tensor is
    materialised exactly once, never produces a non-contiguous view that the
    next op would silently re-materialise.
    """
    B, H, S, D = o_spa.shape
    N_q = R_n.shape[-1]

    # Broadcast R_n [B, H, N_q] → [B, H, S, 1]: one materialisation, contiguous.
    R_tok = (
        R_n.unsqueeze(-1).unsqueeze(-1)
        .expand(B, H, N_q, chunk_size_q, 1)
        .contiguous()
        .reshape(B, H, S, 1)
    )
    o_cri = R_tok * o_spa

    # Non-critical contribution: weighted sum of pooled V over rectifiable blocks.
    A_rect = A_pool * rectify_mask.to(A_pool.dtype)
    o_ncri_block = torch.matmul(A_rect, v_cmp)              # [B, H, N_q, D]

    # Broadcast block-level output to per-token, again contiguous.
    o_ncri = (
        o_ncri_block.unsqueeze(3)
        .expand(B, H, N_q, chunk_size_q, D)
        .contiguous()
        .reshape(B, H, S, D)
    )
    return (o_cri + o_ncri).to(o_spa.dtype).contiguous()


def bsa_rectified_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size_q: int,
    chunk_size_k: int,
    sparsity: Optional[float],
    cdf_threshold: Optional[float],
    sm_scale: float,
    *,
    kv_valid_mask: Optional[torch.Tensor] = None,
    q_valid_mask: Optional[torch.Tensor] = None,
    score_modifier: Optional[torch.Tensor] = None,
    use_triton_inference: bool = True,
) -> torch.Tensor:
    # --- (a) pooling (autograd-tracked) ---
    q_cmp, k_cmp = _pool_qk(q, k, chunk_size_q, chunk_size_k, kv_valid_mask, q_valid_mask)

    # --- (b) raw scoring (autograd-tracked) ---
    raw_score = cal_score(q_cmp, k_cmp)

    # --- (c) selection (no grad) ---
    with torch.no_grad():
        sel_score = raw_score * score_modifier if score_modifier is not None else raw_score
        block_indices, block_indices_lens = _select_blocks(sel_score, sparsity, cdf_threshold, sm_scale)

    # --- (d) BSA Triton (autograd via _bsa_kernel_only) ---
    o_spa, _ = _bsa_kernel(
        q, k, v, sm_scale, block_indices, block_indices_lens,
        chunk_size_q, chunk_size_k, sparsity, kv_valid_mask,
    )

    # --- (e) sel_mask / rectify_mask (no grad; bool tensors, contiguous) ---
    block_valid = _compute_block_valid_mask(kv_valid_mask, chunk_size_k) if kv_valid_mask is not None else None
    with torch.no_grad():
        sel_mask = create_mask_from_indices_varlen(block_indices, raw_score.shape[-1])
        rectify_mask = ~sel_mask                              # all non-selected blocks
        if block_valid is not None:
            rectify_mask = rectify_mask & block_valid.view(1, 1, 1, -1)
        rectify_mask = rectify_mask.contiguous()

    # --- (f) A_pool & R_n (autograd-tracked: ∂L/∂q,∂L/∂k flow through these) ---
    A_pool = _masked_softmax_along_last(raw_score, sm_scale, block_valid)
    R_n = (1.0 - (A_pool * rectify_mask.to(A_pool.dtype)).sum(dim=-1)).clamp(min=1e-6)

    # --- (g) v_cmp (autograd-tracked: ∂L/∂v flows through pooling) ---
    v_cmp = _pool_v(v, chunk_size_k, kv_valid_mask)

    # --- (h) correction (fused Triton kernel for inference, PyTorch for training) ---
    use_triton = (
        use_triton_inference
        and not torch.is_grad_enabled()
        and q.is_cuda
    )
    if use_triton:
        try:
            from .bias_rectification_triton import rectified_correction_triton
            return rectified_correction_triton(
                o_spa, R_n.detach(), A_pool.detach(), rectify_mask, v_cmp.detach(),
                chunk_size_q,
            )
        except Exception as e:
            print(f"[bias_rectification] rectified_correction_triton fallback: {e}")

    return _rectified_correction_pytorch(
        o_spa, R_n, A_pool, rectify_mask, v_cmp, chunk_size_q,
    )


__all__ = [
    "bsa_taylor_sparse_attn",
    "bsa_rectified_sparse_attn",
]
