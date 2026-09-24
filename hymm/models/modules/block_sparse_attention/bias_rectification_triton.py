import torch
import triton
import triton.language as tl


# =====================================================================
# Rectified-SpaAttn: fused (R_n * o_spa + (A_pool * rectify_mask) @ v_cmp)
# =====================================================================
@triton.jit
def _rectified_correction_fwd_kernel(
    O_spa, R_n, A_pool, rectify_mask, V_cmp, Out,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_rz, stride_rh, stride_rn,                 # R_n: [B, H, N_q]
    stride_az, stride_ah, stride_an, stride_ak,      # A_pool: [B, H, N_q, N_k]
    stride_mz, stride_mh, stride_mn, stride_mk,      # rectify_mask: [B, H, N_q, N_k]
    stride_vz, stride_vh, stride_vk, stride_vd,      # V_cmp: [B, H, N_k, D]
    stride_cz, stride_ch, stride_cm, stride_cd,      # Out: [B, H, S_q, D]
    H, N_q, N_k,
    CHUNK_SIZE_Q: tl.constexpr,                       # tokens per Q block
    BLOCK_M: tl.constexpr,                            # tokens per program (<= CHUNK_SIZE_Q)
    BLOCK_N: tl.constexpr,                            # K blocks per inner step
    HEAD_DIM: tl.constexpr,
):
    """Compute Out[b, h, s, :] = R_n[b, h, n_q(s)] * O_spa[b, h, s, :]
                                + sum_{j} A_pool[b, h, n_q(s), j]
                                         * rectify_mask[b, h, n_q(s), j]
                                         * V_cmp[b, h, j, :]
    where n_q(s) = s // CHUNK_SIZE_Q.

    Each program handles BLOCK_M consecutive query tokens (within one Q block,
    BLOCK_M <= CHUNK_SIZE_Q so every token in the program shares the same n_q).
    """
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    off_z = pid_zh // H
    off_h = pid_zh % H

    # All BLOCK_M rows belong to the same Q block (BLOCK_M divides CHUNK_SIZE_Q).
    n_q = (pid_m * BLOCK_M) // CHUNK_SIZE_Q

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # -------- Load R_n and the BSA output --------
    r_ptr = R_n + off_z * stride_rz + off_h * stride_rh + n_q * stride_rn
    r = tl.load(r_ptr).to(tl.float32)

    o_spa_ptrs = (
        O_spa
        + off_z * stride_oz + off_h * stride_oh
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    )
    out_ptrs = (
        Out
        + off_z * stride_cz + off_h * stride_ch
        + offs_m[:, None] * stride_cm + offs_d[None, :] * stride_cd
    )

    s_q_total = pid_m * BLOCK_M + BLOCK_M  # valid rows are those < S_q_total in caller
    # We don't know S_q here; rely on caller to guarantee S_q is a multiple of CHUNK_SIZE_Q
    # via padding. But we still do a row mask to be safe for the last program.
    row_mask = offs_m[:, None] < (n_q + 1) * CHUNK_SIZE_Q

    o_spa = tl.load(o_spa_ptrs, mask=row_mask, other=0.0).to(tl.float32)
    acc = r * o_spa  # broadcast scalar * [BLOCK_M, HEAD_DIM]

    # -------- Accumulate non-critical contribution: sum_j A * mask * V --------
    a_base = (
        A_pool
        + off_z * stride_az + off_h * stride_ah + n_q * stride_an
    )
    m_base = (
        rectify_mask
        + off_z * stride_mz + off_h * stride_mh + n_q * stride_mn
    )
    v_base = V_cmp + off_z * stride_vz + off_h * stride_vh

    for j_start in range(0, N_k, BLOCK_N):
        offs_n = j_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N_k

        a = tl.load(a_base + offs_n * stride_ak, mask=n_mask, other=0.0).to(tl.float32)
        m = tl.load(m_base + offs_n * stride_mk, mask=n_mask, other=0).to(tl.float32)
        coef = a * m  # [BLOCK_N], zero where mask is False

        v_ptrs = v_base + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

        # acc += coef[:, None] * v summed over BLOCK_N (one Q row uses all K blocks)
        # Contribution is the same for all Q tokens in this block, so we add a row-wise
        # contribution: acc[i, :] += sum_j coef[j] * v[j, :]
        contrib = tl.sum(coef[:, None] * v, axis=0)  # [HEAD_DIM]
        acc += contrib[None, :]

    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=row_mask)


def rectified_correction_triton(
    o_spa: torch.Tensor,            # [B, H, S_q, D]
    R_n: torch.Tensor,              # [B, H, N_q]
    A_pool: torch.Tensor,           # [B, H, N_q, N_k] (probabilities, fp32)
    rectify_mask: torch.Tensor,     # [B, H, N_q, N_k] (bool)
    v_cmp: torch.Tensor,            # [B, H, N_k, D]
    chunk_size_q: int,
) -> torch.Tensor:
    """Fused Rectified-SpaAttn correction without materialising the
    [B, H, S_q, D] o_ncri intermediate.

    Output: out = R_n_token_level * o_spa + (A_pool * rectify_mask) @ v_cmp,
    where R_n is broadcast from N_q to S_q via the chunk index.
    """
    B, H, S_q, D = o_spa.shape
    N_q = R_n.shape[-1]
    N_k = A_pool.shape[-1]
    assert S_q == N_q * chunk_size_q, f"S_q={S_q} must equal N_q*chunk_size_q={N_q*chunk_size_q}"

    # Choose BLOCK_M = chunk_size_q for chunk_size_q <= 64; else use 64.
    if chunk_size_q <= 64:
        BLOCK_M = chunk_size_q
    else:
        # When chunk_size_q is large, split each Q block into multiple programs.
        BLOCK_M = 64
        assert chunk_size_q % BLOCK_M == 0, (
            f"chunk_size_q={chunk_size_q} must be divisible by BLOCK_M={BLOCK_M}")

    BLOCK_N = 64 if N_k >= 64 else max(1, triton.next_power_of_2(N_k))

    o_spa = o_spa.contiguous()
    A_pool = A_pool.contiguous()
    rectify_mask = rectify_mask.contiguous()
    v_cmp = v_cmp.contiguous()
    R_n = R_n.contiguous()
    out = torch.empty_like(o_spa)

    grid = (triton.cdiv(S_q, BLOCK_M), B * H, 1)

    _rectified_correction_fwd_kernel[grid](
        o_spa, R_n, A_pool, rectify_mask, v_cmp, out,
        o_spa.stride(0), o_spa.stride(1), o_spa.stride(2), o_spa.stride(3),
        R_n.stride(0), R_n.stride(1), R_n.stride(2),
        A_pool.stride(0), A_pool.stride(1), A_pool.stride(2), A_pool.stride(3),
        rectify_mask.stride(0), rectify_mask.stride(1), rectify_mask.stride(2), rectify_mask.stride(3),
        v_cmp.stride(0), v_cmp.stride(1), v_cmp.stride(2), v_cmp.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H, N_q, N_k,
        CHUNK_SIZE_Q=chunk_size_q,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def _taylor_merge_fwd_kernel(
    Q, K_cmp, V_cmp,
    O_spa, LSE_spa,
    Out,
    block_token_counts,        # [N_k]   (fp32, log already taken outside)
    nonsel_mask,               # [B, H, N_q, N_k]   bool
    is_flat,                   # [B, H, N_q]        bool
    sm_scale_log2,             # scalar, in log2
    sm_scale,                  # scalar, natural
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kk, stride_kd,
    stride_vz, stride_vh, stride_vk, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_lz, stride_lh, stride_lm,                 # LSE_spa [B, H, S_q]
    stride_cz, stride_ch, stride_cm, stride_cd,      # Out
    stride_btc,                                       # block_token_counts [N_k]
    stride_mz, stride_mh, stride_mn, stride_mk,
    stride_fz, stride_fh, stride_fn,
    H, N_q, N_k,
    CHUNK_SIZE_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """For each query token i in a Q block n_q:
       z_taylor[j] = q_i · K_cmp[j] * sm_scale + log_count[j]   (only for j in nonsel)
       lse_taylor = logsumexp_j z_taylor                        (over nonsel j)
       o_taylor   = sum_j softmax(z_taylor)[j] * V_cmp[j]
    Then if is_flat[n_q]:
       w_spa = exp(lse_spa - lse_total)
       w_taylor = exp(lse_taylor - lse_total)
       out = w_spa * o_spa + w_taylor * o_taylor
    else:
       out = o_spa
    """
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    off_z = pid_zh // H
    off_h = pid_zh % H

    n_q = (pid_m * BLOCK_M) // CHUNK_SIZE_Q
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    flat_ptr = is_flat + off_z * stride_fz + off_h * stride_fh + n_q * stride_fn
    flat_val = tl.load(flat_ptr).to(tl.int32)

    # Load Q rows for this program
    q_ptrs = (
        Q + off_z * stride_qz + off_h * stride_qh
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs).to(tl.float32)

    # Always read o_spa
    o_spa_ptrs = (
        O_spa + off_z * stride_oz + off_h * stride_oh
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    )
    o_spa = tl.load(o_spa_ptrs).to(tl.float32)

    out_ptrs = (
        Out + off_z * stride_cz + off_h * stride_ch
        + offs_m[:, None] * stride_cm + offs_d[None, :] * stride_cd
    )

    if flat_val == 0:
        # Sharp queries: keep o_spa unchanged.
        tl.store(out_ptrs, o_spa.to(Out.type.element_ty))
        return

    # --- Streaming softmax over nonsel K blocks for Taylor ---
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    mask_base = nonsel_mask + off_z * stride_mz + off_h * stride_mh + n_q * stride_mn
    k_base = K_cmp + off_z * stride_kz + off_h * stride_kh
    v_base = V_cmp + off_z * stride_vz + off_h * stride_vh

    for j_start in range(0, N_k, BLOCK_N):
        offs_n = j_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N_k

        # Load K_cmp [BLOCK_N, HEAD_DIM]
        k_ptrs = k_base + offs_n[:, None] * stride_kk + offs_d[None, :] * stride_kd
        k_cmp_b = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

        # Score q @ k_cmp^T : [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k_cmp_b)) * sm_scale

        # Add log(block_token_count[j]) to make Taylor weights commensurate.
        log_cnt = tl.load(block_token_counts + offs_n * stride_btc, mask=n_mask, other=0.0)
        qk = qk + log_cnt[None, :]

        # Mask out non-eligible blocks (selected, padded, etc.)
        nm = tl.load(mask_base + offs_n * stride_mk, mask=n_mask, other=0).to(tl.int32)
        valid = (nm != 0) & n_mask
        qk = tl.where(valid[None, :], qk, float("-inf"))

        # Streaming online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2((m_i - m_ij) * 1.4426950408889634)
        # exp in natural log: convert qk (already natural) to natural exp
        # We want p = exp(qk - m_ij). Use tl.math.exp.
        p = tl.math.exp(qk - m_ij[:, None])
        p = tl.where(valid[None, :], p, 0.0)

        # Load V_cmp [BLOCK_N, HEAD_DIM]
        v_ptrs = v_base + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd
        v_cmp_b = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v_cmp_b)
        m_i = m_ij

    # lse_taylor = m_i + log(l_i); guard l_i = 0 (no nonsel blocks → o_taylor = 0, lse = -inf)
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    lse_taylor = tl.where(l_i > 0.0, m_i + tl.log(l_safe), tl.full([BLOCK_M], -float("inf"), tl.float32))
    o_taylor = tl.where(l_i[:, None] > 0.0, acc / l_safe[:, None], tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32))

    # Load lse_spa for these tokens
    lse_spa_ptrs = LSE_spa + off_z * stride_lz + off_h * stride_lh + offs_m * stride_lm
    lse_spa = tl.load(lse_spa_ptrs)

    # Stable lse merge: lse_total = max + log(exp(lse_a - max) + exp(lse_b - max))
    lse_max = tl.maximum(lse_spa, lse_taylor)
    finite = (lse_max > -float("inf"))
    a_ext = tl.where(finite, lse_spa - lse_max, 0.0)
    b_ext = tl.where(finite, lse_taylor - lse_max, 0.0)
    lse_total = tl.where(finite, lse_max + tl.log(tl.math.exp(a_ext) + tl.math.exp(b_ext)), lse_max)

    w_spa = tl.where(finite, tl.math.exp(lse_spa - lse_total), 1.0)
    w_taylor = tl.where(finite, tl.math.exp(lse_taylor - lse_total), 0.0)

    out = w_spa[:, None] * o_spa + w_taylor[:, None] * o_taylor
    tl.store(out_ptrs, out.to(Out.type.element_ty))


def taylor_merge_triton(
    q: torch.Tensor,                # [B, H, S_q, D]   float (bf16/fp16/fp32)
    k_cmp: torch.Tensor,            # [B, H, N_k, D]
    v_cmp: torch.Tensor,            # [B, H, N_k, D]
    o_spa: torch.Tensor,            # [B, H, S_q, D]
    lse_spa: torch.Tensor,          # [B, H, S_q]   natural-log LSE from BSA
    nonsel_mask: torch.Tensor,      # [B, H, N_q, N_k] bool
    is_flat: torch.Tensor,          # [B, H, N_q]      bool
    block_token_counts: torch.Tensor,  # [N_k] fp32 — number of valid tokens per K block
    chunk_size_q: int,
    sm_scale: float,
) -> torch.Tensor:
    """Fused Taylor merge.

    Materialises NONE of the [B, H, S_q, N_k] taylor_scores tensor.
    Sharp queries (is_flat=False) skip the streaming softmax entirely.
    """
    B, H, S_q, D = o_spa.shape
    N_q = nonsel_mask.shape[-2]
    N_k = nonsel_mask.shape[-1]
    assert S_q == N_q * chunk_size_q

    if chunk_size_q <= 64:
        BLOCK_M = chunk_size_q
    else:
        BLOCK_M = 64
        assert chunk_size_q % BLOCK_M == 0

    BLOCK_N = 64 if N_k >= 64 else max(1, triton.next_power_of_2(N_k))

    # Take log of token counts once; pad blocks (count=0) → log(1) = 0, never reached
    # because they're already excluded via nonsel_mask & block_valid in the caller.
    log_counts = torch.log(block_token_counts.float().clamp(min=1.0)).contiguous()

    q = q.contiguous()
    k_cmp = k_cmp.contiguous()
    v_cmp = v_cmp.contiguous()
    o_spa = o_spa.contiguous()
    lse_spa = lse_spa.contiguous()
    nonsel_mask = nonsel_mask.contiguous()
    is_flat = is_flat.contiguous()
    out = torch.empty_like(o_spa)

    grid = (triton.cdiv(S_q, BLOCK_M), B * H, 1)

    _taylor_merge_fwd_kernel[grid](
        q, k_cmp, v_cmp,
        o_spa, lse_spa, out,
        log_counts, nonsel_mask, is_flat,
        sm_scale * 1.4426950408889634, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_cmp.stride(0), k_cmp.stride(1), k_cmp.stride(2), k_cmp.stride(3),
        v_cmp.stride(0), v_cmp.stride(1), v_cmp.stride(2), v_cmp.stride(3),
        o_spa.stride(0), o_spa.stride(1), o_spa.stride(2), o_spa.stride(3),
        lse_spa.stride(0), lse_spa.stride(1), lse_spa.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        log_counts.stride(0),
        nonsel_mask.stride(0), nonsel_mask.stride(1), nonsel_mask.stride(2), nonsel_mask.stride(3),
        is_flat.stride(0), is_flat.stride(1), is_flat.stride(2),
        H, N_q, N_k,
        CHUNK_SIZE_Q=chunk_size_q,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        num_warps=4,
        num_stages=2,
    )
    return out
