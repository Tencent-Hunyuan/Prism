
from typing import List, Optional, Tuple

import torch

CANDIDATE_SHAPES_64: List[Tuple[int, int, int]] = [
    (4, 4, 4),  # isotropic, default
    (8, 2, 4),  # long-T, narrow-H (lip/face)
    (8, 4, 2),  # long-T, narrow-W
    (2, 4, 8),  # short-T, wide-W (fast motion)
    (2, 8, 4),  # short-T, wide-H
    (4, 2, 8),  # mid-T, narrow-H wide-W
    (4, 8, 2),  # mid-T, wide-H narrow-W
]

CANDIDATE_SHAPES_128: List[Tuple[int, int, int]] = [
    (2, 8, 8),  # thin-T, large spatial
    (8, 2, 8),  # long-T, narrow-H wide-W
    (8, 8, 2),  # long-T, wide-H narrow-W
    (4, 4, 8),  # mid-T, wide-W
    (4, 8, 4),  # mid-T, wide-H
    (8, 4, 4),  # long-T, mid spatial (static sky)
]

ALL_CANDIDATE_SHAPES: List[Tuple[int, int, int]] = (
    CANDIDATE_SHAPES_64 + CANDIDATE_SHAPES_128
)
NUM_CANDIDATES_64 = len(CANDIDATE_SHAPES_64)
NUM_CANDIDATES = len(ALL_CANDIDATE_SHAPES)

ZONE_SIZE = 8  # macro-zone edge length (Section 8.4.1); lcm(2,4,8)

_EPS = 1e-6


def candidate_tensors(device, dtype=torch.float32):
    """Return (shapes[NUM_CANDIDATES,3], log2_shapes[NUM_CANDIDATES,3],
    is_128[NUM_CANDIDATES], product[NUM_CANDIDATES]) on `device`.

    Cached per (device, dtype) to avoid re-allocation each layer/step.
    """
    key = (device, dtype)
    cache = candidate_tensors._cache
    if key not in cache:
        shapes = torch.tensor(ALL_CANDIDATE_SHAPES, device=device, dtype=dtype)  # [C,3]
        log2_shapes = torch.log2(shapes)
        prod = shapes.prod(dim=-1)
        is_128 = prod > 64.5
        cache[key] = (shapes, log2_shapes, is_128, prod)
    return cache[key]


candidate_tensors._cache = {}

def _zone_reshape(x_thw_d: torch.Tensor, ZT: int, ZH: int, ZW: int) -> torch.Tensor:
    """[T,H,W,D] -> [num_zones, ZS,ZS,ZS, D] with zone order zt-major.

    Memory-contiguous output (one permute+contiguous). T,H,W must be multiples
    of ZONE_SIZE.
    """
    T, H, W, D = x_thw_d.shape
    zs = ZONE_SIZE
    x = x_thw_d.reshape(ZT, zs, ZH, zs, ZW, zs, D)
    x = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous()  # [ZT,ZH,ZW, zs,zs,zs, D]
    return x.view(ZT * ZH * ZW, zs, zs, zs, D)


def compute_zone_aggregates(
    v_headavg_thw_d: torch.Tensor,
    ZT: int, ZH: int, ZW: int,
    valid_mask_thw: Optional[torch.Tensor] = None,
):
    """Compute the per-zone *masked mean* aggregates that are linear in V.

    Padded tokens (valid_mask == False) are excluded from every mean. Because
    the valid mask is identical on every SP rank (it is a token-level grid
    mask, not a head shard), the per-zone valid token *counts* are
    rank-independent — so a masked-mean computed per rank and then AVG-all-
    reduced across the SP group yields the exact global (head- and token-)
    masked mean. This keeps the SP reduction a single cheap collective.

    Returns (packed[num_zones, D + 3*ZS*D], frame_valid[num_zones, ZS]).
    Packed last-dim layout:
        [ 0        : D        ]  channel_means  (masked mean over all tokens)
        [ D        : D + ZS*D ]  frame_means_T  (masked mean over H,W per t)
        [ D + ZS*D : D + 2ZS*D]  axis_means_H   (masked mean over T,W per h)
        [ D +2ZS*D : D + 3ZS*D]  axis_means_W   (masked mean over T,H per w)
    `frame_valid[z,t]` marks t-frames that contain >=1 valid token (used to
    exclude padded frames from the temporal-variance diff downstream).
    """
    zs = ZONE_SIZE
    D = v_headavg_thw_d.shape[-1]
    xz = _zone_reshape(v_headavg_thw_d, ZT, ZH, ZW).float()  # [Nz, t,h,w, D]
    Nz = xz.shape[0]

    if valid_mask_thw is not None:
        mz = _zone_reshape(valid_mask_thw.reshape(*v_headavg_thw_d.shape[:3], 1).float(),
                           ZT, ZH, ZW)                       # [Nz, t,h,w, 1]
        c_all = mz.sum(dim=(1, 2, 3)).clamp(min=1.0)         # [Nz,1]
        channel_means = (xz * mz).sum(dim=(1, 2, 3)) / c_all
        cT = mz.sum(dim=(2, 3)).clamp(min=1.0)               # [Nz, t, 1]
        frame_means_T = (xz * mz).sum(dim=(2, 3)) / cT
        cH = mz.sum(dim=(1, 3)).clamp(min=1.0)
        axis_means_H = (xz * mz).sum(dim=(1, 3)) / cH
        cW = mz.sum(dim=(1, 2)).clamp(min=1.0)
        axis_means_W = (xz * mz).sum(dim=(1, 2)) / cW
        m = mz.squeeze(-1)
        t_valid = m.sum(dim=(2, 3)) > 0                      # [Nz, t]
        h_valid = m.sum(dim=(1, 3)) > 0                      # [Nz, h]
        w_valid = m.sum(dim=(1, 2)) > 0                      # [Nz, w]
        axis_valid = (t_valid, h_valid, w_valid)
    else:
        channel_means = xz.mean(dim=(1, 2, 3))               # [Nz, D]
        frame_means_T = xz.mean(dim=(2, 3))                  # [Nz, t, D]
        axis_means_H = xz.mean(dim=(1, 3))                   # [Nz, h, D]
        axis_means_W = xz.mean(dim=(1, 2))                   # [Nz, w, D]
        ones = torch.ones(Nz, zs, dtype=torch.bool, device=xz.device)
        axis_valid = (ones, ones, ones)

    packed = torch.cat(
        [
            channel_means,
            frame_means_T.reshape(Nz, zs * D),
            axis_means_H.reshape(Nz, zs * D),
            axis_means_W.reshape(Nz, zs * D),
        ],
        dim=-1,
    ).contiguous()
    return packed, axis_valid


def _maybe_sp_average(packed: torch.Tensor) -> torch.Tensor:
    """AVG-all-reduce the packed aggregates across the SP group (no-op if SP off).

    Mirrors the FSDP/SP convention used by bias_rectification: only the SP path
    splits heads across ranks, so the head-agnostic shape decision needs this
    reduction to be globally consistent. FSDP-only (no SP) sees identical data
    on all ranks already → no collective.
    """
    # Not wrapped in try/except on purpose: a swallowed import error would let one
    # rank return early while the rest block inside the all_reduce, turning a
    # loud failure into a silent NCCL hang.
    import torch.distributed as dist
    from hymm.utils.parallel_states import nccl_info, get_sequence_parallel_state

    if (not get_sequence_parallel_state()) or nccl_info.sp_size <= 1:
        return packed
    out = packed.contiguous()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=nccl_info.sp_group)
    out /= float(nccl_info.sp_size)
    return out


def _unpack_aggregates(packed: torch.Tensor, D: int):
    zs = ZONE_SIZE
    Nz = packed.shape[0]
    channel_means = packed[:, :D]
    off = D
    frame_means_T = packed[:, off:off + zs * D].reshape(Nz, zs, D); off += zs * D
    axis_means_H = packed[:, off:off + zs * D].reshape(Nz, zs, D); off += zs * D
    axis_means_W = packed[:, off:off + zs * D].reshape(Nz, zs, D); off += zs * D
    return channel_means, frame_means_T, axis_means_H, axis_means_W


def _axis_var(axis_means: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Variance across the ZS axis positions, averaged over channels. [Nz,ZS,D]->[Nz].

    If `valid` [Nz,ZS] is given, only valid positions contribute (padded
    axis positions are excluded so they cannot fabricate variance).
    """
    if valid is None:
        var_per_channel = axis_means.var(dim=1, unbiased=False)   # [Nz, D]
        return var_per_channel.mean(dim=-1)                       # [Nz]
    vf = valid.unsqueeze(-1).to(axis_means.dtype)                 # [Nz,ZS,1]
    n = vf.sum(dim=1).clamp(min=1.0)                              # [Nz,1]
    mean = (axis_means * vf).sum(dim=1, keepdim=True) / n.unsqueeze(1)
    var_per_channel = ((axis_means - mean) ** 2 * vf).sum(dim=1) / n
    return var_per_channel.mean(dim=-1)                           # [Nz]


def _masked_var_1d(vals: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Variance over the last axis restricted to valid positions. [Nz,L]->[Nz]."""
    vf = valid.to(vals.dtype)
    n = vf.sum(dim=-1).clamp(min=1.0)
    mean = (vals * vf).sum(dim=-1) / n
    return ((vals - mean.unsqueeze(-1)) ** 2 * vf).sum(dim=-1) / n


def compute_audio_directional(
    audio_norms_thw: Optional[torch.Tensor],
    ZT: int, ZH: int, ZW: int,
    valid_mask_thw: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
):
    """Per-zone audio descriptor (Section 8.4.3).

    Returns (a_bar[Nz], v_hat_T[Nz], v_hat_H[Nz], v_hat_W[Nz]) where a_bar is
    the (max-normalized) mean audio-absorption and v_hat_d is the normalized
    directional variance along axis d. All in float32.

    `audio_norms_thw` is identical on every SP rank (all-gathered upstream), so
    no collective is needed here.
    """
    zs = ZONE_SIZE
    Nz = ZT * ZH * ZW
    if device is None:
        device = (audio_norms_thw.device if audio_norms_thw is not None
                  else (valid_mask_thw.device if valid_mask_thw is not None else 'cpu'))
    if audio_norms_thw is None:
        z = torch.zeros(Nz, device=device, dtype=torch.float32)
        return z, z.clone(), z.clone(), z.clone()

    a = audio_norms_thw.reshape(ZT, zs, ZH, zs, ZW, zs).float()
    a = a.permute(0, 2, 4, 1, 3, 5).reshape(Nz, zs, zs, zs)  # [Nz,t,h,w]

    if valid_mask_thw is not None:
        m = valid_mask_thw.reshape(ZT, zs, ZH, zs, ZW, zs).float()
        m = m.permute(0, 2, 4, 1, 3, 5).reshape(Nz, zs, zs, zs)
        cnt = m.sum(dim=(1, 2, 3)).clamp(min=1.0)
        a_bar = (a * m).sum(dim=(1, 2, 3)) / cnt
        # Per-axis masked mean over the other two axes, then masked variance over
        # the axis. Padded positions are excluded so they cannot fabricate audio
        # directional structure (mirrors the V-feature masking).
        cT = m.sum(dim=(2, 3)).clamp(min=1.0); aT = (a * m).sum(dim=(2, 3)) / cT  # [Nz,t]
        cH = m.sum(dim=(1, 3)).clamp(min=1.0); aH = (a * m).sum(dim=(1, 3)) / cH
        cW = m.sum(dim=(1, 2)).clamp(min=1.0); aW = (a * m).sum(dim=(1, 2)) / cW
        t_valid = m.sum(dim=(2, 3)) > 0
        h_valid = m.sum(dim=(1, 3)) > 0
        w_valid = m.sum(dim=(1, 2)) > 0
        var_T = _masked_var_1d(aT, t_valid)
        var_H = _masked_var_1d(aH, h_valid)
        var_W = _masked_var_1d(aW, w_valid)
    else:
        a_bar = a.mean(dim=(1, 2, 3))
        var_T = a.mean(dim=(2, 3)).var(dim=1, unbiased=False)  # [Nz]
        var_H = a.mean(dim=(1, 3)).var(dim=1, unbiased=False)
        var_W = a.mean(dim=(1, 2)).var(dim=1, unbiased=False)

    denom = (var_T + var_H + var_W).clamp(min=_EPS)
    v_hat_T = var_T / denom
    v_hat_H = var_H / denom
    v_hat_W = var_W / denom

    a_bar_norm = a_bar / a_bar.max().clamp(min=_EPS)
    return a_bar_norm, v_hat_T, v_hat_H, v_hat_W


def compute_effective_gradient(
    v_headavg_thw_d: torch.Tensor,
    audio_norms_thw: Optional[torch.Tensor],
    ZT: int, ZH: int, ZW: int,
    lambda_a: float = 0.5,
    valid_mask_thw: Optional[torch.Tensor] = None,
    sp_reduce: bool = True,
):
    """Compute (g[Nz,3], rho[Nz]) — Section 8.4.4 effective gradient + density.

    g_d = aniso_ratio_d  +  lambda_a * a_bar_z * v_hat_d^audio        (d in T,H,W)
    rho = normalized (spatial + temporal) channel-variance information density.
    """
    D = v_headavg_thw_d.shape[-1]
    packed, axis_valid = compute_zone_aggregates(
        v_headavg_thw_d, ZT, ZH, ZW, valid_mask_thw=valid_mask_thw
    )
    if sp_reduce:
        packed = _maybe_sp_average(packed)
    t_valid, h_valid, w_valid = axis_valid
    channel_means, frame_means_T, axis_means_H, axis_means_W = _unpack_aggregates(packed, D)

    # --- channel-variance information density rho (Section 3.3 @ zone level) ---
    global_cm = channel_means.mean(dim=-1, keepdim=True)
    spatial_var = ((channel_means - global_cm) ** 2).mean(dim=-1)            # [Nz]
    # Temporal diff only between adjacent frames that are BOTH valid (mirrors
    # compute_channel_variance_density) so padded frames cannot fabricate motion.
    both_valid = (t_valid[:, 1:] & t_valid[:, :-1]).unsqueeze(-1).to(frame_means_T.dtype)  # [Nz,ZS-1,1]
    n_pairs = both_valid.sum(dim=1).clamp(min=1.0)                           # [Nz,1]
    frame_diffs = frame_means_T[:, 1:, :] - frame_means_T[:, :-1, :]         # [Nz,ZS-1,D]
    temporal_var = ((frame_diffs.pow(2) * both_valid).sum(dim=1) / n_pairs).mean(dim=-1)  # [Nz]
    rho_raw = spatial_var + temporal_var
    rho = rho_raw / rho_raw.max().clamp(min=_EPS)                            # [Nz] in [0,1]

    # --- anisotropy ratios (Section 8.4.2), padded axis positions excluded ---
    var_T = _axis_var(frame_means_T, t_valid)
    var_H = _axis_var(axis_means_H, h_valid)
    var_W = _axis_var(axis_means_W, w_valid)
    asum = (var_T + var_H + var_W).clamp(min=_EPS)
    aniso_T = var_T / asum
    aniso_H = var_H / asum
    aniso_W = var_W / asum

    # --- audio directional guidance (Section 8.4.3) ---
    a_bar, v_hat_T, v_hat_H, v_hat_W = compute_audio_directional(
        audio_norms_thw, ZT, ZH, ZW, valid_mask_thw,
        device=v_headavg_thw_d.device,
    )

    g_T = aniso_T + lambda_a * a_bar * v_hat_T
    g_H = aniso_H + lambda_a * a_bar * v_hat_H
    g_W = aniso_W + lambda_a * a_bar * v_hat_W
    g = torch.stack([g_T, g_H, g_W], dim=-1).clamp(min=1e-4)  # [Nz,3]
    return g, rho

def select_shapes_ivpq(
    g: torch.Tensor,         # [Nz, 3]
    rho: torch.Tensor,       # [Nz]
    tau_128: float = 0.15,
) -> torch.Tensor:
    """IVPQ (Section 8.5): b_d* ∝ g_d^{-2/3}; nearest candidate in log2-space.

    The candidate pool gating (rho vs tau_128) restricts high-info zones to the
    7 64-token shapes and lets low-info zones also consider the 6 128-token
    shapes (Section 8.5.3). Returns shape_id [Nz] (index into ALL_CANDIDATE_SHAPES).
    """
    device = g.device
    _, log2_shapes, is_128, _ = candidate_tensors(device, dtype=g.dtype)  # [C,3], [C]

    log2_ratio = (-2.0 / 3.0) * torch.log2(g)               # [Nz, 3], ratio only
    log2_ideal = log2_ratio + (6.0 - log2_ratio.sum(dim=-1, keepdim=True)) / 3.0

    # log-space squared distance to each candidate: [Nz, C]
    dist = ((log2_shapes.unsqueeze(0) - log2_ideal.unsqueeze(1)) ** 2).sum(dim=-1)

    # Pool gating: forbid 128-token candidates for high-info zones (rho >= tau_128).
    forbid = (rho.unsqueeze(1) >= tau_128) & is_128.unsqueeze(0)  # [Nz, C]
    dist = dist.masked_fill(forbid, float('inf'))

    return dist.argmin(dim=-1).to(torch.int64)


def select_shapes_penalty(
    g: torch.Tensor,         # [Nz, 3]
    rho: torch.Tensor,       # [Nz]
    tau_128: float = 0.15,
    lambda_128: float = 1.0,
) -> torch.Tensor:
    """Penalty Matching (Section 8.6): argmin_c sum_d g_d * b_d (+128 bonus).

    128-token candidates receive a density bonus ``-lambda_128 * (1 - rho)`` so
    low-info zones can fairly select them; high-info zones (rho >= tau_128) are
    still restricted to the 64-token pool. Returns shape_id [Nz].
    """
    device = g.device
    shapes, _, is_128, _ = candidate_tensors(device, dtype=g.dtype)  # [C,3], [C]

    penalty = (g.unsqueeze(1) * shapes.unsqueeze(0)).sum(dim=-1)     # [Nz, C]

    # 128-token density bonus (lowers penalty for low-info zones).
    bonus = (lambda_128 * (1.0 - rho)).unsqueeze(1) * is_128.unsqueeze(0).to(g.dtype)
    penalty = penalty - bonus

    # Pool gating identical to IVPQ.
    forbid = (rho.unsqueeze(1) >= tau_128) & is_128.unsqueeze(0)
    penalty = penalty.masked_fill(forbid, float('inf'))

    return penalty.argmin(dim=-1).to(torch.int64)


def select_zone_shapes(
    v_headavg_thw_d: torch.Tensor,
    audio_norms_thw: Optional[torch.Tensor],
    grid_padded: Tuple[int, int, int],
    method: str,
    lambda_a: float = 0.5,
    tau_128: float = 0.15,
    lambda_128: float = 1.0,
    valid_mask_thw: Optional[torch.Tensor] = None,
    sp_reduce: bool = True,
) -> torch.Tensor:
    """End-to-end: features -> g_d -> shape_id per zone. Returns int64 [num_zones].

    Args:
        v_headavg_thw_d: [T,H,W,D] head-averaged V (already padded). float ok.
        audio_norms_thw: [T*H*W] per-token L2 norm of A->V bridge residual, or None.
        grid_padded: (T,H,W), each a multiple of ZONE_SIZE.
        method: "ivpq" or "penalty".
    """
    T, H, W = grid_padded
    assert T % ZONE_SIZE == 0 and H % ZONE_SIZE == 0 and W % ZONE_SIZE == 0, (
        f"grid {grid_padded} must be multiple of zone size {ZONE_SIZE}"
    )
    ZT, ZH, ZW = T // ZONE_SIZE, H // ZONE_SIZE, W // ZONE_SIZE

    g, rho = compute_effective_gradient(
        v_headavg_thw_d, audio_norms_thw, ZT, ZH, ZW,
        lambda_a=lambda_a, valid_mask_thw=valid_mask_thw, sp_reduce=sp_reduce,
    )
    if method == "ivpq":
        return select_shapes_ivpq(g, rho, tau_128=tau_128)
    elif method == "penalty":
        return select_shapes_penalty(g, rho, tau_128=tau_128, lambda_128=lambda_128)
    raise ValueError(f"unknown dynamic-block method: {method!r}")


__all__ = [
    "CANDIDATE_SHAPES_64",
    "CANDIDATE_SHAPES_128",
    "ALL_CANDIDATE_SHAPES",
    "NUM_CANDIDATES_64",
    "NUM_CANDIDATES",
    "ZONE_SIZE",
    "candidate_tensors",
    "compute_effective_gradient",
    "select_shapes_ivpq",
    "select_shapes_penalty",
    "select_zone_shapes",
]
