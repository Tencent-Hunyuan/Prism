
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch.distributed.tensor import DTensor
from torch.nn import RMSNorm

from hymm.utils.parallel_states import get_sequence_parallel_state, nccl_info, sp_seq_ctx
from hymm.utils.communications import (
    all_to_all_4D, all_gather,
    sp_pad_num_heads, sp_pad_heads, sp_num_local_real_heads,
    sp_split, assert_sp_splittable,
)
from hymm.models.modules.block_sparse_attention import flash_attn_bsa_3d

_DYN_BLOCK_FWD_ID = 0


def advance_dynamic_block_pass_id():
    """Call once at the start of each real outer forward (never from recompute)."""
    global _DYN_BLOCK_FWD_ID
    _DYN_BLOCK_FWD_ID += 1
    return _DYN_BLOCK_FWD_ID


def current_dynamic_block_pass_id():
    return _DYN_BLOCK_FWD_ID


try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    print("flash_attn_interface loaded from flash_attn_interface")
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False
    try:
        from kernels import get_kernel
        flash_attn_kernel = get_kernel("kernels-community/flash-attn3")
        flash_attn_3_func = flash_attn_kernel.flash_attn_func
        print("flash_attn_interface loaded from kernels-community/flash-attn3")
        FLASH_ATTN_3_AVAILABLE = True
    except ModuleNotFoundError:
        FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    from yunchang import LongContextAttention
    from yunchang.kernels import AttnType
    LONG_CONTEXT_ATTN_AVAILABLE = True
except Exception:
    LONG_CONTEXT_ATTN_AVAILABLE = False
    LongContextAttention = None
    # Fallback so module import succeeds even when yunchang isn't available.
    # `USPAttention` will error if instantiated without LongContextAttention.
    class AttnType:  # type: ignore
        FA = None
        FA3 = None


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_3_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = sageattn(q, k, v, tensor_layout="NHD")
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


@torch.compile(fullgraph=True)
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def _rope_rotate(x_4d, freqs):
    """RoPE via sin/cos rotation in the input dtype — avoids float64 allocation.

    Complex multiply (a+bi)(c+di) = (ac-bd) + (ad+bc)i is equivalent to:
      out_even = x_even * cos - x_odd * sin
      out_odd  = x_even * sin + x_odd * cos
    where freqs = cos + i*sin (complex64).

    Args:
        x_4d: [B, S, n, head_dim] in bf16/fp32.
        freqs: [S, 1, head_dim//2] complex64.
    Returns:
        [B, S, n*head_dim] in same dtype as x_4d.
    """
    cos_f = freqs.real.unsqueeze(0).to(x_4d.dtype)   # [1, S, 1, head_dim//2]
    sin_f = freqs.imag.unsqueeze(0).to(x_4d.dtype)   # [1, S, 1, head_dim//2]
    x_pairs = x_4d.unflatten(-1, (-1, 2))             # [..., head_dim//2, 2]
    x_even = x_pairs[..., 0]                          # view
    x_odd = x_pairs[..., 1]                           # view
    out_even = x_even * cos_f
    out_even.sub_(x_odd * sin_f)
    out_odd = x_even * sin_f
    out_odd.add_(x_odd * cos_f)
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2).flatten(2)


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)
    # return _rope_rotate(x, freqs)


@torch.amp.autocast('cuda', enabled=False)
def rope_apply_head_dim(x, freqs, head_dim):
    x = rearrange(x, "b s (n d) -> b s n d", d=head_dim)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)
    # return _rope_rotate(x, freqs)


class SlowRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return (self.norm(x.float()) * self.weight).to(dtype)


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6,
                 enable_bsa: bool = False, bsa_params: dict = None,
                 enable_audio_guidance: bool = False,
                 enable_audio_concentration_gate: bool = False,
                 enable_timestep_reliability_gate: bool = False,
                 enable_variance_guidance: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)
        self.enable_bsa = enable_bsa
        self.bsa_params = bsa_params or {}

        # Which SP-sharded stream this attention operates on. MOVABridge retags
        # the audio tower to "audio"; see sp_seq_ctx in parallel_states.
        self.sp_stream = "visual"

        # --- Audio Guidance (Section 3.5.1 Path A: Post-QK Score Modulation) ---
        self.enable_audio_guidance = enable_audio_guidance
        self.enable_audio_concentration_gate = enable_audio_concentration_gate
        self.enable_timestep_reliability_gate = enable_timestep_reliability_gate
        # --- Audio Weighted Pooling (Section 3.5.2 Path B: Pre-QK Representation Enhancement) ---
        self.enable_audio_weighted_pooling = False

        # --- Channel-Variance Guidance (non-learnable, no gate) ---
        self.enable_variance_guidance = enable_variance_guidance

        # --- Bias correction (two independent methods, at most one enabled) ---
        self.enable_taylor_sparse_attn = False    # LIVEditor method
        self.enable_rectified_sparse_attn = False  # Rectified SpaAttn method

        # --- Anisotropic Dynamic Block Shape (Section 8; two methods, at most
        #     one enabled). When EITHER is on, ALL of audio-guidance,
        #     variance-guidance and bias-correction are disabled (full isolation).
        self.enable_ivpq_dynamic_block = False     # Section 8.5
        self.enable_penalty_dynamic_block = False  # Section 8.6

        # (fwd_id, grid_padded, shape_ids) cache; training-only, reused on recompute.
        self._dyn_shape_cache = None

        # --- Layer-Adaptive Dynamic Block Shape (Section 8.2) ---
        # Optional wrapper on top of a dynamic-block method. When enabled:
        #   * shallow layers (idx < _la_split) use FIXED uniform-shape BSA;
        #   * deep   layers (idx >= _la_split) use the dynamic-block method;
        #   * deep layers WITH their own a2v bridge use their own audio; deep
        #     layers WITHOUT a bridge (tail beyond audio_layers) reuse the cached
        #     audio norms (seeded by the deepest bridge layer = cache-source) to
        #     drive the audio-directional gradient, combined with the deep layer's
        #     own video channel-variance.
        # All fields are plain attributes set by configure_layer_adaptive_dynamic_block.
        self.enable_layer_adaptive_dynamic_block = False
        self._la_layer_idx = None          # this self-attn's global video-layer index
        self._la_split = None              # shallow/deep boundary (num_video_layers // 2)
        self._la_is_cache_source = False   # deepest shallow bridge layer → seeds the cache
        self._la_audio_cache = None        # shared {'norms': tensor|None} across all video self-attns

    _bsa_fallback_logged = False
    _path_b_in_correction_warned = False

    def _check_bsa(self, grid_size):
        """
        Check if BSA can be applied. Returns (can_use, fallback_reason).
        Divisibility is NOT required — internal padding + masking handles arbitrary T/H/W.
        When sparsity=0, BSA selects ALL blocks and produces results identical to full attention.
        """
        if not self.enable_bsa:
            return False, None
        if grid_size is None:
            return False, "grid_size is None"
        T, H, W = grid_size
        if T <= 1:
            return False, f"T={T} <= 1 (image mode)"
        return True, None

    def _prepare_audio_guidance(self, audio_token_norms, q_bhsd, valid_mask,
                                T, H, W, pad_t, pad_h, pad_w, need_pad,
                                timestep_ratio, use_path_a, use_path_b):
        """Prepare audio guidance parameters (isolated from variance guidance)."""
        from hymm.models.modules.block_sparse_attention import (
            compute_audio_spatial_concentration_gate,
            compute_timestep_reliability_gate,
        )
        if need_pad:
            norms_3d = audio_token_norms.view(T, H, W)
            norms_padded = F.pad(norms_3d, (0, pad_w, 0, pad_h, 0, pad_t), value=0.0)
            audio_norms_padded = norms_padded.reshape(-1).contiguous()
        else:
            audio_norms_padded = audio_token_norms.contiguous()

        audio_gate = torch.ones(1, device=q_bhsd.device, dtype=q_bhsd.dtype)
        audio_boost_gamma = torch.zeros(1, device=q_bhsd.device, dtype=q_bhsd.dtype)
        audio_weighted_lambda = 0.0

        if use_path_a:
            if self.enable_timestep_reliability_gate and timestep_ratio is not None:
                r_t = compute_timestep_reliability_gate(
                    timestep_ratio,
                    mode=self.bsa_params.get('timestep_gate_mode', 'power'),
                    power=self.bsa_params.get('timestep_gate_power', 1.5),
                    scale=self.bsa_params.get('timestep_gate_scale', 10.0),
                    threshold=self.bsa_params.get('timestep_gate_threshold', 0.7),
                )
            else:
                r_t = torch.ones(1, device=q_bhsd.device, dtype=q_bhsd.dtype)

            if self.enable_audio_concentration_gate:
                g_a = compute_audio_spatial_concentration_gate(
                    audio_norms_padded, valid_mask=valid_mask)
            else:
                g_a = torch.ones(1, device=q_bhsd.device, dtype=q_bhsd.dtype)

            audio_gate = r_t * g_a
            audio_boost_gamma = torch.tensor(
                self.bsa_params.get('audio_boost_gamma', 1.0),
                device=q_bhsd.device, dtype=q_bhsd.dtype,
            )

        if use_path_b:
            audio_weighted_lambda = self.bsa_params.get('audio_weighted_lambda', 1.0)

        return audio_norms_padded, audio_gate, audio_boost_gamma, audio_weighted_lambda

    def _run_bsa_dynamic(self, q_bhsd, k_bhsd, v_bhsd, grid_size, audio_token_norms=None):
        """Anisotropic Dynamic Block Shape BSA (Section 8) — video self-attn only.

        Fully isolated from `_run_bsa`'s uniform path. Pads T/H/W to a multiple of
        the macro-zone size (8) using the SAME internal padding+mask+unpad
        contract, builds a `valid_mask` so padded tokens never participate, then
        defers all the heavy lifting (per-zone shape selection, anisotropic
        rearrange, logical-block selection, Triton kernel reuse, scatter back) to
        `flash_attn_bsa_3d_dynamic`.

        `audio_token_norms` (per-token ||Δh_v||_2 of the A→V bridge residual, in
        THW order, full sequence) drives the audio-directional gradient when
        available; it is optional (silence/no-bridge → video-only shape decision).
        """
        from hymm.models.modules.block_sparse_attention.dynamic_block_attention import (
            flash_attn_bsa_3d_dynamic,
        )
        from hymm.models.modules.block_sparse_attention.dynamic_block_shape import ZONE_SIZE

        T, H, W = grid_size
        B, n_heads, S, D = q_bhsd.shape

        pad_t = (ZONE_SIZE - T % ZONE_SIZE) % ZONE_SIZE
        pad_h = (ZONE_SIZE - H % ZONE_SIZE) % ZONE_SIZE
        pad_w = (ZONE_SIZE - W % ZONE_SIZE) % ZONE_SIZE
        need_pad = pad_t > 0 or pad_h > 0 or pad_w > 0

        valid_mask = None
        audio_norms_p = audio_token_norms
        if need_pad:
            T_p, H_p, W_p = T + pad_t, H + pad_h, W + pad_w
            idx_t = torch.arange(T_p, device=q_bhsd.device)
            idx_h = torch.arange(H_p, device=q_bhsd.device)
            idx_w = torch.arange(W_p, device=q_bhsd.device)
            valid_mask = (
                (idx_t[:, None, None] < T)
                & (idx_h[None, :, None] < H)
                & (idx_w[None, None, :] < W)
            ).reshape(-1).contiguous()

            def _pad_3d(t):
                t_3d = t.view(B, n_heads, T, H, W, D)
                t_padded = F.pad(t_3d, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
                return t_padded.reshape(B, n_heads, -1, D).contiguous()

            q_bhsd = _pad_3d(q_bhsd)
            k_bhsd = _pad_3d(k_bhsd)
            v_bhsd = _pad_3d(v_bhsd)
            if audio_token_norms is not None:
                norms_3d = audio_token_norms.reshape(T, H, W)
                audio_norms_p = F.pad(
                    norms_3d, (0, pad_w, 0, pad_h, 0, pad_t), value=0.0
                ).reshape(-1).contiguous()
            grid_padded = (T_p, H_p, W_p)
        else:
            grid_padded = grid_size

        method = "ivpq" if self.enable_ivpq_dynamic_block else "penalty"

        # Training-only: reuse this forward's shape decision on the recompute
        # (cache hit when fwd_id+grid match). Eval bypasses the cache entirely.
        use_shape_cache = self.training
        cached_shape_ids = None
        if use_shape_cache:
            cur_fwd_id = current_dynamic_block_pass_id()
            cached = self._dyn_shape_cache
            if (cached is not None and cached[0] == cur_fwd_id
                    and cached[1] == grid_padded):
                cached_shape_ids = cached[2]

        out, used_shape_ids = flash_attn_bsa_3d_dynamic(
            q_bhsd, k_bhsd, v_bhsd, grid_padded,
            method=method,
            sparsity=self.bsa_params.get('sparsity', 0.9375),
            cdf_threshold=self.bsa_params.get('cdf_threshold', None),
            audio_token_norms=audio_norms_p,
            valid_mask=valid_mask,
            lambda_a=self.bsa_params.get('dynamic_block_lambda_a', 0.5),
            tau_128=self.bsa_params.get('dynamic_block_tau_128', 0.15),
            lambda_128=self.bsa_params.get('dynamic_block_lambda_128', 1.0),
            shape_ids=cached_shape_ids,
            return_shape_ids=True,
        )

        if use_shape_cache:
            self._dyn_shape_cache = (cur_fwd_id, grid_padded, used_shape_ids)

        if need_pad:
            out_3d = out.view(B, n_heads, T_p, H_p, W_p, D)
            out = out_3d[:, :, :T, :H, :W, :].contiguous().reshape(B, n_heads, T * H * W, D)
        return out

    def _run_bsa(self, q_bhsd, k_bhsd, v_bhsd, grid_size, audio_token_norms=None, timestep_ratio=None):
        """Run BSA with internal temporal+spatial padding/masking/unpadding.

        Execution order (when all features are enabled):
          1. Padding (temporal + spatial) + valid_mask creation
          2. Audio/Variance score modulation (multiplicative boost)
          3. Block selection (top-k / top-p)
          4. BSA Triton kernel → o_spa
          5. **Bias rectification** (post-hoc, training-free, Section 3.6):
             a. Reproduce block-level scoring (O(N_q*N_k), negligible cost)
             b. Compute R_n (selected weight fraction), M_n (query sharpness)
             c. Compute O_nonsel_pool (Taylor approximation from non-selected blocks)
             d. flat:  O = R_n * O_spa + O_nonsel_pool
                sharp: O = R_n * O_spa
          6. Unpadding

        Bias correction controlled by enable_taylor_sparse_attn / enable_rectified_sparse_attn.
        Isolated from audio/variance guidance code paths.
        """
        # === Anisotropic Dynamic Block Shape (Section 8) — fully isolated path ===
        # When a dynamic-block trigger is on we route to a dedicated method and
        # never touch the uniform-shape / audio / variance / bias-correction code
        # below. The two dynamic methods are mutually exclusive with each other
        # and with all the other BSA features (enforced in configure_* + asserts).
        use_dynamic = self.enable_ivpq_dynamic_block or self.enable_penalty_dynamic_block
        if use_dynamic and self.enable_layer_adaptive_dynamic_block \
                and self._la_layer_idx is not None and self._la_split is not None:
            # Layer-adaptive (Section 8.2): shallow half → FIXED uniform-shape BSA
            # (falls through to the uniform dispatch below); deep half → dynamic.
            if self._la_layer_idx < self._la_split:
                use_dynamic = False
        if use_dynamic:
            return self._run_bsa_dynamic(
                q_bhsd, k_bhsd, v_bhsd, grid_size, audio_token_norms=audio_token_norms,
            )

        from hymm.models.modules.block_sparse_attention import flash_attn_bsa_3d_audio_guided

        T, H, W = grid_size
        cq = self.bsa_params.get('chunk_3d_shape_q', [4, 4, 4])
        ck = self.bsa_params.get('chunk_3d_shape_k', [4, 4, 4])
        chunk_t = max(cq[0], ck[0])
        chunk_h = max(cq[1], ck[1])
        chunk_w = max(cq[2], ck[2])

        pad_t = (chunk_t - T % chunk_t) % chunk_t
        pad_h = (chunk_h - H % chunk_h) % chunk_h
        pad_w = (chunk_w - W % chunk_w) % chunk_w

        B, n_heads, S, D = q_bhsd.shape
        need_pad = pad_t > 0 or pad_h > 0 or pad_w > 0

        if need_pad:
            T_p, H_p, W_p = T + pad_t, H + pad_h, W + pad_w

            idx_t = torch.arange(T_p, device=q_bhsd.device)
            idx_h = torch.arange(H_p, device=q_bhsd.device)
            idx_w = torch.arange(W_p, device=q_bhsd.device)
            valid_mask = (
                (idx_t[:, None, None] < T)
                & (idx_h[None, :, None] < H)
                & (idx_w[None, None, :] < W)
            ).reshape(-1).contiguous()

            def _pad_3d(t):
                t_3d = t.view(B, n_heads, T, H, W, D)
                t_padded = F.pad(t_3d, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
                return t_padded.reshape(B, n_heads, -1, D).contiguous()

            q_bhsd = _pad_3d(q_bhsd)
            k_bhsd = _pad_3d(k_bhsd)
            v_bhsd = _pad_3d(v_bhsd)
            grid_padded = (T_p, H_p, W_p)
        else:
            grid_padded = grid_size
            valid_mask = None

        # --- Separate guidance-specific keys from core BSA params ---
        _GUIDANCE_ONLY_KEYS = ('audio_boost_gamma', 'audio_weighted_lambda',
                               'timestep_gate_mode', 'timestep_gate_power',
                               'timestep_gate_scale', 'timestep_gate_threshold',
                               'variance_boost_gamma',
                               'taylor_alpha_f',
                               # dynamic-block knobs — never go to the uniform kernel
                               # (used by shallow layers under layer-adaptive mode)
                               'dynamic_block_lambda_a', 'dynamic_block_tau_128',
                               'dynamic_block_lambda_128')
        bsa_kwargs = {k: v for k, v in self.bsa_params.items() if k not in _GUIDANCE_ONLY_KEYS}

        has_audio_norms = (audio_token_norms is not None and audio_token_norms.numel() > 0)
        use_path_a = self.enable_audio_guidance and has_audio_norms
        use_path_b = self.enable_audio_weighted_pooling and has_audio_norms
        use_audio = use_path_a or use_path_b
        use_variance = self.enable_variance_guidance

        # =====================================================================
        # Prepare audio guidance data (shared by both dispatch and bias_rect).
        # Done once, used in whichever path is active.
        # =====================================================================
        audio_norms_padded = audio_gate = audio_boost_gamma = audio_weighted_lambda = None
        if use_audio:
            audio_norms_padded, audio_gate, audio_boost_gamma, audio_weighted_lambda = (
                self._prepare_audio_guidance(
                    audio_token_norms, q_bhsd, valid_mask, T, H, W,
                    pad_t, pad_h, pad_w, need_pad, timestep_ratio,
                    use_path_a, use_path_b,
                )
            )

        # =====================================================================
        # Dispatch: 3 mutually exclusive modes
        #   1. Taylor sparse attn (LIVEditor) — selected + Taylor in one softmax
        #   2. Rectified sparse attn (Rectified SpaAttn) — R_n * o_spa + o_ncri
        #   3. Plain BSA (default) — original paths, unchanged
        # Audio/variance guidance affects block SELECTION only (score_modifier).
        # =====================================================================
        use_correction = self.enable_taylor_sparse_attn or self.enable_rectified_sparse_attn

        if use_correction:
            # Bias rectification lives in a fully-isolated module so that the
            # audio-guidance / variance-guidance code paths above are untouched.
            from hymm.models.modules.block_sparse_attention.bias_rectification import (
                bsa_taylor_sparse_attn, bsa_rectified_sparse_attn,
            )
            from hymm.models.modules.block_sparse_attention.bsa_interface import (
                rearrange_THW_to_3d_block, rearrange_3d_block_to_THW,
                rearrange_THW_to_3d_block_1d,
            )

            _ck = bsa_kwargs.get('chunk_3d_shape_k', [4, 4, 4])
            _cq = bsa_kwargs.get('chunk_3d_shape_q', [4, 4, 4])
            _tk, _hk, _wk = _ck[0], _ck[1], _ck[2]
            _tq, _hq, _wq = _cq[0], _cq[1], _cq[2]
            _Tp, _Hp, _Wp = grid_padded
            _Ntk, _Nhk, _Nwk = _Tp // _tk, _Hp // _hk, _Wp // _wk
            _Ntq, _Nhq, _Nwq = _Tp // _tq, _Hp // _hq, _Wp // _wq

            # Rearrange once, reuse for variance/audio/correction
            q_ro = rearrange_THW_to_3d_block(q_bhsd, _Ntq, _Nhq, _Nwq, _tq, _hq, _wq, D)
            k_ro = rearrange_THW_to_3d_block(k_bhsd, _Ntk, _Nhk, _Nwk, _tk, _hk, _wk, D)
            v_ro = rearrange_THW_to_3d_block(v_bhsd, _Ntk, _Nhk, _Nwk, _tk, _hk, _wk, D)
            kv_vm_ro = rearrange_THW_to_3d_block_1d(valid_mask, _Ntk, _Nhk, _Nwk, _tk, _hk, _wk) if valid_mask is not None else None

            score_mod = None
            if use_variance:
                from hymm.models.modules.block_sparse_attention.bsa_interface import compute_channel_variance_density
                with torch.no_grad():
                    _rho = compute_channel_variance_density(v_ro, _tk*_hk*_wk, _tk, _hk, _wk, kv_vm_ro)
                score_mod = self.bsa_params.get('variance_boost_gamma', 1.0) * _rho.to(q_bhsd.dtype).view(1, 1, 1, -1)

            if use_audio and audio_norms_padded is not None:
                # Path A boost: γ_audio * (g_a * S̃_j + (1 - g_a) / N_bk)
                # All ops stay on-GPU. Avoid `.item()` here — `audio_gate` and
                # `audio_boost_gamma` are 1-element CUDA tensors, and `.item()`
                # synchronises the GPU pipeline (each call drains the queue;
                # 30 layers x 50 steps x 2 (checkpoint recompute) = 3000 syncs
                # per training step). Use plain tensor broadcasting instead.
                from hymm.models.modules.block_sparse_attention.bsa_interface import compute_block_audio_saliency
                _anorms_bo = rearrange_THW_to_3d_block_1d(audio_norms_padded, _Ntk, _Nhk, _Nwk, _tk, _hk, _wk)
                _bsal = compute_block_audio_saliency(_anorms_bo, _tk*_hk*_wk, valid_mask=kv_vm_ro)
                _N_bk = _Ntk * _Nhk * _Nwk
                # Cast scalar/Python audio_gate to a 1-elem tensor for uniform broadcast.
                if not torch.is_tensor(audio_gate):
                    audio_gate = torch.tensor(audio_gate, device=q_bhsd.device, dtype=q_bhsd.dtype)
                if not torch.is_tensor(audio_boost_gamma):
                    audio_boost_gamma = torch.tensor(audio_boost_gamma, device=q_bhsd.device, dtype=q_bhsd.dtype)
                _gv = audio_gate.to(q_bhsd.dtype)
                _bg = audio_boost_gamma.to(q_bhsd.dtype)
                _gated = _gv * _bsal.view(1, 1, 1, -1).to(q_bhsd.dtype) + (1.0 - _gv) / _N_bk
                audio_term = _bg * _gated
                score_mod = (score_mod + audio_term) if score_mod is not None else audio_term

                # Warn-once: Path B (audio-weighted K pooling) has no effect in the
                # bias-rectification path. Taylor / Rectified methods use plain mean
                # pooling for K (so autograd through pooling is correct). Path A
                # boost (audio_boost_gamma) still applies.
                if self.enable_audio_weighted_pooling and not SelfAttention._path_b_in_correction_warned:
                    print(
                        "[BSA] enable_audio_weighted_pooling (Path B) is silently ignored "
                        "when enable_taylor_sparse_attn / enable_rectified_sparse_attn is "
                        "active. Path A boost still applies."
                    )
                    SelfAttention._path_b_in_correction_warned = True

            if score_mod is not None:
                score_mod = (1.0 + score_mod).contiguous()

            _sm = 1.0 / (D ** 0.5)
            _sp = bsa_kwargs.get('sparsity', 0.9375)
            _cdf = bsa_kwargs.get('cdf_threshold', None)
            if self.enable_taylor_sparse_attn:
                out = bsa_taylor_sparse_attn(
                    q_ro, k_ro, v_ro,
                    chunk_size_q=_tq * _hq * _wq,
                    chunk_size_k=_tk * _hk * _wk,
                    sparsity=_sp, cdf_threshold=_cdf, sm_scale=_sm,
                    alpha_f=self.bsa_params.get('taylor_alpha_f', 0.5),
                    kv_valid_mask=kv_vm_ro, score_modifier=score_mod,
                )
            else:
                out = bsa_rectified_sparse_attn(
                    q_ro, k_ro, v_ro,
                    chunk_size_q=_tq * _hq * _wq,
                    chunk_size_k=_tk * _hk * _wk,
                    sparsity=_sp, cdf_threshold=_cdf, sm_scale=_sm,
                    kv_valid_mask=kv_vm_ro, score_modifier=score_mod,
                )
            out = rearrange_3d_block_to_THW(out, _Ntq, _Nhq, _Nwq, _tq, _hq, _wq, D)

        else:
            # ---------- Original dispatch paths (completely unchanged) ----------
            if use_variance and not use_audio:
                from hymm.models.modules.block_sparse_attention import flash_attn_bsa_3d_variance_guided
                out = flash_attn_bsa_3d_variance_guided(
                    q_bhsd, k_bhsd, v_bhsd, grid_padded, grid_padded,
                    variance_boost_gamma=self.bsa_params.get('variance_boost_gamma', 1.0),
                    valid_mask=valid_mask, **bsa_kwargs,
                )
            elif use_audio and not use_variance:
                out = flash_attn_bsa_3d_audio_guided(
                    q_bhsd, k_bhsd, v_bhsd, grid_padded, grid_padded,
                    audio_token_norms=audio_norms_padded,
                    audio_gate_value=audio_gate,
                    audio_boost_gamma=audio_boost_gamma,
                    valid_mask=valid_mask,
                    audio_weighted_lambda=audio_weighted_lambda,
                    **bsa_kwargs,
                )
            elif use_audio and use_variance:
                # Pass variance_boost_gamma so ρ is computed INSIDE on the already-
                # rearranged V — saves one large `rearrange_THW_to_3d_block(v)` per
                # layer per step (significant on 2K where V is ~600K tokens).
                # Mathematically identical to the explicit-score_modifier route.
                out = flash_attn_bsa_3d_audio_guided(
                    q_bhsd, k_bhsd, v_bhsd, grid_padded, grid_padded,
                    audio_token_norms=audio_norms_padded,
                    audio_gate_value=audio_gate,
                    audio_boost_gamma=audio_boost_gamma,
                    valid_mask=valid_mask,
                    audio_weighted_lambda=audio_weighted_lambda,
                    variance_boost_gamma=self.bsa_params.get('variance_boost_gamma', 1.0),
                    **bsa_kwargs,
                )
            else:
                out = flash_attn_bsa_3d(
                    q_bhsd, k_bhsd, v_bhsd, grid_padded, grid_padded,
                    valid_mask=valid_mask, **bsa_kwargs,
                )

        if need_pad:
            out_3d = out.view(B, n_heads, T_p, H_p, W_p, D)
            out = out_3d[:, :, :T, :H, :W, :].contiguous().reshape(B, n_heads, T * H * W, D)

        return out

    def forward(self, x, freqs, grid_size=None, a2v_bridge_residual=None, timestep_ratio=None):
        """
        Args:
            a2v_bridge_residual: [B, S, D] — A→V bridge cross-attention output (before scale).
                Used for audio-guided BSA scoring. Only effective when
                enable_audio_guidance=True and BSA is active.
            timestep_ratio: scalar in [0,1] — normalized timestep (0=clean, 1=noise).
                Used for Timestep Reliability Gate. Only effective when
                enable_timestep_reliability_gate=True.
        """
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        if isinstance(freqs, DTensor):
            freqs = freqs.to_local()
        q = rope_apply_head_dim(q, freqs, self.head_dim)
        k = rope_apply_head_dim(k, freqs, self.head_dim)

        use_bsa, fallback_reason = self._check_bsa(grid_size)

        if self.enable_bsa and not use_bsa and not SelfAttention._bsa_fallback_logged:
            print(f"[BSA] Falling back to full attention: {fallback_reason} "
                  f"(grid_size={grid_size})")
            SelfAttention._bsa_fallback_logged = True

        # Compute per-token audio norms for audio guidance (before SP reshuffling).
        # Also needed by the dynamic-block path for the audio-directional gradient.
        audio_token_norms = None
        # Layer-adaptive dynamic block (Section 8.2): only effective with a dynamic method.
        la_on = (self.enable_layer_adaptive_dynamic_block
                 and (self.enable_ivpq_dynamic_block or self.enable_penalty_dynamic_block)
                 and self._la_layer_idx is not None and self._la_split is not None)
        la_is_deep = la_on and (self._la_layer_idx >= self._la_split)

        use_any_audio = use_bsa and (
            self.enable_audio_guidance or self.enable_audio_weighted_pooling
            or self.enable_ivpq_dynamic_block or self.enable_penalty_dynamic_block
        )
        # Decide whether THIS layer needs to compute audio norms (a per-layer
        # all-gather under SP). Under layer-adaptive mode we avoid wasted work:
        #   * shallow layers run FIXED BSA (ignore audio) → only the designated
        #     cache-source layer computes audio (to seed the shallow→deep cache);
        #   * deep layers compute their own only if they actually have a bridge
        #     (otherwise audio stays None and they reuse the cache below).
        if la_on:
            need_audio = True if la_is_deep else self._la_is_cache_source
        else:
            need_audio = use_any_audio
        sp_full_len = sp_seq_ctx.get(self.sp_stream)

        if need_audio and a2v_bridge_residual is not None:
            # a2v_bridge_residual: [B, S_local, D] (SP-split) or [B, S, D] (no SP)
            # Compute per-token L2 norm: ||Δh_v[m]||_2
            with torch.no_grad():
                local_norms = a2v_bridge_residual.norm(dim=-1)  # [B, S_local]

            if get_sequence_parallel_state():
                # All-gather norms across SP ranks to get full sequence
                local_norms_4d = local_norms.unsqueeze(-1)  # [B, S_local, 1]
                full_norms = all_gather(local_norms_4d, dim=1, group=nccl_info.sp_group,
                                        full_seq_len=sp_full_len)
                audio_token_norms = full_norms.squeeze(-1)[0]  # [S]
            else:
                audio_token_norms = local_norms[0]  # [S]

            # Seed the shallow→deep cache (detached, full-seq THW order, SP-consistent).
            # `audio_token_norms` is computed under no_grad above (no graph retained).
            if la_on and self._la_is_cache_source and self._la_audio_cache is not None:
                self._la_audio_cache['norms'] = audio_token_norms.contiguous()

        # Deep layers under layer-adaptive: reuse cached shallow audio when this layer
        # has no bridge of its own (own audio_token_norms is None for those layers).
        if la_is_deep and audio_token_norms is None and self._la_audio_cache is not None:
            audio_token_norms = self._la_audio_cache.get('norms', None)

        if get_sequence_parallel_state():
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup
            # Ulysses requires num_heads % sp_size == 0. When it is not (e.g. audio
            # DiT = 12 heads with sp_size=8), pad the head dim with ZERO heads to the
            # next multiple. Padded heads are sliced off before any attention compute
            # and truncated after the inverse all_to_all, so they never enter the
            # BSA Triton / flash kernel and never affect the real heads' fwd/bwd.
            # When num_heads is already divisible (e.g. video = 40 heads), padded_heads
            # == num_heads and this path is bit-identical to the original code.
            padded_heads = sp_pad_num_heads(self.num_heads, sp_size)

            q, k, v = [rearrange(t, "b s (n d) -> b s n d", n=self.num_heads) for t in (q, k, v)]
            if padded_heads != self.num_heads:
                q = sp_pad_heads(q, padded_heads)
                k = sp_pad_heads(k, padded_heads)
                v = sp_pad_heads(v, padded_heads)

            q = all_to_all_4D(q, nccl_info.sp_group, scatter_dim=2, gather_dim=1,
                              full_seq_len=sp_full_len)
            k = all_to_all_4D(k, nccl_info.sp_group, scatter_dim=2, gather_dim=1,
                              full_seq_len=sp_full_len)
            v = all_to_all_4D(v, nccl_info.sp_group, scatter_dim=2, gather_dim=1,
                              full_seq_len=sp_full_len)

            hpr = q.shape[2]  # heads-per-rank after scatter (= padded_heads // sp_size)
            local_heads = sp_num_local_real_heads(self.num_heads, padded_heads, sp_size, sp_rank)
            if local_heads > 0:
                if local_heads != hpr:
                    # Mixed rank: keep only the real-head prefix. The slice stays
                    # connected to the all_to_all output, so its backward yields grad
                    # for all hpr local heads (zeros for the padded tail) and the
                    # q/k/v all_to_all collectives still fire here.
                    q = q[:, :, :local_heads, :]
                    k = k[:, :, :local_heads, :]
                    v = v[:, :, :local_heads, :]
                if use_bsa:
                    q_bsa = q.permute(0, 2, 1, 3).contiguous()
                    k_bsa = k.permute(0, 2, 1, 3).contiguous()
                    v_bsa = v.permute(0, 2, 1, 3).contiguous()
                    x_bsa = self._run_bsa(q_bsa, k_bsa, v_bsa, grid_size,
                                          audio_token_norms=audio_token_norms,
                                          timestep_ratio=timestep_ratio)
                    x = rearrange(x_bsa, "b n s d -> b s (n d)")
                else:
                    x = flash_attention(q.flatten(2), k.flatten(2), v.flatten(2), num_heads=local_heads)
                x = rearrange(x, "b s (n d) -> b s n d", n=local_heads)
                if x.shape[2] != hpr:
                    # Re-pad to hpr so the inverse all_to_all (head gather) is balanced.
                    x = sp_pad_heads(x, hpr)
            else:
                # Padding-only rank: this rank holds NO real heads, so NO attention
                # kernel is run (padding never enters the BSA Triton / flash kernel).
                # CRITICAL: x must stay CONNECTED to q/k/v (value forced to zero) so the
                # backward all_to_all for q/k/v still fires on this rank. This keeps the
                # NCCL collectives symmetric across ranks (no deadlock) AND lets this
                # rank's real tokens receive their correct q/k/v gradients, which are
                # computed on the head-owner ranks and routed back through the all_to_all.
                # `* 0.0` guarantees zero contribution to forward/backward of real heads.
                x = (q + k + v) * 0.0
            x = all_to_all_4D(x, nccl_info.sp_group, scatter_dim=1, gather_dim=2,
                              full_seq_len=sp_full_len)
            if padded_heads != self.num_heads:
                x = x[:, :, :self.num_heads, :]
            x = x.flatten(2)
        else:
            if use_bsa:
                q_bsa = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads).contiguous()
                k_bsa = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads).contiguous()
                v_bsa = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads).contiguous()
                x_bsa = self._run_bsa(q_bsa, k_bsa, v_bsa, grid_size,
                                      audio_token_norms=audio_token_norms,
                                      timestep_ratio=timestep_ratio)
                x = rearrange(x_bsa, "b n s d -> b s (n d)")
            else:
                x = self.attn(q, k, v)

        return self.o(x)


class USPAttention(nn.Module):
    def __init__(self, num_heads: int, attn_type=AttnType.FA):
        super().__init__()
        if not LONG_CONTEXT_ATTN_AVAILABLE or LongContextAttention is None:
            raise RuntimeError(
                "USPAttention requires `yunchang` (LongContextAttention). "
                "Please install/enable yunchang or avoid using USPAttention."
            )
        self.num_heads = num_heads
        self.attn = LongContextAttention(ring_impl_type="basic", attn_type=attn_type)

    def forward(self, q, k, v):
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)
        x = self.attn(q, k, v)
        return rearrange(x, "b s n d -> b s (n d)")


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

        # Query stream for SP; the text context is replicated, not sharded.
        self.sp_stream = "visual"

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)

        sp_enabled = get_sequence_parallel_state()

        if sp_enabled:
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup
            # Same Ulysses head-padding contract as SelfAttention: pad to a multiple
            # of sp_size, slice real heads before compute, truncate after. The k/v
            # context is NOT all_to_all'd (replicated); it is padded then sliced by
            # head_start so each rank takes the same global heads as its q.
            padded_heads = sp_pad_num_heads(self.num_heads, sp_size)

            q_4d = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
            k_4d = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
            v_4d = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)
            if padded_heads != self.num_heads:
                q_4d = sp_pad_heads(q_4d, padded_heads)
                k_4d = sp_pad_heads(k_4d, padded_heads)
                v_4d = sp_pad_heads(v_4d, padded_heads)

            sp_full_len = sp_seq_ctx.get(self.sp_stream)
            q_4d = all_to_all_4D(q_4d, nccl_info.sp_group, scatter_dim=2, gather_dim=1,
                                 full_seq_len=sp_full_len)
            hpr = q_4d.shape[2]
            head_start = sp_rank * hpr
            k_4d = k_4d[:, :, head_start:head_start + hpr, :]
            v_4d = v_4d[:, :, head_start:head_start + hpr, :]

            local_heads = sp_num_local_real_heads(self.num_heads, padded_heads, sp_size, sp_rank)
            if local_heads > 0:
                if local_heads != hpr:
                    q_4d = q_4d[:, :, :local_heads, :]
                    k_4d = k_4d[:, :, :local_heads, :]
                    v_4d = v_4d[:, :, :local_heads, :]
                out = flash_attention(q_4d.flatten(2), k_4d.flatten(2), v_4d.flatten(2), num_heads=local_heads)
                out = rearrange(out, "b s (n d) -> b s n d", n=local_heads)
                if self.has_image_input:
                    k_img = self.norm_k_img(self.k_img(img))
                    v_img = self.v_img(img)
                    k_img_4d = rearrange(k_img, "b s (n d) -> b s n d", n=self.num_heads)
                    v_img_4d = rearrange(v_img, "b s (n d) -> b s n d", n=self.num_heads)
                    if padded_heads != self.num_heads:
                        k_img_4d = sp_pad_heads(k_img_4d, padded_heads)
                        v_img_4d = sp_pad_heads(v_img_4d, padded_heads)
                    k_img_4d = k_img_4d[:, :, head_start:head_start + local_heads, :]
                    v_img_4d = v_img_4d[:, :, head_start:head_start + local_heads, :]
                    img_out = flash_attention(q_4d.flatten(2), k_img_4d.flatten(2), v_img_4d.flatten(2), num_heads=local_heads)
                    img_out = rearrange(img_out, "b s (n d) -> b s n d", n=local_heads)
                    out = out + img_out
                if out.shape[2] != hpr:
                    out = sp_pad_heads(out, hpr)
            else:
                # Padding-only rank: no kernel. Keep out CONNECTED to q_4d (the only
                # all_to_all'd tensor here; k/v are local context slices with no
                # collective) so q's backward all_to_all fires on this rank too.
                # `* 0.0` forces zero so it never affects real heads' fwd/bwd.
                out = q_4d * 0.0
            out = all_to_all_4D(out, nccl_info.sp_group, scatter_dim=1, gather_dim=2,
                                full_seq_len=sp_full_len)
            if padded_heads != self.num_heads:
                out = out[:, :, :self.num_heads, :]
            x = out.flatten(2)
        else:
            x = self.attn(q, k, v)
            if self.has_image_input:
                k_img = self.norm_k_img(self.k_img(img))
                v_img = self.v_img(img)
                y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
                x = x + y

        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6,
                 enable_bsa: bool = False, bsa_params: dict = None,
                 enable_audio_guidance: bool = False,
                 enable_audio_concentration_gate: bool = False,
                 enable_variance_guidance: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(
            dim, num_heads, eps,
            enable_bsa=enable_bsa, bsa_params=bsa_params,
            enable_audio_guidance=enable_audio_guidance,
            enable_audio_concentration_gate=enable_audio_concentration_gate,
            enable_variance_guidance=enable_variance_guidance,
        )
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, grid_size=None, a2v_bridge_residual=None, timestep_ratio=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(
            input_x, freqs, grid_size=grid_size,
            a2v_bridge_residual=a2v_bridge_residual,
            timestep_ratio=timestep_ratio,
        ))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(ModelMixin, ConfigMixin):
    _repeated_blocks = ("DiTBlock",)

    @register_to_config
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        enable_variance_guidance: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.enable_bsa = enable_bsa
        self.bsa_params = bsa_params
        self.enable_variance_guidance = enable_variance_guidance

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps,
                     enable_bsa=enable_bsa, bsa_params=bsa_params,
                     enable_audio_guidance=False,
                     enable_audio_concentration_gate=False,
                     enable_variance_guidance=enable_variance_guidance)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

    def patchify(self, x: torch.Tensor,control_camera_latents_input: torch.Tensor = None):
        # NOTE(dhyu): avoid slow_conv
        x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        advance_dynamic_block_pass_id()  # once per real forward (not on recompute)
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        visual_seq_len = f * h * w
        if get_sequence_parallel_state():
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup
            assert_sp_splittable(visual_seq_len, sp_size, "visual sequence")
            sp_seq_ctx.set(visual=visual_seq_len)
            x = sp_split(x, sp_size, sp_rank, dim=1)
            freqs = sp_split(freqs, sp_size, sp_rank, dim=0)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        grid_size = (f, h, w) if self.enable_bsa else None

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, grid_size,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, grid_size,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs, grid_size=grid_size)

        x = self.head(x, t)
        if get_sequence_parallel_state():
            x = all_gather(x, dim=1, group=nccl_info.sp_group,
                           full_seq_len=visual_seq_len)
        x = self.unpatchify(x, (f, h, w))
        return x

    def configure_bsa(self, enable_bsa: bool, bsa_params: dict = None):
        """Runtime configuration of Block Sparse Attention on all video self-attention blocks."""
        self.enable_bsa = enable_bsa
        self.bsa_params = bsa_params
        for block in self.blocks:
            block.self_attn.enable_bsa = enable_bsa
            block.self_attn.bsa_params = bsa_params or {}

    def configure_audio_guidance(self, enable: bool, enable_concentration_gate: bool = False):
        """Runtime configuration of audio-guided BSA on all video self-attention blocks.

        Audio guidance modulates block-level QK scores with audio saliency
        from the A→V bridge residual. Only effective when BSA is also enabled.

        The boost strength γ (audio_boost_gamma) is read from bsa_params at runtime.
        Set bsa_params['audio_boost_gamma'] to control the modulation strength
        (default 1.0; 0.0 disables modulation).

        Args:
            enable: whether to enable audio-guided score modulation.
            enable_concentration_gate: whether to enable Gate 2 (Audio Spatial
                Concentration Gate) which down-weights audio saliency when the
                bridge residual is spatially uniform (silence/ambient scenarios).
        """
        for block in self.blocks:
            sa = block.self_attn
            sa.enable_audio_guidance = enable
            sa.enable_audio_concentration_gate = enable_concentration_gate

    def configure_variance_guidance(self, enable: bool):
        """Runtime configuration of channel-variance guidance on all video self-attention blocks.

        Non-learnable, no gate. When enabled, block-level QK scores are boosted:
          score_{i,j} *= (1 + γ * ρ_j)
        where ρ_j is the V-block channel-variance density ∈ [0,1] (key/value side only).
        Query-awareness comes from P^QK being per-query (zero-gate effect):
        each query block has its own QK scores → different top-k after boost.

        Completely independent from audio guidance — no shared code or state.
        Only effective when BSA (enable_bsa) is also enabled.
        Boost strength γ is read from bsa_params['variance_boost_gamma'] (default 1.0).

        Args:
            enable: whether to enable channel-variance guidance.
        """
        self.enable_variance_guidance = enable
        for block in self.blocks:
            block.self_attn.enable_variance_guidance = enable

    def configure_taylor_sparse_attn(self, enable: bool):
        """LIVEditor: selected (exact) + non-selected (Taylor) in one unified softmax.
        Weights sum to exactly 1. No R_n. Mutually exclusive with rectified."""
        for block in self.blocks:
            block.self_attn.enable_taylor_sparse_attn = enable
            if enable:
                block.self_attn.enable_rectified_sparse_attn = False

    def configure_rectified_sparse_attn(self, enable: bool):
        """Rectified SpaAttn: o = R_n * o_spa + A_pool[nonsel] · V_pool.
        Mutually exclusive with taylor."""
        for block in self.blocks:
            block.self_attn.enable_rectified_sparse_attn = enable
            if enable:
                block.self_attn.enable_taylor_sparse_attn = False

    def _disable_non_dynamic_bsa_features(self, sa):
        """Turn off every other BSA feature on a SelfAttention module so the
        dynamic-block path is fully isolated (Section 8 requirement)."""
        sa.enable_audio_guidance = False
        sa.enable_audio_concentration_gate = False
        sa.enable_timestep_reliability_gate = False
        sa.enable_audio_weighted_pooling = False
        sa.enable_variance_guidance = False
        sa.enable_taylor_sparse_attn = False
        sa.enable_rectified_sparse_attn = False

    def configure_ivpq_dynamic_block(self, enable: bool):
        """IVPQ anisotropic dynamic block shape (Section 8.5). Video self-attn only.
        Mutually exclusive with penalty-matching AND with all other BSA features."""
        for block in self.blocks:
            sa = block.self_attn
            sa.enable_ivpq_dynamic_block = enable
            if enable:
                sa.enable_penalty_dynamic_block = False
                self._disable_non_dynamic_bsa_features(sa)

    def configure_penalty_dynamic_block(self, enable: bool):
        """Penalty-Matching anisotropic dynamic block shape (Section 8.6). Video
        self-attn only. Mutually exclusive with IVPQ AND all other BSA features."""
        for block in self.blocks:
            sa = block.self_attn
            sa.enable_penalty_dynamic_block = enable
            if enable:
                sa.enable_ivpq_dynamic_block = False
                self._disable_non_dynamic_bsa_features(sa)
