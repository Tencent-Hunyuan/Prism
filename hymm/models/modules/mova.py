import gc
import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint
from diffusers import ModelMixin, ConfigMixin

from hymm.models.modules.wan_video_dit import (
    DiTBlock,
    WanModel,
    sinusoidal_embedding_1d,
    advance_dynamic_block_pass_id,
)
from hymm.models.modules.wan_audio_dit import WanAudioModel
from hymm.models.modules.interactionv2 import (
    ConditionalCrossAttentionBlock,
    DualTowerConditionalBridge,
)
from hymm.utils.parallel_states import get_sequence_parallel_state, nccl_info, sp_seq_ctx
from hymm.utils.communications import all_gather, sp_split, assert_sp_splittable


def assemble_visual_freqs(freqs_tuple, t, h, w, device):
    """Assemble 3D RoPE complex-valued freqs for the video DiT.

    Matches the inline assembly in WanModel.forward() and
    MOVA/mova/diffusion/pipelines/pipeline_mova.py.

    Args:
        freqs_tuple: (f_freqs_cis, h_freqs_cis, w_freqs_cis) from
            precompute_freqs_cis_3d.  Each is a complex tensor.
        t, h, w: temporal, height, width grid dimensions (after patchify).
        device: target device.

    Returns:
        Complex tensor of shape [t*h*w, 1, head_dim//2].
    """
    freqs = tuple(f.to(device) for f in freqs_tuple)
    return torch.cat([
        freqs[0][:t].view(t, 1, 1, -1).expand(t, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(t, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(t, h, w, -1),
    ], dim=-1).reshape(t * h * w, 1, -1)


def assemble_audio_freqs(freqs_tuple, f, device):
    """Assemble 1D RoPE complex-valued freqs for the audio DiT.

    Matches the inline assembly in WanAudioModel.forward() and
    MOVA/mova/diffusion/pipelines/pipeline_mova.py.

    Args:
        freqs_tuple: (f0, f1, f2) from precompute_freqs_cis.
        f: sequence length (after audio patchify).
        device: target device.

    Returns:
        Complex tensor of shape [f, 1, head_dim//2].
    """
    return torch.cat([
        freqs_tuple[0][:f].view(f, -1).expand(f, -1),
        freqs_tuple[1][:f].view(f, -1).expand(f, -1),
        freqs_tuple[2][:f].view(f, -1).expand(f, -1),
    ], dim=-1).reshape(f, 1, -1).to(device)



class FusedMOVABlock(nn.Module):

    def __init__(self, video_block, audio_block,
                 a2v_conditioner=None, v2a_conditioner=None):
        super().__init__()
        self.video_block = video_block
        self.audio_block = audio_block
        self.a2v_conditioner = a2v_conditioner
        self.v2a_conditioner = v2a_conditioner

        # Mark inner DiTBlocks so apply_fsdp_checkpointing skips them,
        # avoiding nested checkpointing (outer FusedMOVABlock already covers them).
        self.video_block._inside_fused_block = True
        self.audio_block._inside_fused_block = True

        # Set to True by _apply_mova_fsdp_checkpointing (only when the YAML trigger fine_grained_gc is enabled) for blocks that ALSO get the outer block-level checkpoint
        # wrapper. When on (and training), the a2v/v2a bridge submodules are
        # additionally inner-checkpointed (deferred) so they are NOT co-resident
        # with the video block at the video-FFN recompute peak. The video/audio
        # DiT blocks run inline (Version B — see forward()). Pure recompute-
        # schedule change: fwd/bwd numerics and loss are bit-identical to the
        # default whole-block checkpointing.
        self.fine_grained_gc = False

    def forward(
        self,
        visual_x, audio_x,
        visual_context, audio_context,
        visual_t_mod, audio_t_mod,
        visual_freqs, audio_freqs,
        visual_rope_cos_sin=None, audio_rope_cos_sin=None,
        a2v_scale=1.0, v2a_scale=1.0,
        video_grid_size=None,
        override_video_block=None,
        timestep_ratio=None,
    ):
        # Fine-grained (nested) checkpointing helper. wraps ONLY the a2v/v2a bridges; the video/audio
        # blocks run inline. All arguments are passed POSITIONALLY in
        # the submodule's signature order, so the computation is byte-for-byte identical to the un-checkpointed calls (checkpointing only re-runs forward; there are no RNG ops here).
        inner_gc = self.training and self.fine_grained_gc

        def _maybe_ckpt(fn, *args):
            if inner_gc:
                return torch_checkpoint.checkpoint(fn, *args, use_reentrant=False)
            return fn(*args)

        # Bridge conditioning — must use ORIGINAL states for both directions,
        # matching DualTowerConditionalBridge.forward which computes a2v and v2a
        # from the same unmodified inputs before returning both conditioned outputs.
        visual_x_orig = visual_x

        a2v_bridge_residual = None
        if self.a2v_conditioner is not None:
            # Positional order matches ConditionalCrossAttentionBlock.forward:
            # (x, y, x_freqs, y_freqs, video_grid_size, q_structure, k_structure,
            #  q_grid_size, k_grid_size)
            cond = _maybe_ckpt(
                self.a2v_conditioner,
                visual_x, audio_x,
                visual_rope_cos_sin, audio_rope_cos_sin,
                video_grid_size,
                "3d", "1d",
                video_grid_size, None,
            )
            # Store the raw bridge residual (before scaling) for audio guidance
            a2v_bridge_residual = cond
            visual_x = visual_x + cond * a2v_scale

        if self.v2a_conditioner is not None:
            cond = _maybe_ckpt(
                self.v2a_conditioner,
                audio_x, visual_x_orig,
                audio_rope_cos_sin, visual_rope_cos_sin,
                video_grid_size,
                "1d", "3d",
                None, video_grid_size,
            )
            audio_x = audio_x + cond * v2a_scale

        # DiT blocks (allow video block override for video_dit_2).
        # If this OOMs -> wrapping BOTH calls below in _maybe_ckpt(...) (positional order as in the bridges above).
        # Numerically bit-identical either way (checkpointing only re-runs fwd).
        vblock = override_video_block if override_video_block is not None else self.video_block

        visual_x = vblock(
            visual_x, visual_context, visual_t_mod, visual_freqs,
            grid_size=video_grid_size,
            a2v_bridge_residual=a2v_bridge_residual,
            timestep_ratio=timestep_ratio,
        )
        audio_x = self.audio_block(audio_x, audio_context, audio_t_mod, audio_freqs)

        return visual_x, audio_x


# ---------------------------------------------------------------------------
# MOVABridge — top-level wrapper
# ---------------------------------------------------------------------------

class MOVABridge(ModelMixin, ConfigMixin):

    def __init__(
        self,
        video_dit: WanModel,
        video_dit_2: Optional[WanModel],
        audio_dit: WanAudioModel,
        dual_tower_bridge: DualTowerConditionalBridge,
        boundary_ratio: float = 0.9,
        gradient_checkpoint: bool = False,
    ):
        super().__init__()
        self.boundary_ratio = boundary_ratio
        self.gradient_checkpoint = gradient_checkpoint

        min_layers = min(len(video_dit.blocks), len(audio_dit.blocks))

        # --- Build fused blocks (video + audio + bridge per layer) ---
        self.fusion_blocks = nn.ModuleList()
        for i in range(min_layers):
            key = str(i)
            a2v = dual_tower_bridge.audio_to_video_conditioners[key] if key in dual_tower_bridge.audio_to_video_conditioners else None
            v2a = dual_tower_bridge.video_to_audio_conditioners[key] if key in dual_tower_bridge.video_to_audio_conditioners else None
            fused = FusedMOVABlock(
                video_block=video_dit.blocks[i],
                audio_block=audio_dit.blocks[i],
                a2v_conditioner=a2v,
                v2a_conditioner=v2a,
            )
            self.fusion_blocks.append(fused)

        # Remaining video-only blocks (primary dit, beyond min_layers)
        self.remaining_video_blocks = nn.ModuleList(
            [video_dit.blocks[i] for i in range(min_layers, len(video_dit.blocks))]
        )

        # --- Clear extracted blocks to avoid double registration ---
        video_dit.blocks = nn.ModuleList()
        audio_dit.blocks = nn.ModuleList()
        dual_tower_bridge.audio_to_video_conditioners = nn.ModuleDict()
        dual_tower_bridge.video_to_audio_conditioners = nn.ModuleDict()

        # Keep DiTs for utility functions (patchify, unpatchify, head, embeddings, freqs)
        self.video_dit = video_dit
        self.audio_dit = audio_dit

        # Keep bridge for rotary / build_aligned_freqs / condition_scale
        self.dual_tower_bridge = dual_tower_bridge

        # Secondary video dit (kept as-is with its own blocks, not fused).
        # Mark override blocks (0..min_layers-1) so they are skipped by
        # apply_fsdp_checkpointing — they run inside a checkpointed
        # FusedMOVABlock and must not be double-wrapped.
        if video_dit_2 is not None:
            for i in range(min(min_layers, len(video_dit_2.blocks))):
                video_dit_2.blocks[i]._inside_fused_block = True
        self.video_dit_2 = video_dit_2

        # When True, ALL sparse-attention features (top-k/top-p BSA, audio guidance,
        # variance guidance, taylor/rectified bias correction, dynamic block shape)
        # are applied ONLY to the high-noise expert (video_dit = fusion_blocks +
        # remaining_video_blocks). The low-noise expert (video_dit_2) then keeps the
        # backbone's original dense full attention. Default False = sparse on BOTH
        # experts (legacy behavior). Plain python attr (not an FSDP param/buffer).
        self.sparse_high_noise_only = False

        self._tag_sp_streams()

        gc.collect()

    def _tag_sp_streams(self):
        """Tell every attention which SP-sharded stream its tensors belong to.

        Under sequence parallelism the visual and audio towers are chunked to
        different lengths, and the Ulysses all-to-all has to undo the exact
        ``torch.chunk`` split of the stream it is handling. Attentions default to
        "visual", so only the audio tower and the two bridge directions need
        retagging here.
        """
        from hymm.models.modules.wan_audio_dit import tag_audio_sp_stream

        for fused in self.fusion_blocks:
            tag_audio_sp_stream([fused.audio_block])
            if fused.a2v_conditioner is not None:
                # a2v: query = visual tokens, context = audio tokens
                fused.a2v_conditioner.set_sp_streams("visual", "audio")
            if fused.v2a_conditioner is not None:
                # v2a: query = audio tokens, context = visual tokens
                fused.v2a_conditioner.set_sp_streams("audio", "visual")

    @classmethod
    def from_mova_pretrained(cls, ckpt_path, torch_dtype=torch.bfloat16, logger=None):
        """
        Load from a MOVA HuggingFace-style checkpoint directory.

        The directory should contain subfolders:
            video_dit/, video_dit_2/, audio_dit/, dual_tower_bridge/,
            video_vae/, audio_vae/, text_encoder/, tokenizer/, scheduler/

        Returns:
            (mova_bridge, extra_components) where extra_components is a dict
            containing {video_vae, audio_vae, text_encoder, tokenizer, scheduler,
            boundary_ratio, audio_vae_type}.
        """
        from diffusers.models.autoencoders import AutoencoderKLWan
        from transformers import T5TokenizerFast, UMT5EncoderModel

        from hymm.models.modules.dac_vae import DAC
        from hymm.diffusion.schedulers.flow_match_pair import FlowMatchPairScheduler

        _log = logger.info if logger else print

        import time as _time
        _load_start = _time.time()
        _log(f"[MOVABridge] Loading from {ckpt_path}")

        # Trainable components: load in requested dtype on CPU
        _t0 = _time.time()
        video_dit = WanModel.from_pretrained(
            ckpt_path, subfolder="video_dit", torch_dtype=torch_dtype,
        ).cpu()
        _log(f"[MOVABridge] Loaded video_dit ({_time.time()-_t0:.1f}s)")

        video_dit_2_path = os.path.join(ckpt_path, "video_dit_2")
        if os.path.isdir(video_dit_2_path):
            _t0 = _time.time()
            video_dit_2 = WanModel.from_pretrained(
                ckpt_path, subfolder="video_dit_2", torch_dtype=torch_dtype,
            ).cpu()
            _log(f"[MOVABridge] Loaded video_dit_2 ({_time.time()-_t0:.1f}s)")
        else:
            video_dit_2 = None
            _log("[MOVABridge] No video_dit_2 found, skipping")

        _t0 = _time.time()
        audio_dit = WanAudioModel.from_pretrained(
            ckpt_path, subfolder="audio_dit", torch_dtype=torch_dtype,
        ).cpu()
        _log(f"[MOVABridge] Loaded audio_dit ({_time.time()-_t0:.1f}s)")

        _t0 = _time.time()
        dual_tower_bridge = DualTowerConditionalBridge.from_pretrained(
            ckpt_path, subfolder="dual_tower_bridge", torch_dtype=torch_dtype,
        ).cpu()
        _log(f"[MOVABridge] Loaded dual_tower_bridge ({_time.time()-_t0:.1f}s)")

        # Frozen components: load in bf16 on CPU to save memory
        frozen_dtype = torch.bfloat16

        _t0 = _time.time()
        video_vae = AutoencoderKLWan.from_pretrained(
            ckpt_path, subfolder="video_vae", torch_dtype=frozen_dtype,
        ).cpu()
        _log(f"[MOVABridge] Loaded video_vae (bf16, frozen, {_time.time()-_t0:.1f}s)")

        _t0 = _time.time()
        audio_vae = DAC.from_pretrained(
            ckpt_path, subfolder="audio_vae", torch_dtype=frozen_dtype,
        ).cpu()
        _log(f"[MOVABridge] Loaded audio_vae (bf16, frozen, {_time.time()-_t0:.1f}s)")

        _t0 = _time.time()
        text_encoder = UMT5EncoderModel.from_pretrained(
            ckpt_path, subfolder="text_encoder", torch_dtype=frozen_dtype,
        ).to("cpu")
        _log(f"[MOVABridge] Loaded text_encoder (bf16, frozen, {_time.time()-_t0:.1f}s)")
        _log(f"[MOVABridge] Total model loading time: {_time.time()-_load_start:.1f}s")

        tokenizer = T5TokenizerFast.from_pretrained(
            ckpt_path, subfolder="tokenizer",
        )
        _log("[MOVABridge] Loaded tokenizer")

        scheduler_path = os.path.join(ckpt_path, "scheduler")
        if os.path.isdir(scheduler_path):
            scheduler = FlowMatchPairScheduler.from_pretrained(
                ckpt_path, subfolder="scheduler",
            )
            _log("[MOVABridge] Loaded scheduler from pretrained")
        else:
            scheduler = FlowMatchPairScheduler(shift=5.0)
            _log("[MOVABridge] Using default FlowMatchPairScheduler(shift=5.0)")

        model_index_path = os.path.join(ckpt_path, "model_index.json")
        boundary_ratio = 0.9
        audio_vae_type = "dac"
        if os.path.exists(model_index_path):
            with open(model_index_path, "r") as f:
                model_index = json.load(f)
            boundary_ratio = model_index.get("boundary_ratio", 0.9)
            audio_vae_type = model_index.get("audio_vae_type", "dac")

        bridge = cls(
            video_dit=video_dit,
            video_dit_2=video_dit_2,
            audio_dit=audio_dit,
            dual_tower_bridge=dual_tower_bridge,
            boundary_ratio=boundary_ratio,
        )

        extra_components = {
            "video_vae": video_vae,
            "audio_vae": audio_vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "boundary_ratio": boundary_ratio,
            "audio_vae_type": audio_vae_type,
        }

        return bridge, extra_components

    # -----------------------------------------------------------------------
    # FSDP helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def get_fsdp_no_split_modules():
        """Return module types that should not be split by FSDP."""
        return (FusedMOVABlock, DiTBlock)

    def get_fsdp_block_list(self):
        """Return a flat list of all blocks for FSDP wrapping."""
        blocks = list(self.fusion_blocks) + list(self.remaining_video_blocks)
        if self.video_dit_2 is not None:
            blocks.extend(self.video_dit_2.blocks)
        return blocks

    def get_fsdp_nested_block_list(self):
        """Sub-blocks that need an FSDP unit of their own inside a parent block.

        Each FusedMOVABlock owns the high-noise expert's video block, but on a
        low-noise step that block is bypassed in favour of ``video_dit_2``. If it
        stays in the fused block's parameter group it is still all-gathered on
        every one of those steps -- twice, since activation checkpointing
        re-runs the forward during backward -- and then discarded unused. Its own
        group is simply never entered, the same way ``video_dit_2.blocks`` are
        never entered on a high-noise step.

        Only worth doing when a second expert exists; otherwise the video block
        runs every step and splitting it just buys an extra collective.
        """
        if self.video_dit_2 is None:
            return []
        return [fused.video_block for fused in self.fusion_blocks]

    def get_fsdp_forward_tail_blocks(self):
        """Blocks that can be the LAST one executed in a forward pass.

        FSDP re-gathers the final block almost immediately for backward, so
        resharding it after forward is a wasted round trip. Which block ends the
        forward depends on the expert this step routed to, so both candidates are
        returned rather than just the last entry of ``get_fsdp_block_list()`` --
        that entry belongs to ``video_dit_2`` and is idle on high-noise steps.
        """
        tails = []
        high_noise_tail = self.remaining_video_blocks or self.fusion_blocks
        if len(high_noise_tail) > 0:
            tails.append(high_noise_tail[-1])
        if self.video_dit_2 is not None and len(self.video_dit_2.blocks) > 0:
            tails.append(self.video_dit_2.blocks[-1])
        return tails

    # -----------------------------------------------------------------------
    # BSA configuration
    # -----------------------------------------------------------------------

    def set_sparse_high_noise_only(self, flag: bool):
        """Restrict ALL video self-attn sparse features to the high-noise expert.

        When ``flag`` is True, every subsequent ``configure_*`` call (BSA top-k/top-p,
        audio guidance, variance guidance, taylor/rectified bias correction, IVPQ /
        penalty dynamic block, layer-adaptive) skips the low-noise expert
        (``video_dit_2``), leaving it on the backbone's dense full attention. The
        high-noise expert (``fusion_blocks`` + ``remaining_video_blocks``) is
        unaffected. Must be called BEFORE the ``configure_*`` methods. Default False
        = sparse applied to BOTH experts (legacy behavior).
        """
        self.sparse_high_noise_only = bool(flag)

    def configure_bsa(self, enable_bsa: bool, bsa_params: dict = None,
                       enable_bsa_v2a: bool = False, bsa_params_v2a: dict = None):
        """Runtime configuration of Block Sparse Attention.

        Args:
            enable_bsa: enable BSA on video self-attention
            bsa_params: params for video self-attn BSA
            enable_bsa_v2a: enable BSA on v2a bridge cross-attention
                (Q=audio 1D blocked internally, K=video 3D blocked)
                a2v does NOT use BSA — audio K is too short.
            bsa_params_v2a: params for v2a BSA
        """
        self._bsa_enabled = enable_bsa
        self._bsa_params = bsa_params or {}
        for fused_block in self.fusion_blocks:
            fused_block.video_block.self_attn.enable_bsa = enable_bsa
            fused_block.video_block.self_attn.bsa_params = bsa_params or {}
            if fused_block.v2a_conditioner is not None:
                fused_block.v2a_conditioner.inner.enable_bsa = enable_bsa_v2a
                fused_block.v2a_conditioner.inner.bsa_params = bsa_params_v2a or {}
        for block in self.remaining_video_blocks:
            block.self_attn.enable_bsa = enable_bsa
            block.self_attn.bsa_params = bsa_params or {}
        if self.video_dit_2 is not None and not self.sparse_high_noise_only:
            for block in self.video_dit_2.blocks:
                block.self_attn.enable_bsa = enable_bsa
                block.self_attn.bsa_params = bsa_params or {}

    # -----------------------------------------------------------------------
    # Audio Guidance configuration
    # -----------------------------------------------------------------------

    def configure_audio_guidance(self, enable: bool, enable_concentration_gate: bool = False,
                                enable_timestep_reliability_gate: bool = False,
                                enable_audio_weighted_pooling: bool = False):
        """Runtime configuration of audio-guided BSA on video self-attention.

        Two independent paths:
          - Path A (enable=True): Multiplicative score fusion with optional gates.
          - Path B (enable_audio_weighted_pooling=True): Audio-weighted K pooling.

        Either or both can be enabled independently.

        Args:
            enable: whether to enable Path A (score modulation).
            enable_concentration_gate: Gate 2 for Path A.
            enable_timestep_reliability_gate: Gate 1 for Path A.
            enable_audio_weighted_pooling: whether to enable Path B (weighted K pooling).
        """
        for fused_block in self.fusion_blocks:
            sa = fused_block.video_block.self_attn
            sa.enable_audio_guidance = enable
            sa.enable_audio_concentration_gate = enable_concentration_gate
            sa.enable_timestep_reliability_gate = enable_timestep_reliability_gate
            sa.enable_audio_weighted_pooling = enable_audio_weighted_pooling
        for block in self.remaining_video_blocks:
            sa = block.self_attn
            sa.enable_audio_guidance = enable
            sa.enable_audio_concentration_gate = enable_concentration_gate
            sa.enable_timestep_reliability_gate = enable_timestep_reliability_gate
            sa.enable_audio_weighted_pooling = enable_audio_weighted_pooling
        if self.video_dit_2 is not None and not self.sparse_high_noise_only:
            for block in self.video_dit_2.blocks:
                sa = block.self_attn
                sa.enable_audio_guidance = enable
                sa.enable_audio_concentration_gate = enable_concentration_gate
                sa.enable_timestep_reliability_gate = enable_timestep_reliability_gate
                sa.enable_audio_weighted_pooling = enable_audio_weighted_pooling

    # -----------------------------------------------------------------------
    # Variance Guidance configuration (independent from audio guidance)
    # -----------------------------------------------------------------------

    def configure_variance_guidance(self, enable: bool):
        """Runtime configuration of channel-variance guided BSA on video self-attention.

        Non-learnable, no gate. Computes ρ_j on V blocks (key/value side only):
          score_{i,j} *= (1 + γ * ρ_j)
        Query-awareness via P^QK zero-gate: same ρ_j, different QK scores per query
        → different top-k per query block. γ read from bsa_params['variance_boost_gamma'].
        Completely independent from audio guidance — no shared code.

        Args:
            enable: whether to enable channel-variance guidance.
        """
        for fused_block in self.fusion_blocks:
            fused_block.video_block.self_attn.enable_variance_guidance = enable
        for block in self.remaining_video_blocks:
            block.self_attn.enable_variance_guidance = enable
        if self.video_dit_2 is not None and not self.sparse_high_noise_only:
            for block in self.video_dit_2.blocks:
                block.self_attn.enable_variance_guidance = enable

    def configure_taylor_sparse_attn(self, enable: bool):
        """LIVEditor: selected + non-selected (Taylor) in one unified softmax."""
        for fused_block in self.fusion_blocks:
            fused_block.video_block.self_attn.enable_taylor_sparse_attn = enable
            if enable:
                fused_block.video_block.self_attn.enable_rectified_sparse_attn = False
        for block in self.remaining_video_blocks:
            block.self_attn.enable_taylor_sparse_attn = enable
            if enable:
                block.self_attn.enable_rectified_sparse_attn = False
        if self.video_dit_2 is not None and not self.sparse_high_noise_only:
            for block in self.video_dit_2.blocks:
                block.self_attn.enable_taylor_sparse_attn = enable
                if enable:
                    block.self_attn.enable_rectified_sparse_attn = False

    def configure_rectified_sparse_attn(self, enable: bool):
        """Rectified SpaAttn: o = R_n * o_spa + A_pool[nonsel] · V_pool."""
        for fused_block in self.fusion_blocks:
            fused_block.video_block.self_attn.enable_rectified_sparse_attn = enable
            if enable:
                fused_block.video_block.self_attn.enable_taylor_sparse_attn = False
        for block in self.remaining_video_blocks:
            block.self_attn.enable_rectified_sparse_attn = enable
            if enable:
                block.self_attn.enable_taylor_sparse_attn = False
        if self.video_dit_2 is not None and not self.sparse_high_noise_only:
            for block in self.video_dit_2.blocks:
                block.self_attn.enable_rectified_sparse_attn = enable
                if enable:
                    block.self_attn.enable_taylor_sparse_attn = False

    def _video_self_attns(self):
        """Yield EVERY video self-attn (both experts). NOT gated by
        sparse_high_noise_only — this enumerates all attention modules and is used
        for setting numeric bsa_params (which are inert unless a feature is enabled).
        Never used in the forward pass.
        """
        for fused_block in self.fusion_blocks:
            yield fused_block.video_block.self_attn
        for block in self.remaining_video_blocks:
            yield block.self_attn
        if self.video_dit_2 is not None:
            for block in self.video_dit_2.blocks:
                yield block.self_attn

    def _video_self_attns_with_idx(self):
        """Yield (global_video_layer_idx, self_attn) for every video self-attn.

        Layer indexing matches the forward loop: fusion_blocks[i] -> i;
        remaining_video_blocks[j] -> n_fused + j; video_dit_2.blocks[i] -> i
        (the boundary model mirrors the same per-layer indices). NOT gated.
        """
        n_fused = len(self.fusion_blocks)
        for i, fused_block in enumerate(self.fusion_blocks):
            yield i, fused_block.video_block.self_attn
        for j, block in enumerate(self.remaining_video_blocks):
            yield n_fused + j, block.self_attn
        if self.video_dit_2 is not None:
            for i, block in enumerate(self.video_dit_2.blocks):
                yield i, block.self_attn

    def _sparse_cfg_video_self_attns(self):

        for fused_block in self.fusion_blocks:
            yield fused_block.video_block.self_attn
        for block in self.remaining_video_blocks:
            yield block.self_attn
        if self.video_dit_2 is not None and not self.sparse_high_noise_only:
            for block in self.video_dit_2.blocks:
                yield block.self_attn

    def _sparse_cfg_video_self_attns_with_idx(self):
        """Indexed variant of ``_sparse_cfg_video_self_attns`` (low-noise expert
        omitted when ``sparse_high_noise_only`` is set)."""
        n_fused = len(self.fusion_blocks)
        for i, fused_block in enumerate(self.fusion_blocks):
            yield i, fused_block.video_block.self_attn
        for j, block in enumerate(self.remaining_video_blocks):
            yield n_fused + j, block.self_attn
        if self.video_dit_2 is not None and not self.sparse_high_noise_only:
            for i, block in enumerate(self.video_dit_2.blocks):
                yield i, block.self_attn

    @property
    def _num_video_layers(self):
        return len(self.fusion_blocks) + len(self.remaining_video_blocks)

    @staticmethod
    def _disable_non_dynamic_bsa_features(sa):
        sa.enable_audio_guidance = False
        sa.enable_audio_concentration_gate = False
        sa.enable_timestep_reliability_gate = False
        sa.enable_audio_weighted_pooling = False
        sa.enable_variance_guidance = False
        sa.enable_taylor_sparse_attn = False
        sa.enable_rectified_sparse_attn = False

    def configure_ivpq_dynamic_block(self, enable: bool):
        """IVPQ anisotropic dynamic block shape (Section 8.5)."""
        for sa in self._sparse_cfg_video_self_attns():
            sa.enable_ivpq_dynamic_block = enable
            if enable:
                sa.enable_penalty_dynamic_block = False
                self._disable_non_dynamic_bsa_features(sa)

    def configure_penalty_dynamic_block(self, enable: bool):
        """Penalty-Matching anisotropic dynamic block shape (Section 8.6)."""
        for sa in self._sparse_cfg_video_self_attns():
            sa.enable_penalty_dynamic_block = enable
            if enable:
                sa.enable_ivpq_dynamic_block = False
                self._disable_non_dynamic_bsa_features(sa)

    def configure_layer_adaptive_dynamic_block(self, enable: bool):
        """Layer-Adaptive Dynamic Block Shape wrapper (Section 8.2).

        Must be configured AFTER ``configure_ivpq/penalty_dynamic_block`` (it only
        takes effect when one of those is active). When ``enable``:

          * shallow half of the video self-attn layers (idx < split) run FIXED
            uniform-shape BSA;
          * deep half (idx >= split, split = num_video_layers // 2) run the
            dynamic-block method;
          * a deep layer that HAS its own a2v bridge uses its OWN audio; only the
            deep layers WITHOUT a bridge (the tail beyond audio_layers) reuse the
            cache. The cache is seeded by the DEEPEST bridge layer — its
            full-sequence ``||Δh_v||_2`` is cached (per forward) and reused by those
            no-bridge deep layers so the audio-directional gradient still guides
            their 64/128 + triple split, combined with each deep layer's own video
            channel-variance.

        Returns (split, cache_source_idx) for logging. Idempotent / safe to call
        with ``enable=False`` to turn the wrapper off (back to all-layer dynamic).
        """
        n_fused = len(self.fusion_blocks)
        num_video_layers = self._num_video_layers
        split = num_video_layers // 2


        bridge_layers = [
            i for i, fb in enumerate(self.fusion_blocks)
            if getattr(fb, "a2v_conditioner", None) is not None
        ]
        cache_source_idx = max(bridge_layers) if bridge_layers else None

        # One mutable holder shared by reference across ALL video self-attns; reset
        # at the start of every forward. Plain python attr → not an FSDP param/buffer.
        if getattr(self, "_la_audio_cache", None) is None:
            self._la_audio_cache = {"norms": None}
        cache = self._la_audio_cache

        for idx, sa in self._sparse_cfg_video_self_attns_with_idx():
            sa.enable_layer_adaptive_dynamic_block = enable
            sa._la_layer_idx = idx
            sa._la_split = split
            sa._la_is_cache_source = bool(
                enable and cache_source_idx is not None and idx == cache_source_idx
            )
            sa._la_audio_cache = cache if enable else None

        return split, cache_source_idx

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        visual_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        audio_context: Optional[torch.Tensor] = None,
        audio_timestep: Optional[torch.Tensor] = None,
        video_fps: float = 24.0,
        num_train_timesteps: int = 1000,
        condition_scale: float = 1.0,
        a2v_condition_scale: Optional[float] = None,
        v2a_condition_scale: Optional[float] = None,
        use_video_dit_2: bool = False,
    ):
        """
        Single denoising step.  Reproduces MOVA's inference_single_step exactly.
        """
        advance_dynamic_block_pass_id()  # once per real forward (not on recompute)

        use_dit_2 = use_video_dit_2 and self.video_dit_2 is not None
        visual_dit = self.video_dit_2 if use_dit_2 else self.video_dit

        if audio_context is None:
            audio_context = context
        if audio_timestep is None:
            audio_timestep = timestep

        # Timestep embeddings (force fp32 for numerical stability, same as MOVA)
        with torch.autocast("cuda", dtype=torch.float32):
            visual_t = visual_dit.time_embedding(
                sinusoidal_embedding_1d(visual_dit.freq_dim, timestep)
            )
            visual_t_mod = visual_dit.time_projection(visual_t).unflatten(1, (6, visual_dit.dim))

            audio_t = self.audio_dit.time_embedding(
                sinusoidal_embedding_1d(self.audio_dit.freq_dim, audio_timestep)
            )
            audio_t_mod = self.audio_dit.time_projection(audio_t).unflatten(1, (6, self.audio_dit.dim))

        model_dtype = visual_dit.dtype
        visual_t = visual_t.to(model_dtype)
        visual_t_mod = visual_t_mod.to(model_dtype)
        audio_t = audio_t.to(model_dtype)
        audio_t_mod = audio_t_mod.to(model_dtype)

        visual_context_emb = visual_dit.text_embedding(context)
        audio_context_emb = self.audio_dit.text_embedding(audio_context)

        visual_x = visual_latents.to(model_dtype)
        audio_x = audio_latents.to(model_dtype)

        # Patchify
        visual_x, (t, h, w) = visual_dit.patchify(visual_x)
        grid_size = (t, h, w)
        visual_freqs = assemble_visual_freqs(visual_dit.freqs, t, h, w, visual_x.device)

        audio_x, (f,) = self.audio_dit.patchify(audio_x, None)
        audio_freqs = assemble_audio_freqs(self.audio_dit.freqs, f, audio_x.device)

        # Compute normalized timestep ratio for Timestep Reliability Gate
        # In flow matching: timestep ∈ [0, 1] where 0=clean, 1=noise
        # If timestep > 1 (e.g., in [0, 1000] convention), normalize
        t_val = timestep.float().mean()
        timestep_ratio = t_val if t_val <= 1.0 else t_val / float(num_train_timesteps)

        # Forward through fused dual-tower blocks
        visual_x, audio_x = self._forward_dual_tower(
            visual_x=visual_x,
            audio_x=audio_x,
            visual_context=visual_context_emb,
            audio_context=audio_context_emb,
            visual_t_mod=visual_t_mod,
            audio_t_mod=audio_t_mod,
            visual_freqs=visual_freqs,
            audio_freqs=audio_freqs,
            grid_size=grid_size,
            video_fps=video_fps,
            condition_scale=condition_scale,
            a2v_condition_scale=a2v_condition_scale,
            v2a_condition_scale=v2a_condition_scale,
            use_dit_2=use_dit_2,
            timestep_ratio=timestep_ratio,
        )

        # Head projections (operate on SP-split sequence for efficiency)
        visual_output = visual_dit.head(visual_x, visual_t)
        audio_output = self.audio_dit.head(audio_x, audio_t)

        # SP: gather back full sequences after head
        if get_sequence_parallel_state():
            visual_output = all_gather(visual_output, dim=1, group=nccl_info.sp_group,
                                       full_seq_len=t * h * w)
            audio_output = all_gather(audio_output, dim=1, group=nccl_info.sp_group,
                                      full_seq_len=f)

        visual_output = visual_dit.unpatchify(visual_output, grid_size)
        audio_output = self.audio_dit.unpatchify(audio_output, (f,))

        return visual_output, audio_output

    def _forward_dual_tower(
        self,
        visual_x,
        audio_x,
        visual_context,
        audio_context,
        visual_t_mod,
        audio_t_mod,
        visual_freqs,
        audio_freqs,
        grid_size,
        video_fps,
        condition_scale=1.0,
        a2v_condition_scale=None,
        v2a_condition_scale=None,
        use_dit_2=False,
        timestep_ratio=None,
    ):
        """
        Forward through fused dual-tower blocks + remaining visual-only blocks.

        Ulysses SP flow:
          - Before blocks: chunk sequences and freqs
          - Inside each attention: all_to_all_4D scatter heads / gather seq
          - After all blocks: all_gather to reconstruct full sequences
        """
        sp_enabled = get_sequence_parallel_state()

        # Build bridge cross-modal RoPE
        if self.dual_tower_bridge.apply_cross_rope:
            (visual_rope_cos_sin, audio_rope_cos_sin) = (
                self.dual_tower_bridge.build_aligned_freqs(
                    video_fps=video_fps,
                    grid_size=grid_size,
                    audio_steps=audio_x.shape[1],
                    device=visual_x.device,
                    dtype=visual_x.dtype,
                )
            )
        else:
            visual_rope_cos_sin = None
            audio_rope_cos_sin = None

        # Resolve condition scales
        bridge_scale = self.dual_tower_bridge.condition_scale
        a2v_scale = (a2v_condition_scale if a2v_condition_scale is not None
                     else (condition_scale if condition_scale is not None
                           else bridge_scale))
        v2a_scale = (v2a_condition_scale if v2a_condition_scale is not None
                     else (condition_scale if condition_scale is not None
                           else bridge_scale))

        # --- SP: chunk sequences and freqs ---
        if sp_enabled:
            if getattr(nccl_info, 'use_dynamic_ring_attention', False):
                raise NotImplementedError(
                    "MOVABridge does not support dynamic_ring_attention. "
                    "Please set use_dynamic_ring_attention=False."
                )
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup

            # Publish the pre-shard lengths of both streams. Every Ulysses
            # all-to-all below reads them from here instead of renegotiating the
            # split with a per-call all_gather_object.
            visual_seq_len = visual_x.shape[1]
            audio_seq_len = audio_x.shape[1]
            assert_sp_splittable(visual_seq_len, sp_size, "visual sequence")
            assert_sp_splittable(audio_seq_len, sp_size, "audio sequence")
            sp_seq_ctx.set(visual=visual_seq_len, audio=audio_seq_len)

            visual_x = sp_split(visual_x, sp_size, sp_rank, dim=1)
            audio_x = sp_split(audio_x, sp_size, sp_rank, dim=1)
            visual_freqs = sp_split(visual_freqs, sp_size, sp_rank, dim=0)
            audio_freqs = sp_split(audio_freqs, sp_size, sp_rank, dim=0)

            if visual_rope_cos_sin is not None:
                v_cos, v_sin = visual_rope_cos_sin
                v_cos = sp_split(v_cos, sp_size, sp_rank, dim=1)
                v_sin = sp_split(v_sin, sp_size, sp_rank, dim=1)
                visual_rope_cos_sin = (v_cos, v_sin)
            if audio_rope_cos_sin is not None:
                a_cos, a_sin = audio_rope_cos_sin
                a_cos = sp_split(a_cos, sp_size, sp_rank, dim=1)
                a_sin = sp_split(a_sin, sp_size, sp_rank, dim=1)
                audio_rope_cos_sin = (a_cos, a_sin)

        use_gc = self.training and self.gradient_checkpoint

        # Reset the layer-adaptive audio cache for this forward. The deepest-bridge
        # cache-source layer seeds it; the no-bridge tail layers (which run AFTER it,
        # sequentially) read it. During checkpoint recompute it still holds the
        # forward value because the tail layers recompute first (reverse order) and
        # the cache is detached/no-grad, so the seed→read order is preserved.
        if getattr(self, "_la_audio_cache", None) is not None:
            self._la_audio_cache["norms"] = None

        # --- Fused dual-tower layers ---
        for i, fused_block in enumerate(self.fusion_blocks):
            override = None
            if use_dit_2:
                override = self.video_dit_2.blocks[i]

            if use_gc:
                visual_x, audio_x = torch_checkpoint.checkpoint(
                    fused_block,
                    visual_x, audio_x,
                    visual_context, audio_context,
                    visual_t_mod, audio_t_mod,
                    visual_freqs, audio_freqs,
                    visual_rope_cos_sin, audio_rope_cos_sin,
                    a2v_scale, v2a_scale,
                    grid_size,
                    override,
                    timestep_ratio,
                    use_reentrant=False,
                )
            else:
                visual_x, audio_x = fused_block(
                    visual_x, audio_x,
                    visual_context, audio_context,
                    visual_t_mod, audio_t_mod,
                    visual_freqs, audio_freqs,
                    visual_rope_cos_sin=visual_rope_cos_sin,
                    audio_rope_cos_sin=audio_rope_cos_sin,
                    a2v_scale=a2v_scale, v2a_scale=v2a_scale,
                    video_grid_size=grid_size,
                    override_video_block=override,
                    timestep_ratio=timestep_ratio,
                )

        # --- Remaining video-only layers ---
        if use_dit_2:
            remaining = [
                self.video_dit_2.blocks[i]
                for i in range(len(self.fusion_blocks), len(self.video_dit_2.blocks))
            ]
        else:
            remaining = self.remaining_video_blocks

        for block in remaining:
            if use_gc:
                visual_x = torch_checkpoint.checkpoint(
                    block, visual_x, visual_context, visual_t_mod, visual_freqs, grid_size,
                    None, timestep_ratio,
                    use_reentrant=False,
                )
            else:
                visual_x = block(
                    visual_x, visual_context, visual_t_mod, visual_freqs,
                    grid_size=grid_size, timestep_ratio=timestep_ratio,
                )

        return visual_x, audio_x
