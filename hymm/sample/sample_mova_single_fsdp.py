
import argparse
import datetime
import functools
import os
import subprocess

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file as safetensors_load_file
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from hymm.diffusion.pipelines.mova_pipeline import MOVAPipeline
from hymm.models.modules.mova import MOVABridge, FusedMOVABlock, DiTBlock
from hymm.utils.parallel_states import init_distributed, nccl_info
from hymm.utils.file_utils import rank0_logger

from hymm.sample.sample_mova_single import (
    NEGATIVE_PROMPT,
    crop_and_resize,
    init_mova_sp,
    save_video_with_audio,
)


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def parse_fsdp_args():
    """Parse arguments for FSDP-based MOVA inference."""
    parser = argparse.ArgumentParser("MOVA FSDP inference", add_help=False)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--audio_prompt", type=str, default=None)
    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./data/samples/mova_fsdp_output.mp4")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--visual_shift", type=float, default=None)
    parser.add_argument("--audio_shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--offload", type=str, default="cpu", choices=("none", "cpu"),
                        help="CPU offload for text_encoder/VAE only. The FSDP transformer "
                             "stays sharded on GPU regardless of this flag.")
    parser.add_argument("--resume_ckpt", type=str, default=None)

    # --- VAE tiling knobs (isolated; default OFF). Reduces peak VAE-decode VRAM
    #     for high-res (e.g. 2K) videos. Mirrors decode_wan_vae.py defaults. ---
    parser.add_argument("--enable_tiling", type=str, default="false",
                        help="Enable tiled VAE decode to reduce VRAM for high-res decode. Default: false.")
    parser.add_argument("--tile_sample_min_size", type=int, default=None,
                        help="Tile size (height & width) in pixels. Default: diffusers/Wan default 256.")
    parser.add_argument("--tile_sample_stride", type=int, default=None,
                        help="Stride between tiles in pixels. overlap = tile_size - stride. Default: 192.")

    # BSA knobs (same as sample_mova_single.py)
    parser.add_argument("--enable_bsa", type=str, default="false")
    parser.add_argument("--bsa_sparsity", type=float, default=0.9375)
    parser.add_argument("--bsa_chunk_3d_shape_q", type=int, nargs=3, default=[4, 4, 4])
    parser.add_argument("--bsa_chunk_3d_shape_k", type=int, nargs=3, default=[4, 4, 4])
    parser.add_argument("--bsa_cdf_threshold", type=float, default=None)
    parser.add_argument("--enable_bsa_v2a", type=str, default="false")
    parser.add_argument("--bsa_v2a_sparsity", type=float, default=0.875)
    parser.add_argument("--bsa_v2a_audio_chunk_size", type=int, default=64)
    parser.add_argument("--bsa_v2a_chunk_3d_shape_k", type=int, nargs=3, default=[4, 4, 4])
    parser.add_argument("--bsa_v2a_cdf_threshold", type=float, default=None)
    parser.add_argument("--enable_audio_guidance", type=str, default="false")
    parser.add_argument("--enable_audio_concentration_gate", type=str, default="false")
    parser.add_argument("--enable_timestep_reliability_gate", type=str, default="false")
    parser.add_argument("--audio_boost_gamma", type=float, default=1.0)
    parser.add_argument("--enable_audio_weighted_pooling", type=str, default="false")
    parser.add_argument("--audio_weighted_lambda", type=float, default=1.0)
    parser.add_argument("--enable_variance_guidance", type=str, default="false")
    parser.add_argument("--variance_boost_gamma", type=float, default=1.0)
    parser.add_argument("--enable_taylor_sparse_attn", type=str, default="false")
    parser.add_argument("--taylor_alpha_f", type=float, default=0.5)
    parser.add_argument("--enable_rectified_sparse_attn", type=str, default="false")
    parser.add_argument("--enable_ivpq_dynamic_block", type=str, default="false")
    parser.add_argument("--enable_penalty_dynamic_block", type=str, default="false")
    parser.add_argument("--dynamic_block_lambda_a", type=float, default=0.5)
    parser.add_argument("--dynamic_block_tau_128", type=float, default=0.15)
    parser.add_argument("--dynamic_block_lambda_128", type=float, default=1.0)
    parser.add_argument("--enable_layer_adaptive_dynamic_block", type=str, default="false")
    parser.add_argument("--sparse_high_noise_only", type=str, default="false",
                        help="If 'true', apply all sparse-attn features ONLY to the high-noise "
                             "expert (video_dit); video_dit_2 keeps dense full attention. "
                             "Default: false = sparse on both video experts.")

    args, _ = parser.parse_known_args()
    return args


def _as_bool(v):
    return v.lower() == "true" if isinstance(v, str) else bool(v)


def configure_bridge_features(bridge, args, log):
    """Configure BSA / guidance / dynamic-block (reused from sample_mova_single)."""
    # Must run BEFORE any configure_* call: gates whether sparse features touch
    # the low-noise expert (video_dit_2) or only the high-noise expert (video_dit).
    sparse_high_noise_only = _as_bool(getattr(args, "sparse_high_noise_only", "false"))
    bridge.set_sparse_high_noise_only(sparse_high_noise_only)
    log.info(
        f"[Sparse Attn Scope] sparse_high_noise_only={sparse_high_noise_only} "
        f"({'video_dit only; video_dit_2 = full attn' if sparse_high_noise_only else 'both video experts'})"
    )

    enable_bsa = _as_bool(args.enable_bsa)
    enable_bsa_v2a = _as_bool(args.enable_bsa_v2a)

    if enable_bsa or enable_bsa_v2a:
        bsa_params = {
            "sparsity": args.bsa_sparsity,
            "cdf_threshold": args.bsa_cdf_threshold,
            "chunk_3d_shape_q": list(args.bsa_chunk_3d_shape_q),
            "chunk_3d_shape_k": list(args.bsa_chunk_3d_shape_k),
        } if enable_bsa else None
        bsa_params_v2a = {
            "sparsity": args.bsa_v2a_sparsity,
            "cdf_threshold": args.bsa_v2a_cdf_threshold,
            "chunk_size_q": args.bsa_v2a_audio_chunk_size,
            "chunk_3d_shape_k": list(args.bsa_v2a_chunk_3d_shape_k),
        } if enable_bsa_v2a else None
        bridge.configure_bsa(
            enable_bsa=enable_bsa, bsa_params=bsa_params,
            enable_bsa_v2a=enable_bsa_v2a, bsa_params_v2a=bsa_params_v2a,
        )
        log.info(f"BSA configured: video_self={enable_bsa}, v2a={enable_bsa_v2a}")

    enable_audio_guidance = _as_bool(args.enable_audio_guidance)
    enable_audio_weighted_pooling = _as_bool(args.enable_audio_weighted_pooling)
    if enable_audio_guidance or enable_audio_weighted_pooling:
        audio_boost_gamma = float(args.audio_boost_gamma)
        audio_weighted_lambda = float(args.audio_weighted_lambda)
        for fused_block in bridge.fusion_blocks:
            fused_block.video_block.self_attn.bsa_params["audio_boost_gamma"] = audio_boost_gamma
            fused_block.video_block.self_attn.bsa_params["audio_weighted_lambda"] = audio_weighted_lambda
        for block in bridge.remaining_video_blocks:
            block.self_attn.bsa_params["audio_boost_gamma"] = audio_boost_gamma
            block.self_attn.bsa_params["audio_weighted_lambda"] = audio_weighted_lambda
        bridge.configure_audio_guidance(
            enable=enable_audio_guidance,
            enable_concentration_gate=_as_bool(args.enable_audio_concentration_gate),
            enable_timestep_reliability_gate=_as_bool(args.enable_timestep_reliability_gate),
            enable_audio_weighted_pooling=enable_audio_weighted_pooling,
        )

    enable_variance_guidance = _as_bool(args.enable_variance_guidance)
    if enable_variance_guidance:
        variance_boost_gamma = float(args.variance_boost_gamma)
        for fused_block in bridge.fusion_blocks:
            fused_block.video_block.self_attn.bsa_params["variance_boost_gamma"] = variance_boost_gamma
        for block in bridge.remaining_video_blocks:
            block.self_attn.bsa_params["variance_boost_gamma"] = variance_boost_gamma
        bridge.configure_variance_guidance(enable=True)

    enable_taylor_sa = _as_bool(args.enable_taylor_sparse_attn)
    enable_rectified_sa = _as_bool(args.enable_rectified_sparse_attn)
    if enable_taylor_sa and enable_rectified_sa:
        raise ValueError("enable_taylor_sparse_attn and enable_rectified_sparse_attn are mutually exclusive.")
    if enable_taylor_sa:
        _alpha = float(args.taylor_alpha_f)
        for fb in bridge.fusion_blocks:
            fb.video_block.self_attn.bsa_params["taylor_alpha_f"] = _alpha
        for blk in bridge.remaining_video_blocks:
            blk.self_attn.bsa_params["taylor_alpha_f"] = _alpha
        bridge.configure_taylor_sparse_attn(enable=True)
    if enable_rectified_sa:
        bridge.configure_rectified_sparse_attn(enable=True)

    enable_ivpq = _as_bool(args.enable_ivpq_dynamic_block)
    enable_penalty = _as_bool(args.enable_penalty_dynamic_block)
    if enable_ivpq and enable_penalty:
        raise ValueError("enable_ivpq_dynamic_block and enable_penalty_dynamic_block are mutually exclusive.")
    if enable_ivpq or enable_penalty:
        if (enable_audio_guidance or enable_audio_weighted_pooling
                or enable_variance_guidance or enable_taylor_sa or enable_rectified_sa):
            raise ValueError(
                "Dynamic block shape (IVPQ/Penalty) is mutually exclusive with audio guidance, "
                "variance guidance and taylor/rectified bias correction."
            )
        _la = float(args.dynamic_block_lambda_a)
        _t128 = float(args.dynamic_block_tau_128)
        _l128 = float(args.dynamic_block_lambda_128)
        for _sa in bridge._video_self_attns():
            _sa.bsa_params["dynamic_block_lambda_a"] = _la
            _sa.bsa_params["dynamic_block_tau_128"] = _t128
            _sa.bsa_params["dynamic_block_lambda_128"] = _l128
        if enable_ivpq:
            bridge.configure_ivpq_dynamic_block(enable=True)
        else:
            bridge.configure_penalty_dynamic_block(enable=True)
        log.info(f"[Dynamic Block Shape] method={'IVPQ' if enable_ivpq else 'Penalty'}")

        if _as_bool(getattr(args, "enable_layer_adaptive_dynamic_block", "false")):
            _split, _cache_src = bridge.configure_layer_adaptive_dynamic_block(enable=True)
            log.info(
                f"[Layer-Adaptive Dynamic Block] shallow(<{_split}) fixed / "
                f"deep(>={_split}) dynamic; audio cache-source layer={_cache_src}"
            )


def wrap_with_fsdp(model):
    """Wrap the MOVABridge transformer with FSDP FULL_SHARD for inference.

    Only the transformer (MOVABridge) is sharded.  Text encoder, video VAE, and
    audio VAE are handled separately via manual CPU offload in the pipeline.

    FSDP's own CPUOffload is NOT used here — it conflicts with the pipeline's
    manual .to(device)/.to("cpu") offload logic for text_encoder/VAE.  FSDP
    FULL_SHARD already reduces per-GPU memory to ~1/N of the parameters (plus
    one layer's worth during forward), which is sufficient for 1080p.
    """
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(FusedMOVABlock, DiTBlock),
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=True,
    )
    return model


def main():
    # --- Distributed init (torchrun) ---
    # Binds the device so the default NCCL communicator is built eagerly.
    local_rank = init_distributed(timeout_seconds=7200)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.set_grad_enabled(False)

    log = rank0_logger(rank)
    args = parse_fsdp_args()

    # --- SP init ---
    sp_size = args.sp_size
    if sp_size > 1:
        init_mova_sp(sp_size)
        log.info(f"Sequence parallelism enabled: sp_size={sp_size}, world_size={world_size}")

    # --- Config ---
    config_path = args.config
    if config_path and os.path.exists(config_path):
        config = OmegaConf.load(config_path)
    else:
        config = OmegaConf.create({})

    infer_cfg = config.get("inference", OmegaConf.create({}))
    height = args.height or infer_cfg.get("height", 352)
    width = args.width or infer_cfg.get("width", 640)
    num_frames = args.num_frames or infer_cfg.get("num_frames", 193)
    fps = args.fps or infer_cfg.get("fps", 24.0)
    num_inference_steps = args.num_inference_steps or infer_cfg.get("num_inference_steps", 50)
    cfg_scale = args.cfg_scale or infer_cfg.get("cfg_scale", 5.0)
    visual_shift = args.visual_shift or infer_cfg.get("visual_shift", 5.0)
    audio_shift = args.audio_shift or infer_cfg.get("audio_shift", 5.0)
    seed = args.seed if args.seed is not None else infer_cfg.get("seed", 42)
    negative_prompt = args.negative_prompt or NEGATIVE_PROMPT

    # --- Load model on CPU (all ranks load, FSDP sync_module_states ensures consistency) ---
    ckpt_path = args.ckpt
    log.info(f"Loading MOVA from {ckpt_path}")
    bridge, extras = MOVABridge.from_mova_pretrained(ckpt_path, torch_dtype=torch.bfloat16)

    if args.resume_ckpt:
        resume_path = args.resume_ckpt
        log.info(f"Loading resume checkpoint from {resume_path}")
        if resume_path.endswith(".safetensors"):
            state_dict = safetensors_load_file(resume_path)
        elif resume_path.endswith(".pt"):
            state_dict = torch.load(resume_path, map_location="cpu")
            if "module" in state_dict:
                state_dict = state_dict["module"]
        else:
            raise ValueError(f"Unsupported checkpoint format: {resume_path}")
        m, u = bridge.load_state_dict(state_dict, strict=False)
        log.info(f"Resume checkpoint loaded. Missing keys: {m}, Unexpected keys: {u}")

    configure_bridge_features(bridge, args, log)
    bridge.eval()

    # --- Wrap transformer with FSDP FULL_SHARD ---
    # FSDP shards parameters across GPUs.  During forward each layer does:
    #   all-gather params → compute → reshard
    # So all ranks see the SAME full parameters and produce the SAME output.
    log.info("Wrapping MOVABridge with FSDP FULL_SHARD...")
    bridge = wrap_with_fsdp(bridge)
    log.info("FSDP wrapping complete.")

    device_t = torch.device("cuda", local_rank)
    use_offload = args.offload == "cpu"


    bridge.to = lambda *a, **kw: bridge

    if use_offload:
        log.info("CPU offload: text_encoder/VAE will be moved to GPU on demand; "
                 "transformer stays FSDP-sharded on GPU")
        pipeline = MOVAPipeline(
            transformer=bridge,
            video_vae=extras["video_vae"],
            audio_vae=extras["audio_vae"],
            text_encoder=extras["text_encoder"],
            tokenizer=extras["tokenizer"],
            scheduler=extras["scheduler"],
            boundary_ratio=extras["boundary_ratio"],
            device=device_t,
        )
        pipeline.enable_cpu_offload(gpu_device=device_t)
    else:
        log.info("No CPU offload — all components on GPU")
        pipeline = MOVAPipeline(
            transformer=bridge,
            video_vae=extras["video_vae"].to(device_t),
            audio_vae=extras["audio_vae"].to(device_t),
            text_encoder=extras["text_encoder"].to(device_t),
            tokenizer=extras["tokenizer"],
            scheduler=extras["scheduler"],
            boundary_ratio=extras["boundary_ratio"],
            device=device_t,
        )

    if not os.path.exists(args.ref_path):
        raise FileNotFoundError(f"Reference image not found: {args.ref_path}")

    ref_img = Image.open(args.ref_path).convert("RGB")
    ref_img = crop_and_resize(ref_img, height=height, width=width)


    torch.manual_seed(seed)

    enable_tiling = _as_bool(getattr(args, "enable_tiling", "false"))
    if enable_tiling:
        _ts = args.tile_sample_min_size if args.tile_sample_min_size is not None else 256
        _st = args.tile_sample_stride if args.tile_sample_stride is not None else 192
        log.info(
            f"[VAE Tiling] ENABLED: tile_size={_ts}, stride={_st}, "
            f"overlap={_ts - _st}px ({(_ts - _st) / _ts * 100:.0f}%)"
        )

    log.info(f"Starting FSDP inference: {height}x{width}, {num_frames} frames, seed={seed}")
    log.info(f"Prompt: {args.prompt}")
    log.info(f"Audio prompt: {args.audio_prompt}")

    video, audio = pipeline(
        prompt=args.prompt,
        image=ref_img,
        audio_prompt=args.audio_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        video_fps=fps,
        num_inference_steps=num_inference_steps,
        visual_shift=visual_shift,
        audio_shift=audio_shift,
        cfg_scale=cfg_scale,
        enable_vae_tiling=enable_tiling,
        vae_tile_sample_min_size=args.tile_sample_min_size,
        vae_tile_sample_stride=args.tile_sample_stride,
    )

    if is_main_process():
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        audio_save = audio[0].cpu().squeeze()
        save_video_with_audio(
            video[0],
            audio_save,
            args.output_path,
            fps=fps,
            sample_rate=pipeline.audio_sample_rate,
        )
        log.info(f"Saved to {args.output_path}")

    dist.barrier()
    log.info("Done.")


if __name__ == "__main__":
    main()
