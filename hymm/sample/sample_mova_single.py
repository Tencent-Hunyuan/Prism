
import argparse
import os
import subprocess

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file as safetensors_load_file

from hymm.config import parse_eval_initial_args
from hymm.diffusion.pipelines.mova_pipeline import MOVAPipeline
from hymm.models.modules.mova import MOVABridge
from hymm.sample.base_sampler import setup_distributed_initialize
from hymm.utils.file_utils import rank0_logger
from hymm.utils.parallel_states import initialize_parallel_state, nccl_info


def snap_num_frames(num_frames, temporal_scale):
    """Round a clip length down to the nearest value the video VAE accepts.

    MOVAPipeline.check_inputs requires `num_frames % temporal_scale == 1`.
    """
    num_frames = int(num_frames)
    if num_frames < temporal_scale + 1:
        return temporal_scale + 1
    return num_frames - (num_frames - 1) % temporal_scale


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"
)


def crop_and_resize(img, height, width):
    """Center-crop and resize to (height, width) preserving aspect ratio."""
    w, h = img.size
    target_ratio = width / height
    img_ratio = w / h

    if img_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    elif img_ratio < target_ratio:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    img = img.resize((width, height), Image.LANCZOS)
    return img


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def save_video_with_audio(frames, audio, save_path, fps, sample_rate=44100):
    """Save PIL frames + audio tensor to mp4 using ffmpeg."""
    import tempfile
    import wave
    import imageio

    with tempfile.TemporaryDirectory(prefix="mova_save_") as tmp_dir:
        tmp_video = os.path.join(tmp_dir, "video.mp4")
        tmp_audio = os.path.join(tmp_dir, "audio.wav")

        writer = imageio.get_writer(tmp_video, fps=fps, quality=9)
        for frame in frames:
            writer.append_data(np.array(frame))
        writer.close()

        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio)
        if audio_np.ndim == 1:
            audio_np = audio_np[None, :]
        channels, samples = audio_np.shape[0], audio_np.shape[1]
        if channels > 2:
            audio_np = audio_np[:2, :]
            channels = 2
        if np.issubdtype(audio_np.dtype, np.floating):
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767.0).astype(np.int16)
        elif audio_np.dtype != np.int16:
            audio_np = np.clip(audio_np, -32768, 32767).astype(np.int16)
        if channels == 1:
            interleaved = audio_np.reshape(-1)
        else:
            interleaved = audio_np.T.reshape(-1)
        with wave.open(tmp_audio, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(interleaved.tobytes(order="C"))

        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_video,
            "-i", tmp_audio,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-shortest",
            save_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e.stderr.decode(errors='ignore')[:500]}")
            import shutil
            shutil.copyfile(tmp_video, save_path)


def parse_single_args():
    """Parse extra arguments for single-case MOVA inference."""
    parser = argparse.ArgumentParser("MOVA single-case inference", add_help=False)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--audio_prompt", type=str, default=None)
    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./data/samples/mova_output.mp4")
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
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--offload", type=str, default="none", choices=("none", "cpu"),
                        help="CPU offload: move text_encoder/VAEs to CPU during denoising to save GPU memory")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Path to a training checkpoint (.safetensors or .pt) to override model weights")
    # --- VAE tiling knobs (isolated; default OFF). Reduces peak VAE-decode VRAM
    #     for high-res (e.g. 2K) videos. Mirrors decode_wan_vae.py defaults. ---
    parser.add_argument("--enable_tiling", type=str, default="false",
                        help="Enable tiled VAE decode to reduce VRAM for high-res decode. Default: false.")
    parser.add_argument("--tile_sample_min_size", type=int, default=None,
                        help="Tile size (height & width) in pixels. Default: diffusers/Wan default 256.")
    parser.add_argument("--tile_sample_stride", type=int, default=None,
                        help="Stride between tiles in pixels. overlap = tile_size - stride. Default: 192.")
    parser.add_argument("--enable_bsa", type=str, default="false",
                        help="Enable Block Sparse Attention for video self-attn. Default: false.")
    parser.add_argument("--bsa_sparsity", type=float, default=0.9375,
                        help="BSA sparsity ratio. Default: 0.9375.")
    parser.add_argument("--bsa_chunk_3d_shape_q", type=int, nargs=3, default=[4, 4, 4],
                        help="BSA 3D chunk shape for query (T H W). Default: 4 4 4.")
    parser.add_argument("--bsa_chunk_3d_shape_k", type=int, nargs=3, default=[4, 4, 4],
                        help="BSA 3D chunk shape for key (T H W). Default: 4 4 4.")
    parser.add_argument("--bsa_cdf_threshold", type=float, default=None,
                        help="BSA Top-p threshold for hybrid Top-k+Top-p masking. Default: None (Top-k only).")
    parser.add_argument("--enable_bsa_v2a", type=str, default="false",
                        help="Enable BSA for v2a bridge cross-attn (Q=audio, K=video). Default: false.")
    parser.add_argument("--bsa_v2a_sparsity", type=float, default=0.875,
                        help="BSA sparsity for v2a. Default: 0.875.")
    parser.add_argument("--bsa_v2a_audio_chunk_size", type=int, default=64,
                        help="BSA audio Q block size for v2a (multiple of 64). Default: 64.")
    parser.add_argument("--bsa_v2a_chunk_3d_shape_k", type=int, nargs=3, default=[4, 4, 4],
                        help="BSA 3D chunk for v2a K (video side). Default: 4 4 4.")
    parser.add_argument("--bsa_v2a_cdf_threshold", type=float, default=None,
                        help="BSA Top-p threshold for v2a hybrid masking. Default: None.")
    parser.add_argument("--enable_audio_guidance", type=str, default="false",
                        help="Enable audio-guided BSA score modulation. Default: false.")
    parser.add_argument("--enable_audio_concentration_gate", type=str, default="false",
                        help="Enable Gate 2 (Audio Spatial Concentration Gate). Default: false.")
    parser.add_argument("--enable_timestep_reliability_gate", type=str, default="false",
                        help="Enable Gate 1 (Timestep Reliability Gate). Default: false.")
    parser.add_argument("--audio_boost_gamma", type=float, default=1.0,
                        help="Audio guidance boost strength γ (Path A). Default: 1.0.")
    parser.add_argument("--enable_audio_weighted_pooling", type=str, default="false",
                        help="Enable Path B: Audio-weighted K pooling. Default: false.")
    parser.add_argument("--audio_weighted_lambda", type=float, default=1.0,
                        help="Audio-weighted K pooling strength λ (Path B). Default: 1.0.")
    parser.add_argument("--enable_variance_guidance", type=str, default="false",
                        help="Enable channel-variance guided BSA (independent from audio). Default: false.")
    parser.add_argument("--variance_boost_gamma", type=float, default=1.0,
                        help="Variance guidance boost strength γ. Default: 1.0.")
    parser.add_argument("--enable_taylor_sparse_attn", type=str, default="false",
                        help="LIVEditor: Taylor sparse attn (selected + non-selected in one softmax). Default: false.")
    parser.add_argument("--taylor_alpha_f", type=float, default=0.5,
                        help="Taylor flat ratio: 0.5 = 50%% flat queries get Taylor, 50%% sharp keep BSA.")
    parser.add_argument("--enable_rectified_sparse_attn", type=str, default="false",
                        help="Rectified SpaAttn: R_n * o_spa + A_pool[nonsel] · V_pool. Default: false.")
    # Anisotropic Dynamic Block Shape (Section 8) — video self-attn only. Two
    # mutually-exclusive methods; enabling either disables all other BSA features.
    parser.add_argument("--enable_ivpq_dynamic_block", type=str, default="false",
                        help="IVPQ dynamic block shape (Section 8.5). Default: false.")
    parser.add_argument("--enable_penalty_dynamic_block", type=str, default="false",
                        help="Penalty-Matching dynamic block shape (Section 8.6). Default: false.")
    parser.add_argument("--dynamic_block_lambda_a", type=float, default=0.5,
                        help="Audio-directional influence strength λ_a for g_d. Default: 0.5.")
    parser.add_argument("--dynamic_block_tau_128", type=float, default=0.15,
                        help="Info-density threshold τ_128 gating the 128-token pool. Default: 0.15.")
    parser.add_argument("--dynamic_block_lambda_128", type=float, default=1.0,
                        help="128-token density bonus λ_128 (Penalty Matching only). Default: 1.0.")
    parser.add_argument("--enable_layer_adaptive_dynamic_block", type=str, default="false",
                        help="Layer-adaptive dynamic block (Section 8.2): shallow half fixed / "
                             "deep half dynamic + shallow→deep audio cache. Default: false.")
    parser.add_argument("--sparse_high_noise_only", type=str, default="false",
                        help="If 'true', apply all sparse-attn features ONLY to the high-noise "
                             "expert (video_dit); video_dit_2 keeps dense full attention. "
                             "Default: false = sparse on both video experts.")
    args, _ = parser.parse_known_args()
    return args


def init_mova_sp(sp_size):
    """Set up Ulysses sequence parallelism for inference.

    Same entry point the trainer uses, so the device meshes, the nccl_info view
    the model reads, and the communicator warm-up are identical in both. The
    warm-up matters here too: it creates every NCCL communicator up front
    instead of on the first collective inside the denoising loop.
    """
    world_size = dist.get_world_size()
    assert world_size % sp_size == 0, (
        f"world_size ({world_size}) must be divisible by sp_size ({sp_size})"
    )
    return initialize_parallel_state(sp=sp_size, dp_replicate=world_size // sp_size)


def main():
    initial_args, mode = parse_eval_initial_args()
    world_size, rank, device = setup_distributed_initialize(initial_args, mode)
    log = rank0_logger(rank)

    single_args = parse_single_args()

    sp_size = single_args.sp_size
    if sp_size > 1:
        assert dist.is_initialized(), "SP requires distributed init (use --deepspeed or --ddp)"
        init_mova_sp(sp_size)
        log.info(f"Sequence parallelism enabled: sp_size={sp_size}, world_size={world_size}")

    config_path = initial_args.config
    if config_path and os.path.exists(config_path):
        config = OmegaConf.load(config_path)
    else:
        config = OmegaConf.create({})

    infer_cfg = config.get("inference", OmegaConf.create({}))
    height = single_args.height or infer_cfg.get("height", 352)
    width = single_args.width or infer_cfg.get("width", 640)
    num_frames = single_args.num_frames or infer_cfg.get("num_frames", 193)
    fps = single_args.fps or infer_cfg.get("fps", 24.0)
    num_inference_steps = single_args.num_inference_steps or infer_cfg.get("num_inference_steps", 50)
    cfg_scale = single_args.cfg_scale or infer_cfg.get("cfg_scale", 5.0)
    visual_shift = single_args.visual_shift or infer_cfg.get("visual_shift", 5.0)
    audio_shift = single_args.audio_shift or infer_cfg.get("audio_shift", 5.0)
    seed = single_args.seed if single_args.seed is not None else infer_cfg.get("seed", 42)
    negative_prompt = single_args.negative_prompt or NEGATIVE_PROMPT

    ckpt_path = initial_args.ckpt
    torch_dtype = torch.bfloat16
    use_offload = single_args.offload == "cpu"

    log.info(f"Loading MOVA from {ckpt_path}")
    bridge, extras = MOVABridge.from_mova_pretrained(ckpt_path, torch_dtype=torch_dtype)

    if single_args.resume_ckpt:
        resume_path = single_args.resume_ckpt
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

    # Gate sparse features to high-noise expert only (must precede configure_* calls).
    sparse_high_noise_only = getattr(single_args, 'sparse_high_noise_only', 'false')
    if isinstance(sparse_high_noise_only, str):
        sparse_high_noise_only = sparse_high_noise_only.lower() == 'true'
    bridge.set_sparse_high_noise_only(sparse_high_noise_only)
    log.info(
        f"[Sparse Attn Scope] sparse_high_noise_only={sparse_high_noise_only} "
        f"({'video_dit only; video_dit_2 = full attn' if sparse_high_noise_only else 'both video experts'})"
    )

    enable_bsa = getattr(single_args, 'enable_bsa', 'false')
    if isinstance(enable_bsa, str):
        enable_bsa = enable_bsa.lower() == 'true'
    enable_bsa_v2a = getattr(single_args, 'enable_bsa_v2a', 'false')
    if isinstance(enable_bsa_v2a, str):
        enable_bsa_v2a = enable_bsa_v2a.lower() == 'true'

    if enable_bsa or enable_bsa_v2a:
        bsa_params = {
            'sparsity': single_args.bsa_sparsity,
            'cdf_threshold': getattr(single_args, 'bsa_cdf_threshold', None),
            'chunk_3d_shape_q': list(single_args.bsa_chunk_3d_shape_q),
            'chunk_3d_shape_k': list(single_args.bsa_chunk_3d_shape_k),
        } if enable_bsa else None
        bsa_params_v2a = {
            'sparsity': single_args.bsa_v2a_sparsity,
            'cdf_threshold': getattr(single_args, 'bsa_v2a_cdf_threshold', None),
            'chunk_size_q': single_args.bsa_v2a_audio_chunk_size,
            'chunk_3d_shape_k': list(single_args.bsa_v2a_chunk_3d_shape_k),
        } if enable_bsa_v2a else None
        bridge.configure_bsa(
            enable_bsa=enable_bsa, bsa_params=bsa_params,
            enable_bsa_v2a=enable_bsa_v2a, bsa_params_v2a=bsa_params_v2a,
        )
        vid_cdf = bsa_params.get('cdf_threshold') if bsa_params else None
        v2a_cdf = bsa_params_v2a.get('cdf_threshold') if bsa_params_v2a else None
        vid_mode = "Hybrid Top-k+Top-p" if vid_cdf else "Top-k only"
        v2a_mode = "Hybrid Top-k+Top-p" if v2a_cdf else "Top-k only"
        log.info(
            f"BSA configured: video_self={enable_bsa} [{vid_mode}] "
            f"(sparsity={bsa_params['sparsity'] if bsa_params else 'N/A'}, cdf_threshold={vid_cdf}), "
            f"v2a={enable_bsa_v2a} [{v2a_mode}] "
            f"(sparsity={bsa_params_v2a['sparsity'] if bsa_params_v2a else 'N/A'}, cdf_threshold={v2a_cdf})"
        )

    # --- Audio Guidance for BSA ---
    enable_audio_guidance = getattr(single_args, 'enable_audio_guidance', 'false')
    if isinstance(enable_audio_guidance, str):
        enable_audio_guidance = enable_audio_guidance.lower() == 'true'
    enable_audio_concentration_gate = getattr(single_args, 'enable_audio_concentration_gate', 'false')
    if isinstance(enable_audio_concentration_gate, str):
        enable_audio_concentration_gate = enable_audio_concentration_gate.lower() == 'true'
    enable_timestep_reliability_gate = getattr(single_args, 'enable_timestep_reliability_gate', 'false')
    if isinstance(enable_timestep_reliability_gate, str):
        enable_timestep_reliability_gate = enable_timestep_reliability_gate.lower() == 'true'
    enable_audio_weighted_pooling = getattr(single_args, 'enable_audio_weighted_pooling', 'false')
    if isinstance(enable_audio_weighted_pooling, str):
        enable_audio_weighted_pooling = enable_audio_weighted_pooling.lower() == 'true'
    audio_boost_gamma = float(getattr(single_args, 'audio_boost_gamma', 1.0))
    audio_weighted_lambda = float(getattr(single_args, 'audio_weighted_lambda', 1.0))
    log.info(
        f"[Audio Guidance] Path A: enable={enable_audio_guidance}, "
        f"concentration_gate={enable_audio_concentration_gate}, "
        f"timestep_gate={enable_timestep_reliability_gate}, "
        f"boost_gamma={audio_boost_gamma} | "
        f"Path B: enable={enable_audio_weighted_pooling}, "
        f"weighted_lambda={audio_weighted_lambda}"
    )
    if enable_audio_guidance or enable_audio_weighted_pooling:
        for fused_block in bridge.fusion_blocks:
            fused_block.video_block.self_attn.bsa_params['audio_boost_gamma'] = audio_boost_gamma
            fused_block.video_block.self_attn.bsa_params['audio_weighted_lambda'] = audio_weighted_lambda
        for block in bridge.remaining_video_blocks:
            block.self_attn.bsa_params['audio_boost_gamma'] = audio_boost_gamma
            block.self_attn.bsa_params['audio_weighted_lambda'] = audio_weighted_lambda
        bridge.configure_audio_guidance(
            enable=enable_audio_guidance,
            enable_concentration_gate=enable_audio_concentration_gate,
            enable_timestep_reliability_gate=enable_timestep_reliability_gate,
            enable_audio_weighted_pooling=enable_audio_weighted_pooling,
        )

    # --- Channel-Variance Guidance for BSA (independent from audio guidance) ---
    enable_variance_guidance = getattr(single_args, 'enable_variance_guidance', 'false')
    if isinstance(enable_variance_guidance, str):
        enable_variance_guidance = enable_variance_guidance.lower() == 'true'
    variance_boost_gamma = float(getattr(single_args, 'variance_boost_gamma', 1.0))
    log.info(
        f"[Variance Guidance] enable={enable_variance_guidance}, "
        f"boost_gamma={variance_boost_gamma}"
    )
    if enable_variance_guidance:
        for fused_block in bridge.fusion_blocks:
            fused_block.video_block.self_attn.bsa_params['variance_boost_gamma'] = variance_boost_gamma
        for block in bridge.remaining_video_blocks:
            block.self_attn.bsa_params['variance_boost_gamma'] = variance_boost_gamma
        bridge.configure_variance_guidance(enable=True)

    # --- Bias Correction: two independent methods ---
    enable_taylor_sa = getattr(single_args, 'enable_taylor_sparse_attn', 'false')
    if isinstance(enable_taylor_sa, str):
        enable_taylor_sa = enable_taylor_sa.lower() == 'true'
    enable_rectified_sa = getattr(single_args, 'enable_rectified_sparse_attn', 'false')
    if isinstance(enable_rectified_sa, str):
        enable_rectified_sa = enable_rectified_sa.lower() == 'true'
    if enable_taylor_sa and enable_rectified_sa:
        raise ValueError(
            "enable_taylor_sparse_attn and enable_rectified_sparse_attn are mutually "
            "exclusive bias-correction methods — enable at most one."
        )
    log.info(
        f"[Bias Correction] taylor_sparse_attn={enable_taylor_sa}, "
        f"rectified_sparse_attn={enable_rectified_sa}"
    )
    if enable_taylor_sa:
        _alpha = float(getattr(single_args, 'taylor_alpha_f', 0.5))
        for fb in bridge.fusion_blocks:
            fb.video_block.self_attn.bsa_params['taylor_alpha_f'] = _alpha
        for blk in bridge.remaining_video_blocks:
            blk.self_attn.bsa_params['taylor_alpha_f'] = _alpha
        bridge.configure_taylor_sparse_attn(enable=True)
    if enable_rectified_sa:
        bridge.configure_rectified_sparse_attn(enable=True)

    # --- Anisotropic Dynamic Block Shape (Section 8): video self-attn only ---
    enable_ivpq = getattr(single_args, 'enable_ivpq_dynamic_block', 'false')
    if isinstance(enable_ivpq, str):
        enable_ivpq = enable_ivpq.lower() == 'true'
    enable_penalty = getattr(single_args, 'enable_penalty_dynamic_block', 'false')
    if isinstance(enable_penalty, str):
        enable_penalty = enable_penalty.lower() == 'true'
    if enable_ivpq and enable_penalty:
        raise ValueError(
            "enable_ivpq_dynamic_block and enable_penalty_dynamic_block are mutually exclusive."
        )
    if enable_ivpq or enable_penalty:
        if (enable_audio_guidance or enable_audio_weighted_pooling
                or enable_variance_guidance or enable_taylor_sa or enable_rectified_sa):
            raise ValueError(
                "Dynamic block shape (IVPQ/Penalty) is mutually exclusive with audio guidance, "
                "variance guidance and taylor/rectified bias correction. Disable those triggers."
            )
        _la = float(getattr(single_args, 'dynamic_block_lambda_a', 0.5))
        _t128 = float(getattr(single_args, 'dynamic_block_tau_128', 0.15))
        _l128 = float(getattr(single_args, 'dynamic_block_lambda_128', 1.0))
        for _sa in bridge._video_self_attns():
            _sa.bsa_params['dynamic_block_lambda_a'] = _la
            _sa.bsa_params['dynamic_block_tau_128'] = _t128
            _sa.bsa_params['dynamic_block_lambda_128'] = _l128
        if enable_ivpq:
            bridge.configure_ivpq_dynamic_block(enable=True)
        else:
            bridge.configure_penalty_dynamic_block(enable=True)
        log.info(
            f"[Dynamic Block Shape] method={'IVPQ' if enable_ivpq else 'Penalty'}, "
            f"lambda_a={_la}, tau_128={_t128}, lambda_128={_l128}"
        )

        # Layer-adaptive (Section 8.2): shallow fixed / deep dynamic + shallow→deep
        # audio cache. Default off. Must run after the ivpq/penalty configure above.
        _enable_la = getattr(single_args, 'enable_layer_adaptive_dynamic_block', 'false')
        if isinstance(_enable_la, str):
            _enable_la = _enable_la.lower() == 'true'
        if _enable_la:
            _split, _cache_src = bridge.configure_layer_adaptive_dynamic_block(enable=True)
            log.info(
                f"[Layer-Adaptive Dynamic Block] shallow(<{_split}) fixed / "
                f"deep(>={_split}) dynamic; audio cache-source layer={_cache_src}"
            )

    device_t = torch.device("cuda", device)
    bridge.eval()

    if use_offload:
        log.info("CPU offload enabled — components will be moved to GPU on demand")
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
        bridge = bridge.to(device_t)
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

    if not os.path.exists(single_args.ref_path):
        raise FileNotFoundError(f"Reference image not found: {single_args.ref_path}")

    ref_img = Image.open(single_args.ref_path).convert("RGB")
    ref_img = crop_and_resize(ref_img, height=height, width=width)

    torch.manual_seed(seed)

    enable_tiling = getattr(single_args, 'enable_tiling', 'false')
    if isinstance(enable_tiling, str):
        enable_tiling = enable_tiling.lower() == 'true'
    if enable_tiling:
        _ts = single_args.tile_sample_min_size if single_args.tile_sample_min_size is not None else 256
        _st = single_args.tile_sample_stride if single_args.tile_sample_stride is not None else 192
        log.info(
            f"[VAE Tiling] ENABLED: tile_size={_ts}, stride={_st}, "
            f"overlap={_ts - _st}px ({(_ts - _st) / _ts * 100:.0f}%)"
        )

    raw_num_frames = num_frames
    num_frames = snap_num_frames(num_frames, pipeline.vae_scale_factor_temporal)
    if num_frames != int(raw_num_frames):
        log.info(
            f"NOTE: num_frames {raw_num_frames} -> {num_frames} "
            f"(must satisfy (n-1) % {pipeline.vae_scale_factor_temporal} == 0)"
        )

    log.info(f"Starting inference: {height}x{width}, {num_frames} frames, seed={seed}")
    log.info(f"Prompt: {single_args.prompt}")
    log.info(f"Audio prompt: {single_args.audio_prompt}")

    video, audio = pipeline(
        prompt=single_args.prompt,
        image=ref_img,
        audio_prompt=single_args.audio_prompt,
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
        vae_tile_sample_min_size=single_args.tile_sample_min_size,
        vae_tile_sample_stride=single_args.tile_sample_stride,
    )

    if is_main_process():
        os.makedirs(os.path.dirname(single_args.output_path) or ".", exist_ok=True)
        audio_save = audio[0].cpu().squeeze()
        save_video_with_audio(
            video[0],
            audio_save,
            single_args.output_path,
            fps=fps,
            sample_rate=pipeline.audio_sample_rate,
        )
        log.info(f"Saved to {single_args.output_path}")

    if dist.is_initialized():
        dist.barrier()

    log.info("Done.")


if __name__ == "__main__":
    main()


# export node_ip=$(echo ${NODE_IP_LIST} | sed 's/:8//g')
# pdsh -f 256 -w $node_ip "pip install ipdb"
# pdsh -f 256 -w $node_ip "pip install yunchang"
# pdsh -f 256 -w $node_ip "pip install mmengine"
# pdsh -f 256 -w $node_ip "pip uninstall diffusers -y"
# pdsh -f 256 -w $node_ip "pip install diffusers==0.33.0"
# pdsh -f 256 -w $node_ip "pip install descript-audiotools"
# pdsh -f 256 -w $node_ip "pip install imageio[ffmpeg]"
# pdsh -f 256 -w $node_ip "pip install imageio[pyav]"

# nohup bash prism_infer.sh > infer_output.txt 2>&1 &
# nohup bash prism_infer_fsdp.sh > infer_fsdp_output.txt 2>&1 &
