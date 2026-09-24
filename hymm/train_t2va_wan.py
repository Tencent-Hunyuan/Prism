import math
import os
import sys
import gc
import re
import ftfy
import html
import warnings

from torch.distributed.tensor import DTensor

gc.set_threshold(7000, 100, 100)
import time
import random
from omegaconf import OmegaConf
from accelerate.utils import set_seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from hymm.utils.load import load_ti2va_wan_transformer
from hymm.config import parse_args
from hymm.constants import C_SCALE
from hymm.dataset.csv_video_audio_dataset import CsvVideoAudioDataset, collate_video_audio
from hymm.dataset.bucket_sampler import (
    DistributedBucketBatchSampler,
    EpochCyclingBatchIterator,
)
from hymm.utils.torch_utils import set_worker_seed_builder
from hymm.utils.parallel_states import (
    init_distributed,
    initialize_parallel_state,
    set_collective_timeout,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    sync_data_for_sp,
    sync_random_states,
)
from hymm.utils.fsdp_util import (
    get_dit_fsdp_kwargs_v2,
    apply_fsdp2
)
from hymm.utils.logging_ import setup_logger
from hymm.utils.checkpoint import (
    fsdp_save_checkpoint_without_optim,
    resume_wan_training,
    save_optimizer_state,
    load_optimizer_state,
)
from hymm.utils.helpers import ScalarStates, CycleStates

check_min_version("0.31.0")

# Conditioning modes this trainer emits. Fixed so the per-log-interval reduction
# has an identical tensor length on every rank.
MASK_TYPES = ("i2v",)


# ---------------------------------------------------------------------------
# MOVA-specific helpers (self-contained, no codebase modifications needed)
# ---------------------------------------------------------------------------

def _basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def _prompt_clean(text):
    return _whitespace_clean(_basic_clean(text))


def _normalize_video_latents(video_vae_config, latents):
    """Normalize video latents using mean/std from the WAN VAE config."""
    z_dim = video_vae_config.z_dim
    mean = torch.tensor(
        video_vae_config.latents_mean, device=latents.device, dtype=latents.dtype
    ).view(1, z_dim, 1, 1, 1)
    inv_std = (
        1.0
        / torch.tensor(
            video_vae_config.latents_std, device=latents.device, dtype=latents.dtype
        )
    ).view(1, z_dim, 1, 1, 1)
    return (latents - mean) * inv_std


def _compute_density_for_timestep_sampling(
    weighting_scheme,
    batch_size,
    logit_mean=None,
    logit_std=None,
    mode_scale=None,
    min_timestep_boundary=0.0,
    max_timestep_boundary=1.0,
):
    if weighting_scheme == "logit_normal":
        u = torch.zeros(size=(batch_size,), device="cpu")
        a = torch.logit(torch.tensor(min_timestep_boundary))
        b = torch.logit(torch.tensor(max_timestep_boundary))
        torch.nn.init.trunc_normal_(u, mean=logit_mean, std=logit_std, a=a, b=b)
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
        u = min_timestep_boundary + (u * (max_timestep_boundary - min_timestep_boundary))
    return u


def _sample_timestep_id(scheduler, max_timestep_boundary=1.0, min_timestep_boundary=0.0, weighting_scheme="uniform"):
    total_timesteps = scheduler.num_train_timesteps
    u = _compute_density_for_timestep_sampling(
        weighting_scheme=weighting_scheme,
        batch_size=1,
        min_timestep_boundary=min_timestep_boundary,
        max_timestep_boundary=max_timestep_boundary,
    )
    timestep_id = torch.floor(u * total_timesteps).to(dtype=torch.long)
    int_min = int(min_timestep_boundary * total_timesteps)
    int_max = int(max_timestep_boundary * total_timesteps)
    timestep_id = torch.clamp(timestep_id, min=int_min, max=int_max - 1)
    return timestep_id


def _sample_timestep_pair(scheduler, device, boundary_ratio=0.9, global_step=0):
    """
    Sample a (visual_timestep, audio_timestep) pair with boundary alternation.
    Even steps sample high-noise (early indices where timestep >= boundary),
    odd steps sample low-noise (later indices where timestep < boundary).
    Note: scheduler.timesteps is descending, so index [0, boundary) = high noise.
    """
    boundary = (
        (scheduler.timesteps >= boundary_ratio * scheduler.num_train_timesteps)
        .sum()
        .item()
        / scheduler.num_train_timesteps
    )
    if global_step % 2 == 0:
        max_tb = boundary
        min_tb = 0.0
    else:
        max_tb = 1.0
        min_tb = boundary

    timestep_id = _sample_timestep_id(scheduler, max_tb, min_tb, "uniform")
    base_timestep = scheduler.timesteps[timestep_id].to(device=device)

    pair_timesteps = getattr(scheduler, "pair_timesteps", None)
    if pair_timesteps is None:
        return base_timestep.clone(), base_timestep.clone()

    pair_matrix = scheduler.get_pairs("timesteps")
    pair_row = pair_matrix[timestep_id].to(device=device, dtype=base_timestep.dtype)
    return pair_row[:, 0], pair_row[:, 1]


@torch.no_grad()
def _get_t5_prompt_embeds(text_encoder, tokenizer, prompts, device, max_length=512):
    """Encode text prompts with UMT5, matching MOVA's _get_t5_prompt_embeds."""
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = [_prompt_clean(p) for p in prompts]
    batch_size = len(prompts)

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    mask = text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(
        text_input_ids.to(device), mask.to(device)
    ).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds


@torch.no_grad()
def _encode_first_frame(ref_image, video_latents, video_vae, video_vae_config, device, dtype):
    """
    Encode the reference first frame to produce the i2v conditioning tensor y.

    Args:
        ref_image: [B, 3, H_pixel, W_pixel] normalized to [-1, 1]
        video_latents: [B, C, T_lat, H_lat, W_lat] pre-cached video latents
        video_vae: AutoencoderKLWan instance (frozen, on device)
        video_vae_config: video_vae.config
        device, dtype: target device and dtype

    Returns:
        y: [B, 20, T_lat, H_lat, W_lat]  (4ch mask + 16ch encoded first frame)
    """
    B = video_latents.shape[0]
    C_latent = video_latents.shape[1]
    T_lat = video_latents.shape[2]
    H_lat = video_latents.shape[3]
    W_lat = video_latents.shape[4]

    spr = getattr(video_vae_config, "spatial_compression_ratio", None) or video_vae_config.scale_factor_spatial
    tpr = getattr(video_vae_config, "temporal_compression_ratio", None) or getattr(video_vae_config, "scale_factor_temporal", 4)

    target_h = H_lat * spr
    target_w = W_lat * spr
    num_frames = (T_lat - 1) * tpr + 1

    ref = F.interpolate(
        ref_image.to(dtype=dtype, device=device),
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
    )

    vae_input = torch.zeros(B, 3, num_frames, target_h, target_w, device=device, dtype=dtype)
    vae_input[:, :, 0, :, :] = ref

    with torch.autocast("cuda", dtype=dtype):
        encoded = video_vae.encode(vae_input).latent_dist.mode()
        encoded = _normalize_video_latents(video_vae_config, encoded)

    msk = torch.zeros(B, 4, T_lat, H_lat, W_lat, device=device, dtype=dtype)
    msk[:, :, 0, :, :] = 1.0

    y = torch.cat([msk, encoded], dim=1)
    return y


def _mova_params_count(model):
    """Compute parameter counts for MOVABridge (compatible with print_training_configuration)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "attn+mlp": total,
    }


def _get_t2va_wan_dataloader(ori_args, args, dp_degree, dp_rank, logger):
    """Build the CSV dataset, the bucket batch sampler and the DataLoader."""
    batch_size = args.video_micro_batch_size
    if not isinstance(batch_size, int):
        # The old temporal sampler took a per-bucket list; a single value is
        # enough now that a batch is always inside one bucket.
        batch_size = int(batch_size[0])

    video_dataset = CsvVideoAudioDataset(
        csv_files=args.video_csv_file,
        args=args,
        video_fps=getattr(args, "video_fps", 24.0),
        vae_spatial_ratio=args.vae_spatial_ratio,
        multireso=args.video_multireso,
        multitemp=args.video_multitemp,
        bucket_hw_base_size=args.video_bucket_hw_base_size,
        bucket_hw_bucket_stride=args.video_bucket_hw_bucket_stride,
        bucket_temporal_min_length=args.video_bucket_temporal_min_length,
        bucket_temporal_max_length=args.video_bucket_temporal_max_length,
        bucket_temporal_interval=args.video_bucket_temporal_interval,
        latent_resolution=getattr(args, "latent_resolution", None),
        caption_sample_ratio=args.video_caption_sample_ratio,
        caption_processor=getattr(args, "video_caption_processor", "caption_process_v1"),
        ocr_only_long_caption=args.ocr_only_long_caption,
        num_replace_rate=args.video_num_replace_rate,
        video_uncond_p=args.video_uncond_p,
        audio_uncond_p=args.audio_uncond_p,
        audio_sr=args.audio_sr,
        crop_latent_to_bucket=bool(getattr(args, "crop_latent_to_bucket", True)),
        global_seed=ori_args.global_seed,
        logger=logger,
    )

    video_batch_sampler = DistributedBucketBatchSampler(
        bucket_keys=video_dataset.bucket_keys,
        batch_size=batch_size,
        num_replicas=dp_degree,
        rank=dp_rank,
        seed=ori_args.global_seed,
        shuffle=True,
        drop_last=True,
    )
    if video_batch_sampler.batches_per_epoch() == 0:
        raise ValueError(
            f"The bucket plan yields {video_batch_sampler.num_global_batches()} global "
            f"batches for {dp_degree} DP ranks at micro_batch_size={batch_size}. "
            f"Every rank must get at least one batch -- add data, lower the batch size, "
            f"or coarsen the buckets (larger --temporal-interval / smaller sp_size)."
        )

    video_loader = DataLoader(
        video_dataset,
        batch_sampler=video_batch_sampler,
        collate_fn=collate_video_audio,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=None if args.num_workers == 0 else args.prefetch_factor,
        worker_init_fn=set_worker_seed_builder(dp_rank),
        # Workers are re-forked each epoch so they pick up dataset.set_epoch().
        persistent_workers=False,
    )

    return video_dataset, video_batch_sampler, video_loader


# ---------------------------------------------------------------------------
# Utilities (kept from original)
# ---------------------------------------------------------------------------

def sync_cuda_time(sync=False, barrier=False):
    """Timestamp for the step timers.

    Both the device sync and the barrier default to off. The old default put a
    ``dist.barrier()`` plus a ``cuda.synchronize()`` at five points inside every
    training step, which pinned the whole job to the slowest rank five times per
    step and gave the NCCL watchdog five extra places to trip once buckets made
    ranks uneven. The timers are only read at log boundaries, where the caller
    asks for a real sync explicitly.
    """
    if barrier and dist.is_initialized():
        dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def print_training_configuration(args, logger, params_count, model, world_size, local_rank, rank,
                                 dp_degree, dp_rank,
                                 video_audio_total_batch_size=0, video_audio_num=0,
                                 video_audio_sampler=None, video_audio_dataset=None,
                                 scalar_states=None, t2va_args=None):
    logger.info("****************************** Running training ******************************")

    # ===============================================================================
    # System & Hardware Configuration
    # ===============================================================================
    logger.info("=" * 80)
    logger.info("SYSTEM & HARDWARE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Number of GPUs                                   : {world_size}")
    logger.info(f"Local rank                                       : {local_rank}")
    logger.info(f"Global rank                                      : {rank}")
    logger.info(f"Device                                           : cuda:{local_rank}")
    logger.info(f"Master weight dtype                              : {model.parameters().__next__().dtype}")

    # ===============================================================================
    # Model Parameters & Architecture
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("MODEL PARAMETERS & ARCHITECTURE")
    logger.info("-" * 80)
    for k, v in params_count.items():
        logger.info(f"Number of {k:<25}                     : {v:,}")
    total_params_b = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    logger.info(f"Total trainable parameters per FSDP shard       : {total_params_b:.3f}B")
    if t2va_args is not None:
        pretrained_path = getattr(t2va_args.training_config, 'pretrained_model_name_or_path', 'N/A')
        logger.info(f"Pretrained model path                            : {pretrained_path}")
    logger.info(f"Gradient checkpointing                           : {args.gradient_checkpointing}")
    logger.info(f"Selective checkpointing                          : {args.selective_checkpointing}")
    logger.info(f"Dynamic ring attention                           : {args.use_dynamic_ring_attention}")

    # ===============================================================================
    # Distributed Training Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("DISTRIBUTED TRAINING CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"World size                                       : {world_size}")
    logger.info(f"DP degree                                        : {dp_degree}")
    logger.info(f"DP rank                                          : {dp_rank}")
    logger.info(f"SP size (Ulysses)                                : {args.sp_size}")
    logger.info(f"FSDP sharding strategy                           : {args.fsdp_sharding_strategy}")
    logger.info(f"CPU offload                                      : {args.use_cpu_offload}")

    # ===============================================================================
    # Batch Size & Data Flow Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("BATCH SIZE & DATA FLOW CONFIGURATION")
    logger.info("-" * 80)
    per_gpu_batch = args.micro_batch_size[0] if isinstance(args.micro_batch_size, list) else args.micro_batch_size
    dp_batch = per_gpu_batch * dp_degree
    effective_batch_video_audio = video_audio_total_batch_size * args.gradient_accumulation_steps
    logger.info(f"Micro batch size                                 : {args.micro_batch_size}")
    logger.info(f"Per GPU batch size                               : {per_gpu_batch}")
    logger.info(f"DP degree batch size                             : {dp_batch}")
    logger.info(f"Total batch size (video_audio)                   : {video_audio_total_batch_size}")
    logger.info(f"Gradient accumulation steps                      : {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size (with grad accum)           : {effective_batch_video_audio}")
    logger.info(f"Samples per optimizer update                     : {effective_batch_video_audio}")

    # ===============================================================================
    # Dataset & DataLoader Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("DATASET & DATALOADER CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Data type                                        : {args.data_type}")
    logger.info(f"Video audio examples                             : {video_audio_num:,}")
    logger.info(f"Video sampling probability                       : {args.video_sampling_prob}")
    logger.info(f"Video multireso                                  : {args.video_multireso}")

    if video_audio_dataset is not None:
        logger.info(f"Video audio dataset length                       : {len(video_audio_dataset)}")
        logger.info(f"Bucket plan                                      : {video_audio_dataset.bucket_plan.describe()}")
    if video_audio_sampler is not None:
        logger.info(f"Global batches per epoch                         : {video_audio_sampler.num_global_batches()}")
        logger.info(f"Batches per epoch per DP rank                    : {video_audio_sampler.batches_per_epoch()}")
        logger.info(f"Batch sampler batch size                         : {video_audio_sampler.batch_size}")

    # ===============================================================================
    # Training Schedule & Steps Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("TRAINING SCHEDULE & STEPS CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Number of epochs                                 : {args.num_train_epochs}")
    logger.info(f"Max training steps                               : {args.max_train_steps}")
    total_update_steps = args.max_train_steps // args.gradient_accumulation_steps
    logger.info(f"Total forward/backward steps                     : {args.max_train_steps}")
    logger.info(f"Total optimizer update steps                     : {total_update_steps}")
    logger.info(f"Forward steps per optimizer update               : {args.gradient_accumulation_steps}")

    # ===============================================================================
    # Optimization Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("OPTIMIZATION CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Optimizer                                        : {args.optimizer}")
    logger.info(f"Learning rate                                    : {args.learning_rate}")
    logger.info(f"Weight decay                                     : {args.weight_decay}")
    logger.info(f"Max gradient norm                                : {args.max_grad_norm}")
    logger.info(f"LR scheduler                                     : {args.lr_scheduler}")
    logger.info(f"LR warmup steps                                  : {args.lr_warmup_steps}")
    logger.info(f"LR num cycles                                    : {args.lr_num_cycles}")
    logger.info(f"LR power                                         : {args.lr_power}")

    # ===============================================================================
    # Logging & Checkpointing Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("LOGGING & CHECKPOINTING CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Output directory                                 : {args.output_dir}")
    logger.info(f"Checkpointing steps                              : {args.checkpointing_steps}")
    logger.info(f"Log interval                                     : {args.log_interval}")
    logger.info(f"Resume training                                  : {args.resume}")
    logger.info(f"Profiler enabled                                 : {args.is_profiler}")
    logger.info(f"Dry run enabled                                  : {getattr(args, 'dry_run', False)}")

    # ===============================================================================
    # Seeds & Reproducibility Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("SEEDS & REPRODUCIBILITY CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Global seed                                      : {args.global_seed}")
    logger.info(f"Local seed                                       : {args.local_seed}")

    # ===============================================================================
    # MOVA-specific Configuration
    # ===============================================================================
    if t2va_args is not None:
        logger.info("-" * 80)
        logger.info("MOVA-SPECIFIC CONFIGURATION")
        logger.info("-" * 80)
        logger.info(f"Video loss weight                                : {getattr(t2va_args.training_config, 'video_loss_weight', 'N/A')}")
        logger.info(f"Audio loss weight                                : {getattr(t2va_args.training_config, 'audio_loss_weight', 'N/A')}")

    # ===============================================================================
    # Performance Estimates
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("PERFORMANCE ESTIMATES")
    logger.info("-" * 80)
    logger.info(f"Target throughput per update                     : {effective_batch_video_audio} (video_audio)")

    # ===============================================================================
    # Mask Type Statistics
    # ===============================================================================
    if scalar_states and scalar_states.consumed_samples_by_mask_type_total:
        logger.info("-" * 80)
        logger.info("MASK TYPE STATISTICS (RESUMED TRAINING)")
        logger.info("-" * 80)
        mask_type_stats = dict(scalar_states.consumed_samples_by_mask_type_total)
        total_mask_samples = sum(mask_type_stats.values())
        for mt, count in mask_type_stats.items():
            percentage = (count / total_mask_samples * 100) if total_mask_samples > 0 else 0
            logger.info(f"Mask type '{mt}' samples                            : {count:,} ({percentage:.1f}%)")
        logger.info(f"Total mask type samples                          : {total_mask_samples:,}")

    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)


def print_memory_usage(stage):
    if torch.distributed.get_rank() == 0:
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"{stage}: Allocated: {allocated:.2f} MiB")
        print(f"{stage}: Reserved: {reserved:.2f} MiB")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# MOVA-specific FSDP activation checkpointing
# ---------------------------------------------------------------------------

def _apply_mova_fsdp_checkpointing(model, p=1, fine_grained=False):
    """Apply activation checkpointing following the same pattern as
    ``apply_fsdp_checkpointing`` but aware of MOVA's block hierarchy.

    FusedMOVABlock contains DiTBlock children (video_block, audio_block).
    The standard ``apply_fsdp_checkpointing`` with ``(FusedMOVABlock, DiTBlock)``
    would wrap both the outer fused block AND its inner DiTBlocks, causing
    nested (double) checkpointing.

    This version skips any DiTBlock that carries an ``_inside_fused_block``
    marker (set by FusedMOVABlock.__init__ and MOVABridge.__init__ for
    video_dit_2 override blocks), so only standalone DiTBlocks in
    ``remaining_video_blocks`` and ``video_dit_2`` remaining layers are
    individually checkpointed.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
        CheckpointImpl,
    )
    from functools import partial
    from hymm.models.modules.mova import FusedMOVABlock
    from hymm.models.modules.wan_video_dit import DiTBlock

    non_reentrant_wrapper = partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    print("--> applying MOVA-specific FSDP activation checkpointing...")
    block_idx = 0
    cut_off = 1 / 2
    p = eval(p) if isinstance(p, str) else p

    def selective_checkpointing(submodule):
        nonlocal block_idx, cut_off
        should_ckpt = False
        if isinstance(submodule, FusedMOVABlock):
            should_ckpt = True
        elif isinstance(submodule, DiTBlock) and not getattr(
            submodule, '_inside_fused_block', False
        ):
            should_ckpt = True

        if should_ckpt:
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                # A FusedMOVABlock that gets the OUTER block-level checkpoint may
                # ALSO opt into fine-grained (nested) inner checkpointing: each of
                # its submodules (a2v / v2a / video / audio) then recomputes
                # independently during the outer recompute, dropping the recompute
                # peak from the SUM of submodule activations to the MAX. Numerically
                # identical (recompute-schedule only). Gated by `fine_grained`;
                # when False, behavior is exactly the original whole-block
                # checkpointing. Set on the raw module BEFORE it is wrapped below.
                if fine_grained and isinstance(submodule, FusedMOVABlock):
                    submodule.fine_grained_gc = True
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main(args):
    # ============================= Setup ==============================
    torch.backends.cuda.matmul.allow_tf32 = True
    warnings.filterwarnings("ignore", message=".*an autograd kernel was not registered to the Autograd key.*")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # device_id is bound inside init_distributed so the default NCCL communicator
    # is created eagerly rather than on the first collective. The startup timeout
    # is deliberately generous -- ranks legitimately wait a long time in the
    # staggered-load barrier below while rank 0 streams the checkpoint off CephFS.
    # It is tightened to --nccl-timeout just before the training loop.
    local_rank = init_distributed(
        timeout_seconds=int(getattr(args, "init_timeout", 5400))
    )
    device = torch.cuda.current_device()

    if args.fsdp_sharding_strategy == 'full':
        dp_replicate = 1
    elif args.fsdp_sharding_strategy == 'none':
        dp_replicate = world_size
    else:
        assert args.fsdp_sharding_strategy == 'hybrid'
        # One FSDP replica per node; parameters are sharded within a node.
        dp_replicate = -1

    # Builds world_mesh [dp, sp] + fsdp_mesh [dp_replicate, fsdp_shard], publishes
    # them to nccl_info, and warms up every process group so no communicator is
    # ever created lazily mid-step.
    parallel_dims = initialize_parallel_state(
        sp=args.sp_size,
        dp_replicate=dp_replicate,
        use_dynamic_ring_attention=args.use_dynamic_ring_attention,
    )
    dp_degree, dp_rank = parallel_dims.dp_size, parallel_dims.dp_rank

    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.output_dir is not None:
        obj_list = [args.output_dir]
        dist.broadcast_object_list(obj_list, src=0)
        args.output_dir = obj_list[0]

    logger, _ = setup_logger(args.output_dir)
    if rank <= 0:
        tb_dir = os.path.join(args.output_dir, "tb")
        tb_writer = SummaryWriter(log_dir=tb_dir)
    else:
        tb_writer = None
    logger.info(f"--> args {args}")
    logger.info(f"dp_degree, dp_rank: {dp_degree, dp_rank}")

    logger.info(f'--> Inject config: {args.inject_config}')
    t2va_args = OmegaConf.load(args.inject_config)

    args.audio_micro_batch_size = args.micro_batch_size

    if args.is_profiler:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        sort_by_keyword = "self_" + str(device) + "_time_total"

    # =========================== Build main model ===========================
    logger.info(f"--> loading model from {t2va_args.training_config.pretrained_model_name_or_path}")
    _load_dtype = torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16
    factor_kwargs = {
        'device': device,
        'dtype': _load_dtype,
    }

    # ---- Stagger model loading to avoid CephFS I/O saturation ----
    if local_rank != 0:
        dist.barrier()

    print(f"[Rank {rank}] >>> STEP 1: Loading model from pretrained...", flush=True)
    _t_load = time.time()
    model, extra_components = load_ti2va_wan_transformer(
        t2va_args,
        factor_kwargs,
        '',
        logger
    )
    print(f"[Rank {rank}] >>> STEP 2: Model loaded in {time.time()-_t_load:.1f}s, moving to device", flush=True)

    if local_rank == 0:
        dist.barrier()

    model = model.to(device)
    model.train()
    print(f"[Rank {rank}] >>> STEP 3: Model on device, train mode set", flush=True)

    # When enabled, restrict ALL sparse-attention features below to the high-noise
    # expert (video_dit); the low-noise expert (video_dit_2) keeps dense full attn.
    # Must be set BEFORE any configure_* call. Default false = sparse on both experts.
    sparse_high_noise_only = getattr(args, 'sparse_attn_high_noise_only', 'false')
    if isinstance(sparse_high_noise_only, str):
        sparse_high_noise_only = sparse_high_noise_only.lower() == 'true'
    model.set_sparse_high_noise_only(sparse_high_noise_only)
    logger.info(
        f"[Sparse Attn Scope] sparse_high_noise_only={sparse_high_noise_only} "
        f"({'high-noise expert (video_dit) only; video_dit_2 = full attn' if sparse_high_noise_only else 'both video experts'})"
    )

    enable_bsa = getattr(args, 'enable_bsa', 'false')
    if isinstance(enable_bsa, str):
        enable_bsa = enable_bsa.lower() == 'true'
    enable_bsa_v2a = getattr(args, 'enable_bsa_v2a', 'false')
    if isinstance(enable_bsa_v2a, str):
        enable_bsa_v2a = enable_bsa_v2a.lower() == 'true'

    if enable_bsa or enable_bsa_v2a:
        bsa_cdf = getattr(args, 'bsa_cdf_threshold', None)
        bsa_params = {
            'sparsity': getattr(args, 'bsa_sparsity', 0.9375),
            'cdf_threshold': bsa_cdf,
            'chunk_3d_shape_q': list(getattr(args, 'bsa_chunk_3d_shape_q', [4, 4, 4])),
            'chunk_3d_shape_k': list(getattr(args, 'bsa_chunk_3d_shape_k', [4, 4, 4])),
        } if enable_bsa else None
        bsa_v2a_cdf = getattr(args, 'bsa_v2a_cdf_threshold', None)
        bsa_params_v2a = {
            'sparsity': getattr(args, 'bsa_v2a_sparsity', 0.875),
            'cdf_threshold': bsa_v2a_cdf,
            'chunk_size_q': getattr(args, 'bsa_v2a_audio_chunk_size', 64),
            'chunk_3d_shape_k': list(getattr(args, 'bsa_v2a_chunk_3d_shape_k', [4, 4, 4])),
        } if enable_bsa_v2a else None
        model.configure_bsa(
            enable_bsa=enable_bsa, bsa_params=bsa_params,
            enable_bsa_v2a=enable_bsa_v2a, bsa_params_v2a=bsa_params_v2a,
        )
        vid_mode = "Hybrid Top-k+Top-p" if bsa_cdf else "Top-k only"
        v2a_mode = "Hybrid Top-k+Top-p" if bsa_v2a_cdf else "Top-k only"
        logger.info(
            f"BSA configured: video_self={enable_bsa} [{vid_mode}] "
            f"(sparsity={bsa_params['sparsity'] if bsa_params else 'N/A'}, cdf_threshold={bsa_cdf}), "
            f"v2a={enable_bsa_v2a} [{v2a_mode}] "
            f"(sparsity={bsa_params_v2a['sparsity'] if bsa_params_v2a else 'N/A'}, cdf_threshold={bsa_v2a_cdf})"
        )

    # --- Audio Guidance for BSA (Gate 2: Audio Spatial Concentration Gate) ---
    enable_audio_guidance = getattr(args, 'enable_audio_guidance', 'false')
    if isinstance(enable_audio_guidance, str):
        enable_audio_guidance = enable_audio_guidance.lower() == 'true'
    enable_audio_concentration_gate = getattr(args, 'enable_audio_concentration_gate', 'false')
    if isinstance(enable_audio_concentration_gate, str):
        enable_audio_concentration_gate = enable_audio_concentration_gate.lower() == 'true'

    enable_timestep_reliability_gate = getattr(args, 'enable_timestep_reliability_gate', 'false')
    if isinstance(enable_timestep_reliability_gate, str):
        enable_timestep_reliability_gate = enable_timestep_reliability_gate.lower() == 'true'
    enable_audio_weighted_pooling = getattr(args, 'enable_audio_weighted_pooling', 'false')
    if isinstance(enable_audio_weighted_pooling, str):
        enable_audio_weighted_pooling = enable_audio_weighted_pooling.lower() == 'true'
    audio_boost_gamma = float(getattr(args, 'audio_boost_gamma', 1.0))
    audio_weighted_lambda = float(getattr(args, 'audio_weighted_lambda', 1.0))
    logger.info(
        f"[Audio Guidance] Path A: enable={enable_audio_guidance}, "
        f"concentration_gate={enable_audio_concentration_gate}, "
        f"timestep_gate={enable_timestep_reliability_gate}, "
        f"boost_gamma={audio_boost_gamma} | "
        f"Path B: enable={enable_audio_weighted_pooling}, "
        f"weighted_lambda={audio_weighted_lambda}"
    )
    if enable_audio_guidance or enable_audio_weighted_pooling:
        for _sa in model._video_self_attns():
            _sa.bsa_params['audio_boost_gamma'] = audio_boost_gamma
            _sa.bsa_params['audio_weighted_lambda'] = audio_weighted_lambda
        model.configure_audio_guidance(
            enable=enable_audio_guidance,
            enable_concentration_gate=enable_audio_concentration_gate,
            enable_timestep_reliability_gate=enable_timestep_reliability_gate,
            enable_audio_weighted_pooling=enable_audio_weighted_pooling,
        )

    # --- Channel-Variance Guidance for BSA (independent from audio guidance) ---
    enable_variance_guidance = getattr(args, 'enable_variance_guidance', 'false')
    if isinstance(enable_variance_guidance, str):
        enable_variance_guidance = enable_variance_guidance.lower() == 'true'
    variance_boost_gamma = float(getattr(args, 'variance_boost_gamma', 1.0))
    logger.info(
        f"[Variance Guidance] enable={enable_variance_guidance}, "
        f"boost_gamma={variance_boost_gamma}"
    )
    if enable_variance_guidance:
        for _sa in model._video_self_attns():
            _sa.bsa_params['variance_boost_gamma'] = variance_boost_gamma
        model.configure_variance_guidance(enable=True)

    # --- Bias Correction: two independent methods (at most one enabled) ---
    enable_taylor_sa = getattr(args, 'enable_taylor_sparse_attn', 'false')
    if isinstance(enable_taylor_sa, str):
        enable_taylor_sa = enable_taylor_sa.lower() == 'true'
    enable_rectified_sa = getattr(args, 'enable_rectified_sparse_attn', 'false')
    if isinstance(enable_rectified_sa, str):
        enable_rectified_sa = enable_rectified_sa.lower() == 'true'
    if enable_taylor_sa and enable_rectified_sa:
        raise ValueError(
            "enable_taylor_sparse_attn and enable_rectified_sparse_attn are mutually "
            "exclusive bias-correction methods — enable at most one. "
            "(Previously the rectified path would silently win and Taylor be disabled.)"
        )
    logger.info(
        f"[Bias Correction] taylor_sparse_attn={enable_taylor_sa}, "
        f"rectified_sparse_attn={enable_rectified_sa}"
    )
    if enable_taylor_sa:
        _alpha = float(getattr(args, 'taylor_alpha_f', 0.5))
        for _sa in model._video_self_attns():
            _sa.bsa_params['taylor_alpha_f'] = _alpha
        model.configure_taylor_sparse_attn(enable=True)
    if enable_rectified_sa:
        model.configure_rectified_sparse_attn(enable=True)

    # --- Anisotropic Dynamic Block Shape (Section 8): two mutually-exclusive ---
    # methods on the video self-attn only. Enabling either DISABLES audio/variance
    # guidance + bias correction (fully isolated; v2a bridge BSA untouched).
    enable_ivpq = getattr(args, 'enable_ivpq_dynamic_block', 'false')
    if isinstance(enable_ivpq, str):
        enable_ivpq = enable_ivpq.lower() == 'true'
    enable_penalty = getattr(args, 'enable_penalty_dynamic_block', 'false')
    if isinstance(enable_penalty, str):
        enable_penalty = enable_penalty.lower() == 'true'
    if enable_ivpq and enable_penalty:
        raise ValueError(
            "enable_ivpq_dynamic_block and enable_penalty_dynamic_block are mutually "
            "exclusive — enable at most one."
        )
    if enable_ivpq or enable_penalty:
        if not enable_bsa:
            raise ValueError("Dynamic block shape requires enable_bsa=true (video self-attn BSA).")
        if (enable_audio_guidance or enable_audio_weighted_pooling
                or enable_variance_guidance or enable_taylor_sa or enable_rectified_sa):
            raise ValueError(
                "Dynamic block shape (IVPQ/Penalty) is mutually exclusive with audio guidance, "
                "variance guidance and taylor/rectified bias correction. Disable those triggers."
            )
        _la = float(getattr(args, 'dynamic_block_lambda_a', 0.5))
        _t128 = float(getattr(args, 'dynamic_block_tau_128', 0.15))
        _l128 = float(getattr(args, 'dynamic_block_lambda_128', 1.0))
        for _sa in model._video_self_attns():
            _sa.bsa_params['dynamic_block_lambda_a'] = _la
            _sa.bsa_params['dynamic_block_tau_128'] = _t128
            _sa.bsa_params['dynamic_block_lambda_128'] = _l128
        if enable_ivpq:
            model.configure_ivpq_dynamic_block(enable=True)
        else:
            model.configure_penalty_dynamic_block(enable=True)
        logger.info(
            f"[Dynamic Block Shape] method={'IVPQ' if enable_ivpq else 'Penalty'}, "
            f"lambda_a={_la}, tau_128={_t128}, lambda_128={_l128}"
        )

        # --- Layer-Adaptive Dynamic Block (Section 8.2): shallow fixed / deep dynamic,
        #     shallow-audio cached → deep layers. Default off. Must run AFTER the
        #     ivpq/penalty configure above (it only takes effect with a dynamic method).
        enable_la_dyn = getattr(args, 'enable_layer_adaptive_dynamic_block', 'false')
        if isinstance(enable_la_dyn, str):
            enable_la_dyn = enable_la_dyn.lower() == 'true'
        if enable_la_dyn:
            _split, _cache_src = model.configure_layer_adaptive_dynamic_block(enable=True)
            logger.info(
                f"[Layer-Adaptive Dynamic Block] enabled: shallow(<{_split}) fixed / "
                f"deep(>={_split}) dynamic; audio cache-source layer={_cache_src}"
            )

    # ---- Trainable parameter selection -------------------------------------------
    train_full_model = str(getattr(args, 'train_full_model', 'false')).lower() == 'true'
    if train_full_model:
        model.requires_grad_(True)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = total_params
        logger.info(
            f"Full model training: {trainable_params/1e6:.1f}M trainable / "
            f"{total_params/1e6:.1f}M total (100.0%)"
        )
    else:
        from hymm.models.modules.wan_video_dit import SelfAttention, CrossAttention
        from hymm.models.modules.interactionv2 import ConditionalCrossAttentionBlock
        model.requires_grad_(False)
        attn_module_types = (SelfAttention, CrossAttention, ConditionalCrossAttentionBlock)
        for module in model.modules():
            if isinstance(module, attn_module_types):
                for p in module.parameters():
                    p.requires_grad = True
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Attention-only training: {trainable_params/1e6:.1f}M trainable / "
            f"{total_params/1e6:.1f}M total ({trainable_params/total_params*100:.1f}%)"
        )

    args.local_seed = args.global_seed + rank // args.sp_size
    set_seed(args.local_seed)
    random.seed(args.local_seed)
    print(f"dp_rank: {dp_rank}, rank: {rank},  local_seed: {args.local_seed}")

    print_memory_usage('Model load')

    # ============================= Build extra models =========================
    # Extract components from MOVABridge's extra_components (all loaded in load_ti2va_wan_transformer)
    mova_text_encoder = extra_components["text_encoder"]
    mova_tokenizer = extra_components["tokenizer"]
    mova_audio_vae = extra_components["audio_vae"]
    mova_video_vae = extra_components["video_vae"]
    mova_scheduler = extra_components["scheduler"]
    boundary_ratio = extra_components.get("boundary_ratio", 0.9)
    audio_vae_type = extra_components.get("audio_vae_type", "dac")

    # Keep frozen encoders on CPU — they are moved to GPU on-demand in the
    # training loop and offloaded back to CPU after encoding.  Loading them
    # to GPU here would create a ~49 GB peak (full model + frozen encoders)
    # before FSDP can shard, causing severe CUDA memory fragmentation.
    mova_text_encoder = mova_text_encoder.requires_grad_(False).eval()
    mova_audio_vae = mova_audio_vae.requires_grad_(False).eval()
    mova_video_vae = mova_video_vae.requires_grad_(False).eval()
    video_vae_config = mova_video_vae.config

    num_train_ts = mova_scheduler.config.num_train_timesteps
    mova_scheduler.set_timesteps(num_train_ts, training=True)

    _visual_shift = getattr(t2va_args.training_config, 'visual_shift', None)
    _audio_shift = getattr(t2va_args.training_config, 'audio_shift', None)
    if _visual_shift is not None and _audio_shift is not None:
        mova_scheduler.set_pair_postprocess_by_name(
            "dual_sigma_shift",
            visual_shift=float(_visual_shift),
            audio_shift=float(_audio_shift),
        )
        logger.info(f"--> Scheduler set_timesteps({num_train_ts}, training=True) — "
                    f"timesteps len={len(mova_scheduler.timesteps)}, "
                    f"dual_sigma_shift(visual={_visual_shift}, audio={_audio_shift})")
    else:
        logger.info(f"--> Scheduler set_timesteps({num_train_ts}, training=True) — "
                    f"timesteps len={len(mova_scheduler.timesteps)}, "
                    f"no dual_sigma_shift (same shift for video & audio)")

    logger.info(f"--> Loaded MOVA text_encoder, audio_vae, video_vae, scheduler from extra_components")

    # Validate audio sampling rate: dataloader audio_sr MUST match DAC's expected rate.
    # Mismatch silently corrupts audio latents — DAC's preprocess only asserts its own
    # metadata, not the actual data rate.
    dac_sr = getattr(mova_audio_vae, 'sample_rate', None)
    cfg_sr = getattr(t2va_args.dataloader_config, 'audio_sr', None)
    if dac_sr is not None and cfg_sr is not None and int(dac_sr) != int(cfg_sr):
        raise ValueError(
            f"audio_sr mismatch: dataloader_config.audio_sr={cfg_sr} but "
            f"DAC audio_vae.sample_rate={dac_sr}. The dataloader resamples "
            f"audio to {cfg_sr} Hz, but DAC expects {dac_sr} Hz. "
            f"Fix: set dataloader_config.audio_sr to {dac_sr} in the YAML."
        )
    logger.info(f"--> DAC audio_vae sample_rate: {dac_sr}, dataloader audio_sr: {cfg_sr}")

    _spr = getattr(video_vae_config, "spatial_compression_ratio", None) or video_vae_config.scale_factor_spatial
    logger.info(f"--> video_vae spatial_compression_ratio: {_spr}")
    logger.info(f"--> video_vae z_dim: {video_vae_config.z_dim}")

    print_memory_usage('Extra model load')

    logger.info(
        f"  Total training parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6} M"
    )

    # ============================== resume ==============================
    init_steps = 0
    if args.resume and args.resume != "None":
        if not os.path.exists(args.resume):
            raise ValueError("Invalid resume path: %s" % args.resume)

        logger.info(f"loading the resumed checkpoint: {args.resume}")
        (model, _scalar_states_list) = resume_wan_training(
            model,
            args.resume,
        )
        logger.info(f"loading the resumed checkpoint successfully: {args.resume}")

        # Resume ScalarStates
        scalar_states = ScalarStates.from_pretrained(
            _scalar_states_list,
            rank=rank,
            world_size=world_size,
            default_rank0_ss=False,
        )
        print(f'--> resume training scalar states {scalar_states}')
        init_steps = scalar_states.train_steps # 续训学习率参数

        # init_scalar_state = {
        #     'lr': args.learning_rate,
        # }
        # scalar_states = ScalarStates(**init_scalar_state)
    else:
        init_scalar_state = {
            'lr': args.learning_rate,
        }
        scalar_states = ScalarStates(**init_scalar_state)

    # ============================= Build FSDP =========================
    print(f"[Rank {rank}] >>> STEP 4: Pre-FSDP barrier...", flush=True)
    # dist.barrier()
    print(f"[Rank {rank}] >>> STEP 5: Pre-FSDP barrier passed", flush=True)

    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs_v2(
        model,
        parallel_dims,
        args.fsdp_sharding_strategy,
        cpu_offload=args.use_cpu_offload,
        master_weight_type=args.master_weight_type,
    )

    print_memory_usage('Before FSDP')

    if args.gradient_checkpointing:
        _fine_grained_gc = getattr(args, 'fine_grained_gc', 'false')
        if isinstance(_fine_grained_gc, str):
            _fine_grained_gc = _fine_grained_gc.lower() == 'true'
        _apply_mova_fsdp_checkpointing(
            model, args.selective_checkpointing, fine_grained=_fine_grained_gc
        )
        logger.info(
            f"Activation checkpointing: whole-block"
            f"{' + fine-grained (nested submodule) ' if _fine_grained_gc else ' '}"
            f"(fine_grained_gc={_fine_grained_gc})"
        )

    print(f"[Rank {rank}] >>> STEP 6: Applying FSDP2...", flush=True)
    apply_fsdp2(
        model=model,
        dp_mesh=fsdp_kwargs["device_mesh"],
        param_dtype=torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=fsdp_kwargs["cpu_offload"],
        reshard_after_forward_policy="default",
    )
    print(f"[Rank {rank}] >>> STEP 7: FSDP2 applied", flush=True)
    torch.cuda.empty_cache()
    logger.info(f"--> model loaded")

    print_memory_usage('After FSDP')

    if args.fsdp_sharding_strategy == 'hybrid':
        replicate_group = parallel_dims.fsdp_mesh['dp_replicate'].get_group()
        group_rank0_rank = dist.get_process_group_ranks(replicate_group)[0]
        for name, param in model.named_parameters():
            if isinstance(param, DTensor):
                dist.broadcast(
                    param._local_tensor,
                    src=group_rank0_rank,
                    group=replicate_group,
                )
            else:
                dist.broadcast(
                    param,
                    src=group_rank0_rank,
                    group=replicate_group,
                )

    # ============================= Optimizer =========================
    if args.optimizer != 'adamw':
        raise ValueError(f"Unsupported optimizer '{args.optimizer}'. Only 'adamw' is supported.")
    params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    # Toggle (from yaml -> shell -> argparse): persist/restore the sharded AdamW
    # optimizer state via DCP. Default on. Reshards across different node counts.
    save_optim_state = str(getattr(args, "save_optimizer_state", "true")).lower() == "true"

    if args.resume and args.resume != "None":
        # Restore AdamW moments/step (sharded, reshardable). MUST run AFTER FSDP
        # wrap + optimizer creation. No-op (cold optimizer) if disabled or if the
        # checkpoint has no optimizer_state subdir.
        if save_optim_state:
            load_optimizer_state(model, optimizer, args.resume)
        for p in optimizer.param_groups:
            p["initial_lr"] = args.learning_rate

    logger.info(f"optimizer: {optimizer}")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        # last_epoch=init_steps - 1,
        last_epoch=-1,
    )

    # ============================== Load dataset ==============================
    logger.info(f"[Rank {rank}] Creating dataloader...")

    print(f"[Rank {rank}] >>> STEP 8: Creating dataloader...", flush=True)
    videoaudio_dataset, videoaudio_sampler, videoaudio_loader = _get_t2va_wan_dataloader(
        args, t2va_args.dataloader_config, dp_degree, dp_rank, logger
    )
    print(f"[Rank {rank}] >>> STEP 9: Dataloader created", flush=True)

    # Full-epoch count on purpose: the resume offset is what we are about to
    # derive from it, so it must not already have that offset subtracted.
    steps_per_epoch = videoaudio_sampler.batches_per_epoch()
    resume_epoch, resume_batch = divmod(int(scalar_states.train_steps), max(steps_per_epoch, 1))
    loader = EpochCyclingBatchIterator(
        dataloader=videoaudio_loader,
        batch_sampler=videoaudio_sampler,
        dataset=videoaudio_dataset,
        start_epoch=resume_epoch,
        start_batch=resume_batch,
        logger=logger,
    )
    data_keys = ['t2va']
    for key in data_keys:
        scalar_states.epoch.setdefault(key, resume_epoch)
        scalar_states.consumed_samples_per_dp.setdefault(key, 0)

    # ============================== Print Key Info ==============================
    micro_bs = t2va_args.dataloader_config.video_micro_batch_size
    if not isinstance(micro_bs, int):
        micro_bs = int(micro_bs[0])
    video_audio_total_batch_size = micro_bs * dp_degree
    video_audio_num = videoaudio_dataset.total_length

    params_count = _mova_params_count(model)
    print_training_configuration(
        args, logger, params_count, model, world_size, local_rank, rank,
        dp_degree, dp_rank,
        video_audio_total_batch_size=video_audio_total_batch_size,
        video_audio_num=video_audio_num,
        video_audio_dataset=videoaudio_dataset,
        video_audio_sampler=videoaudio_sampler,
        scalar_states=scalar_states,
        t2va_args=t2va_args,
    )

    torch.cuda.empty_cache()

    # Determine text max length from config
    text_max_length = getattr(t2va_args.dataloader_config, 'text_len', 512)
    video_loss_weight = getattr(t2va_args.training_config, 'video_loss_weight', 0.85)
    audio_loss_weight = getattr(t2va_args.training_config, 'audio_loss_weight', 0.15)
    video_fps = getattr(t2va_args.dataloader_config, 'video_fps', 24.0)
    num_train_timesteps = mova_scheduler.config.num_train_timesteps

    # DiT MoE high/low-noise expert routing trigger (from --swap-dit-moe-high-noise).
    # Default False keeps current order (matching inference):
    #   even step = high-noise -> video_dit,  odd step = low-noise -> video_dit_2.
    # True swaps it (breaks inference alignment):
    #   even step = high-noise -> video_dit_2,  odd step = low-noise -> video_dit.
    swap_dit_moe_high_noise = str(getattr(args, 'swap_dit_moe_high_noise', 'false')).lower() == 'true'
    if swap_dit_moe_high_noise:
        _moe_order = "high-noise -> video_dit_2 | low-noise -> video_dit  (SWAPPED, misaligned with inference)"
    else:
        _moe_order = "high-noise -> video_dit | low-noise -> video_dit_2  (default, aligned with inference)"
    logger.info(f"[DiT MoE] training order: {_moe_order} "
                f"(swap_dit_moe_high_noise={swap_dit_moe_high_noise})")

    # Per-step CPU<->GPU shuttling of the frozen encoders (UMT5 + DAC + Wan VAE)
    # is the single largest fixed cost in a step. Keeping them resident trades
    # roughly 12 GB of VRAM for that traffic; both knobs default to the original
    # behaviour so memory headroom is unchanged unless you opt in.
    offload_frozen_encoders = str(
        getattr(args, 'offload_frozen_encoders', 'true')
    ).lower() == 'true'
    empty_cache_interval = int(getattr(args, 'empty_cache_interval', 1))
    logger.info(
        f"[Frozen encoders] offload_per_step={offload_frozen_encoders}, "
        f"empty_cache_interval={empty_cache_interval}"
    )
    if not offload_frozen_encoders:
        mova_text_encoder.to(device)
        mova_audio_vae.to(device)
        mova_video_vae.to(device)

    # ============================= Start training =============================
    # Startup (checkpoint load, FSDP wrap, dataset scan) is done, so swap the
    # generous startup deadline for a tight one. From here a collective that
    # blocks for minutes means ranks have desynced, and we want the watchdog to
    # say so with a stack trace rather than let the job idle.
    set_collective_timeout(int(getattr(args, "nccl_timeout", 1800)), logger=logger)

    print(f"[Rank {rank}] >>> STEP 10: Entering training loop", flush=True)
    global_step = 0
    if True:
        cycle_states = CycleStates()
        data_iter = iter(loader)
        optimizer.zero_grad()
        # One barrier here, before the first step, so every rank starts the timer
        # from the same point. There are none inside the step itself.
        print(f"[Rank {rank}] >>> STEP 11: Pre-training barrier...", flush=True)
        global_start_time = sync_cuda_time(sync=True, barrier=True)
        print(f"[Rank {rank}] >>> STEP 12: Training started!", flush=True)

        if getattr(args, 'dry_run', False) and rank == 0:
            logger.info("=" * 80)
            logger.info("DRY RUN MODE ENABLED")
            logger.info("Will only iterate through data without training.")
            logger.info("=" * 80)

        # The batch iterator cycles epochs forever, so max_train_steps is the only
        # stopping condition. It is a plain counter that advances identically on
        # every rank, which keeps the exit collective-safe.
        while scalar_states.train_steps < args.max_train_steps:
            start_time = sync_cuda_time()
            # EpochCyclingBatchIterator never raises StopIteration: it rolls over
            # to the next epoch itself. Every DP rank sees the same number of
            # batches per epoch, so the rollover happens on the same step
            # everywhere and no rank is left alone inside a collective.
            batch = next(data_iter)

            if getattr(args, 'dry_run', False):
                data_type = 't2va'
                scalar_states.add(train_steps=1, update_steps=1)

                if scalar_states.update_steps % args.log_interval == 0:
                    end_time = sync_cuda_time(sync=True)
                    sec_per_step = (end_time - global_start_time) / max(cycle_states.log_steps, 1)
                    if rank == 0:
                        logger.info(
                            f"[Dry Run] Progress: {scalar_states.update_steps}/{args.max_train_steps} | "
                            f"Step time: {sec_per_step:.2f}s | "
                            f"latent {tuple(batch['video_latents'].shape)} | "
                            f"Data type: {data_type}"
                        )

                    cycle_states.reset_epoch_based_states()
                    global_start_time = end_time

                cycle_states.add(log_steps=1)
                continue

            dataloader_time_ = sync_cuda_time()

            # -------------------- Prepare model inputs --------------------
            # Move frozen encoders back to GPU for encoding
            if offload_frozen_encoders:
                mova_text_encoder.to(device)
                mova_audio_vae.to(device)
                mova_video_vae.to(device)

            video_latents = batch["video_latents"].to(device=device, dtype=torch.bfloat16)
            ref_image = batch["ref_image"].to(device=device, dtype=torch.bfloat16)
            waveform = batch["waveform"].to(device=device, dtype=torch.float32)
            video_caption = batch["video_caption"]
            audio_caption = batch["audio_caption"]
            current_videoid = batch.get("videoid", None)

            # Belt and braces: the dataset seeds every random choice from the row
            # index, so ranks in one SP group already build identical batches. The
            # broadcast guarantees it, and runs after the device transfer so it is
            # a single NVLink hop rather than a CPU round trip.
            video_latents, ref_image, waveform = sync_data_for_sp(
                [video_latents, ref_image, waveform], parallel_dims=parallel_dims
            )
            video_caption, audio_caption = sync_data_for_sp(
                [video_caption, audio_caption], parallel_dims=parallel_dims,
                force_object=True,
            )

            if video_latents.dim() == 4:
                video_latents = video_latents.unsqueeze(0)

            # NOTE: pre-cached latents from wan_vae_latent_extraction.py are
            # already normalized (normalize_wan_latents applied before saving).
            # Do NOT normalize again here — double normalization would corrupt
            # the latent distribution.

            # Encode text (video and audio captions separately, matching MOVA inference)
            context = _get_t5_prompt_embeds(
                mova_text_encoder, mova_tokenizer, video_caption, device, max_length=text_max_length
            )
            audio_context = _get_t5_prompt_embeds(
                mova_text_encoder, mova_tokenizer, audio_caption, device, max_length=text_max_length
            )

            # Encode audio waveform through DAC
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float32):
                    if audio_vae_type == "dac":
                        audio_sr = getattr(mova_audio_vae, 'sample_rate', 48000)
                        x_pad = mova_audio_vae.preprocess(waveform, sample_rate=audio_sr)
                        z, codes, latents_out, commitment_loss, codebook_loss = mova_audio_vae.encode(x_pad)
                        audio_latents = z.mode()
                    else:
                        audio_latents = mova_audio_vae.encode(waveform).latent_dist.sample()
                audio_latents = audio_latents.to(dtype=torch.bfloat16)

            # Encode first frame for i2v conditioning
            y = _encode_first_frame(
                ref_image, video_latents, mova_video_vae, video_vae_config, device, torch.bfloat16
            )

            # Offload frozen encoders to CPU to free GPU memory for FSDP forward/backward
            if offload_frozen_encoders:
                mova_text_encoder.to("cpu")
                mova_audio_vae.to("cpu")
                mova_video_vae.to("cpu")
            # empty_cache() is a device-wide sync that also drops the allocator's
            # pooled blocks, so at interval 1 it costs a full stall plus a round of
            # cudaMalloc every step. With expandable_segments (set in the launch
            # script) the pool already reuses the freed encoder memory, so raising
            # the interval is usually free throughput.
            if empty_cache_interval > 0 and (global_step % empty_cache_interval == 0):
                torch.cuda.empty_cache()

            B = video_latents.shape[0]
            data_type = 't2va'
            mask_type = 'i2v'

            # Compute n_tokens (visual sequence length after patchify)
            T_lat = video_latents.shape[2]
            H_lat = video_latents.shape[3]
            W_lat = video_latents.shape[4]
            n_tokens = T_lat * (H_lat // 2) * (W_lat // 2)

            # scalar_states advances identically on every rank (the skip flags
            # below are all-reduced), so it needs no cross-rank sync.
            micro_samples = video_latents.shape[0]
            # Every rank of an SP group holds the SAME samples, so divide by
            # sp_size to keep the world-wide SUM below equal to the real sample
            # count rather than counting each clip sp_size times.
            cur_batch_size = micro_samples / parallel_dims.sp
            data_time_ = sync_cuda_time()

            # -------------------- Forward pass --------------------
            with torch.autocast("cuda", torch.bfloat16):
                if args.is_profiler and global_step >= 20:
                    with profile(activities=activities) as prof:
                        # Sample timesteps with boundary alternation (matching mova_train.py)
                        visual_timestep, audio_timestep = _sample_timestep_pair(
                            mova_scheduler, device, boundary_ratio, scalar_states.train_steps
                        )
                        video_noise = torch.randn_like(video_latents)
                        audio_noise = torch.randn_like(audio_latents)
                        noisy_video = mova_scheduler.add_noise(video_latents, video_noise, visual_timestep).to(device)
                        noisy_audio = mova_scheduler.add_noise(audio_latents, audio_noise, audio_timestep).to(device)
                        _high_noise_dit2 = (scalar_states.train_steps % 2 == 1)
                        if swap_dit_moe_high_noise:
                            _high_noise_dit2 = not _high_noise_dit2
                        use_video_dit_2 = _high_noise_dit2 and (model.video_dit_2 is not None)
                        visual_input = torch.cat([noisy_video, y], dim=1)
                        video_pred, audio_pred = model(
                            visual_latents=visual_input,
                            audio_latents=noisy_audio,
                            context=context,
                            audio_context=audio_context,
                            timestep=visual_timestep.unsqueeze(0) if visual_timestep.dim() == 0 else visual_timestep[:1],
                            audio_timestep=audio_timestep.unsqueeze(0) if audio_timestep.dim() == 0 else audio_timestep[:1],
                            video_fps=video_fps,
                            num_train_timesteps=num_train_timesteps,
                            use_video_dit_2=use_video_dit_2,
                        )
                        video_target = video_noise - video_latents
                        audio_target = audio_noise - audio_latents
                        v_loss = F.mse_loss(video_pred.to(video_target.dtype), video_target)
                        a_loss = F.mse_loss(audio_pred.to(audio_target.dtype), audio_target)
                        diffusion_loss = video_loss_weight * v_loss + audio_loss_weight * a_loss
                        video_loss = v_loss
                        audio_loss = a_loss
                    prof.export_chrome_trace(f"{args.output_dir}/trace-{rank}-step{global_step}.json")
                    if global_step > 22:
                        sys.exit(0)
                else:
                    # Align the RNGs so every SP rank draws the same noise. The
                    # seed is a pure function of (global_seed, dp_rank, step), so
                    # this costs no collective and no device sync while still
                    # giving distinct DP groups distinct noise.
                    sync_random_states(
                        step=scalar_states.train_steps,
                        base_seed=args.global_seed,
                        dp_rank=dp_rank,
                    )

                    # Sample timesteps with boundary alternation (matching mova_train.py)
                    visual_timestep, audio_timestep = _sample_timestep_pair(
                        mova_scheduler, device, boundary_ratio, scalar_states.train_steps
                    )

                    # Add noise (Flow Matching)
                    video_noise = torch.randn_like(video_latents)
                    audio_noise = torch.randn_like(audio_latents)

                    noisy_video = mova_scheduler.add_noise(video_latents, video_noise, visual_timestep).to(device)
                    noisy_audio = mova_scheduler.add_noise(audio_latents, audio_noise, audio_timestep).to(device)

                    # Select video_dit or video_dit_2 based on step parity.
                    # Default: even step (high-noise) -> video_dit; odd step (low-noise) -> video_dit_2.
                    # swap_dit_moe_high_noise flips this assignment.
                    _high_noise_dit2 = (scalar_states.train_steps % 2 == 1)
                    if swap_dit_moe_high_noise:
                        _high_noise_dit2 = not _high_noise_dit2
                    use_video_dit_2 = _high_noise_dit2 and (model.video_dit_2 is not None)

                    # Concatenate noisy_video with i2v conditioning y
                    visual_input = torch.cat([noisy_video, y], dim=1)

                    # Forward through MOVABridge
                    video_pred, audio_pred = model(
                        visual_latents=visual_input,
                        audio_latents=noisy_audio,
                        context=context,
                        audio_context=audio_context,
                        timestep=visual_timestep.unsqueeze(0) if visual_timestep.dim() == 0 else visual_timestep[:1],
                        audio_timestep=audio_timestep.unsqueeze(0) if audio_timestep.dim() == 0 else audio_timestep[:1],
                        video_fps=video_fps,
                        num_train_timesteps=num_train_timesteps,
                        use_video_dit_2=use_video_dit_2,
                    )

                    # Flow Matching target: v = noise - sample
                    video_target = video_noise - video_latents
                    audio_target = audio_noise - audio_latents

                    v_loss = F.mse_loss(video_pred.to(video_target.dtype), video_target)
                    a_loss = F.mse_loss(audio_pred.to(audio_target.dtype), audio_target)

                    diffusion_loss = video_loss_weight * v_loss + audio_loss_weight * a_loss
                    video_loss = v_loss
                    audio_loss = a_loss

            loss = diffusion_loss

            loss_threshold = getattr(args, 'loss_spike_threshold', 5.0)

            # Spike detection needs (a) this rank's loss for the log line and
            # (b) a cluster-wide OR of the skip flag so FSDP parameters stay
            # consistent. Both are packed into one collective and read back with
            # a single device sync instead of the four `.item()` calls this used
            # to cost per step.
            _loss_det = loss.detach().float().reshape(())
            _local_spike = (~torch.isfinite(_loss_det)) | (_loss_det > loss_threshold)
            _probe = torch.stack([_loss_det, _local_spike.float()])
            _reduced = _probe.clone()
            dist.all_reduce(_reduced, op=dist.ReduceOp.MAX)
            current_loss, _local_spike_f, _max_loss, _spike_f = torch.cat(
                [_probe, _reduced]
            ).tolist()
            skip_this_step = _spike_f > 0.5

            # Reuses the values already pulled off the device above, so the
            # diagnostics cost no extra syncs on the happy path.
            if not math.isfinite(current_loss):
                print(f"Rank {rank}: NaN/Inf loss detected at step {scalar_states.train_steps}. "
                      f"Loss: {current_loss}, Data type: {data_type}, Batch size: {cur_batch_size}, "
                      f"Mask type: {mask_type}")
            elif current_loss > loss_threshold:
                print(f"Rank {rank}: High loss detected at step {scalar_states.train_steps}. "
                      f"Loss: {current_loss:.6f} > {loss_threshold}, Data type: {data_type}, "
                      f"Batch size: {cur_batch_size} "
                      f"visual_t: {visual_timestep.item():.1f} audio_t: {audio_timestep.item():.1f} "
                      f"Mask type: {mask_type}, Diffusion loss: {diffusion_loss.item():.6f}")
                print(f"  video_loss: {video_loss.item():.6f}, audio_loss: {audio_loss.item():.6f}")
                print(f"  Video Latent stats - Mean: {video_latents.mean():.6f}, Std: {video_latents.std():.6f}, "
                      f"Max: {video_latents.max():.6f}, Min: {video_latents.min():.6f}")
                print(f"  Audio Latent stats - Mean: {audio_latents.mean():.6f}, Std: {audio_latents.std():.6f}, "
                      f"Max: {audio_latents.max():.6f}, Min: {audio_latents.min():.6f}")

            forward_time_ = sync_cuda_time()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            scalar_states.add(train_steps=1)

            data_dtype = "t2va"
            scalar_states.consumed_samples_per_dp[data_dtype] += micro_samples
            scalar_states.epoch[data_dtype] = loader.epoch
            is_update_step = scalar_states.train_steps % args.gradient_accumulation_steps == 0

            if is_update_step:
                loss.backward()

                if skip_this_step:
                    # Spike detected across cluster — discard gradients, freeze params
                    optimizer.zero_grad()
                    grad_norm = 0.0
                    logger.info(
                        f"[Loss Spike Skip] Rank {rank} | train_step "
                        f"{scalar_states.train_steps} | "
                        f"Local loss: {current_loss:.6f} | "
                        f"threshold: {loss_threshold} | "
                        f"videoid: {current_videoid} | "
                        f"latent_shape: {video_latents.shape} | "
                        f"video_caption: {video_caption[:80] if isinstance(video_caption, str) else video_caption}"
                    )
                else:
                    grad_norm_ = nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                        foreach=True
                    )
                    # Under FSDP2 the gradients are DTensors, so the norm comes back
                    # as a DTensor that is still Partial over the shard mesh: it holds
                    # this rank's contribution, not the global norm. It has to be
                    # materialised before anything else touches it -- comparing it
                    # would read one shard's value, and c10d cannot take a tensor
                    # subclass at all, so the all_reduce below would fail outright.
                    if isinstance(grad_norm_, DTensor):
                        grad_norm_ = grad_norm_.full_tensor()
                    grad_norm_threshold = getattr(args, 'grad_norm_spike_threshold', 60.0)

                    # Same fused pattern as the loss probe: one collective, one
                    # device sync, and a skip decision every rank agrees on.
                    _gn = grad_norm_.detach().float().reshape(())
                    _gn_spike = (~torch.isfinite(_gn)) | (_gn > grad_norm_threshold)
                    _gprobe = torch.stack([_gn, _gn_spike.float()])
                    _greduced = _gprobe.clone()
                    dist.all_reduce(_greduced, op=dist.ReduceOp.MAX)
                    grad_norm, _, _max_grad_norm, _gspike_f = torch.cat(
                        [_gprobe, _greduced]
                    ).tolist()
                    skip_by_grad = _gspike_f > 0.5

                    if not math.isfinite(grad_norm):
                        print(f"Rank {rank}: NaN/Inf gradient norm detected at step {scalar_states.train_steps}. "
                              f"Grad norm: {grad_norm}, Loss: {current_loss:.6f}, Data type: {data_type}, "
                              f"Latent shape: {video_latents.shape}.")

                    if skip_by_grad:
                        optimizer.zero_grad()
                        logger.info(
                            f"[Grad Spike Skip] Rank {rank} | train_step "
                            f"{scalar_states.train_steps} | "
                            f"Local grad_norm: {grad_norm:.4f} | "
                            f"threshold: {grad_norm_threshold} | "
                            f"videoid: {current_videoid} | "
                            f"latent_shape: {video_latents.shape} | "
                            f"loss: {current_loss:.6f} | "
                            f"video_loss: {video_loss.item():.6f} | "
                            f"audio_loss: {audio_loss.item():.6f} | "
                            f"video_caption: {video_caption[:80] if isinstance(video_caption, str) else video_caption}"
                        )
                    else:
                        optimizer.step()

                    lr_scheduler.step()
                    optimizer.zero_grad()
                    scalar_states.add(update_steps=1, current_run_update_steps=1)
                    scalar_states.lr = optimizer.param_groups[0]["lr"]
            else:
                loss.backward()
            backward_time_ = sync_cuda_time()

            cycle_states.add(log_steps=1, running_loss=loss.item())
            cycle_states.running_samples[data_dtype] += cur_batch_size
            cycle_states.running_tokens[data_dtype] += cur_batch_size * n_tokens
            cycle_states.add_mask_type_samples(mask_type, cur_batch_size)
            cycle_states.add_mask_type_loss(mask_type, loss.item())

            scalar_states.consumed_samples_by_mask_type_per_dp[mask_type] += cur_batch_size

            if is_update_step and scalar_states.update_steps % args.log_interval == 0:
                # Mask types are static for this task, so the per-step
                # all_gather_object that used to discover them is gone; and every
                # scalar below travels in ONE all_reduce with ONE device sync,
                # instead of ~10 separate all_reduce + .item() round trips.
                # The list is a fixed constant rather than the dict's keys so the
                # reduced tensor has the same length on every rank by construction.
                all_mask_types = MASK_TYPES
                reduce_vals = [
                    cycle_states.running_loss / max(cycle_states.log_steps, 1),
                    cycle_states.running_samples[data_dtype],
                    cycle_states.running_tokens[data_dtype],
                ]
                for mt in all_mask_types:
                    reduce_vals.append(cycle_states.running_samples_by_mask_type.get(mt, 0))
                    reduce_vals.append(cycle_states.running_loss_by_mask_type.get(mt, 0.0))
                    reduce_vals.append(cycle_states.running_loss_count_by_mask_type.get(mt, 0))

                reduced = torch.tensor(reduce_vals, device=device, dtype=torch.float64)
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                reduced = reduced.tolist()

                global_loss = reduced[0] / world_size
                cum_samples = {data_dtype: reduced[1]}
                cum_tokens = {data_dtype: reduced[2]}

                global_cycle_samples_by_mask_type = {}
                global_avg_loss_by_mask_type = {}
                for i, mt in enumerate(all_mask_types):
                    samples, loss_sum, loss_count = reduced[3 + i * 3: 6 + i * 3]
                    scalar_states.consumed_samples_by_mask_type_total[mt] += samples
                    global_cycle_samples_by_mask_type[mt] = samples
                    global_avg_loss_by_mask_type[mt] = (
                        loss_sum / loss_count if loss_count > 0 else 0.0
                    )

                cum_samples_total = sum(cum_samples.values())
                cum_tokens_total = sum(cum_tokens.values())

                end_time = sync_cuda_time(sync=True)
                elapsed = max(end_time - global_start_time, 1e-6)
                samples_per_sec = cum_samples_total / elapsed
                sec_per_step = elapsed / max(cycle_states.log_steps, 1)
                steps_per_sec = cycle_states.log_steps / elapsed

                for data_key in data_keys:
                    scalar_states.consumed_samples_total[data_key] += cum_samples[data_key]
                    scalar_states.epoch_consumed_samples[data_key] += cum_samples[data_key]
                    scalar_states.consumed_tokens_total[data_key] += cum_tokens[data_key]
                scalar_states.add(
                    consumed_computations_attn=6 * params_count['attn+mlp'] * cum_tokens_total / C_SCALE,
                    consumed_computations_total=6 * params_count['total'] * cum_tokens_total / C_SCALE,
                )

                cycle_states.reset_epoch_based_states()
                global_start_time = end_time

            if rank == 0 and (is_update_step and scalar_states.update_steps % args.log_interval == 0):
                if 'global_cycle_samples_by_mask_type' not in locals():
                    global_cycle_samples_by_mask_type = {}
                if 'global_avg_loss_by_mask_type' not in locals():
                    global_avg_loss_by_mask_type = {}

                progress_info = {
                    "avg_loss": f"{global_loss:.8f}",
                    "step_loss": f"{loss.item():.8f}",
                    "video_loss": f"{video_loss.item():.8f}",
                    "audio_loss": f"{audio_loss.item():.8f}",
                    "diffusion_loss": f"{diffusion_loss.item():.8f}",
                    "grad_norm": f"{grad_norm:.10f}",
                    "step_time": f"{sec_per_step:.2f}s",
                    "steps_per_sec": f"{steps_per_sec:.2f}",
                    "samples_per_sec": f"{samples_per_sec:.2f}",
                    "dataloader_time": f"{dataloader_time_ - start_time:.2f}s",
                    "data_time": f"{data_time_ - dataloader_time_:.2f}s",
                    "forward_time": f"{forward_time_ - data_time_:.2f}s",
                    "backward_time": f"{backward_time_ - forward_time_:.2f}s",
                    "learning_rate": f"{lr_scheduler.get_last_lr()[0]:.8f}",
                    "current_batch_n_tokens": f"{n_tokens}",
                    "consumed_computations_attn": f"{scalar_states.consumed_computations_attn}",
                    "consumed_computations_total": f"{scalar_states.consumed_computations_total}",
                    "current_mask_type": mask_type,
                    "visual_timestep": f"{visual_timestep.item():.1f}",
                    "audio_timestep": f"{audio_timestep.item():.1f}",
                    "use_video_dit_2": f"{use_video_dit_2}",
                }
                for data_key in data_keys:
                    progress_info[f"cum_samples_{data_key}"] = f"{cum_samples[data_key]}"
                    progress_info[f"consumed_tokens_{data_key}_total"] = f"{scalar_states.consumed_tokens_total[data_key]}"
                    progress_info[f"consumed_samples_{data_key}_total"] = f"{scalar_states.consumed_samples_total[data_key]}"
                    progress_info[f"consumed_samples_{data_key}_epoch"] = f"{scalar_states.epoch_consumed_samples[data_key]}"
                    progress_info[f"consumed_samples_{data_key}_per_dp"] = f"{scalar_states.consumed_samples_per_dp[data_key]}"

                mask_type_stats = dict(scalar_states.consumed_samples_by_mask_type_total)
                for mt, count in mask_type_stats.items():
                    progress_info[f"consumed_samples_{mt}_total"] = f"{count}"
                for mt, count in global_cycle_samples_by_mask_type.items():
                    progress_info[f"cycle_samples_{mt}"] = f"{count}"
                for mt, avg_loss_val in global_avg_loss_by_mask_type.items():
                    progress_info[f"cycle_avg_loss_{mt}"] = f"{avg_loss_val:.8f}"

                logger.info(f"Progress: {scalar_states.update_steps}/{args.max_train_steps} | Details: {progress_info}")

                # Log training metrics
                tb_writer.add_scalar("Train/Steps/train_loss", global_loss, scalar_states.update_steps)
                tb_writer.add_scalar("Train/Steps/video_loss", video_loss.cpu().item(), scalar_states.update_steps)
                tb_writer.add_scalar("Train/Steps/audio_loss", audio_loss.cpu().item(), scalar_states.update_steps)
                tb_writer.add_scalar("Train/Steps/step_loss", loss.cpu().item(), scalar_states.update_steps)
                tb_writer.add_scalar("Train/Steps/grad_norm", grad_norm, scalar_states.update_steps)
                for data_key in data_keys:
                    tb_writer.add_scalar(f"Train/Tokens/{data_key}_train_loss", global_loss,
                                         scalar_states.consumed_tokens_total[data_key])
                tb_writer.add_scalar("Train/ComputationsAttn/train_loss", global_loss,
                                     scalar_states.consumed_computations_attn)
                tb_writer.add_scalar("Train/ComputationsTotal/train_loss", global_loss,
                                     scalar_states.consumed_computations_total)

                # Log learning rate
                tb_writer.add_scalar("LR/learning_rate", lr_scheduler.get_last_lr()[0], scalar_states.update_steps)

                # Log timing metrics
                tb_writer.add_scalar("Time/step_time", sec_per_step, scalar_states.update_steps)
                tb_writer.add_scalar("Time/steps_per_sec", steps_per_sec, scalar_states.update_steps)
                tb_writer.add_scalar("Time/samples_per_sec", int(samples_per_sec), scalar_states.update_steps)
                tb_writer.add_scalar("Time/dataloader_time", dataloader_time_ - start_time, scalar_states.update_steps)
                tb_writer.add_scalar("Time/data_time", data_time_ - dataloader_time_, scalar_states.update_steps)
                tb_writer.add_scalar("Time/forward_time", forward_time_ - data_time_, scalar_states.update_steps)
                tb_writer.add_scalar("Time/backward_time", backward_time_ - forward_time_, scalar_states.update_steps)

                # Consumed samples
                for data_key in data_keys:
                    tb_writer.add_scalar(f"Data/{data_key}_samples_total",
                                         scalar_states.consumed_samples_total[data_key], scalar_states.update_steps)
                    tb_writer.add_scalar(f"Data/{data_key}_samples_epoch",
                                         scalar_states.epoch_consumed_samples[data_key], scalar_states.update_steps)

                for data_key in data_keys:
                    tb_writer.add_scalar(f"Data/{data_key}_tokens_total", scalar_states.consumed_tokens_total[data_key],
                                         scalar_states.update_steps)

                # Log mask type statistics to tensorboard
                for mt, count in mask_type_stats.items():
                    tb_writer.add_scalar(f"Data/MaskType/{mt}_samples_total", count, scalar_states.update_steps)

                # Log mask type loss statistics to tensorboard (using global averages)
                for mt, avg_loss_val in global_avg_loss_by_mask_type.items():
                    tb_writer.add_scalar(f"Train/MaskType/{mt}_avg_loss", avg_loss_val, scalar_states.update_steps)

                # Log mask type current cycle sample statistics to tensorboard
                for mt, count in global_cycle_samples_by_mask_type.items():
                    tb_writer.add_scalar(f"Data/MaskType/{mt}_samples", count, scalar_states.update_steps)

                # Log MOVA-specific metrics
                tb_writer.add_scalar("Train/Steps/visual_timestep", visual_timestep.item(), scalar_states.update_steps)
                tb_writer.add_scalar("Train/Steps/audio_timestep", audio_timestep.item(), scalar_states.update_steps)

            # Save checkpoint
            if scalar_states.train_steps % args.checkpointing_steps == 0 and scalar_states.train_steps > 0:
                logger.info(f"--> save checkpoint at step {scalar_states.train_steps}, {args.output_dir}")
                fsdp_save_checkpoint_without_optim(model, rank, args.output_dir, scalar_states)
                # Collective DCP save of the sharded optimizer state (all ranks).
                if save_optim_state:
                    save_optimizer_state(model, optimizer, args.output_dir, scalar_states)
                dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

            global_step += 1

    if getattr(args, 'dry_run', False):
        logger.info("Dry run finished.")
    else:
        fsdp_save_checkpoint_without_optim(
            model, rank, args.output_dir, scalar_states
        )
        if save_optim_state:
            save_optimizer_state(model, optimizer, args.output_dir, scalar_states)

    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
    destroy_sequence_parallel_group()


def save_args(args):
    tccl_file = "/dockerdata/.tccl/tccl.data"
    exist = os.path.exists(tccl_file)
    if exist is not True:
        print(f"file {tccl_file} doesnot exist, no need save args")
        return

    rank = int(os.environ["RANK"])
    if rank > 0:
        print(f"only the rank 0 write the tccl data")
        return
    output_lines = []
    if hasattr(args, "pipeline_model_parallel_size") is not True:
        output_lines.append(f"'--pipeline-model-parallel-size', '1',")
    if hasattr(args, "tensor_model_parallel_size") is not True:
        output_lines.append(f"'--tensor-model-parallel-size', '1',")

    for key, value in vars(args).items():
        if key.startswith('_'):
            continue
        arg_name = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                output_lines.append(f"'{arg_name}',")
        elif value is None:
            continue
        elif isinstance(value, (list, tuple, dict)):
            continue
        else:
            output_lines.append(f"'{arg_name}', '{value}',")

    fo = None
    try:
        fo = open(tccl_file, "a")
        fo.write("\n")
        for line in output_lines:
            fo.write(f"{line}\n")
    except Exception as e:
        print(f"rank {rank} fail to save the arguments to tccl file {tccl_file}, reason:{e}")
    except:
        print(f"rank {rank} fail to save the arguments to tcc file {tccl_file} for unknown reason")

    if fo is not None:
        fo.close()
    print(f"rank {rank} save the arguments to tccl file {tccl_file} successfully")


if __name__ == "__main__":
    args = parse_args()
    save_args(args)
    main(args)


# export node_ip=$(echo ${NODE_IP_LIST} | sed 's/:8//g')
# pdsh -f 256 -w $node_ip "pip install ipdb"
# pdsh -f 256 -w $node_ip "pip uninstall diffusers -y"
# pdsh -f 256 -w $node_ip "pip install diffusers==0.33.0"
# pdsh -f 256 -w $node_ip "pip install decord"

# dataset preparation (offline bucket assignment)
# python tools/build_dataset_csv.py --input <raw>.csv --output <bucketed>.csv \
#     --latent-resolution 720p --temporal-min-length 49 --temporal-max-length 289 \
#     --temporal-interval 12 --scan-latents

# training script
# nohup bash scripts/pretrain/pdsh_train.sh "scripts/pretrain/train_wan_ti2va.sh configs/train/wan_ti2va.yaml wan_15B_ti2va_720p_init_top_k_p" > train_output_wan_720p.txt 2>&1 &
# nohup bash scripts/pretrain/pdsh_train.sh "scripts/pretrain/train_wan_ti2va.sh configs/train/wan_ti2va.yaml wan_15B_ti2va_1080p_init_top_k_p" > train_output_wan_1080p.txt 2>&1 &
# nohup bash scripts/pretrain/pdsh_train.sh "scripts/pretrain/train_wan_ti2va.sh configs/train/wan_ti2va.yaml wan_15B_ti2va_2k_init_top_k_p" > train_output_wan_2k.txt 2>&1 &

# inference script
# nohup bash prism_infer.sh > infer_output.txt 2>&1 &
# nohup bash prism_infer_fsdp.sh > infer_fsdp_output.txt 2>&1 &

