import argparse


def parse_eval_initial_args():
    """
    Note: Arguments in this section are duplicated from other sections for procedure initialization
          before the main parser is called.
    """
    parser = argparse.ArgumentParser("Initial arguments for evaluation.")
    parser.add_argument("--ddp", action="store_true", help="Enable Distributed Data Parallel (DDP) during sampling.")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for DDP.")
    parser.add_argument("--node-index", type=int, default=0, help="Node index for DDP.")
    parser.add_argument("--deepspeed", action="store_true", help="Enable Deepspeed during sampling.")

    parser.add_argument("--reproduce", action="store_true", help="Enable reproducibility by setting random seeds and deterministic algorithms.")
    parser.add_argument("--global-seed", type=int, default=1, help="Global seed for reproducibility.")

    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--config", type=str, default="", help="Config yaml file.")

    args, _ = parser.parse_known_args()

    if args.ddp:
        mode = "ddp"
    elif args.deepspeed:
        mode = "deepspeed"
    else:
        mode = "none"
    return args, mode


def parse_args(mode="train", namespace=None):
    parser = argparse.ArgumentParser(description="Hunyuan Multimodal training/inference script")

    parser.add_argument("--inject-config", type=str, default="configs/train/t2va_config/wan_15B_ti2va_normal_720p_init.yaml",
                        help="MOVA/WAN inject config (model, training and dataloader sections).")

    parser = add_common_args(parser)
    parser = add_data_args(parser)
    parser = add_network_args(parser)
    parser = add_training_args(parser)
    parser = add_fsdp_args(parser)
    args = parser.parse_args(namespace=namespace)

    return args


def add_common_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Common")
    group.add_argument(
        "--global-seed", type=int, default=42, help="A seed for reproducible training."
    )
    group.add_argument("--output-dir", type=str, default='./outputs', help="Directory to save logs and models")
    group.add_argument("--is-profiler", action="store_true", help="Enable profiler.")
    return parser


def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Data")

    group.add_argument("--data-type", type=str, default="image", help="Type of the dataset.")
    group.add_argument("--fast-shuffle", action="store_true", help="Enable fast shuffle for data loading.")
    group.set_defaults(fast_shuffle=True)
    group.add_argument("--rank-assign-mode", type=str, default="fixed", choices=["fixed", "random"],
                       help="Rank assign mode for data loading.")
    group.add_argument("--video-multireso", action="store_true", help="Use multiple resolution for video training.")
    group.add_argument("--video-sampling-prob", type=float, default=1.0,
                       help="The prob to sample a video from image and video dataset.")

    group.add_argument("--use-ll-cap-dataset", action="store_true", help="Use long long caption dataset for training.")
    group.add_argument("--ll-cap-sample-ratio", type=float, default=0.2, help="Sampling ratio for ll_cap dataset.")
    group.add_argument("--use-video-ll-cap-dataset", action="store_true",
                       help="Use long long caption dataset for video training.")
    group.add_argument("--video-ll-cap-sample-ratio", type=float, default=0.2,
                       help="Sampling ratio for video ll_cap dataset.")

    return parser


def add_training_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Training")
    group.add_argument("--optimizer", type=str, default='adamw',
                       help="Optimizer type. Only 'adamw' is supported.")
    group.add_argument("--micro-batch-size", type=int, default=1, nargs='*',
                       help="Batch size per model instance (local batch size).")
    group.add_argument(
        "--checkpointing-steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    group.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    group.add_argument(
        "--num-train-epochs",
        type=int,
        default=100
    )
    group.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    group.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    group.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    group.add_argument(
        "--max-grad-norm", default=1.0, type=float, help="Max gradient norm."
    )
    group.add_argument(
        "--save-optimizer-state",
        type=str,
        default="true",
        help=(
            "Whether to save & resume the (sharded, reshardable via DCP) AdamW "
            "optimizer state alongside the model checkpoint. 'true' or 'false'. "
            "Default 'true'. Works across a different node/GPU count on resume."
        ),
    )
    group.add_argument("--selective-checkpointing", type=float, default=1.0)
    group.add_argument(
        "--fine-grained-gc", type=str, default="false",
        help="Fine-grained (nested) activation checkpointing inside each "
             "FusedMOVABlock: checkpoint a2v/v2a/video/audio submodules "
             "independently so the recompute peak drops from SUM to MAX of "
             "submodule activations. Numerically identical to the default "
             "whole-block checkpointing. Default: false (original behavior).",
    )
    group.add_argument(
        "--lr-scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    group.add_argument(
        "--lr-num-cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    group.add_argument(
        "--lr-power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    group.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay to apply."
    )
    group.add_argument(
        "--master-weight-type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    group.add_argument("--log-interval", type=int, default=20, help="Interval of logging.")

    group.add_argument(
        "--offload-frozen-encoders", type=str, default="true",
        help="Move the frozen text encoder / audio VAE / video VAE back to CPU "
             "after each step. Default 'true' keeps the original memory profile; "
             "'false' keeps them resident (~12GB more VRAM) and removes the "
             "largest fixed per-step cost.",
    )
    group.add_argument(
        "--empty-cache-interval", type=int, default=1,
        help="Call torch.cuda.empty_cache() every N steps (0 disables). It is a "
             "device-wide sync that also drops the allocator pool, so 1 stalls "
             "every step. Raise it if you are not memory bound.",
    )

    group.add_argument("--dry-run", action="store_true", help="Dry run mode.")
    group.add_argument(
        "--loss-spike-threshold",
        type=float,
        default=5.0,
        help="Skip optimizer update if loss exceeds this threshold."
    )
    group.add_argument(
        "--grad-norm-spike-threshold",
        type=float,
        default=60.0,
        help="Skip optimizer update if pre-clip grad norm exceeds this threshold (synchronized across all ranks via all_reduce MAX)."
    )
    group.add_argument(
        "--train-full-model",
        type=str,
        default="false",
        help="When 'true', train all model parameters (full fine-tuning). "
             "When 'false' (default), only train attention modules (SelfAttention, CrossAttention, Bridge)."
    )
    return parser


def add_network_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Network")

    # DiT MoE high/low-noise expert routing during training.
    # Default (false): even/high-noise step -> video_dit, odd/low-noise step -> video_dit_2 (aligned with inference).
    # When true: SWAP, i.e. even/high-noise step -> video_dit_2, odd/low-noise step -> video_dit (misaligned).
    group.add_argument("--swap-dit-moe-high-noise", type=str, default="false",
                       help="Swap which video DiT expert is trained at high noise. "
                            "Default false: high noise -> video_dit (aligned with inference).")

    # Block Sparse Attention (BSA) config — video self-attention
    group.add_argument("--enable-bsa", type=str, default="false",
                       help="Enable Block Sparse Attention for video self-attn. Default: false.")
    group.add_argument("--bsa-sparsity", type=float, default=0.9375,
                       help="BSA sparsity ratio (fraction of blocks to skip). Default: 0.9375.")
    group.add_argument("--bsa-chunk-3d-shape-q", type=int, nargs=3, default=[4, 4, 4],
                       help="BSA 3D chunk shape for query (T H W). Default: 4 4 4.")
    group.add_argument("--bsa-chunk-3d-shape-k", type=int, nargs=3, default=[4, 4, 4],
                       help="BSA 3D chunk shape for key (T H W). Default: 4 4 4.")
    group.add_argument("--bsa-cdf-threshold", type=float, default=None,
                       help="BSA Top-p cumulative probability threshold for hybrid Top-k+Top-p masking. "
                            "When set together with --bsa-sparsity, enables hybrid masking: "
                            "M = Top-k(P, sparsity) ∪ Top-p(P, cdf_threshold). Default: None (Top-k only).")
    # BSA config — v2a bridge cross-attention (Q=audio 1D blocked, K=video 3D blocked)
    # a2v is full attention (audio K too short for sparse)
    group.add_argument("--enable-bsa-v2a", type=str, default="false",
                       help="Enable BSA for v2a bridge cross-attn (Q=audio, K=video). Default: false.")
    group.add_argument("--bsa-v2a-sparsity", type=float, default=0.875,
                       help="BSA sparsity for v2a cross-attn. Default: 0.875.")
    group.add_argument("--bsa-v2a-audio-chunk-size", type=int, default=64,
                       help="BSA audio Q block size for v2a (must be multiple of 64). Default: 64.")
    group.add_argument("--bsa-v2a-chunk-3d-shape-k", type=int, nargs=3, default=[4, 4, 4],
                       help="BSA 3D chunk for v2a K (video side). Default: 4 4 4.")
    group.add_argument("--bsa-v2a-cdf-threshold", type=float, default=None,
                       help="BSA Top-p threshold for v2a hybrid Top-k+Top-p masking. Default: None.")

    # Audio Guidance for BSA
    group.add_argument("--enable-audio-guidance", type=str, default="false",
                       help="Enable audio-guided BSA score modulation. Default: false.")
    group.add_argument("--enable-audio-concentration-gate", type=str, default="false",
                       help="Enable Gate 2 (Audio Spatial Concentration Gate). Default: false.")
    group.add_argument("--enable-timestep-reliability-gate", type=str, default="false",
                       help="Enable Gate 1 (Timestep Reliability Gate, non-learnable). Default: false.")
    group.add_argument("--audio-boost-gamma", type=float, default=1.0,
                       help="Audio guidance boost strength γ (Path A). Default: 1.0.")
    group.add_argument("--enable-audio-weighted-pooling", type=str, default="false",
                       help="Enable Path B: Audio-weighted K pooling. Default: false.")
    group.add_argument("--audio-weighted-lambda", type=float, default=1.0,
                       help="Audio-weighted K pooling strength λ (Path B). Default: 1.0.")

    # Channel-Variance Guidance for BSA (independent from audio guidance)
    group.add_argument("--enable-variance-guidance", type=str, default="false",
                       help="Enable channel-variance guided BSA score modulation. Default: false.")
    group.add_argument("--variance-boost-gamma", type=float, default=1.0,
                       help="Variance guidance boost strength γ. Default: 1.0.")

    # Bias Correction for BSA (two independent methods, at most one enabled at a time)
    group.add_argument("--enable-taylor-sparse-attn", type=str, default="false",
                       help="LIVEditor: Taylor sparse attn (selected exact + non-selected Taylor in one softmax). Default: false.")
    group.add_argument("--taylor-alpha-f", type=float, default=0.5,
                       help="Taylor flat ratio: fraction of queries routed to Taylor branch. Default: 0.5.")
    group.add_argument("--enable-rectified-sparse-attn", type=str, default="false",
                       help="Rectified SpaAttn: R_n * o_spa + A_pool[nonsel] · V_pool. Default: false.")

    # Anisotropic Dynamic Block Shape — video self-attn only.
    # Two mutually-exclusive methods. Enabling EITHER disables audio/variance
    # guidance and taylor/rectified bias correction (fully isolated path).
    group.add_argument("--enable-ivpq-dynamic-block", type=str, default="false",
                       help="IVPQ dynamic block shape. Default: false.")
    group.add_argument("--enable-penalty-dynamic-block", type=str, default="false",
                       help="Penalty-Matching dynamic block shape. Default: false.")
    group.add_argument("--dynamic-block-lambda-a", type=float, default=0.5,
                       help="Audio-directional influence strength λ_a for g_d. Default: 0.5.")
    group.add_argument("--dynamic-block-tau-128", type=float, default=0.15,
                       help="Info-density threshold τ_128 gating the 128-token shape pool. Default: 0.15.")
    group.add_argument("--dynamic-block-lambda-128", type=float, default=1.0,
                       help="128-token density bonus λ_128 (Penalty Matching only). Default: 1.0.")
    group.add_argument("--enable-layer-adaptive-dynamic-block", type=str, default="false",
                       help="Layer-adaptive dynamic block: shallow half FIXED / "
                            "deep half dynamic + shallow→deep audio cache. Only effective with "
                            "ivpq/penalty. Default: false.")
    group.add_argument("--sparse-attn-high-noise-only", type=str, default="false",
                       help="If 'true', apply ALL video self-attn sparse features (top-k/top-p "
                            "BSA, audio/variance guidance, taylor/rectified, dynamic block shape) "
                            "ONLY to the high-noise expert (video_dit); the low-noise expert "
                            "(video_dit_2) keeps the backbone's dense full attention. "
                            "Default: false = sparse on BOTH video experts.")

    group.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )
    return parser


def add_fsdp_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="FSDP")

    group.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    group.add_argument("--sp-size", type=int, default=1, help="For sequence parallel")
    group.add_argument("--use-dynamic-ring-attention", action="store_true", help="Use dynamic Ring Attention.")
    group.add_argument("--fsdp-sharding-strategy", default="full", choices=['full', 'hybrid', 'none'])
    group.add_argument(
        "--nccl-timeout", type=int, default=1800,
        help="Per-collective NCCL timeout in seconds, applied once the training "
             "loop starts. Lower is better there: a genuine desync surfaces as a "
             "watchdog abort with a stack trace in minutes instead of stalling.",
    )
    group.add_argument(
        "--init-timeout", type=int, default=5400,
        help="NCCL timeout in seconds during startup only. Must be generous: "
             "ranks sit in the staggered-load barrier while a peer streams the "
             "checkpoint off a network filesystem.",
    )

    return parser
