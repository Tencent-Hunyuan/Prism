# MOVA FSDP inference — uses FSDP FULL_SHARD to shard model parameters across
# GPUs, enabling 1080p inference that would OOM with full-model replication.
# Launched via torchrun (no deepspeed dependency).

# Block Sparse Attention (BSA) triggers - set to "true" to enable, "false" to disable
# Video self-attention BSA
ENABLE_BSA="true"
BSA_SPARSITY=0.75  # default=0.9375-->0.75-->0.85
BSA_CHUNK_3D_SHAPE_Q="4 4 4"
BSA_CHUNK_3D_SHAPE_K="4 4 4"
BSA_CDF_THRESHOLD=0.20  # [0.16, 0.2, 0.3, 0.4]
# =========================================================================================================
# Bridge v2a cross-attention BSA (Q=audio, K=video 3D blocked) BSA_V2A_SPARSITY: 0.0-->0.50-->0.75-->0.875
# a2v is full attention (audio K too short for sparse to help)
# Hybrid Top-k+Top-p: uncomment to enable Top-p component
ENABLE_BSA_V2A="false"
BSA_V2A_SPARSITY=0.875
BSA_V2A_AUDIO_CHUNK_SIZE=64
BSA_V2A_CHUNK_3D_SHAPE_K="4 4 4"
#BSA_V2A_CDF_THRESHOLD=0.4
# =========================================================================================================
# Audio Guidance for BSA (Gate 2: Audio Spatial Concentration Gate)
# Uses A→V bridge residual to modulate block selection scores in video self-attention.
ENABLE_AUDIO_GUIDANCE="false"
ENABLE_AUDIO_CONCENTRATION_GATE="false"
ENABLE_TIMESTEP_RELIABILITY_GATE="false"
AUDIO_BOOST_GAMMA=1.0  # Path A boost strength γ: 0.0=no effect, 1.0=moderate, 2.0=strong
# Path B: Audio-Weighted K Pooling (independent from Path A)
ENABLE_AUDIO_WEIGHTED_POOLING="false"
AUDIO_WEIGHTED_LAMBDA=1.0  # Path B weighted K pooling λ: 1.0=moderate
# =========================================================================================================
# Channel-Variance Guidance for BSA (independent from audio guidance, can be used together)
ENABLE_VARIANCE_GUIDANCE="false"
VARIANCE_BOOST_GAMMA=1.0  # boost strength: 0.5=mild, 1.0=moderate, 2.0=strong
# =========================================================================================================
# Bias Correction for BSA (two independent methods, at most one enabled)
ENABLE_TAYLOR_SPARSE_ATTN="false"
TAYLOR_ALPHA_F=0.5  # flat ratio: 0.5 = 50% flat queries get Taylor, 50% sharp keep BSA
ENABLE_RECTIFIED_SPARSE_ATTN="false"
# =========================================================================================================
# Anisotropic Dynamic Block Shape — video self-attn only. Two MUTUALLY-EXCLUSIVE methods.
# Enabling EITHER disables audio/variance guidance + taylor/rectified (fully isolated). Default: both off.
ENABLE_IVPQ_DYNAMIC_BLOCK="true"      # Inverse-Variance Proportional Quantization
ENABLE_PENALTY_DYNAMIC_BLOCK="false"   # Audio-Directional Penalty Matching
DYNAMIC_BLOCK_LAMBDA_A=0.5             # audio-directional influence on g_d
DYNAMIC_BLOCK_TAU_128=0.25            # info-density gate for the 128-token shape pool
DYNAMIC_BLOCK_LAMBDA_128=1.0          # 128-token density bonus (Penalty Matching only)
# shallow half FIXED / deep half dynamic. Only effective with IVPQ/Penalty. Default off.
ENABLE_LAYER_ADAPTIVE_DYNAMIC_BLOCK="false"

# =========================================================================================================
# Sparse-attn scope across the two video MoE experts.
# "true": sparse-attn ops apply ONLY to video_dit (high-noise expert); video_dit_2 (low-noise expert) keeps the backbone's original dense full attention.
SPARSE_HIGH_NOISE_ONLY="false"

# =========================================================================================================
# Inference shift (controls denoising timestep schedule; higher = more steps at high noise)
# 720p recommend: VISUAL_SHIFT=7.0, 1080p: 9.0~13.0, 2K: 13.0~17.0
VISUAL_SHIFT=9.0
AUDIO_SHIFT=7.0
# CFG scale (classifier-free guidance strength; 1.0 = no guidance) default=5.0-->7.5-->9.0
CFG_SCALE=5.0
# 720p: (height=720, width=1280) --> 1080p: (height=1072, width=1920) --> 2k: (height=1440, width=2560)

# =========================================================================================================
# VAE tiling (decode-only) — tile the video VAE decode spatially to avoid OOM
ENABLE_TILING="false"          # set "true" only when the VAE decode OOMs (e.g. 2K)
TILE_SAMPLE_MIN_SIZE=256       # tile size (H & W) in pixels
TILE_SAMPLE_STRIDE=192         # stride between tiles; overlap = tile_size - stride


bash scripts/inference/inference_mova_single_fsdp.sh \
    --ckpt /path/pretrained_models/MOVA-360p \
    --resume_ckpt /path/checkpoints/preview_beta/diffusion_pytorch_model.safetensors \
    --config configs/train/t2va_config/mova_infer.yaml \
    --sp_size 8 \
    --offload cpu \
    --height 1072 \
    --width 1920 \
    --prompt "No background music, quiet recording environment, close-mic capture, no noticeable reverb. Close-up, eye-level view, softly lit with frontal lighting. The frame features an elderly white woman with short gray curly hair, blue eyes, deeply lined face, flushed cheeks, and red lipstick. She wears a black coat and a red scarf with a white floral pattern, a black microphone clipped to her collar. She sits in front of a brown wooden piece of furniture, with a pale yellow textured wall behind her, on which hangs a miniature model depicting a canal and Dutch-style buildings on both sides. Initially, the elderly white woman faces the camera, lips moving, speaking in a hoarse voice with a heavy expression<speech>weg haden ze nog gezegd, ze liepen weg.</speech>. Then, she briefly lowers her head, her gaze moving downward, mouth slightly open as if sighing. Immediately afterward, she raises her head to look at the camera again, continuing<speech>Nou, dan kwam de directie er bij, want de directeur moest oepen.</speech>, blinking slowly throughout, her expression solemn." \
    --audio_prompt "No background music, quiet recording environment, close-mic capture, no noticeable reverb. Close-up, eye-level view, softly lit with frontal lighting. The frame features an elderly white woman with short gray curly hair, blue eyes, deeply lined face, flushed cheeks, and red lipstick. She wears a black coat and a red scarf with a white floral pattern, a black microphone clipped to her collar. She sits in front of a brown wooden piece of furniture, with a pale yellow textured wall behind her, on which hangs a miniature model depicting a canal and Dutch-style buildings on both sides. Initially, the elderly white woman faces the camera, lips moving, speaking in a hoarse voice with a heavy expression<speech>weg haden ze nog gezegd, ze liepen weg.</speech>. Then, she briefly lowers her head, her gaze moving downward, mouth slightly open as if sighing. Immediately afterward, she raises her head to look at the camera again, continuing<speech>Nou, dan kwam de directie er bij, want de directeur moest oepen.</speech>, blinking slowly throughout, her expression solemn." \
    --ref_path "/path/reference_images/3b5f82e5b8d93df57c28bfce997443d7.png" \
    --output_path "/path/prism_samples/i2va_sft_fsdp.mp4" \
    --seed 42 \
    --enable_bsa $ENABLE_BSA \
    --bsa_sparsity $BSA_SPARSITY \
    --bsa_chunk_3d_shape_q $BSA_CHUNK_3D_SHAPE_Q \
    --bsa_chunk_3d_shape_k $BSA_CHUNK_3D_SHAPE_K \
    ${BSA_CDF_THRESHOLD:+--bsa_cdf_threshold $BSA_CDF_THRESHOLD} \
    --enable_bsa_v2a $ENABLE_BSA_V2A \
    --bsa_v2a_sparsity $BSA_V2A_SPARSITY \
    --bsa_v2a_audio_chunk_size $BSA_V2A_AUDIO_CHUNK_SIZE \
    --bsa_v2a_chunk_3d_shape_k $BSA_V2A_CHUNK_3D_SHAPE_K \
    ${BSA_V2A_CDF_THRESHOLD:+--bsa_v2a_cdf_threshold $BSA_V2A_CDF_THRESHOLD} \
    --enable_audio_guidance $ENABLE_AUDIO_GUIDANCE \
    --enable_audio_concentration_gate $ENABLE_AUDIO_CONCENTRATION_GATE \
    --enable_timestep_reliability_gate $ENABLE_TIMESTEP_RELIABILITY_GATE \
    --audio_boost_gamma $AUDIO_BOOST_GAMMA \
    --enable_audio_weighted_pooling $ENABLE_AUDIO_WEIGHTED_POOLING \
    --audio_weighted_lambda $AUDIO_WEIGHTED_LAMBDA \
    --enable_variance_guidance $ENABLE_VARIANCE_GUIDANCE \
    --variance_boost_gamma $VARIANCE_BOOST_GAMMA \
    --enable_taylor_sparse_attn $ENABLE_TAYLOR_SPARSE_ATTN \
    --taylor_alpha_f $TAYLOR_ALPHA_F \
    --enable_rectified_sparse_attn $ENABLE_RECTIFIED_SPARSE_ATTN \
    --enable_ivpq_dynamic_block $ENABLE_IVPQ_DYNAMIC_BLOCK \
    --enable_penalty_dynamic_block $ENABLE_PENALTY_DYNAMIC_BLOCK \
    --dynamic_block_lambda_a $DYNAMIC_BLOCK_LAMBDA_A \
    --dynamic_block_tau_128 $DYNAMIC_BLOCK_TAU_128 \
    --dynamic_block_lambda_128 $DYNAMIC_BLOCK_LAMBDA_128 \
    --enable_layer_adaptive_dynamic_block $ENABLE_LAYER_ADAPTIVE_DYNAMIC_BLOCK \
    --sparse_high_noise_only $SPARSE_HIGH_NOISE_ONLY \
    --visual_shift $VISUAL_SHIFT \
    --audio_shift $AUDIO_SHIFT \
    --cfg_scale $CFG_SCALE \
    --enable_tiling $ENABLE_TILING \
    --tile_sample_min_size $TILE_SAMPLE_MIN_SIZE \
    --tile_sample_stride $TILE_SAMPLE_STRIDE


# nohup bash prism_infer_fsdp.sh > infer_fsdp_output.txt 2>&1 &