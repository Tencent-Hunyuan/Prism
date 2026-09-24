# Video self-attention BSA
ENABLE_BSA="true"
BSA_SPARSITY=0.75
BSA_CHUNK_3D_SHAPE_Q="4 4 4" # optimal = 4 4 4
BSA_CHUNK_3D_SHAPE_K="4 4 4"
BSA_CDF_THRESHOLD=0.20  # [0.16, 0.2, 0.3, 0.4] (optimal=0.2)
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
AUDIO_BOOST_GAMMA=2.0  # Path A boost strength γ: 0.0=no effect, 1.0=moderate, 2.0=strong
# Path B: Audio-Weighted K Pooling (independent from Path A)
ENABLE_AUDIO_WEIGHTED_POOLING="false"
AUDIO_WEIGHTED_LAMBDA=2.0  # Path B weighted K pooling λ: 1.0=moderate
# =========================================================================================================
# Channel-Variance Guidance for BSA (independent from audio guidance, can be used together)
ENABLE_VARIANCE_GUIDANCE="false"
VARIANCE_BOOST_GAMMA=2.0  # boost strength: 0.5=mild, 1.0=moderate, 2.0=strong
# =========================================================================================================
# Bias Correction for BSA (two independent methods, at most one enabled)
ENABLE_TAYLOR_SPARSE_ATTN="false"
TAYLOR_ALPHA_F=0.5  # flat ratio: 0.5 = 50% flat queries get Taylor, 50% sharp keep BSA
ENABLE_RECTIFIED_SPARSE_ATTN="false"
# =========================================================================================================
# Anisotropic Dynamic Block Shape — video self-attn only. Two MUTUALLY-EXCLUSIVE methods.
# Enabling EITHER disables audio/variance guidance + taylor/rectified (fully isolated). Default: both off.
ENABLE_IVPQ_DYNAMIC_BLOCK="false"      # Inverse-Variance Proportional Quantization
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
# 480p: (height=480, width=848) --> 720p: (height=720, width=1280) --> 1080p: (height=1072, width=1920) --> 2k: (height=1440, width=2560)
# num_frames: must satisfy (n-1) % temporal_scale == 0; auto-snapped if not (default=193)
NUM_FRAMES=205

# =========================================================================================================
# VAE tiling (decode-only) — tile the video VAE decode spatially to avoid OOM
ENABLE_TILING="false"          # set "true" only when the VAE decode OOMs (e.g. 2K)
TILE_SAMPLE_MIN_SIZE=256       # tile size (H & W) in pixels
TILE_SAMPLE_STRIDE=192         # stride between tiles; overlap = tile_size - stride

bash scripts/inference/inference_mova_single.sh --deepspeed \
    --ckpt /path/pretrained_models/MOVA-360p \
    --resume_ckpt /path/checkpoints/preview_alpha/diffusion_pytorch_model.safetensors \
    --config configs/train/t2va_config/mova_infer.yaml \
    --sp_size 8 \
    --offload cpu \
    --height 480 \
    --width 848 \
    --num_frames $NUM_FRAMES \
    --prompt "Continuous <music>light and soothing instrumental music played on guitar</music>, accompanied by <sfx>constant wave sounds</sfx>. Wide shot, eye-level view. Daytime, clear outdoor ocean scene with high saturation turquoise-blue tones. A young white male surfer with damp dark brown short hair wears a full-length black tight-fitting wetsuit with a gray panel on the left arm, black knee-length shorts, and barefoot on a white surfboard featuring a black brand logo <text>ROXY</text> and a black fin. Behind him is a massive turquoise wave, with breaking white foam to the left. In the distance, a vast deep blue ocean stretches out, and far to the right, another person is lying on a surfboard. Foreground features white sandy foam. Initially, the camera pans right, following the black-clad surfer gliding rightward on his board, accompanied by <sfx>loud crashing wave sounds</sfx>. Then he crouches, cuts left into the wave tube, creating <sfx>intense splashing sounds</sfx>. He continues riding through the tube, then forcefully carves with his back edge, nearly engulfed by a massive white splash. After the carve, he bursts out of the spray, adjusts his stance over a small wave, then rides upward along the wave face toward the crest, making a large cutback to the left, producing a huge <sfx>spattering splash sound</sfx>. Finally, he regains his balance from the cutback and lands on the breaking white foam, continuing forward." \
    --audio_prompt "Continuous <music>light and soothing instrumental music played on guitar</music>, accompanied by <sfx>constant wave sounds</sfx>. Wide shot, eye-level view. Daytime, clear outdoor ocean scene with high saturation turquoise-blue tones. A young white male surfer with damp dark brown short hair wears a full-length black tight-fitting wetsuit with a gray panel on the left arm, black knee-length shorts, and barefoot on a white surfboard featuring a black brand logo <text>ROXY</text> and a black fin. Behind him is a massive turquoise wave, with breaking white foam to the left. In the distance, a vast deep blue ocean stretches out, and far to the right, another person is lying on a surfboard. Foreground features white sandy foam. Initially, the camera pans right, following the black-clad surfer gliding rightward on his board, accompanied by <sfx>loud crashing wave sounds</sfx>. Then he crouches, cuts left into the wave tube, creating <sfx>intense splashing sounds</sfx>. He continues riding through the tube, then forcefully carves with his back edge, nearly engulfed by a massive white splash. After the carve, he bursts out of the spray, adjusts his stance over a small wave, then rides upward along the wave face toward the crest, making a large cutback to the left, producing a huge <sfx>spattering splash sound</sfx>. Finally, he regains his balance from the cutback and lands on the breaking white foam, continuing forward." \
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


# nohup bash prism_infer.sh > infer_output.txt 2>&1 &
