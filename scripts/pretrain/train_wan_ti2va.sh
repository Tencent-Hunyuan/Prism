#!/bin/bash
source scripts/utils/env.sh
# 公共环境变量
export PYTHONPATH=`pwd`
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NCCL_DEBUG=INFO
#export NCCL_BLOCKING_WAIT=0     # 设置为0, 最大化计算与通信的重叠，提升训练效率
export NCCL_IB_TIMEOUT=22
# 让 watchdog 在 collective 超时后直接 abort 并打印堆栈, 而不是无限期挂住;
# flight recorder 会 dump 出每个 rank 最后执行到哪个 collective, 便于定位 desync.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export NCCL_SOCKET_TIMEOUT=600
export NCCL_PRIMS_PROFILE_VERSION=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#################################PARSE FROM YAML#################################
pip3 install shyaml
config_file=$1
config_name=$2
echo $config_file
echo $config_name

config_names=()
config_count=$(shyaml get-length configs < "$config_file")
for ((i=0; i<config_count; i++)); do
    config_names+=("$(shyaml get-value "configs.$i.name" < "$config_file")")
done

config_index=-1
for i in "${!config_names[@]}"; do
    if [[ "${config_names[i]}" == "$config_name" ]]; then
        config_index=$i
        break
    fi
done
if [ $config_index -eq -1 ]; then
    echo "--------------------------------------------------------------------------"
    echo "Error: Config '$config_name' not found. Available configs: ${config_names[@]}"
    echo "--------------------------------------------------------------------------"
    exit 1
fi

# Inject config: model / training / dataloader sections for the MOVA bridge
inject_config=$(shyaml get-value "configs.$config_index.inject_config" < "$config_file")

exp_name=$(shyaml get-value "configs.$config_index.name" < "$config_file")
# exp_base: 优先读取 yaml 中对应 config 的 exp_base，缺失时回退到 env.sh 的 $EXP_BASE
exp_base=$(shyaml get-value "configs.$config_index.exp_base" < "$config_file" 2>/dev/null || echo "$EXP_BASE")
if [ -z "$exp_base" ]; then
    exp_base="$EXP_BASE"
fi

sp_size=($(shyaml get-value "configs.$config_index.sp_size" < "$config_file" || echo "1"))
# Per-collective NCCL timeout. Keep it short: a real desync then aborts with a
# stack trace in minutes instead of stalling the job until the 1h default.
nccl_timeout=$(shyaml get-value "configs.$config_index.nccl_timeout" < "$config_file" || echo "1800")
# Startup timeout: covers the staggered checkpoint load, where ranks legitimately
# wait a long time in a barrier. Raise it if loading from a slow filesystem.
init_timeout=$(shyaml get-value "configs.$config_index.init_timeout" < "$config_file" || echo "5400")
# Per-step frozen-encoder shuttling. Defaults reproduce the original behaviour;
# set offload_frozen_encoders=false / empty_cache_interval>1 to trade VRAM for speed.
offload_frozen_encoders=$(shyaml get-value "configs.$config_index.offload_frozen_encoders" < "$config_file" || echo "true")
empty_cache_interval=$(shyaml get-value "configs.$config_index.empty_cache_interval" < "$config_file" || echo "1")
use_profile=($(shyaml get-value "configs.$config_index.use_profile" < "$config_file" || echo "false"))
checkpointing_steps=$(shyaml get-value "configs.$config_index.checkpointing_steps" < "$config_file" || echo "2000")
fsdp_strategy=$(shyaml get-value "configs.$config_index.fsdp_strategy" < "$config_file" || echo "full")
use_cpu_offload=$(shyaml get-value "configs.$config_index.use_cpu_offload" < "$config_file" || echo "false")
# Fine-grained (nested) activation checkpointing. Default "false" = original
# whole-block gradient checkpointing. "true" = checkpoint a2v/v2a/video/audio
# submodules independently inside each FusedMOVABlock (lower recompute peak,
# numerically identical).
fine_grained_gc=$(shyaml get-value "configs.$config_index.fine_grained_gc" < "$config_file" || echo "false")

# Trainable mode: "true" = full model, "false" = attention-only (default)
train_full_model=$(shyaml get-value "configs.$config_index.train_full_model" < "$config_file" || echo "false")

# Spike detection parameters
grad_norm_spike_threshold=$(shyaml get-value "configs.$config_index.grad_norm_spike_threshold" < "$config_file" || echo "60.0")
loss_spike_threshold=$(shyaml get-value "configs.$config_index.loss_spike_threshold" < "$config_file" || echo "1.0")

# DiT MoE high-noise expert swap trigger (default false = aligned with inference: high-noise -> video_dit)
swap_dit_moe_high_noise=$(shyaml get-value "configs.$config_index.swap_dit_moe_high_noise" < "$config_file" || echo "false")

# Block Sparse Attention (BSA) config
enable_bsa=$(shyaml get-value "configs.$config_index.enable_bsa" < "$config_file" || echo "false")
bsa_sparsity=$(shyaml get-value "configs.$config_index.bsa_sparsity" < "$config_file" || echo "0.9375")
bsa_chunk_3d_shape_q=$(shyaml get-value "configs.$config_index.bsa_chunk_3d_shape_q" < "$config_file" || echo "4 4 4")
bsa_chunk_3d_shape_k=$(shyaml get-value "configs.$config_index.bsa_chunk_3d_shape_k" < "$config_file" || echo "4 4 4")
bsa_cdf_threshold=$(shyaml get-value "configs.$config_index.bsa_cdf_threshold" < "$config_file" || echo "")
# BSA for v2a bridge cross-attention (Q=audio 1D blocked, K=video 3D blocked)
enable_bsa_v2a=$(shyaml get-value "configs.$config_index.enable_bsa_v2a" < "$config_file" || echo "false")
bsa_v2a_sparsity=$(shyaml get-value "configs.$config_index.bsa_v2a_sparsity" < "$config_file" || echo "0.875")
bsa_v2a_audio_chunk_size=$(shyaml get-value "configs.$config_index.bsa_v2a_audio_chunk_size" < "$config_file" || echo "64")
bsa_v2a_chunk_3d_shape_k=$(shyaml get-value "configs.$config_index.bsa_v2a_chunk_3d_shape_k" < "$config_file" || echo "4 4 4")
bsa_v2a_cdf_threshold=$(shyaml get-value "configs.$config_index.bsa_v2a_cdf_threshold" < "$config_file" || echo "")

# Audio Guidance for BSA (Dual-Gated: Gate 1 + Gate 2)
enable_audio_guidance=$(shyaml get-value "configs.$config_index.enable_audio_guidance" < "$config_file" || echo "false")
enable_audio_concentration_gate=$(shyaml get-value "configs.$config_index.enable_audio_concentration_gate" < "$config_file" || echo "false")
enable_timestep_reliability_gate=$(shyaml get-value "configs.$config_index.enable_timestep_reliability_gate" < "$config_file" || echo "false")
audio_boost_gamma=$(shyaml get-value "configs.$config_index.audio_boost_gamma" < "$config_file" || echo "1.0")
enable_audio_weighted_pooling=$(shyaml get-value "configs.$config_index.enable_audio_weighted_pooling" < "$config_file" || echo "false")
audio_weighted_lambda=$(shyaml get-value "configs.$config_index.audio_weighted_lambda" < "$config_file" || echo "1.0")

# Channel-Variance Guidance for BSA (independent from audio guidance)
enable_variance_guidance=$(shyaml get-value "configs.$config_index.enable_variance_guidance" < "$config_file" || echo "false")
variance_boost_gamma=$(shyaml get-value "configs.$config_index.variance_boost_gamma" < "$config_file" || echo "1.0")

# Bias Correction for BSA (two independent methods, at most one enabled)
enable_taylor_sparse_attn=$(shyaml get-value "configs.$config_index.enable_taylor_sparse_attn" < "$config_file" || echo "false")
taylor_alpha_f=$(shyaml get-value "configs.$config_index.taylor_alpha_f" < "$config_file" || echo "0.5")
enable_rectified_sparse_attn=$(shyaml get-value "configs.$config_index.enable_rectified_sparse_attn" < "$config_file" || echo "false")

# Anisotropic Dynamic Block Shape (Section 8) — video self-attn only. Two mutually-exclusive methods.
enable_ivpq_dynamic_block=$(shyaml get-value "configs.$config_index.enable_ivpq_dynamic_block" < "$config_file" || echo "false")
enable_penalty_dynamic_block=$(shyaml get-value "configs.$config_index.enable_penalty_dynamic_block" < "$config_file" || echo "false")
dynamic_block_lambda_a=$(shyaml get-value "configs.$config_index.dynamic_block_lambda_a" < "$config_file" || echo "0.5")
dynamic_block_tau_128=$(shyaml get-value "configs.$config_index.dynamic_block_tau_128" < "$config_file" || echo "0.15")
dynamic_block_lambda_128=$(shyaml get-value "configs.$config_index.dynamic_block_lambda_128" < "$config_file" || echo "1.0")
# Layer-Adaptive Dynamic Block (Section 8.2): shallow half fixed / deep half dynamic
enable_layer_adaptive_dynamic_block=$(shyaml get-value "configs.$config_index.enable_layer_adaptive_dynamic_block" < "$config_file" || echo "false")

# Sparse-attn scope: "true" = all sparse-attn features only on high-noise expert (video_dit);
# low-noise expert (video_dit_2) keeps dense full attn. Default "false" = both experts.
sparse_attn_high_noise_only=$(shyaml get-value "configs.$config_index.sparse_attn_high_noise_only" < "$config_file" || echo "false")

resume=$(shyaml get-value "configs.$config_index.resume" < "$config_file" || echo "None")

# 数据集配置参数（实际的数据集由 inject_config 的 dataloader_config 控制）
data_type=$(shyaml get-value "configs.$config_index.data_type" < "$config_file" || echo "image_video")
image_batch_size=$(shyaml get-value "configs.$config_index.image_batch_size" < "$config_file" || echo "64")
video_sampling_prob=$(shyaml get-value "configs.$config_index.video_sampling_prob" < "$config_file" || echo "0.92")

# master weight dtype: read from inject_config's training_config (default bf16)
master_weight_type=$(shyaml get-value "training_config.master_weight_type" < "$inject_config" 2>/dev/null || echo "bf16")
optimizer=$(shyaml get-value "configs.$config_index.optimizer" < "$config_file" || echo "adamw")
lr=$(shyaml get-value "configs.$config_index.lr" < "$config_file" || echo "1e-4")
gradient_accumulation_steps=$(shyaml get-value "configs.$config_index.gradient_accumulation_steps" < "$config_file"|| echo 1)
lr_warmup_steps=$(shyaml get-value "configs.$config_index.lr_warmup_steps" < "$config_file"|| echo 0)
# lr_scheduler: "constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"
lr_scheduler=$(shyaml get-value "configs.$config_index.lr_scheduler" < "$config_file" || echo "constant_with_warmup")
weight_decay=$(shyaml get-value "configs.$config_index.weight_decay" < "$config_file"|| echo 0)
max_grad_norm=$(shyaml get-value "configs.$config_index.max_grad_norm" < "$config_file"|| echo 1.0)
# Toggle DCP save/resume of the (sharded, reshardable) AdamW optimizer state. Default on.
save_optimizer_state=$(shyaml get-value "configs.$config_index.save_optimizer_state" < "$config_file"|| echo "true")

rank_assign_mode=$(shyaml get-value "configs.$config_index.rank_assign_mode" < "$config_file" || echo "fixed")

echo "--------------------------------------------------------------------------"
echo "config_name: ${config_name}"
echo "exp_name: ${exp_name}"
echo "inject_config: ${inject_config}"
echo "sp_size: ${sp_size}"
echo "master_weight_type: ${master_weight_type}"
echo "use_cpu_offload: ${use_cpu_offload}"
echo "lr_scheduler: ${lr_scheduler}"
echo "use_profile: ${use_profile}"
echo "swap_dit_moe_high_noise: ${swap_dit_moe_high_noise}"
echo "enable_bsa: ${enable_bsa}"
echo "bsa_sparsity: ${bsa_sparsity}"
echo "enable_bsa_v2a: ${enable_bsa_v2a}"
echo "bsa_v2a_sparsity: ${bsa_v2a_sparsity}"
echo "enable_audio_guidance: ${enable_audio_guidance}"
echo "enable_audio_concentration_gate: ${enable_audio_concentration_gate}"
echo "enable_timestep_reliability_gate: ${enable_timestep_reliability_gate}"
echo "enable_variance_guidance: ${enable_variance_guidance}"
echo "variance_boost_gamma: ${variance_boost_gamma}"
echo "enable_taylor_sparse_attn: ${enable_taylor_sparse_attn}"
echo "taylor_alpha_f: ${taylor_alpha_f}"
echo "enable_rectified_sparse_attn: ${enable_rectified_sparse_attn}"
echo "enable_ivpq_dynamic_block: ${enable_ivpq_dynamic_block}"
echo "enable_penalty_dynamic_block: ${enable_penalty_dynamic_block}"
echo "enable_layer_adaptive_dynamic_block: ${enable_layer_adaptive_dynamic_block}"
echo "sparse_attn_high_noise_only: ${sparse_attn_high_noise_only}"
echo "--------------------------------------------------------------------------"
#################################PARAMS setup#####################################
if [[ "${CURRENT_TIME}" = "" ]]; then
    CURRENT_TIME=$(date "+%Y.%m.%d-%H.%M.%S")
fi

if [[ "${START_EXPR_TIME}" = "" ]]; then
    START_EXPR_TIME=${CURRENT_TIME}
fi

output_dir=$exp_base/$exp_name/$START_EXPR_TIME
echo "exp_base: ${exp_base}"
echo "output_dir: ${output_dir}"

#################################PARAMS setup#####################################
if [ "$use_cpu_offload" = "true" ]; then
    use_cpu_offload="--use-cpu-offload"
else
    use_cpu_offload=""
fi

profile_config=""
if [ "$use_profile" = "true" ]; then
    profile_config="--is-profiler"
fi

model_params=" \
    --swap-dit-moe-high-noise $swap_dit_moe_high_noise \
    --enable-bsa $enable_bsa \
    --bsa-sparsity $bsa_sparsity \
    --bsa-chunk-3d-shape-q $bsa_chunk_3d_shape_q \
    --bsa-chunk-3d-shape-k $bsa_chunk_3d_shape_k \
    $([ -n "$bsa_cdf_threshold" ] && echo "--bsa-cdf-threshold $bsa_cdf_threshold") \
    --enable-bsa-v2a $enable_bsa_v2a \
    --bsa-v2a-sparsity $bsa_v2a_sparsity \
    --bsa-v2a-audio-chunk-size $bsa_v2a_audio_chunk_size \
    --bsa-v2a-chunk-3d-shape-k $bsa_v2a_chunk_3d_shape_k \
    $([ -n "$bsa_v2a_cdf_threshold" ] && echo "--bsa-v2a-cdf-threshold $bsa_v2a_cdf_threshold") \
    --enable-audio-guidance $enable_audio_guidance \
    --enable-audio-concentration-gate $enable_audio_concentration_gate \
    --enable-timestep-reliability-gate $enable_timestep_reliability_gate \
    --audio-boost-gamma $audio_boost_gamma \
    --enable-audio-weighted-pooling $enable_audio_weighted_pooling \
    --audio-weighted-lambda $audio_weighted_lambda \
    --enable-variance-guidance $enable_variance_guidance \
    --variance-boost-gamma $variance_boost_gamma \
    --enable-taylor-sparse-attn $enable_taylor_sparse_attn \
    --taylor-alpha-f $taylor_alpha_f \
    --enable-rectified-sparse-attn $enable_rectified_sparse_attn \
    --enable-ivpq-dynamic-block $enable_ivpq_dynamic_block \
    --enable-penalty-dynamic-block $enable_penalty_dynamic_block \
    --dynamic-block-lambda-a $dynamic_block_lambda_a \
    --dynamic-block-tau-128 $dynamic_block_tau_128 \
    --dynamic-block-lambda-128 $dynamic_block_lambda_128 \
    --enable-layer-adaptive-dynamic-block $enable_layer_adaptive_dynamic_block \
    --sparse-attn-high-noise-only $sparse_attn_high_noise_only \
    $profile_config \
    "

data_params=" \
    --data-type $data_type \
    --micro-batch-size $image_batch_size \
    --video-multireso \
    --video-sampling-prob $video_sampling_prob \
    --rank-assign-mode $rank_assign_mode \
"

training_params=" \
    --inject-config $inject_config
    --gradient-checkpointing \
    --fine-grained-gc $fine_grained_gc \
    --gradient-accumulation-steps $gradient_accumulation_steps \
    --master-weight-type $master_weight_type \
    --sp-size $sp_size \
    --nccl-timeout $nccl_timeout \
    --init-timeout $init_timeout \
    --offload-frozen-encoders $offload_frozen_encoders \
    --empty-cache-interval $empty_cache_interval \
    --fsdp-sharding-strategy $fsdp_strategy \
    --max-train-steps 1000000 \
    --learning-rate $lr \
    --lr-scheduler $lr_scheduler \
    --lr-warmup-steps $lr_warmup_steps \
    --weight-decay $weight_decay \
    --max-grad-norm $max_grad_norm \
    --save-optimizer-state $save_optimizer_state \
    --checkpointing-steps $checkpointing_steps \
    --global-seed 42 \
    --log-interval 1 \
    --output-dir $output_dir \
    --loss-spike-threshold $loss_spike_threshold \
    --grad-norm-spike-threshold $grad_norm_spike_threshold \
    --train-full-model $train_full_model \
    --optimizer $optimizer \
    $use_cpu_offload \
"

DRY_RUN=0
if [ "$DRY_RUN" = "1" ]; then
    training_params="$training_params --dry-run"
fi

# 检查当前的ckpt目录, 如果有ckpt, 则使用当前的ckpt; 如果没有, 则尝试使用配置
ckpt_cnt=`ls -d ${output_dir}/checkpoints/global_step-* | sort -t- -k2,2 -n -r | wc -l`
if [[ ${ckpt_cnt} -ne 0 ]]; then
    ckpt_dir=`cd ${output_dir}/checkpoints/; ls -d global_step-* | sort -t- -k2,2 -n -r | head -1`
    resume="${output_dir}/checkpoints/${ckpt_dir}"
    echo "there exists ckpt dir, set the resume=${ckpt_dir}"
else
    echo "there exists no ckpt dir, use config resume=${resume}"
fi

if [ -n "$resume" ] && [ "$resume" != "" ]; then
    training_params="$training_params --resume $resume"
fi

# Create logs directory
mkdir -p $output_dir/logs
node_rank=$INDEX
RANK_ID=${INDEX:-0}
CURRENT_LOG_FILE=$output_dir/logs/training_rank_${RANK_ID}.log

# save the arguments to /dockerdata/.tccl/tccl.data for profiling
TCCL_DATA_DIR="/dockerdata/.tccl"
TCCL_DATA_FILE="${TCCL_DATA_DIR}/tccl.data"
rm -f "${TCCL_DATA_FILE}"
if [[ $node_rank = 0 ]]; then
    if [ -d "${TCCL_DATA_DIR}" ]; then
        echo "tccl directory ${TCCL_DATA_DIR} exists, use it!"
    else
        mkdir -p ${TCCL_DATA_DIR}
        echo "tccl directory ${TCCL_DATA_DIR} does not exist, create it!"
    fi

    echo "${NODE_IP_LIST}" | awk -F: -v RS=, '{gsub(/[0-9]+$/, ""); ips = ips ? ips "," : ""; ips = ips "\047" $1 "\047"} END{print "workers[" ips "]"}' >> ${TCCL_DATA_FILE}
    echo "--master_addr=${CHIEF_IP}" >> ${TCCL_DATA_FILE}
    echo "--master_port=29506" >> ${TCCL_DATA_FILE}
fi

# 把config_file cp 到对应的实验目录下
cp "$config_file" "$output_dir/$(basename $config_file)"
echo "==========!!!=========="
echo $TAIJI_HOST_NUM
echo $HOST_GPU_NUM
echo $node_rank
echo $CHIEF_IP
torchrun --nnodes $TAIJI_HOST_NUM --nproc_per_node $HOST_GPU_NUM \
    --node_rank $node_rank \
    --rdzv_endpoint $CHIEF_IP:30068 \
    --rdzv_id 456 \
    hymm/train_t2va_wan.py \
    ${model_params} \
    ${training_params} \
    ${data_params} \
    2>&1 | tee ${CURRENT_LOG_FILE}
