#!/bin/bash

JOBS_DIR=$(dirname "$0")
PROJECT_BASE=$(cd ${JOBS_DIR}/../.. || exit; pwd)
echo "PROJECT_BASE: ${PROJECT_BASE}"
cd ${PROJECT_BASE} || exit 1
export PYTHONPATH=${PROJECT_BASE}:$PYTHONPATH

function find_free_port() {
    local start=23459
    local end=33456
    local free_port=23459
    for port in $(seq $start 100 $end)
    do
        (echo >/dev/tcp/localhost/$port) >/dev/null 2>&1
        if [[ $? -eq 1 ]]; then
            free_port=$port
            break
        fi
    done
    echo $free_port
}

# Parse --num_gpus from arguments (default: use all available GPUs)
NUM_GPUS=${HOST_GPU_NUM:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NUM_GPUS_EXPLICIT=0
for arg in "$@"; do
    if [[ "$prev_arg" == "--num_gpus" ]]; then
        NUM_GPUS=$arg
        NUM_GPUS_EXPLICIT=1
    fi
    prev_arg=$arg
done

if [ "${1}" == "--ddp" ]; then
    echo "torchrun launch (${NUM_GPUS} GPUs)"
    free_port=$(find_free_port)
    echo "Index: ${INDEX}, HostNum: ${TAIJI_HOST_NUM}, NumGPUs: ${NUM_GPUS}, LocalIP: ${LOCAL_IP}, Port: ${free_port}"
    torchrun \
        --nnodes=${TAIJI_HOST_NUM:-1} --nproc_per_node=${NUM_GPUS} --node_rank=${INDEX:-0} \
        --master_addr="${LOCAL_IP:-127.0.0.1}" --master_port=${free_port} \
        hymm/sample/sample_mova_single.py  "$@" --num-nodes ${TAIJI_HOST_NUM:-1} --node-index ${INDEX:-0}
elif [ "${1}" == "--deepspeed" ]; then
    echo "deepspeed launch (${NUM_GPUS} GPUs)"
    hostfile=/etc/taiji/hostfile
    free_port=$(find_free_port)
    if [ -f "$hostfile" ] && [ "$NUM_GPUS_EXPLICIT" -eq 0 ]; then
        deepspeed --hostfile $hostfile --master_addr "${LOCAL_IP}" --master_port $free_port \
        hymm/sample/sample_mova_single.py "$@"
    else
        deepspeed --num_gpus ${NUM_GPUS} --master_port $free_port \
        hymm/sample/sample_mova_single.py "$@"
    fi
else
    echo "Native launch (single GPU, no SP)"
    python hymm/sample/sample_mova_single.py "$@"
fi

# ============================================================================
# Example: 8 GPUs with CPU offload (recommended — saves ~30GB GPU memory)
#   bash scripts/inference/inference_mova_single.sh --deepspeed \
#       --ckpt /path/Prism/checkpoints/pretrained_models/pretrained_models/MOVA-360p \
#       --config configs/train/t2va_config/mova_infer.yaml \
#       --offload cpu \
#       --height 352 \
#       --width 640 \
#       --prompt "A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks." \
#       --audio_prompt "In a quiet, enclosed environment, a male [Speaker A] with a old man's voice speaks in a neutral, moderate tone: \"I would also say that this election in Germany wasn't surprising\"." \
#       --ref_path "./assets/single_person.jpg" \
#       --output_path "./data/samples/single_person.mp4" \
#       --seed 42
#
# Example: 1 GPU with CPU offload
#   bash scripts/inference/inference_mova_single.sh --deepspeed \
#       --num_gpus 1 \
#       --ckpt /path/to/MOVA-360p \
#       --config configs/train/t2va_config/mova_infer.yaml \
#       --offload cpu \
#       --height 352 --width 640 \
#       --prompt "A man speaks." \
#       --audio_prompt "A male speaks in a neutral tone." \
#       --ref_path "./assets/single_person.jpg" \
#       --output_path "./data/samples/output.mp4"
#
# Example: 8 GPUs without offload (needs >80GB GPU memory per card)
#   bash scripts/inference/inference_mova_single.sh --deepspeed \
#       --ckpt /path/to/MOVA-360p \
#       --config configs/train/t2va_config/mova_infer.yaml \
#       --height 352 --width 640 \
#       --prompt "..." --audio_prompt "..." --ref_path "..." \
#       --output_path "./data/samples/output.mp4"
# ============================================================================
