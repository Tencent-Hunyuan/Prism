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
for arg in "$@"; do
    if [[ "$prev_arg" == "--num_gpus" ]]; then
        NUM_GPUS=$arg
    fi
    prev_arg=$arg
done

free_port=$(find_free_port)
echo "torchrun FSDP launch (${NUM_GPUS} GPUs, port ${free_port})"

torchrun \
    --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    --master_addr="127.0.0.1" --master_port=${free_port} \
    hymm/sample/sample_mova_single_fsdp.py "$@"
