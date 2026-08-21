#!/bin/bash

set -euo pipefail

: "${DEBUG_GPU:?Set DEBUG_GPU to an unused GPU index shown by nvidia-smi}"
: "${LSB_DJOB_NUMPROC:?Run this script inside a100sh}"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export LS_SUBCWD="$repo_dir"
export PYTHONPATH="$repo_dir/RL${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

module purge
source jobs/common.sh

export CUDA_VISIBLE_DEVICES="$DEBUG_GPU"
export CUDA_LAUNCH_BLOCKING=1
export DEBUG=1
export JAX_TRACEBACK_FILTERING=off
export WANDB_MODE=disabled

shared_storage="$STORAGE"
model="$shared_storage/models/AMD-Llama-135m"
fixture_dir="$shared_storage/STP_debug_final_model/fixture"
merge_from="$fixture_dir"
run_id=$(date +%Y%m%d_%H%M%S)
exp_dir="$shared_storage/STP_debug_final_model/$run_id"
run_storage="$exp_dir/storage"
login_host=hpclogin1
control_socket="$exp_dir/debug-tunnel.sock"
mkdir -p "$run_storage/data/SFT"
cp "$shared_storage/STP_debug_train/data/SFT/eval.json" "$run_storage/data/SFT/eval.json"
export STORAGE="$run_storage"

echo "Enter your DTU password to create the debug tunnel to $login_host."
ssh -M -S "$control_socket" -fN \
    -o ExitOnForwardFailure=yes \
    -o IdentityAgent=none \
    -o PubkeyAuthentication=no \
    -o PreferredAuthentications=password \
    -R 127.0.0.1:5678:127.0.0.1:5678 \
    "$login_host"
trap 'ssh -S "$control_socket" -O exit "$login_host" >/dev/null 2>&1' EXIT

echo "Waiting for VS Code on hpclogin1:5678"
echo "Debug output: $exp_dir"

python -m debugpy \
    --listen 127.0.0.1:5678 \
    --wait-for-client \
    RL/RL_step3_final_model.py \
    --base_model "$model" \
    --exp_dir "$exp_dir" \
    --sft_dataset "$fixture_dir/sft.json" \
    --dataset_config RL/dataset_configs/leanworkbook.json \
    --epoch 1 \
    --batch_size 8 \
    --lr 1e-4 \
    --include_synthetic_examples \
    --merge_from "$merge_from" \
    --merge_from_rounds 1
