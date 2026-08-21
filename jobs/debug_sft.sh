#!/bin/bash

set -euo pipefail

: "${DEBUG_GPU:?Set DEBUG_GPU to an unused GPU index shown by nvidia-smi}"
: "${LSB_DJOB_NUMPROC:?Run this script inside a100sh}"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export LS_SUBCWD="$repo_dir"
export PYTHONPATH="$repo_dir/levanter:$repo_dir/levanter/src:$repo_dir/levanter/examples${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

module purge
source jobs/common.sh

export CUDA_VISIBLE_DEVICES="$DEBUG_GPU"
export CUDA_LAUNCH_BLOCKING=1
export JAX_TRACEBACK_FILTERING=off
export WANDB_MODE=disabled

shared_storage="$STORAGE"
model="$shared_storage/models/AMD-Llama-135m"
fixture="$shared_storage/STP_debug_train/data/SFT/eval.json"
run_id=$(date +%Y%m%d_%H%M%S)
run_storage="$shared_storage/STP_debug_sft/$run_id"
data_dir="$run_storage/data/SFT"
login_host=hpclogin1
control_socket="$run_storage/debug-tunnel.sock"

mkdir -p "$data_dir"
cp "$fixture" "$data_dir/train.json"
cp "$fixture" "$data_dir/eval.json"

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
echo "SFT smoke-test output: $run_storage"

python -m debugpy \
    --listen 127.0.0.1:5678 \
    --wait-for-client \
    levanter/examples/weighted_lm.py \
    --config_path levanter/config/sft.yaml \
    --model_name_or_path "$model" \
    --tokenizer_name_or_path "$model" \
    --trainer.num_train_steps 24 \
    --trainer.train_batch_size 8 \
    --trainer.mp p=bfloat16,c=bfloat16 \
    --trainer.checkpointer.base_path "$run_storage/SFT_ckpt" \
    --hf_save_path "$run_storage/SFT" \
    --save_freq 24 \
    --train_data "$data_dir/train.json" \
    --train_data_cache_dir "$data_dir/train_cache" \
    --eval_data "$data_dir/eval.json" \
    --eval_data_cache_dir "$data_dir/eval_cache"
