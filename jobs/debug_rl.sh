#!/bin/bash

set -euo pipefail

: "${DEBUG_GPU:?Set DEBUG_GPU to an unused GPU index shown by nvidia-smi}"
: "${LSB_DJOB_NUMPROC:?Run this script inside a100sh}"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export LS_SUBCWD="$repo_dir"
cd "$repo_dir"

module purge
source jobs/common.sh

export CUDA_VISIBLE_DEVICES="$DEBUG_GPU"
export CUDA_LAUNCH_BLOCKING=1
export DEBUG=1
export JAX_TRACEBACK_FILTERING=off
export WANDB_MODE=offline

model="$STORAGE/models/AMD-Llama-135m"
run_id=$(date +%Y%m%d_%H%M%S)
exp_dir="$STORAGE/STP_debug_amd135m/$run_id"
login_host=hpclogin1
control_socket="$exp_dir/debug-tunnel.sock"
mkdir -p "$exp_dir"

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
    RL/run_rl_steps.py \
    --exp-dir "$exp_dir" \
    --base-model "$model" \
    --start-round 0 \
    --total-rounds 1 \
    --statements-per-round 1 \
    --samples-per-statement 1
