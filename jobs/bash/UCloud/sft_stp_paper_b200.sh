#!/usr/bin/env bash

set -euo pipefail

ENV_SH="$(find /work -maxdepth 3 -type f -path '*/master-thesis/env.sh' -print -quit)"
test -n "$ENV_SH"
source "$ENV_SH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATA="$ALPHA_STP/data"
export WANDB_DIR="$DATA"
export TOKENIZERS_PARALLELISM=false
# Ray UNIX sockets must stay on local storage, not the UCloud /work mount.
export RAY_TMPDIR=/tmp
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# The full FP32 Adam state exceeds JAX's default 64 GiB host-transfer pool.
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=128
export OMP_NUM_THREADS=8
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=8589934592

cd "$ALPHA_STP"
source jobs/bash/UCloud/common_b200.sh

# Preserve the upstream hourly resumable checkpoints as well as the final HF export.
python -u RL/run_sft.py --training-checkpoints --config jobs/yaml/UCloud/sft_stp_paper_B200.yaml "$@"
