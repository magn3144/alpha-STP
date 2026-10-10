#!/usr/bin/env bash

set -euo pipefail

ENV_SH="$(find /work -maxdepth 3 -type f -path '*/master-thesis/env.sh' -print -quit)"
test -n "$ENV_SH"
source "$ENV_SH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATA="$ALPHA_STP/data"
export TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR=/tmp
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=128
export OMP_NUM_THREADS=8
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=8589934592

cd "$ALPHA_STP"
source jobs/bash/UCloud/common_b200.sh
python -u -m scripts.benchmark_stp_sft_b200 \
    --directory "$DATA/runs/batch_benchmark_stp_b200/$(date -u +%Y%m%dT%H%M%SZ)" "$@"
