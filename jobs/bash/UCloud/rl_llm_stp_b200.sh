#!/usr/bin/env bash

set -euo pipefail

ENV_SH="$(find /work -maxdepth 3 -type f -path '*/master-thesis/env.sh' -print -quit)"
test -n "$ENV_SH"
source "$ENV_SH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATA="$ALPHA_STP/data"
export STP_EVAL_DATA_CACHE_DIR="$DATA/dataset/prover_sft/eval_cache-deepseek-prover-v1.5"
# Supply the CPU allocation expected by the existing Ray initializer.
export LSB_DJOB_NUMPROC="$(python3 -c 'import json; print(json.load(open("/work/JobParameters.json"))["machineType"]["cpu"])')"
export RAY_TMPDIR=/tmp
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# The 7B optimizer checkpoint needs more than 64 GiB; leave room for HF export.
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=96
export OMP_NUM_THREADS=8
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=8589934592

cd "$ALPHA_STP"
source jobs/bash/UCloud/common_b200.sh

if [[ "${1:-}" == --smoke-test ]]; then
    shift
    exec python -u -m scripts.smoke_stp_llm_ucloud "$@"
fi

start_round="${START_ROUND:-0}"

python -u RL/run_rl_steps.py \
    --config jobs/yaml/UCloud/rl_llm_stp_B200.yaml \
    --start-round "$start_round" "$@"
