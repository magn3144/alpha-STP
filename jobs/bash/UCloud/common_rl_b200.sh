#!/usr/bin/env bash

set -euo pipefail

trap 'status=$?; if (( status == 0 )); then echo "STP experiment succeeded."; else echo "STP experiment FAILED (exit $status)." >&2; fi; exit "$status"' EXIT

# Supply the allocation expected by the existing Ray initializer.
export LSB_DJOB_NUMPROC="$(python3 -c 'import json; print(json.load(open("/work/JobParameters.json"))["machineType"]["cpu"])')"
export RAY_TMPDIR=/tmp
export OMP_NUM_THREADS=8
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=8589934592
# Bound Lean verification RAM independently of the GPU training batch.
export STP_LEAN_MEMORY_LIMIT_GB=16
export STP_LEAN_CPUS_PER_TASK=3

# All UCloud RL LLM components use the same 7B SFT model.
export STP_EVAL_DATA_CACHE_DIR="$DATA/dataset/prover_sft/eval_cache-deepseek-prover-v1.5"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# Leave room for the 7B optimizer checkpoint and HF export.
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=96

source jobs/bash/UCloud/common_b200.sh
