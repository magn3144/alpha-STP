#!/usr/bin/env bash

set -euo pipefail

# PyTorch batch mode, one B200; mount the persistent master-thesis folder.
ENV_SH="$(find /work -maxdepth 3 -type f -path '*/master-thesis/env.sh' -print -quit)"
test -n "$ENV_SH"
source "$ENV_SH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATA="$ALPHA_STP/data"
export OMP_NUM_THREADS=8

cd "$DELTA_PROOF"
source scripts/batch/UCloud/common_b200.sh
cd "$ALPHA_STP"

python -u -m scripts.evaluate_deltaproof_checkpoints \
    --config jobs/yaml/UCloud/eval_deltaproof_run3_B200.yaml "$@"
