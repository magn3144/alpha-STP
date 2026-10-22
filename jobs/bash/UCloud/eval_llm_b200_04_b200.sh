#!/usr/bin/env bash

set -euo pipefail

# Terminal batch processing, one B200; mount the persistent master-thesis folder.
ENV_SH="$(find /work -maxdepth 3 -type f -path '*/master-thesis/env.sh' -print -quit)"
test -n "$ENV_SH"
source "$ENV_SH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATA="$ALPHA_STP/data"

cd "$ALPHA_STP"
source jobs/bash/UCloud/common_rl_b200.sh

python -u -m scripts.evaluate_llm_checkpoints \
    --config jobs/yaml/UCloud/eval_llm_b200_04_B200.yaml "$@"
