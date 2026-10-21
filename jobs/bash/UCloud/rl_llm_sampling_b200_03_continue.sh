#!/usr/bin/env bash

set -euo pipefail

ENV_SH="$(find /work -maxdepth 3 -type f -path '*/master-thesis/env.sh' -print -quit)"
test -n "$ENV_SH"
source "$ENV_SH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATA="$ALPHA_STP/data"

cd "$ALPHA_STP"
source jobs/bash/UCloud/common_rl_b200.sh

# Resume the first unfinished round; completed rounds are skipped.
start_round="${START_ROUND:-6}"

python -u RL/run_rl_steps.py \
    --config jobs/yaml/UCloud/rl_llm_sampling_b200_03_continue.yaml \
    --start-round "$start_round" "$@"
