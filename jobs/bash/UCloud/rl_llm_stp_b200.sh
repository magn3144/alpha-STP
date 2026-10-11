#!/usr/bin/env bash

set -euo pipefail

ENV_SH="$(find /work -maxdepth 3 -type f -path '*/master-thesis/env.sh' -print -quit)"
test -n "$ENV_SH"
source "$ENV_SH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATA="$ALPHA_STP/data"

cd "$ALPHA_STP"
source jobs/bash/UCloud/common_rl_b200.sh

if [[ "${1:-}" == --smoke-test ]]; then
    shift
    exec python -u -m scripts.smoke_stp_llm_ucloud "$@"
fi

start_round="${START_ROUND:-0}"

python -u RL/run_rl_steps.py \
    --config jobs/yaml/UCloud/rl_llm_stp_B200.yaml \
    --start-round "$start_round" "$@"
