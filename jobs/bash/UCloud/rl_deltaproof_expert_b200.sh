#!/usr/bin/env bash

set -euo pipefail

ENV_SH="$(find /work -maxdepth 3 -type f -path '*/master-thesis/env.sh' -print -quit)"
test -n "$ENV_SH"
source "$ENV_SH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATA="$ALPHA_STP/data"

cd "$ALPHA_STP"
source jobs/bash/UCloud/common_b200.sh

# The checked-in YAML retains its non-B200 interpreter path.
(cd "$DELTA_PROOF" && XLA_PYTHON_CLIENT_PREALLOCATE=false .venv-ucloud-b200/bin/python -m scripts.batch.UCloud.preflight_b200)
b200_config="$(mktemp --suffix=.yaml)"
trap 'rm -f "$b200_config"' EXIT
sed 's|/.venv/bin/python|/.venv-ucloud-b200/bin/python|' \
    jobs/yaml/UCloud/rl_deltaproof_expert_B200.yaml > "$b200_config"

start_round="${START_ROUND:-0}"

python -u RL/run_rl_steps.py \
    --config "$b200_config" \
    --start-round "$start_round"
