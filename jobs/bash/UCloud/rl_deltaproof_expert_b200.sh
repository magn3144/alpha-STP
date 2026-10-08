#!/usr/bin/env bash

set -euo pipefail

source /work/master-thesis/env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATA="$ALPHA_STP/data"

cd "$ALPHA_STP"
source "$ALPHA_STP/.venv/bin/activate"

nvidia-smi -L
python --version

start_round="${START_ROUND:-0}"

python -u RL/run_rl_steps.py \
    --config jobs/yaml/UCloud/rl_deltaproof_expert_B200.yaml \
    --start-round "$start_round"
