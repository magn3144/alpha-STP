#!/usr/bin/env bash
set -euo pipefail

# PyTorch batch mode, one B200; mount master-thesis at /work/master-thesis.
source /work/master-thesis/env.sh
cd "$ALPHA_STP"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTHONPATH="$DELTA_PROOF:$ALPHA_STP"
export HF_HUB_OFFLINE=1
trap 'status=$?; echo "RL validation loss evaluation exited with status $status"; exit "$status"' EXIT
"$DELTA_PROOF/.venv-ucloud-b200/bin/python" -u -m scripts.evaluate_deltaproof_loss \
    --config jobs/yaml/UCloud/validation_loss_deltaproof_run3_B200.yaml "$@"
