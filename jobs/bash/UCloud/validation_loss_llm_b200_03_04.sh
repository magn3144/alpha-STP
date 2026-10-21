#!/usr/bin/env bash
set -euo pipefail
source /work/master-thesis/env.sh
cd "$ALPHA_STP"
source .venv-ucloud-b200/bin/activate
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
trap 'status=$?; echo "Validation loss evaluation exited with status $status"; exit "$status"' EXIT
python -u -m scripts.evaluate_checkpoint_loss --config jobs/yaml/UCloud/validation_loss_llm_b200_03_04.yaml
