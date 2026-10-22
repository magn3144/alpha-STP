#!/usr/bin/env bash
set -euo pipefail

# CPU-only preparation; finish this stage before submitting the GPU scorer.
source /work/master-thesis/env.sh
cd "$ALPHA_STP"
export PYTHONPATH="$DELTA_PROOF:$ALPHA_STP"
"$DELTA_PROOF/.venv-ucloud-b200/bin/python" -u -m scripts.prepare_deltaproof_validation \
    --config jobs/yaml/UCloud/validation_loss_deltaproof_run3_B200.yaml "$@"
