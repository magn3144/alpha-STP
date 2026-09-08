#!/bin/bash
# Run inside a100sh, then attach using the VS Code debug configuration.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
module load python3/3.10.18
module load cuda/12.6.3
source .venv/bin/activate

export STORAGE="$PWD/storage"
export DELTA_PROOF="$PWD/../delta-proof"
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false

hostname
nvidia-smi -i 1
echo "Waiting for VS Code on $(hostname):5678. Use this hostname in the debug configuration."
exec python -u -m debugpy --listen "$(hostname):5678" --wait-for-client \
    --configure-subProcess true RL/run_rl_steps.py \
    --config jobs/yaml/rl_deltaproof_stp_debug_a100.yaml "$@"
