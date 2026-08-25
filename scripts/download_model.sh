#!/bin/bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo_root/jobs/config.sh"

: "${STP_VENV:?Set STP_VENV in jobs/config.sh}"
: "${STP_MODULES:?Set STP_MODULES in jobs/config.sh}"
: "${STORAGE:?Set STORAGE in jobs/config.sh}"

read -r -a modules <<< "$STP_MODULES"
module load "${modules[@]}"
source "$STP_VENV/bin/activate"

model=deepseek-ai/deepseek-coder-1.3b-base
destination="$STORAGE/models/deepseek-coder-1.3b-base"

hf download "$model" --local-dir "$destination"
