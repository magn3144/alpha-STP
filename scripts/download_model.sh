#!/bin/bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
module load python3/3.10.18
source "$repo_root/.venv/bin/activate"

model=deepseek-ai/deepseek-coder-1.3b-base
destination="$repo_root/storage/models/deepseek-coder-1.3b-base"

hf download "$model" --local-dir "$destination"
