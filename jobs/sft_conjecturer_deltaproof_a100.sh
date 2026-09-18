#!/bin/bash
#BSUB -J sft-conjecturer-cap5
#BSUB -q gpua100
#BSUB -W 2:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -oo jobs/logs/sft_conjecturer_deltaproof_%J.out
#BSUB -eo jobs/logs/sft_conjecturer_deltaproof_%J.err

source "$LS_SUBCWD/jobs/common.sh"

# Keep a shuffled input copy and isolated tokenization caches for this run.
python - <<'PYTHON'
import json
import os
import random
import shutil
from pathlib import Path

source = Path(os.environ['DATA']) / 'dataset/conjecturer_sft_deltaproof'
output = Path(os.environ['DATA']) / 'runs/sft_conjecturer_deltaproof_deepseek_coder_1.3b_a100/input'
output.mkdir(parents=True, exist_ok=True)
examples = json.loads((source / 'train.json').read_text())
assert len(examples) == 1393, 'Dataset size changed; review the configured training steps.'
random.Random(42).shuffle(examples)
(output / 'train.json').write_text(json.dumps(examples, ensure_ascii=False) + '\n')
shutil.copyfile(source / 'validation.json', output / 'validation.json')
shutil.copyfile(source / 'metadata.json', output / 'source_metadata.json')
PYTHON

python -u RL/run_sft.py \
    --config jobs/yaml/sft_conjecturer_deltaproof_A100_80GB.yaml
