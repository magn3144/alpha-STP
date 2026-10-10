"""Download the model and released splits used by STP's Stage 1 Lean SFT."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


MODEL = 'deepseek-ai/DeepSeek-Prover-V1.5-SFT'
MODEL_REVISION = 'e9a6e6fbb67620d4e9c4944bc51ff7c435af12da'
DATASET = 'kfdong/STP_Lean_SFT'
DATASET_REVISION = '7166a1964c466af947ae804a09857d69498b1b99'


def main(data_dir):
    model_dir = data_dir / 'models' / 'DeepSeek-Prover-V1.5-SFT'
    snapshot_download(MODEL, revision=MODEL_REVISION, local_dir=model_dir)
    dataset_dir = data_dir / 'dataset' / 'stp_paper_sft'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, filename in (('train', 'mathlib_leanworkbook.json'), ('eval', 'eval.json')):
        dataset = load_dataset(
            DATASET, revision=DATASET_REVISION, split=split,
            cache_dir=str(data_dir / 'dataset' / 'huggingface_cache'),
        )
        with (dataset_dir / filename).open('w') as output:
            json.dump(list(dataset), output)
        counts[split] = len(dataset)
        print(f'{split}: {len(dataset)} examples', flush=True)
    metadata = {
        'model': MODEL, 'model_revision': MODEL_REVISION,
        'dataset': DATASET, 'dataset_revision': DATASET_REVISION,
        'split_sizes': counts,
        'upstream_config': 'https://github.com/kfdong/STP/blob/main/levanter/config/sft.yaml',
    }
    (dataset_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Model: {model_dir}\nDataset: {dataset_dir}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    main(parser.parse_args().data_dir)
