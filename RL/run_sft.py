import argparse
import json
import random
from pathlib import Path

from utils.experiment_utils import REPO_DIR, levanter_environment, run_python


def prepare_eval_subset(storage: Path, size: int, seed: int) -> Path:
    data_dir = storage / 'data/SFT'
    source_path = data_dir / 'eval.json'
    subset_path = data_dir / f'eval_{size}_seed{seed}.json'

    if subset_path.exists():
        with subset_path.open() as f:
            subset = json.load(f)
        if len(subset) != size:
            raise ValueError(f'Expected {size} examples in {subset_path}, found {len(subset)}')
        return subset_path

    with source_path.open() as f:
        dataset = json.load(f)
    if size > len(dataset):
        raise ValueError(f'Cannot select {size} examples from the {len(dataset)} examples in {source_path}')

    selected_indices = sorted(random.Random(seed).sample(range(len(dataset)), size))
    subset = [dataset[index] for index in selected_indices]
    temporary_path = subset_path.with_suffix('.json.tmp')
    with temporary_path.open('w') as f:
        json.dump(subset, f)
    temporary_path.replace(subset_path)
    print(f'Created fixed validation subset with {size} examples at {subset_path}', flush=True)
    return subset_path


def main(args):
    storage = Path(args.storage)
    eval_data = storage / 'data/SFT' / f'eval_{args.eval_size}_seed{args.eval_seed}.json'
    if not args.dry_run:
        eval_data = prepare_eval_subset(storage, args.eval_size, args.eval_seed)
    run_python(
        REPO_DIR / 'levanter/examples/weighted_lm.py',
        '--config_path', REPO_DIR / 'levanter/config/sft.yaml',
        '--model_name_or_path', args.base_model,
        '--tokenizer_name_or_path', args.base_model,
        '--trainer.checkpointer.base_path', storage / 'SFT_ckpt',
        '--hf_save_path', storage / 'SFT',
        '--train_data', storage / 'data/SFT/mathlib_leanworkbook.json',
        '--train_data_cache_dir', storage / 'data/SFT/mathlib_leanworkbook_cache',
        '--eval_data', eval_data,
        '--eval_data_cache_dir', storage / f'data/SFT/eval_{args.eval_size}_seed{args.eval_seed}_cache',
        cwd=REPO_DIR,
        env=levanter_environment(),
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run supervised fine-tuning with Levanter.')
    parser.add_argument('--storage', required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--eval-size', type=int, default=1024)
    parser.add_argument('--eval-seed', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
