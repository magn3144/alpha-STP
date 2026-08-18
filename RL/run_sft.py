import argparse
from pathlib import Path

from utils.experiment_utils import REPO_DIR, levanter_environment, run_python


def main(args):
    storage = Path(args.storage)
    run_python(
        REPO_DIR / 'levanter/examples/weighted_lm.py',
        '--config_path', REPO_DIR / 'levanter/config/sft.yaml',
        '--model_name_or_path', args.base_model,
        '--tokenizer_name_or_path', args.base_model,
        '--trainer.checkpointer.base_path', storage / 'SFT_ckpt',
        '--hf_save_path', storage / 'SFT',
        '--train_data', storage / 'data/SFT/mathlib_leanworkbook.json',
        '--train_data_cache_dir', storage / 'data/SFT/mathlib_leanworkbook_cache',
        '--eval_data', storage / 'data/SFT/eval.json',
        '--eval_data_cache_dir', storage / 'data/SFT/eval_cache',
        cwd=REPO_DIR,
        env=levanter_environment(),
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run supervised fine-tuning with Levanter.')
    parser.add_argument('--storage', required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
