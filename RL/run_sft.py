import argparse
import tempfile
from pathlib import Path

import yaml

from utils.experiment_utils import REPO_DIR, levanter_environment, run_python


def merge_configs(base, override):
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path):
    with open(path) as config_file:
        return yaml.safe_load(config_file) or {}


def main(args):
    storage = Path(args.storage)
    base_model = args.base_model or storage / 'models/deepseek-coder-1.3b-base'
    train_data = storage / 'data/SFT/train.json'
    validation_data = storage / 'data/SFT/validation.json'
    if not args.dry_run:
        missing = [path for path in (train_data, validation_data) if not path.exists()]
        if missing:
            paths = ', '.join(str(path) for path in missing)
            raise FileNotFoundError(f'Missing generated SFT split(s): {paths}. Run RL/prepare_datasets.py first.')

    base_config_path = REPO_DIR / 'levanter/config/sft.yaml'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as config_file:
        config = load_config(base_config_path)
        if args.config:
            config = merge_configs(config, load_config(args.config))
        yaml.safe_dump(config, config_file)
        config_file.flush()

        run_python(
            REPO_DIR / 'levanter/examples/weighted_lm.py',
            '--config_path', config_file.name,
            '--model_name_or_path', base_model,
            '--tokenizer_name_or_path', base_model,
            '--trainer.checkpointer.base_path', storage / 'SFT_ckpt',
            '--hf_save_path', storage / 'SFT',
            '--train_data', train_data,
            '--train_data_cache_dir', storage / 'data/SFT/train_cache',
            '--eval_data', validation_data,
            '--eval_data_cache_dir', storage / 'data/SFT/validation_cache',
            cwd=REPO_DIR,
            env=levanter_environment(),
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run supervised fine-tuning with Levanter.')
    parser.add_argument('--storage', required=True)
    parser.add_argument('--base-model')
    parser.add_argument('--config', help='Job-specific YAML values to merge over levanter/config/sft.yaml')
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
