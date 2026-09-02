import argparse
import tempfile
from pathlib import Path

import yaml

from utils.config_utils import load_experiment_config
from utils.experiment_utils import REPO_DIR, levanter_environment, run_python


def main(args):
    config = load_experiment_config(args.config, 'sft')
    experiment = config['experiment']
    training = config['training']
    storage = Path(experiment['storage'])
    base_model = Path(experiment['base_model'])
    train_data = storage / 'data/SFT/train.json'
    validation_data = storage / 'data/SFT/validation.json'
    if not args.dry_run:
        missing = [path for path in (train_data, validation_data) if not path.exists()]
        if missing:
            paths = ', '.join(str(path) for path in missing)
            raise FileNotFoundError(f'Missing generated SFT split(s): {paths}. Run RL/prepare_datasets.py first.')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as config_file:
        if args.cache_only:
            training['cache_only'] = True
        yaml.safe_dump(training, config_file)
        config_file.flush()

        cache_name = f'{base_model.name}_{training["max_tune_length"]}'
        cache_dir = storage / 'data/SFT/cache' / cache_name

        run_python(
            REPO_DIR / 'levanter/examples/weighted_lm.py',
            '--config_path', config_file.name,
            '--model_name_or_path', base_model,
            '--tokenizer_name_or_path', base_model,
            '--trainer.checkpointer.base_path', storage / 'SFT_ckpt',
            '--hf_save_path', storage / 'SFT',
            '--train_data', train_data,
            '--train_data_cache_dir', cache_dir / 'train',
            '--eval_data', validation_data,
            '--eval_data_cache_dir', cache_dir / 'validation',
            cwd=REPO_DIR,
            env=levanter_environment(),
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run supervised fine-tuning with Levanter.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--cache-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
