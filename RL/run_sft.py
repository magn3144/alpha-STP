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
    run_dir = Path(experiment['run_dir'])
    if args.resume_run_id:
        training['trainer']['id'] = args.resume_run_id
        training['trainer']['load_checkpoint'] = True
        training['trainer']['load_checkpoint_path'] = str(run_dir / 'checkpoints' / args.resume_run_id)
    base_model = Path(experiment['base_model'])
    train_data = Path(experiment['train_data'])
    validation_data = Path(experiment['validation_data'])
    if not args.dry_run:
        data_paths = [train_data] if args.no_eval else [train_data, validation_data]
        missing = [path for path in data_paths if not path.exists()]
        if missing:
            paths = ', '.join(str(path) for path in missing)
            raise FileNotFoundError(f'Missing generated SFT split(s): {paths}. Prepare the configured datasets first.')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as config_file:
        if args.cache_only:
            training['cache_only'] = True
        yaml.safe_dump(training, config_file)
        config_file.flush()

        cache_name = f'{base_model.name}_{training["max_tune_length"]}'
        cache_dir = train_data.parent / 'cache' / cache_name
        eval_args = [] if args.no_eval else [
            '--eval_data', validation_data,
            '--eval_data_cache_dir', cache_dir / 'validation',
        ]

        run_python(
            REPO_DIR / 'levanter/examples/weighted_lm.py',
            '--config_path', config_file.name,
            '--model_name_or_path', base_model,
            '--tokenizer_name_or_path', base_model,
            '--trainer.checkpointer.base_path', run_dir / 'checkpoints',
            '--hf_save_path', run_dir / 'models',
            '--train_data', train_data,
            '--train_data_cache_dir', cache_dir / 'train',
            *eval_args,
            cwd=REPO_DIR,
            env=levanter_environment(),
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run supervised fine-tuning with Levanter.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume-run-id', help='Resume an existing run, requiring its saved checkpoint.')
    parser.add_argument('--cache-only', action='store_true')
    parser.add_argument('--no-eval', action='store_true', help='Train without an evaluation split.')
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
