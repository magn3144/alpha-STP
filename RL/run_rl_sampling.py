import argparse
from pathlib import Path

from utils.config_utils import load_experiment_config
from utils.experiment_utils import RL_DIR, run_python


def main(args):
    config = load_experiment_config(args.config, 'parallel_sampling')
    experiment = config['experiment']
    if not 0 <= args.start_round < experiment['total_rounds']:
        raise ValueError('start_round must be between zero and total_rounds - 1')

    exp_dir = Path(experiment['exp_dir'])
    for round_id in range(args.start_round, experiment['total_rounds']):
        round_dir = exp_dir / f'round{round_id}'
        print(f'Starting sampling round {round_id} with model {experiment["base_model"]}', flush=True)
        run_python(
            RL_DIR / 'RL_step1_generate.py',
            '--model', experiment['base_model'],
            '--exp_dir', round_dir,
            '--seed', experiment['seed'] + round_id,
            '--temperature', experiment['temperature'],
            '--dataset_config', experiment['dataset_config'],
            '--sampler', experiment['sampler'],
            '--samples_per_statement', experiment['samples_per_statement'],
            '--dataset_size', experiment['dataset_size'],
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample proofs independently across rounds.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--start-round', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
