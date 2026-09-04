import argparse
from pathlib import Path

from utils.config_utils import load_experiment_config
from utils.experiment_utils import RL_DIR, run_python


def main(args):
    config_path = Path(args.config).resolve()
    config = load_experiment_config(config_path, 'expert_iteration')
    experiment = config['experiment']
    if not 0 <= args.start_round < experiment['total_rounds']:
        raise ValueError('start_round must be between zero and total_rounds - 1')

    exp_dir = Path(experiment['exp_dir'])
    for round_id in range(args.start_round, experiment['total_rounds']):
        if round_id == 0:
            model = experiment['base_model']
        else:
            model = exp_dir / f'round{round_id - 1}' / 'RL_model'

        round_dir = exp_dir / f'round{round_id}'
        print(f'Starting expert-iteration round {round_id} with model {model}', flush=True)
        run_python(
            RL_DIR / 'RL_step1_generate.py',
            '--model', model,
            '--exp_dir', round_dir,
            '--seed', experiment['seed'] + round_id,
            '--temperature', experiment['temperature'],
            '--dataset_config', experiment['dataset_config'],
            '--sampler', experiment['sampler'],
            '--samples_per_statement', experiment['samples_per_statement'],
            '--dataset_size', experiment['dataset_size'],
            dry_run=args.dry_run,
        )
        run_python(
            RL_DIR / 'RL_step3_final_model.py',
            '--base_model', experiment['base_model'],
            '--exp_dir', round_dir,
            '--dataset_config', experiment['dataset_config'],
            '--epoch', experiment['epochs'],
            '--training_config', config_path,
            '--merge_from', exp_dir,
            '--merge_from_rounds', round_id + 1,
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sequential expert-iteration rounds.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--start-round', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
