import argparse
from pathlib import Path

from utils.config_utils import load_experiment_config
from utils.experiment_utils import RL_DIR, run_python
from utils.timing_utils import configure_timing, timer


def main(args):
    config_path = Path(args.config).resolve()
    config = load_experiment_config(config_path, 'rl')
    experiment = config['experiment']
    training = config['training']
    if not 0 <= args.start_round < experiment['total_rounds']:
        raise ValueError('start_round must be between zero and total_rounds - 1')
    exp_dir = Path(experiment['exp_dir'])
    dataset_size = experiment['dataset_size']
    batch_size = training['trainer']['train_batch_size']
    configure_timing(exp_dir, new_session=True)
    print(
        f'Configuration: dataset_size={dataset_size}, batch_size={batch_size}',
        flush=True,
    )
    with timer('stp_run'):
        for round_id in range(args.start_round, experiment['total_rounds']):
            if round_id == 0:
                model = experiment['base_model']
            else:
                model = exp_dir / f'round{round_id - 1}' / 'RL_model'

            samples_key = 'first_round' if round_id == 0 else 'later_rounds'
            samples_per_statement = experiment['samples_per_statement'][samples_key]

            round_dir = exp_dir / f'round{round_id}'
            print(f'Starting self-play round {round_id} with model {model}', flush=True)
            with timer('round', round=round_id):
                with timer('generation_step', round=round_id):
                    run_python(
                        RL_DIR / 'RL_step1_generate.py',
                        '--model', model,
                        '--exp_dir', round_dir,
                        '--seed', round_id,
                        '--temperature', experiment['temperature'],
                        '--dataset_config', experiment['dataset_config'],
                        '--sampler', experiment['sampler'],
                        '--conjecture_multiplier', experiment['conjecture_multiplier'],
                        '--samples_per_statement', samples_per_statement,
                        '--dataset_size', dataset_size,
                        dry_run=args.dry_run,
                    )
                with timer('round_training_step', round=round_id):
                    run_python(
                        RL_DIR / 'RL_step2_train.py',
                        '--base_model', model,
                        '--exp_dir', round_dir,
                        '--epoch', experiment['epochs'],
                        '--training_config', config_path,
                        dry_run=args.dry_run,
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sequential STP self-play rounds.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--start-round', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
