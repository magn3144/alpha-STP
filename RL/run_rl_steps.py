import argparse
from pathlib import Path

from utils.experiment_utils import RL_DIR, run_python
from utils.timing_utils import configure_timing, timer


def main(args):
    configure_timing(args.exp_dir, new_session=True)
    print(
        f'Configuration: dataset_size={args.dataset_size}, batch_size={args.batch_size}',
        flush=True,
    )
    with timer('stp_run'):
        for round_id in range(args.start_round, args.total_rounds):
            if round_id == 0:
                model = args.base_model
            else:
                model = Path(args.exp_dir) / f'round{round_id - 1}' / 'RL_model'

            samples_per_statement = args.samples_per_statement
            if samples_per_statement is None:
                samples_per_statement = 32 if round_id == 0 else 16

            round_dir = Path(args.exp_dir) / f'round{round_id}'
            print(f'Starting self-play round {round_id} with model {model}', flush=True)
            with timer('round', round=round_id):
                with timer('generation_step', round=round_id):
                    run_python(
                        RL_DIR / 'RL_step1_generate.py',
                        '--model', model,
                        '--exp_dir', round_dir,
                        '--seed', round_id,
                        '--temperature', args.temperature,
                        '--dataset_config', args.dataset_config,
                        '--sampler', 'Sampler_base',
                        '--conjecture_multiplier', 1,
                        '--samples_per_statement', samples_per_statement,
                        '--dataset_size', args.dataset_size,
                        dry_run=args.dry_run,
                    )
                with timer('round_training_step', round=round_id):
                    run_python(
                        RL_DIR / 'RL_step2_train.py',
                        '--base_model', model,
                        '--exp_dir', round_dir,
                        '--epoch', args.epochs,
                        '--batch_size', args.batch_size,
                        '--training_config', args.training_config,
                        dry_run=args.dry_run,
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sequential STP self-play rounds.')
    parser.add_argument('--exp-dir', required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--dataset-config', default=RL_DIR / 'dataset_configs/leanworkbook.json')
    parser.add_argument('--start-round', type=int, default=0)
    parser.add_argument('--total-rounds', type=int, default=12)
    parser.add_argument('--dataset-size', type=int, default=0)
    parser.add_argument('--samples-per-statement', type=int)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--training-config', default='levanter/config/RL_base.yaml')
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
