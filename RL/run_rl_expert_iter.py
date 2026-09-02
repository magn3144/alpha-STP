import argparse
from pathlib import Path

from utils.experiment_utils import RL_DIR, run_python


def main(args):
    training_config = Path(args.training_config).resolve()
    for round_id in range(args.start_round, args.total_rounds):
        if round_id == 0:
            model = args.base_model
        else:
            model = Path(args.exp_dir) / f'round{round_id - 1}' / 'RL_model'

        round_dir = Path(args.exp_dir) / f'round{round_id}'
        print(f'Starting expert-iteration round {round_id} with model {model}', flush=True)
        run_python(
            RL_DIR / 'RL_step1_generate.py',
            '--model', model,
            '--exp_dir', round_dir,
            '--seed', round_id,
            '--temperature', args.temperature,
            '--dataset_config', args.dataset_config,
            '--sampler', 'Sampler_naive',
            '--samples_per_statement', args.samples_per_statement,
            '--statements_per_round', args.statements_per_round,
            dry_run=args.dry_run,
        )
        run_python(
            RL_DIR / 'RL_step3_final_model.py',
            '--base_model', args.base_model,
            '--exp_dir', round_dir,
            '--dataset_config', args.dataset_config,
            '--epoch', args.epochs,
            '--training_config', training_config,
            '--merge_from', args.exp_dir,
            '--merge_from_rounds', round_id + 1,
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sequential expert-iteration rounds.')
    parser.add_argument('--exp-dir', required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--dataset-config', default=RL_DIR / 'dataset_configs/leanworkbook.json')
    parser.add_argument('--start-round', type=int, default=0)
    parser.add_argument('--total-rounds', type=int, default=12)
    parser.add_argument('--samples-per-statement', type=int, default=64)
    parser.add_argument('--statements-per-round', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--training-config', required=True)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
