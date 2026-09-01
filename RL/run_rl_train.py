import argparse

from utils.experiment_utils import RL_DIR, run_python
from utils.timing_utils import configure_timing, timer


def main(args):
    configure_timing(args.exp_dir, new_session=True)
    with timer('final_training_run'):
        run_python(
            RL_DIR / 'RL_step3_final_model.py',
            '--base_model', args.train_from,
            '--exp_dir', args.exp_dir,
            '--sft_dataset', args.sft_dataset,
            '--dataset_config', args.dataset_config,
            '--epoch', args.epochs,
            '--training_config', args.training_config,
            '--include_synthetic_examples',
            '--merge_from', args.merge_from,
            '--merge_from_rounds', args.merge_from_rounds,
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the final model from self-play data.')
    parser.add_argument('--exp-dir', required=True)
    parser.add_argument('--train-from', required=True)
    parser.add_argument('--sft-dataset', required=True)
    parser.add_argument('--merge-from', required=True)
    parser.add_argument('--merge-from-rounds', type=int, required=True)
    parser.add_argument('--dataset-config', default=RL_DIR / 'dataset_configs/leanworkbook.json')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--training-config', default='levanter/config/RL_base.yaml')
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
