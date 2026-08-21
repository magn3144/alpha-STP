import json
import os
import random

from datasets import load_dataset

from utils.experiment_utils import REPO_DIR
from utils.file_utils import write_data


DATASET_NAME = "kfdong/STP_Lean_SFT"
SPLIT_SEED = 0
TRAIN_SIZE = 20_096
VALIDATION_SIZE = 1_024
TEST_SIZE = 4_096


def example_key(example):
    return example['prompt']


def take_unique(rows, size, excluded_keys):
    selected = []
    for example in rows:
        key = example_key(example)
        if key in excluded_keys:
            continue
        excluded_keys.add(key)
        selected.append(example)
        if len(selected) == size:
            return selected
    raise ValueError(f'Could only select {len(selected)} unique examples; requested {size}')


if __name__ == '__main__':
    storage = str(REPO_DIR / 'storage')
    huggingface_cache = os.path.join(storage, 'huggingface_cache')
    print(f'Saving downloaded and prepared datasets under {storage}')
    sft_train_dataset = load_dataset(DATASET_NAME, split="train", cache_dir=huggingface_cache)
    sft_eval_dataset = load_dataset(DATASET_NAME, split="eval", cache_dir=huggingface_cache)
    full_train_dataset = [example for example in sft_train_dataset]
    full_eval_dataset = [example for example in sft_eval_dataset]

    # Preserve the released full splits for final retraining and optional full evaluation.
    print(f'Number of examples in the full SFT training split: {len(full_train_dataset)}')
    write_data(json.dumps(full_train_dataset), os.path.join(storage, 'data/SFT/mathlib_leanworkbook.json'), 'json', no_compression=True)
    write_data(json.dumps(full_eval_dataset), os.path.join(storage, 'data/SFT/eval.json'), 'json', no_compression=True)

    # create mathlib dataset
    mathlib_dataset = [example for example in full_train_dataset if 'lean_workbook' not in example['prompt']]
    print(f'Number of examples in the mathlib dataset: {len(mathlib_dataset)}')
    write_data(json.dumps(mathlib_dataset), os.path.join(storage, 'data/SFT/mathlib.json'), 'json', no_compression=True)

    # Deterministically shuffle and extract prompt-disjoint experiment splits.
    shuffled_train = full_train_dataset.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled_train)
    used_keys = set()
    train_dataset = take_unique(shuffled_train, TRAIN_SIZE, used_keys)
    test_dataset = take_unique(shuffled_train, TEST_SIZE, used_keys)

    shuffled_eval = full_eval_dataset.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled_eval)
    validation_dataset = take_unique(shuffled_eval, VALIDATION_SIZE, used_keys)

    train_keys = {example_key(example) for example in train_dataset}
    validation_keys = {example_key(example) for example in validation_dataset}
    test_keys = {example_key(example) for example in test_dataset}
    assert train_keys.isdisjoint(validation_keys)
    assert train_keys.isdisjoint(test_keys)
    assert validation_keys.isdisjoint(test_keys)

    data_dir = os.path.join(storage, 'data/SFT')
    write_data(json.dumps(train_dataset), os.path.join(data_dir, 'train.json'), 'json', no_compression=True)
    write_data(json.dumps(validation_dataset), os.path.join(data_dir, 'validation.json'), 'json', no_compression=True)
    write_data(json.dumps(test_dataset), os.path.join(data_dir, 'test.json'), 'json', no_compression=True)

    metadata = {
        'dataset': DATASET_NAME,
        'seed': SPLIT_SEED,
        'source_train_size': len(full_train_dataset),
        'source_eval_size': len(full_eval_dataset),
        'train_size': len(train_dataset),
        'validation_size': len(validation_dataset),
        'test_size': len(test_dataset),
        'test_source_split': 'train',
        'validation_source_split': 'eval',
        'overlap_key': 'prompt',
    }
    write_data(json.dumps(metadata, indent=2), os.path.join(data_dir, 'split_metadata.json'), 'json', no_compression=True)
    print(f'Created disjoint SFT splits: train={len(train_dataset)}, validation={len(validation_dataset)}, test={len(test_dataset)}')
