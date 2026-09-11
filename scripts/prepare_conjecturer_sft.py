import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc


REPO_DIR = Path(__file__).resolve().parents[1]
REVISION = '7166a1964c466af947ae804a09857d69498b1b99'


def main():
    parser = argparse.ArgumentParser(description='Split the released STP SFT conjecturer examples.')
    parser.add_argument(
        '--source-dir', type=Path,
        default=REPO_DIR / 'data/dataset/huggingface_cache/kfdong___stp_lean_sft/default/0.0.0' / REVISION,
    )
    parser.add_argument('--output-dir', type=Path, default=REPO_DIR / 'data/dataset/conjecturer_sft')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--validation-size', type=int, default=2000)
    args = parser.parse_args()

    examples = []
    source_rows = 0
    for index in range(5):
        path = args.source_dir / f'stp_lean_sft-train-{index:05d}-of-00005.arrow'
        with pa.memory_map(str(path)) as source:
            for batch in ipc.open_stream(source):
                source_rows += batch.num_rows
                mask = pc.ends_with(batch.column('prompt'), '<hard theorem>')
                assert pc.all(pc.equal(mask, pc.match_substring(batch.column('prompt'), '<hard theorem>'))).as_py()
                examples.extend(batch.filter(mask).to_pylist())

    assert source_rows == 235430
    assert len(examples) == 51273
    assert all(example['target'].endswith('</hard theorem>') for example in examples)
    assert all('<easy theorem>' in example['prompt'] and '<lemma>' in example['prompt'] for example in examples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'all.json').open('w') as output:
        json.dump(examples, output, ensure_ascii=False)
        output.write('\n')

    split_examples = examples.copy()
    random.Random(0).shuffle(split_examples)
    train_size = (len(split_examples) - args.validation_size) // args.batch_size * args.batch_size
    assert train_size > 0
    train_examples = split_examples[:train_size]
    validation_examples = split_examples[train_size:]
    with (args.output_dir / 'train.json').open('w') as output:
        json.dump(train_examples, output, ensure_ascii=False)
        output.write('\n')

    with (args.output_dir / 'validation.json').open('w') as output:
        json.dump(validation_examples, output, ensure_ascii=False)
        output.write('\n')

    metadata = {
        'dataset': 'kfdong/STP_Lean_SFT',
        'revision': REVISION,
        'source_split': 'train',
        'source_rows': source_rows,
        'conjecturer_examples': len(examples),
        'selection': 'prompt ends with <hard theorem>',
        'format': 'All original examples retained without deduplication or batch padding.',
        'seed': 0,
        'batch_size': args.batch_size,
        'repeated_examples_for_final_batch': 0,
        'train_rows': len(train_examples),
        'validation_rows': len(validation_examples),
        'optimizer_steps_per_epoch': len(train_examples) // args.batch_size,
        'evaluation': 'Random sample held out from the released training split.',
    }
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
