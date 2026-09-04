import argparse
import json
import random
import urllib.request
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
DELTA_PROOF_DIR = REPO_DIR.parent / 'delta-proof'
LEANTREE_URL = (
    'https://huggingface.co/datasets/ufal/leantree/resolve/main/'
    'leantree_mathlib.jsonl'
)
INSTRUCTION = 'Complete the following Lean 4 code:\n\n```lean4\n'
CONJECTURE_MARKER = '<hard theorem>'
ORIGINAL_PROOF_EXAMPLES = 14_757
ORIGINAL_CONJECTURE_EXAMPLES = 5_339


def download_leantree(path):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    urllib.request.urlretrieve(LEANTREE_URL, temporary_path)
    temporary_path.replace(path)


def line_count(path):
    with path.open('rb') as input_file:
        return sum(1 for _ in input_file)


def whole_proof(theorem, source):
    if theorem.get('error') is not None or theorem['span'] is None:
        return None

    theorem_start = theorem['span']['start']
    theorem_finish = theorem['span']['finish']
    for block in theorem['by_blocks']:
        block_span = block.get('span')
        tree = block.get('tree', {})
        if (
            block_span is None
            or tree.get('error') is not None
            or block_span['finish'] != theorem_finish
        ):
            continue
        marker = source.rfind(':= by', theorem_start, block_span['start'])
        if marker == -1:
            continue
        marker_finish = marker + len(':= by')
        if source[marker_finish:block_span['start']].strip():
            continue
        return {
            'prompt': INSTRUCTION + source[:marker_finish],
            'target': source[marker_finish:theorem_finish],
        }
    return None


def iter_whole_proofs(dataset_path, mathlib_packages, validation_start):
    with dataset_path.open(encoding='utf-8') as dataset_file:
        for file_index, line in enumerate(dataset_file):
            record = json.loads(line)
            source = (mathlib_packages / record['path']).read_text(encoding='utf-8')
            split = 'validation' if file_index >= validation_start else 'train'
            for theorem_index, theorem in enumerate(record['theorems']):
                example = whole_proof(theorem, source)
                if example is not None:
                    yield split, (file_index, theorem_index), example


def select_examples(dataset_path, mathlib_packages, validation_start, fraction, seed):
    candidate_ids = {'train': [], 'validation': []}
    for split, identity, _ in iter_whole_proofs(
        dataset_path, mathlib_packages, validation_start
    ):
        candidate_ids[split].append(identity)

    selected = {}
    for split, identities in candidate_ids.items():
        count = int(len(identities) * fraction)
        selected[split] = set(random.Random(seed).sample(identities, count))
    return candidate_ids, selected


def write_sft_splits(
    dataset_path,
    mathlib_packages,
    output_dir,
    validation_start,
    selected,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = {
        'train': 'proof_train.json',
        'validation': 'validation.json',
    }
    temporary_paths = {
        split: output_dir / f'{output_names[split]}.tmp'
        for split in selected
    }
    output_files = {
        split: path.open('w', encoding='utf-8')
        for split, path in temporary_paths.items()
    }
    first = {split: True for split in selected}
    try:
        for output_file in output_files.values():
            output_file.write('[')
        for split, identity, example in iter_whole_proofs(
            dataset_path, mathlib_packages, validation_start
        ):
            if identity not in selected[split]:
                continue
            output_file = output_files[split]
            if not first[split]:
                output_file.write(',')
            json.dump(example, output_file, ensure_ascii=False)
            first[split] = False
        for output_file in output_files.values():
            output_file.write(']\n')
    finally:
        for output_file in output_files.values():
            output_file.close()

    for split, temporary_path in temporary_paths.items():
        temporary_path.replace(output_dir / output_names[split])


def write_mixed_sft_train(proof_path, conjecture_path, output_path, seed):
    with proof_path.open(encoding='utf-8') as proof_file:
        proofs = json.load(proof_file)
    with conjecture_path.open(encoding='utf-8') as conjecture_file:
        conjecture_pool = [
            example
            for example in json.load(conjecture_file)
            if CONJECTURE_MARKER in example['prompt']
        ]

    conjecture_count = round(
        len(proofs) * ORIGINAL_CONJECTURE_EXAMPLES / ORIGINAL_PROOF_EXAMPLES
    )
    conjectures = random.Random(seed).sample(conjecture_pool, conjecture_count)
    examples = proofs + conjectures
    random.Random(seed).shuffle(examples)

    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as output_file:
        json.dump(examples, output_file, ensure_ascii=False)
        output_file.write('\n')
    temporary_path.replace(output_path)
    return len(proofs), len(conjectures), len(conjecture_pool)


def write_rl_dataset(input_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    count = 0
    with input_path.open(encoding='utf-8') as input_file, temporary_path.open(
        'w', encoding='utf-8'
    ) as output_file:
        output_file.write('[')
        for line in input_file:
            record = json.loads(line)
            if count:
                output_file.write(',')
            json.dump(
                {
                    'formal_statement': record['theorem'],
                    'split': 'numina_math_lean_passing_small',
                    'proof': [],
                },
                output_file,
                ensure_ascii=False,
            )
            count += 1
        output_file.write(']\n')
    temporary_path.replace(output_path)
    return count


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare STP datasets from the data used by DeltaProof.'
    )
    parser.add_argument(
        '--leantree-dataset',
        type=Path,
        default=REPO_DIR / 'storage/DeltaProof/data/leantree_mathlib.jsonl',
    )
    parser.add_argument(
        '--mathlib-packages',
        type=Path,
        default=REPO_DIR / 'storage/DeltaProof/source',
    )
    parser.add_argument(
        '--rl-input',
        type=Path,
        default=(
            DELTA_PROOF_DIR
            / 'data/dataset/numina_math_lean_passing_small/train.jsonl'
        ),
    )
    parser.add_argument(
        '--conjecture-input',
        type=Path,
        default=REPO_DIR / 'storage/data/SFT/train.json',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=REPO_DIR / 'storage/DeltaProof/data',
    )
    parser.add_argument('--fraction', type=float, default=0.25)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    download_leantree(args.leantree_dataset)
    total_files = line_count(args.leantree_dataset)
    validation_files = int(total_files * args.validation_fraction)
    validation_start = total_files - validation_files
    candidate_ids, selected = select_examples(
        args.leantree_dataset,
        args.mathlib_packages,
        validation_start,
        args.fraction,
        args.seed,
    )
    sft_output_dir = args.output_dir / 'SFT'
    write_sft_splits(
        args.leantree_dataset,
        args.mathlib_packages,
        sft_output_dir,
        validation_start,
        selected,
    )
    proof_count, conjecture_count, conjecture_pool_count = write_mixed_sft_train(
        sft_output_dir / 'proof_train.json',
        args.conjecture_input,
        sft_output_dir / 'train.json',
        args.seed,
    )
    rl_output_path = args.output_dir / 'RL/numina_math_lean_passing_small.json'
    rl_count = write_rl_dataset(args.rl_input, rl_output_path)

    metadata = {
        'sft_source': LEANTREE_URL,
        'mathlib_version': 'v4.19.0',
        'mathlib_revision': 'c44e0c8ee63ca166450922a373c7409c5d26b00b',
        'seed': args.seed,
        'fraction': args.fraction,
        'source_files': total_files,
        'validation_source_files': validation_files,
        'train_candidate_proofs': len(candidate_ids['train']),
        'validation_candidate_proofs': len(candidate_ids['validation']),
        'train_proofs': len(selected['train']),
        'validation_proofs': len(selected['validation']),
        'conjecture_source': str(args.conjecture_input.resolve()),
        'conjecture_pool': conjecture_pool_count,
        'train_conjectures': conjecture_count,
        'train_examples': proof_count + conjecture_count,
        'rl_theorems': rl_count,
        'rl_source': str(args.rl_input.resolve()),
    }
    metadata_path = args.output_dir / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
