import argparse
import json
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
DELTA_PROOF_DIR = REPO_DIR.parent / 'delta-proof'


def read_jsonl(path):
    with path.open(encoding='utf-8') as input_file:
        return [json.loads(line) for line in input_file]


def write_json(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as output_file:
        json.dump(records, output_file, ensure_ascii=False)
        output_file.write('\n')
    temporary_path.replace(path)


def prepare_sft_split(deltaproof_path, conjecturer_path, output_path):
    deltaproof_records = read_jsonl(deltaproof_path)
    examples = [
        {'prompt': record['state'], 'target': record['action']}
        for record in deltaproof_records
    ]
    with conjecturer_path.open(encoding='utf-8') as input_file:
        conjecturer_examples = json.load(input_file)
    examples.extend(conjecturer_examples)
    write_json(output_path, examples)
    return len(deltaproof_records), len(conjecturer_examples)


def prepare_rl_split(input_path, output_path, split):
    source_records = read_jsonl(input_path)
    records = [
        {
            'formal_statement': record['theorem'],
            'split': split,
            'proof': [],
        }
        for record in source_records
    ]
    write_json(output_path, records)
    return len(records)


def main():
    parser = argparse.ArgumentParser(
        description='Convert DeltaProof 1/8 datasets to STP LLM format.'
    )
    parser.add_argument(
        '--deltaproof-sft-dir',
        type=Path,
        default=DELTA_PROOF_DIR / 'data/dataset/sft_1_8',
    )
    parser.add_argument(
        '--deltaproof-rl-dir',
        type=Path,
        default=DELTA_PROOF_DIR / 'data/dataset/numina_math_lean_passing_1_8',
    )
    parser.add_argument(
        '--conjecturer-dir',
        type=Path,
        default=REPO_DIR / 'data/dataset/conjecturer_sft',
    )
    parser.add_argument(
        '--sft-output-dir',
        type=Path,
        default=(
            REPO_DIR
            / 'data/dataset/stp_llm_deltaproof_sft_1_8_with_conjecturer'
        ),
    )
    parser.add_argument(
        '--rl-output-dir',
        type=Path,
        default=REPO_DIR / 'data/dataset/stp_llm_deltaproof_rl_1_8',
    )
    args = parser.parse_args()

    sft_counts = {}
    for split in ('train', 'validation'):
        proof_count, conjecturer_count = prepare_sft_split(
            args.deltaproof_sft_dir / f'{split}.jsonl',
            args.conjecturer_dir / f'{split}.json',
            args.sft_output_dir / f'{split}.json',
        )
        sft_counts[split] = {
            'deltaproof_examples': proof_count,
            'conjecturer_examples': conjecturer_count,
            'examples': proof_count + conjecturer_count,
        }

    rl_counts = {}
    for split in ('train', 'validation', 'test'):
        rl_counts[split] = prepare_rl_split(
            args.deltaproof_rl_dir / f'{split}.jsonl',
            args.rl_output_dir / f'{split}.json',
            split,
        )

    sft_metadata = {
        'format': {'prompt': 'DeltaProof state', 'target': 'DeltaProof action'},
        'deltaproof_source': str(args.deltaproof_sft_dir.resolve()),
        'conjecturer_source': str(args.conjecturer_dir.resolve()),
        'splits': sft_counts,
    }
    rl_metadata = {
        'format': {
            'formal_statement': 'DeltaProof theorem',
            'split': 'source split',
            'proof': [],
        },
        'source': str(args.deltaproof_rl_dir.resolve()),
        'splits': rl_counts,
    }
    write_json(args.sft_output_dir / 'metadata.json', sft_metadata)
    write_json(args.rl_output_dir / 'metadata.json', rl_metadata)
    print(json.dumps({'sft': sft_metadata, 'rl': rl_metadata}, indent=2))


if __name__ == '__main__':
    main()
