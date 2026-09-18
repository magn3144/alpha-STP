import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def cap_targets(examples, limit, seed):
    groups = defaultdict(list)
    for example in examples:
        groups[example['target']].append(example)
    rng = random.Random(seed)
    retained = []
    for candidates in groups.values():
        rng.shuffle(candidates)
        used_inputs = set()
        used_lemmas = set()
        for _ in range(min(limit, len(candidates))):
            scores = []
            for example in candidates:
                lemma, input_proof = example['prompt'].split('\n<lemma>\n', 1)[1].split('\n<easy theorem>\n', 1)
                scores.append(int(lemma not in used_lemmas) + int(input_proof not in used_inputs))
            example = candidates.pop(scores.index(max(scores)))
            lemma, input_proof = example['prompt'].split('\n<lemma>\n', 1)[1].split('\n<easy theorem>\n', 1)
            used_lemmas.add(lemma)
            used_inputs.add(input_proof)
            retained.append(example)
    rng.shuffle(retained)
    return retained


def main():
    parser = argparse.ArgumentParser(description='Cap conjecturer pairs per target, preferring varied inputs and lemmas.')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    assert args.limit > 0
    assert args.input_dir.resolve() != args.output_dir.resolve()
    metadata = json.loads((args.input_dir / 'metadata.json').read_text())
    metadata['target_cap'] = {'limit': args.limit, 'seed': args.seed, 'source': str(args.input_dir), 'selection': 'Prefer unused input proofs and shared lemmas; random tie breaking.'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'validation']:
        examples = json.loads((args.input_dir / f'{split}.json').read_text())
        retained = cap_targets(examples, args.limit, args.seed)
        counts = Counter(example['target'] for example in retained)
        assert max(counts.values()) <= args.limit
        assert set(counts) == {example['target'] for example in examples}
        stats = metadata['splits'][split]
        stats['examples_before_target_cap'] = len(examples)
        stats['examples'] = len(retained)
        stats['unique_targets'] = len(counts)
        stats['max_examples_per_target'] = max(counts.values())
        stats['shared_lemmas'] = len({example['prompt'].split('\n<lemma>\n', 1)[1].split('\n<easy theorem>\n', 1)[0] for example in retained})
        (args.output_dir / f'{split}.json').write_text(json.dumps(retained, ensure_ascii=False) + '\n')
        print(f'{split}: {len(examples)} -> {len(retained)} examples, {len(counts)} targets', flush=True)
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
