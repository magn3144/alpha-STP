import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / 'RL'))

from utils.prover.lean.verifier import verify_leantree_files


SOURCE_PATHS = [
    Path('/work3/s204164/delta-proof/data/runs/rl_codet5p_770m_2x_l40s_1_8_01/results.jsonl'),
    Path('/work3/s204164/delta-proof/data/runs/rl_codet5p_770m_2x_l40s_small_02/results.jsonl'),
]
OUTPUT_DIR = REPO_DIR / 'data/dataset/conjecturer_sft_deltaproof'
THEOREM_DICT_PATH = REPO_DIR / 'assets/data/theorem_dict.pkl'
LAKE_PATH = Path.home() / '.elan/bin/lake'
LEAN_PROJECT = Path('/work3/s204164/delta-proof/lean_project')
SEED = 42
PER_LEMMA_LIMIT = 300
PROVER_PROMPT = (
    'Complete the following Lean 4 code:\n\n```lean4\n'
    'import Mathlib\n'
    'import Aesop\n'
    'set_option maxHeartbeats 0\n'
    'open BigOperators Real Nat Topology Rat\n'
)


def read_proofs(paths):
    retained = {}
    source_counts = {}
    eligible_records = 0
    for path in paths:
        records = 0
        eligible = 0
        with path.open(encoding='utf-8') as source:
            for line in source:
                if not line.strip():
                    continue
                records += 1
                record = json.loads(line)
                if not (
                    record['success']
                    and record['final_proof']
                    and not record['disprove']
                    and not record['error']
                ):
                    continue
                eligible += 1
                theorem = record['theorem'].strip()
                proof = record['final_proof'].strip()
                if theorem not in retained or len(proof) < len(retained[theorem]):
                    retained[theorem] = proof
        source_counts[str(path)] = {'records': records, 'eligible_records': eligible}
        eligible_records += eligible
    return retained, source_counts, eligible_records


def theorem_statement(theorem):
    suffix = ':= by sorry'
    assert theorem.endswith(suffix)
    return theorem.removesuffix(suffix).rstrip()


def extract_premises(theorems, lake_path, lean_project, timeout):
    records = list(theorems.items())
    results = verify_leantree_files(
        [proof for theorem, proof in records],
        [None] * len(records),
        str(lake_path),
        str(lean_project),
        timeout,
    )
    failed = [index for index, result in enumerate(results) if 'invokes' not in result]
    if failed:
        raise RuntimeError(f'Premise extraction failed for {len(failed)} proofs: {failed[:10]}')
    return {
        theorem: sorted(set(result['invokes']))
        for (theorem, proof), result in zip(records, results)
    }


def load_lemma_statements(path):
    with path.open('rb') as source:
        theorem_dict = pickle.load(source)
    return {
        name: value[1].split(':=')[0].strip()
        for name, value in theorem_dict.items()
    }


def make_example(lemma_statement, input_proof, target_theorem):
    prompt = (
        f'{PROVER_PROMPT}\n'
        f'<lemma>\n{lemma_statement.strip()}\n<easy theorem>\n'
        f'{input_proof.strip()}\n<hard theorem>'
    )
    target = f'\n{theorem_statement(target_theorem).strip()}\n</hard theorem>'
    return {'prompt': prompt, 'target': target, 'weight': 1}


def build_split(theorem_list, proofs, premises, lemma_statements, seed, per_lemma_limit):
    by_lemma = defaultdict(list)
    for theorem in theorem_list:
        for lemma in premises[theorem]:
            if lemma in lemma_statements:
                by_lemma[lemma].append(theorem)

    examples = []
    pairable_theorems = set()
    shared_lemmas = []
    seen = set()
    rng = random.Random(seed)
    for lemma in sorted(by_lemma):
        lemma_theorems = sorted(set(by_lemma[lemma]))
        if len(lemma_theorems) < 2:
            continue
        shared_lemmas.append(lemma)
        pairable_theorems.update(lemma_theorems)
        pairs = [
            (input_theorem, target_theorem)
            for index, input_theorem in enumerate(lemma_theorems)
            for target_theorem in lemma_theorems[index + 1:]
        ]
        rng.shuffle(pairs)
        for first_theorem, second_theorem in pairs[:per_lemma_limit // 2]:
            for input_theorem, target_theorem in [
                (first_theorem, second_theorem),
                (second_theorem, first_theorem),
            ]:
                key = (lemma, input_theorem, target_theorem)
                if key in seen:
                    continue
                seen.add(key)
                examples.append(make_example(
                    lemma_statements[lemma],
                    proofs[input_theorem],
                    target_theorem,
                ))
    return examples, shared_lemmas, len(theorem_list) - len(pairable_theorems)


def write_json(path, value, indent=None):
    with path.open('w', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=False, indent=indent)
        output.write('\n')


def validate(train, validation, train_theorems, validation_theorems):
    assert set(train_theorems).isdisjoint(validation_theorems)
    assert all(set(example) == {'prompt', 'target', 'weight'} for example in train + validation)
    assert all(example['weight'] == 1 for example in train + validation)
    assert all(example['prompt'].startswith(PROVER_PROMPT + '\n<lemma>\n') for example in train + validation)
    assert all(example['prompt'].endswith('\n<hard theorem>') for example in train + validation)
    assert all(example['target'].startswith('\n') for example in train + validation)
    assert all(example['target'].endswith('\n</hard theorem>') for example in train + validation)


def main():
    parser = argparse.ArgumentParser(description='Build conjecturer SFT data from DeltaProof proofs.')
    parser.add_argument('--sources', nargs='+', type=Path, default=SOURCE_PATHS)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--theorem-dict', type=Path, default=THEOREM_DICT_PATH)
    parser.add_argument('--lake-path', type=Path, default=LAKE_PATH)
    parser.add_argument('--lean-project', type=Path, default=LEAN_PROJECT)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--per-lemma-limit', type=int, default=PER_LEMMA_LIMIT)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    assert args.per_lemma_limit > 0 and args.per_lemma_limit % 2 == 0

    proofs, source_counts, eligible_records = read_proofs(args.sources)
    premises = extract_premises(proofs, args.lake_path, args.lean_project, args.timeout)
    lemma_statements = load_lemma_statements(args.theorem_dict)

    theorems = sorted(proofs)
    random.Random(args.seed).shuffle(theorems)
    train_size = int(len(theorems) * 0.9)
    train_theorems = theorems[:train_size]
    validation_theorems = theorems[train_size:]

    train, train_lemmas, train_unpaired = build_split(
        train_theorems, proofs, premises, lemma_statements, args.seed, args.per_lemma_limit,
    )
    validation, validation_lemmas, validation_unpaired = build_split(
        validation_theorems, proofs, premises, lemma_statements, args.seed, args.per_lemma_limit,
    )
    validate(train, validation, train_theorems, validation_theorems)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'train.json', train)
    write_json(args.output_dir / 'validation.json', validation)
    metadata = {
        'source_paths': [str(path) for path in args.sources],
        'source_counts': source_counts,
        'selection': 'success with final_proof, excluding disproofs and errors',
        'proof_deduplication': 'exact theorem statement; shortest final_proof retained',
        'premise_extraction': 'LeanTree allTactics usedConstants via the existing DeltaProof verifier',
        'lemma_statements': str(args.theorem_dict),
        'seed': args.seed,
        'train_fraction': 0.9,
        'per_lemma_limit': args.per_lemma_limit,
        'eligible_records': eligible_records,
        'unique_theorems': len(theorems),
        'splits': {
            'train': {
                'unique_theorems': len(train_theorems),
                'shared_lemmas': len(train_lemmas),
                'examples': len(train),
                'theorems_without_pairs': train_unpaired,
            },
            'validation': {
                'unique_theorems': len(validation_theorems),
                'shared_lemmas': len(validation_lemmas),
                'examples': len(validation),
                'theorems_without_pairs': validation_unpaired,
            },
        },
        'validation': {
            'theorem_splits_disjoint': True,
            'record_keys': ['prompt', 'target', 'weight'],
            'weight': 1,
            'prompt_and_target_format_checked': True,
            'deduplication': 'shared lemma, input theorem, target theorem triples',
        },
    }
    write_json(args.output_dir / 'metadata.json', metadata, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
