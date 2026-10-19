import os
import ray
import json
import argparse
import logging
import numpy as np
from ray.util import ActorPool
from utils.model_utils import create_inference_actors, init_ray_cluster, get_prompt
from utils.file_utils import read_file, write_data
from utils.RL_utils import generate_and_test, collect_trajectories, insert_lemma, REPO_DIR, NR_FOLD
from utils.prover.lean.verifier import create_ray_deltaproof_lean_actors, TEST_BATCH_SIZE
from utils.timing_utils import configure_timing, timer
from copy import deepcopy

MAX_LENGTH = 1024


def evaluate_with_deltaproof(args, lemmas, inference_actors, collect_traj, lemma_mapping):
    cache_dir = os.path.join(args.exp_dir, 'sampler_ckpt')
    pool = ActorPool(inference_actors)
    generated = []
    batch_size = (len(lemmas) + NR_FOLD - 1) // NR_FOLD
    for shard in range(NR_FOLD):
        batch = lemmas[shard * batch_size:(shard + 1) * batch_size]
        with timer('network_inference', shard=shard):
            generated.extend(collect_traj(
                pool, len(inference_actors), batch, lemma_mapping, args.seed * NR_FOLD + shard,
            ))
    for actor in inference_actors:
        ray.kill(actor)

    workers = create_ray_deltaproof_lean_actors(
        args.lake_path, args.lean_project, args.lean_workers, args.lean_timeout,
    )
    pool = ActorPool(workers)
    results = []
    pending = []
    for start in range(0, len(generated), TEST_BATCH_SIZE):
        cache_path = os.path.join(cache_dir, f'deltaproof_tests_{start}.json')
        cached = read_file(cache_path)
        if cached is None:
            pending.append((cache_path, generated[start:start + TEST_BATCH_SIZE]))
        else:
            results.extend(cached)
    with timer('verification_wall'):
        verification_results = pool.map(
            lambda actor, item: actor.run.remote(item[1]),
            pending,
        )
        for (cache_path, _), batch_results in zip(pending, verification_results):
            verified = [result | {'multiplicity': 1} for result in batch_results]
            write_data(json.dumps(verified), cache_path)
            results.extend(verified)
    for worker in workers:
        ray.kill(worker)
    return results


def main(args: argparse.Namespace):
    configure_timing(args.exp_dir, new_session=True)
    args.save_file_path = os.path.join(args.exp_dir, f'generated_proofs_{args.save_file_name}.jsonl')
    print(args)
    
    os.makedirs(args.exp_dir, exist_ok=True)

    lemmas_to_generate = []
    lemma_mapping = {}

    idx = 0
    dataset_configs = read_file(args.raw_dataset_config)
    for dataset_config in dataset_configs:
        logging.debug(f'Processing dataset: {dataset_config["dataset_path"]}')
        raw_dataset = read_file(os.path.join(REPO_DIR, dataset_config['dataset_path']))
        nr_samples = dataset_config['weight']
        assert (raw_dataset is not None), f"Failed to read {dataset_config['dataset_path']}"
        logging.debug(f'Size of the dataset: {len(raw_dataset)}')
        
        formatted_ds = []
        for raw in raw_dataset:
            test_info = {'lemma_id': idx,
                        'statement': raw['formal_statement'].rsplit('sorry', 1)[0].strip(),
                        'label': [raw['split']] + (raw.get('tags', None) or []),
                        'header': 'import Mathlib\n' if args.lean_project else raw.get('header', None),}
            idx += 1
            formatted_ds.append(test_info)
        
        for it in range(nr_samples):
            lemmas_to_generate += [test_info | {'iter': it} for test_info in deepcopy(formatted_ds)]
        
        for test_info in formatted_ds:
            insert_lemma(lemma_mapping, test_info)

    collect_traj = lambda inference_pool, nr_actors, selected_lemmas, lemma_mapping, seed: \
        collect_trajectories(inference_pool, nr_actors, selected_lemmas, \
                                           MAX_LENGTH, seed, args.temperature, cache_dir=os.path.join(args.exp_dir, 'sampler_ckpt'))
    init_ray_cluster()
    ray_inference_actors, model_dir = create_inference_actors(args.model, args.model, enable_prefix_caching=False)

    rng = np.random.default_rng(0)
    rng.shuffle(lemmas_to_generate)
    if args.lean_project:
        generated_proofs = evaluate_with_deltaproof(
            args, lemmas_to_generate, ray_inference_actors, collect_traj, lemma_mapping,
        )
    else:
        generated_proofs = generate_and_test(lemmas_to_generate, collect_traj, ray_inference_actors, lemma_mapping, args.seed, os.path.join(args.exp_dir, 'sampler_ckpt'),
                                         cpus_per_task=4, cpus_per_task_stage2=6, group_by_header=True, collect_premises=False)
    
    write_data('\n'.join([json.dumps(t) for t in generated_proofs]), args.save_file_path, 'jsonl')

if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s - %(name)s - %(levelname)s] %(message)s', level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        description="Generate and test proofs."
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--nr_samples", type=int, default=16)
    parser.add_argument("--save_file_name", type=str, default=None)
    parser.add_argument("--raw_dataset_config", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--lean-project', help='Use the delta-proof Lean project for evaluation.')
    parser.add_argument('--lake-path')
    parser.add_argument('--lean-workers', type=int)
    parser.add_argument('--lean-timeout', type=float)

    parsed_args = parser.parse_args()
    main(parsed_args)
