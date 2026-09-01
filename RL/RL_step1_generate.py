import os
import ray
import json
import random
import pickle
random.seed(0)  # Set a random seed for reproducibility
import argparse
import logging
import numpy as np
import gc
from collections import defaultdict
from copy import deepcopy
from utils.file_utils import path_exists, read_file, write_data
from utils.model_utils import init_ray_cluster
from utils.RL_utils import collect_trajectories, collect_conjecture, load_ds_from_config
from utils.RL_utils import Sampler_base, Sampler_naive, __DEBUG__, REPO_DIR
from utils.timing_utils import configure_timing, timer

MAX_LENGTH = 1024

if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s - %(name)s - %(levelname)s] %(message)s', level=logging.DEBUG, force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampler", type=str, default='Sampler_conjecture')
    parser.add_argument("--conjecture_multiplier", type=int, default=1)
    parser.add_argument("--samples_per_statement", type=int)
    parser.add_argument("--dataset_size", type=int, default=0)
    args = parser.parse_args()
    logging.debug(str(args))
    rng = np.random.default_rng(args.seed)

    if path_exists(os.path.join(args.exp_dir, 'generated_proofs.json')) or path_exists(os.path.join(args.exp_dir, 'generated_proofs.json.gz')):
        logging.warning(f"Dataset already exists. Exiting...")
        exit(0)

    os.makedirs(args.exp_dir, exist_ok=True)
    round = int(args.exp_dir.rsplit('/', 1)[-1][len('round'):])
    configure_timing(args.exp_dir, round_id=round)

    with timer('generation_dataset_loading'):
        formatted_ds = load_ds_from_config(args.dataset_config)
    if args.dataset_size > 0:
        subset_rng = np.random.default_rng(0)
        selected_idxs = subset_rng.choice(len(formatted_ds), args.dataset_size, replace=False)
        formatted_ds = [formatted_ds[idx] for idx in selected_idxs]
    if __DEBUG__:
        formatted_ds = formatted_ds[:1000]

    logging.info(f'Number of lemmas to generate: {len(formatted_ds)}')
    selected_statements = formatted_ds
    logging.info(f'Selected {len(selected_statements)} statements for this round.')

    if round == 0:
        sampler_dict = None
    else:
        last_round_dir = os.path.join(args.exp_dir.rsplit('/', 1)[0], f'round{round-1}')
        sampler_dict = read_file(os.path.join(last_round_dir, 'sampler.pkl'))
        assert sampler_dict is not None, f"Failed to read {os.path.join(last_round_dir, 'sampler.pkl')}"

    Sampler: Sampler_base = globals()[args.sampler]
    if sampler_dict is not None:
        sampler = Sampler.from_dict(sampler_dict)
    else:
        sampler = Sampler()
        sampler.init_lemma_mapping(formatted_ds)
    
    init_ray_cluster()

    lemmas_to_generate = deepcopy(selected_statements)
    collect_conjecture_fn = lambda inference_pool, nr_actors, selected_lemmas, lemma_mapping, seed: \
        collect_conjecture(inference_pool, nr_actors, selected_lemmas, \
                             lemma_mapping, MAX_LENGTH, seed, args.temperature, cache_dir=os.path.join(args.exp_dir, 'sampler_ckpt'))
    collect_traj = lambda inference_pool, nr_actors, selected_lemmas, lemma_mapping, seed: \
        collect_trajectories(inference_pool, nr_actors, selected_lemmas, \
                                           MAX_LENGTH, seed, args.temperature, cache_dir=os.path.join(args.exp_dir, 'sampler_ckpt'))
    
    with timer('generation_and_verification'):
        generated_proofs, valid_conjecture_examples = sampler.generate(args.model, args.model,
                lemmas_to_generate, args.seed, collect_traj,
                save_dir=os.path.join(args.exp_dir, 'sampler_ckpt'),
                collect_conjecture=collect_conjecture_fn, conjecture_multiplier=args.conjecture_multiplier,
                round_id=round,
                sps=args.samples_per_statement,
                project_to=formatted_ds)

    # log the distribution of succ rates for the generated lemmas
    proof_results = defaultdict(list)
    for test_info in generated_proofs:
        for _ in range(test_info.get('multiplicity', 1)):
            proof_results[test_info['lemma_id']].append(int(test_info.get('complete', False)))
    succ_rates = defaultdict(int)
    for lemma_id in proof_results:
        succ_rates[f'{np.mean(proof_results[lemma_id]):.3f}'] += 1
    # sort the succ rates
    succ_rates = {k: v for k, v in sorted(succ_rates.items(), key=lambda item: item[0])}
    logging.info(f'Success rate distribution: {succ_rates}')

    # force garbage collection
    ray.shutdown()
    gc.collect()

    write_data(pickle.dumps(sampler.to_dict()), os.path.join(args.exp_dir, 'sampler.pkl'), 'pickle')
    write_data(json.dumps([test_info for test_info in sampler.generated_proofs if test_info['round'] >= round - 2]), os.path.join(args.exp_dir, 'generated_proofs.json'), 'json')
    write_data(json.dumps(sampler.valid_conjecture_examples), os.path.join(args.exp_dir, 'conjecture_examples.json'), 'json')
