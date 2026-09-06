import argparse
import gc
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import ray
from ray.util import ActorPool

from utils.config_utils import load_experiment_config
from utils.deltaproof_utils import (
    apply_results,
    build_requests,
    deduplicate_conjectures,
    load_deltaproof_dataset,
    read_jsonl,
    select_conjecture_inputs,
    select_dataset_theorems,
    write_jsonl,
)
from utils.experiment_utils import run_external_python
from utils.file_utils import path_exists, read_file, write_data
from utils.model_utils import (
    create_embedding_actors,
    create_inference_actors,
    init_ray_cluster,
    insert_lemma,
)
from utils.prover.lean.verifier import (
    TEST_BATCH_SIZE,
    create_ray_deltaproof_lean_actors,
)
from utils.RL_utils import (
    Sampler_base,
    collect_conjecture,
    update_succ_lemmas,
    update_succ_rates,
)
from utils.timing_utils import configure_timing, timer


MAX_LENGTH = 1024


def generate_conjectures(sampler, model, dataset, target, config, round_dir, seed):
    experiment = config['experiment']
    solver = experiment['solver']
    inputs = select_conjecture_inputs(sampler, target, seed)
    if len(inputs) < target:
        raise ValueError(
            f'Only {len(inputs)} conjecture inputs are available; {target} are required.'
        )

    actors, _ = create_inference_actors(
        model,
        model,
        enable_prefix_caching=False,
    )
    pool = ActorPool(actors)
    multiplier = experiment['conjecture_multiplier']
    known_statements = set(sampler.lemma_mapping)
    candidates = collect_conjecture(
        pool,
        len(actors),
        inputs * multiplier,
        sampler.lemma_mapping,
        MAX_LENGTH,
        seed,
        experiment['temperature'],
        cache_dir=os.path.join(round_dir, 'sampler_ckpt'),
    )
    for actor in actors:
        ray.kill(actor)
    candidates = deduplicate_conjectures(
        candidates,
        dataset,
        known_statements,
    )

    workers = create_ray_deltaproof_lean_actors(
        solver['lake_path'],
        solver['lean_project'],
        solver['max_concurrent_lean_imports'],
        solver['final_check_timeout'],
    )
    pool = ActorPool(workers)
    validation_inputs = [candidate | {'proof': ' sorry'} for candidate in candidates]
    blocks = [
        validation_inputs[index:index + TEST_BATCH_SIZE]
        for index in range(0, len(validation_inputs), TEST_BATCH_SIZE)
    ]
    pool.map_unordered(
        lambda actor, block: actor.run.remote(block),
        blocks,
    )
    validation_results = [
        result
        for _ in blocks
        for result in pool.get_next_unordered()
    ]
    for worker in workers:
        ray.kill(worker)
    valid = [result for result in validation_results if result.get('pass', False)]
    if len(valid) < target:
        raise ValueError(
            f'Only {len(valid)} valid distinct conjectures were generated; '
            f'{target} are required.'
        )
    valid.sort(key=lambda result: result['lemma_id'])
    np.random.default_rng(seed).shuffle(valid)
    return valid[:target]


def collect_premises(generated_proofs, solver):
    successful = [
        test_info
        for test_info in generated_proofs
        if test_info['complete']
    ]
    if not successful:
        return generated_proofs
    workers = create_ray_deltaproof_lean_actors(
        solver['lake_path'],
        solver['lean_project'],
        solver['max_concurrent_lean_imports'],
        solver['final_check_timeout'],
    )
    pool = ActorPool(workers)
    blocks = [
        successful[index:index + TEST_BATCH_SIZE]
        for index in range(0, len(successful), TEST_BATCH_SIZE)
    ]
    pool.map_unordered(
        lambda actor, block: actor.run.remote(block),
        blocks,
    )
    verified = [
        result
        for _ in blocks
        for result in pool.get_next_unordered()
    ]
    for worker in workers:
        ray.kill(worker)
    failed = [result for result in verified if not result.get('complete', False)]
    if failed:
        raise ValueError(
            f'Premise collection rejected {len(failed)} DeltaProof-verified proofs.'
        )
    by_request = {result['request_id']: result for result in verified}
    return [
        by_request.get(test_info['request_id'], test_info)
        for test_info in generated_proofs
    ]


def filter_conjecture_examples(sampler, examples, dataset, model, round_dir, seed):
    if not examples:
        return []
    actors = create_embedding_actors(model, model)
    filtered = sampler.filtered_conjecture_examples(
        examples,
        project_to=[
            test_info
            for test_info in dataset
            if test_info['lemma_id'] not in sampler.succ_lemmas
        ],
        ray_embedding_actors=actors,
        seed=seed,
        save_dir=os.path.join(round_dir, 'sampler_ckpt'),
    )
    for actor in actors:
        ray.kill(actor)
    return filtered


def save_round(round_dir, sampler, round_id, requests, results):
    write_data(
        json.dumps([
            test_info
            for test_info in sampler.generated_proofs
            if test_info['round'] >= round_id - 2
        ]),
        os.path.join(round_dir, 'generated_proofs.json'),
        'json',
    )
    write_data(
        json.dumps(sampler.valid_conjecture_examples),
        os.path.join(round_dir, 'conjecture_examples.json'),
        'json',
    )
    cumulative_dataset_solved = len(
        sampler.succ_lemmas.intersection(sampler.relevant_lemmas)
    )
    conjecture_ids = {
        result['theorem_id']
        for result in results
        if result['source'] == 'conjecture'
    }
    solved_conjecture_ids = {
        result['theorem_id']
        for result in results
        if result['source'] == 'conjecture' and result['status'] == 'proved'
    }
    metrics = {
        'round': round_id,
        'attempts': len(requests),
        'dataset_attempts': sum(item['source'] == 'dataset' for item in requests),
        'conjecture_attempts': sum(
            item['source'] == 'conjecture'
            for item in requests
        ),
        'dataset_solved': sum(
            result['source'] == 'dataset' and result['status'] == 'proved'
            for result in results
        ),
        'conjectures_generated': len(conjecture_ids),
        'conjecture_attempts_solved': sum(
            result['source'] == 'conjecture' and result['status'] == 'proved'
            for result in results
        ),
        'conjectures_solved': len(solved_conjecture_ids),
        'simulations_allocated': sum(
            result['simulations_allocated']
            for result in results
        ),
        'simulations_used': sum(result['simulations_used'] for result in results),
        'transitions': sum(result['transition_count'] for result in results),
        'dataset_total': len(sampler.relevant_lemmas),
        'cumulative_dataset_solved': cumulative_dataset_solved,
        'dataset_remaining': len(sampler.relevant_lemmas) - cumulative_dataset_solved,
        'conjecture_training_examples': len(sampler.valid_conjecture_examples),
    }
    write_data(
        json.dumps(metrics, indent=2),
        os.path.join(round_dir, 'round_metrics.json'),
        'json',
        no_compression=True,
    )
    write_data(
        pickle.dumps(sampler.to_dict()),
        os.path.join(round_dir, 'sampler.pkl'),
        'pickle',
    )
    Path(round_dir, 'generation_complete').touch()


def main(args):
    logging.basicConfig(
        format='[%(asctime)s - %(name)s - %(levelname)s] %(message)s',
        level=logging.DEBUG,
        force=True,
    )
    config = load_experiment_config(args.config, 'rl')
    experiment = config['experiment']
    solver = experiment['solver']
    round_dir = os.path.abspath(args.exp_dir)
    round_id = int(Path(round_dir).name.removeprefix('round'))
    configure_timing(round_dir, round_id=round_id)
    results_path = os.path.join(round_dir, 'deltaproof_results.jsonl')
    if path_exists(os.path.join(round_dir, 'generation_complete')):
        logging.warning('Round generation is already complete. Exiting.')
        return
    os.makedirs(round_dir, exist_ok=True)

    dataset = load_deltaproof_dataset(
        solver['dataset_path'],
        experiment['dataset_size'],
    )
    if round_id == 0:
        sampler = Sampler_base()
        sampler.init_lemma_mapping(dataset)
    else:
        previous_dir = os.path.join(
            os.path.dirname(round_dir),
            f'round{round_id - 1}',
        )
        sampler_data = read_file(os.path.join(previous_dir, 'sampler.pkl'))
        assert sampler_data is not None
        sampler = Sampler_base.from_dict(sampler_data)
        for test_info in dataset:
            insert_lemma(sampler.lemma_mapping, test_info)

    attempts = solver['attempts_per_round']
    conjectures = []
    if round_id == 0 or solver['conjecture_fraction'] == 0:
        dataset_attempts = attempts
    else:
        dataset_attempts = attempts // 2
        conjecture_target = (
            attempts // 2 // solver['conjecture_attempts']
        )
        init_ray_cluster()
        with timer('conjecture_generation'):
            conjectures = generate_conjectures(
                sampler,
                args.model,
                dataset,
                conjecture_target,
                config,
                round_dir,
                args.seed,
            )
        ray.shutdown()
        gc.collect()

    dataset_theorems = select_dataset_theorems(
        dataset,
        sampler.succ_lemmas,
        dataset_attempts,
        args.seed,
    )
    if not dataset_theorems:
        logging.info('All dataset theorems are solved.')
        Path(round_dir, 'experiment_complete').touch()
        return
    requests, test_infos = build_requests(
        dataset_theorems,
        conjectures,
        solver['conjecture_attempts'],
        round_id,
    )
    requests_path = os.path.join(round_dir, 'deltaproof_requests.jsonl')
    transitions_path = os.path.join(round_dir, 'deltaproof_transitions.jsonl')
    if os.path.isfile(requests_path):
        if read_jsonl(requests_path) != requests:
            raise ValueError('DeltaProof requests changed while resuming the round.')
    else:
        write_jsonl(requests_path, requests)

    run_dir = Path(solver['run_dir'])
    if not (run_dir / 'checkpoints' / 'latest.pt').is_file():
        run_dir = Path(solver['sft_run_dir'])
    inference_args = [
        '--input', requests_path,
        '--output', results_path,
        '--transitions-output', transitions_path,
        '--batch-id', f'round{round_id}',
        '--run-dir', run_dir,
        '--lean-project', solver['lean_project'],
        '--num-simulations', solver['num_simulations'],
        '--num-sampled-actions', solver['num_sampled_actions'],
        '--tactic-timeout', solver['tactic_timeout'],
        '--final-check-timeout', solver['final_check_timeout'],
        '--parallel-searches', solver['parallel_searches'],
        '--max-concurrent-lean-imports', solver['max_concurrent_lean_imports'],
        '--inference-num-gpus', solver['inference_num_gpus'],
        '--inference-batch-size', solver['inference_batch_size'],
        '--inference-batch-timeout', solver['inference_batch_timeout'],
        '--seed', args.seed,
    ]
    for lean_import in solver['imports']:
        inference_args.extend(['--import', lean_import])
    if not os.path.isfile(results_path) or not os.path.isfile(transitions_path):
        with timer('deltaproof_inference'):
            run_external_python(
                solver['python'],
                'alphaproof.inference.infer',
                *inference_args,
                cwd=solver['repo_dir'],
            )

    results = read_jsonl(results_path)
    requests_by_id = {request['request_id']: request for request in requests}
    expected_request_ids = set(requests_by_id)
    result_request_ids = [result['request_id'] for result in results]
    if (
        len(result_request_ids) != len(expected_request_ids)
        or set(result_request_ids) != expected_request_ids
    ):
        raise ValueError('DeltaProof did not return one result per request.')
    for result in results:
        request = requests_by_id[result['request_id']]
        metadata = ('theorem_id', 'source', 'attempt')
        if any(result[name] != request[name] for name in metadata):
            raise ValueError('DeltaProof result metadata does not match its request.')
    transitions = read_jsonl(transitions_path)
    transition_ids = [transition['transition_id'] for transition in transitions]
    if len(transition_ids) != len(set(transition_ids)):
        raise ValueError('DeltaProof returned duplicate transition IDs.')
    transition_counts = {}
    for transition in transitions:
        request_id = transition['request_id']
        if request_id not in requests_by_id:
            raise ValueError('DeltaProof returned a transition for an unknown request.')
        if transition['batch_id'] != f'round{round_id}':
            raise ValueError('DeltaProof returned a transition for another batch.')
        transition_counts[request_id] = transition_counts.get(request_id, 0) + 1
    if any(
        result['transition_count'] != transition_counts.get(result['request_id'], 0)
        for result in results
    ):
        raise ValueError('DeltaProof transition counts do not match its results.')
    rejected = [result for result in results if result['status'] == 'rejected']
    if rejected:
        raise ValueError(f'DeltaProof rejected {len(rejected)} scheduled theorems.')
    generated_proofs = apply_results(results, test_infos)

    init_ray_cluster()
    with timer('premise_collection'):
        generated_proofs = collect_premises(generated_proofs, solver)
    update_succ_lemmas(generated_proofs, sampler.succ_lemmas)
    update_succ_rates(generated_proofs, sampler.succ_rates)
    conjecture_examples = sampler.get_conjecture_examples(generated_proofs)
    sampler.generated_proofs += [
        test_info | {'round': round_id}
        for test_info in generated_proofs
    ]
    sampler.generated_proofs = [
        test_info
        for test_info in sampler.generated_proofs
        if (
            test_info['round'] >= round_id - 2
            or test_info['lemma_id'] in sampler.relevant_lemmas
        )
        and sampler.succ_rates[test_info['lemma_id']] > 0
    ]
    sampler.valid_conjecture_examples = filter_conjecture_examples(
        sampler,
        conjecture_examples,
        dataset,
        args.model,
        round_dir,
        args.seed,
    )
    ray.shutdown()
    gc.collect()
    save_round(
        round_dir,
        sampler,
        round_id,
        requests,
        results,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--seed', type=int, required=True)
    main(parser.parse_args())
