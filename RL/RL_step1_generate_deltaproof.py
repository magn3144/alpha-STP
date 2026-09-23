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
from utils.conjecture_metrics import ConjectureMetrics
from utils.deltaproof_stream import DeltaProofJournal
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


def generate_conjectures(sampler, model, dataset, target, config, round_dir, seed, progress):
    experiment = config['experiment']
    deltaproof = experiment['deltaproof']
    rl = config['deltaproof']['rl']
    inputs = select_conjecture_inputs(sampler, target, seed)
    if len(inputs) < target:
        raise ValueError(
            f'Only {len(inputs)} conjecture inputs are available; {target} are required.'
        )

    actors, _ = create_inference_actors(
        model,
        model,
        enable_prefix_caching=False,
        gpu_memory_utilization=deltaproof['vllm_gpu_memory_utilization'],
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
        deltaproof['conjecturer_temperature'],
        cache_dir=os.path.join(round_dir, 'sampler_ckpt'),
        progress=progress,
    )
    for actor in actors:
        ray.kill(actor)
    distinct = deduplicate_conjectures(
        candidates,
        dataset,
        known_statements,
    )
    progress.set_candidates(candidates, distinct)

    workers = create_ray_deltaproof_lean_actors(
        deltaproof['lake_path'],
        deltaproof['lean_project'],
        rl['max_concurrent_lean_imports'],
        rl['final_check_timeout'],
    )
    pool = ActorPool(workers)
    validation_inputs = [
        candidate | {'statement': candidate['statement'] + ' sorry', 'proof': ''}
        for candidate in distinct
    ]
    blocks = [
        validation_inputs[index:index + TEST_BATCH_SIZE]
        for index in range(0, len(validation_inputs), TEST_BATCH_SIZE)
    ]
    pool.map_unordered(
        lambda actor, block: actor.run.remote(block),
        blocks,
    )
    validation_results = []
    for _ in blocks:
        results = pool.get_next_unordered()
        validation_results.extend(results)
        progress.validation_progress(results)
    for worker in workers:
        ray.kill(worker)
    valid_ids = {
        result['lemma_id']
        for result in validation_results if result.get('pass', False)
    }
    valid = [candidate for candidate in distinct if candidate['lemma_id'] in valid_ids]
    if len(valid) < target:
        raise ValueError(
            f'Only {len(valid)} valid distinct conjectures were generated; '
            f'{target} are required.'
        )
    valid.sort(key=lambda result: result['lemma_id'])
    np.random.default_rng(seed).shuffle(valid)
    progress.select(valid[:target])
    return valid[:target]


def collect_premises(generated_proofs, deltaproof, rl):
    successful = [
        test_info
        for test_info in generated_proofs
        if test_info['complete']
    ]
    if not successful:
        return generated_proofs
    workers = create_ray_deltaproof_lean_actors(
        deltaproof['lake_path'],
        deltaproof['lean_project'],
        rl['max_concurrent_lean_imports'],
        rl['final_check_timeout'],
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


def save_round(round_dir, sampler, round_id, requests, results, progress):
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
        'status': 'complete',
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
        'conjectures_generated': progress.metrics['generated'],
        'conjectures_selected': len(conjecture_ids),
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
    progress.metrics['training_examples'] = len(sampler.valid_conjecture_examples)
    progress.save(metrics)
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
    round_dir = os.path.abspath(args.exp_dir)
    if path_exists(os.path.join(round_dir, 'generation_complete')):
        logging.warning('Round generation is already complete. Exiting.')
        return
    os.makedirs(round_dir, exist_ok=True)
    progress = ConjectureMetrics(config, round_dir, args.model)
    try:
        run_round(args, config, round_dir, progress)
    except BaseException:
        progress.save({'round': progress.round_id, 'status': 'failed'})
        progress.run.finish(exit_code=1, quiet=True)
        raise
    else:
        progress.run.finish(quiet=True)


def run_round(args, config, round_dir, progress):
    experiment = config['experiment']
    deltaproof = experiment['deltaproof']
    rl = config['deltaproof']['rl']
    round_id = progress.round_id
    configure_timing(round_dir, round_id=round_id)
    results_path = os.path.join(round_dir, 'deltaproof_results.jsonl')
    inference_metrics_path = os.path.splitext(results_path)[0] + '_metrics.json'

    dataset = load_deltaproof_dataset(
        deltaproof['dataset_path'],
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
        assert isinstance(sampler_data, dict)
        sampler = Sampler_base.from_dict(sampler_data)
        for test_info in dataset:
            insert_lemma(sampler.lemma_mapping, test_info)

    attempts = deltaproof['attempts_per_round']
    conjectures = []
    if round_id == 0 or deltaproof['conjecture_fraction'] == 0:
        dataset_attempts = attempts
    else:
        conjecture_budget = int(attempts * deltaproof['conjecture_fraction'])
        dataset_attempts = attempts - conjecture_budget
        conjecture_target = conjecture_budget // deltaproof['conjecture_attempts']

    dataset_theorems = select_dataset_theorems(
        dataset,
        sampler.succ_lemmas,
        dataset_attempts,
        args.seed,
    )
    if not dataset_theorems and all(
        test_info['lemma_id'] in sampler.succ_lemmas for test_info in dataset
    ):
        logging.info('All dataset theorems are solved.')
        progress.save({'round': round_id, 'status': 'experiment_complete'})
        Path(round_dir, 'experiment_complete').touch()
        return

    selection_path = Path(round_dir) / 'deltaproof_conjectures.json'
    if round_id > 0 and deltaproof['conjecture_fraction']:
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text(encoding='utf-8'))
            conjectures = selection['conjectures']
            progress.records = selection['records']
            progress.metrics.update(selection['metrics'])
            progress.by_lemma = {
                record['lemma_id']: record for record in progress.records
                if record['distinct_new']
            }
        else:
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
                    progress,
                )
            ray.shutdown()
            gc.collect()
            temporary = selection_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps({
                'conjectures': conjectures,
                'records': progress.records,
                'metrics': progress.metrics,
            }), encoding='utf-8')
            temporary.replace(selection_path)
    requests, test_infos = build_requests(
        dataset_theorems,
        conjectures,
        deltaproof['conjecture_attempts'],
        round_id,
    )
    requests_path = os.path.join(round_dir, 'deltaproof_requests.jsonl')
    transitions_path = os.path.join(round_dir, 'deltaproof_transitions.jsonl')
    if os.path.isfile(requests_path):
        if read_jsonl(requests_path) != requests:
            raise ValueError('DeltaProof requests changed while resuming the round.')
    else:
        write_jsonl(requests_path, requests)

    run_dir = Path(experiment['exp_dir']) / 'deltaproof'
    if round_id == 0:
        run_dir = Path(rl['sft_run_dir'])
    progress.set_requests(requests, run_dir)
    journal = DeltaProofJournal(round_dir, requests, progress)
    unfinished = [
        request for request in requests if request['request_id'] not in journal.results
    ]
    pending_path = Path(round_dir) / 'deltaproof_pending_requests.jsonl'
    write_jsonl(pending_path, unfinished)
    if unfinished:
        # Learner training starts only after generation; reject a changed checkpoint
        # if a partially completed round is restarted with different model files.
        checkpoint_paths = [run_dir / 'network_params.pt'] if round_id == 0 else [
            run_dir / 'checkpoints' / 'latest.pt',
            *sorted((run_dir / 'checkpoints').glob('step_*.pt'))[-1:],
        ]
        checkpoint = [
            [str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns]
            for path in checkpoint_paths
        ]
        checkpoint_path = Path(round_dir) / 'deltaproof_inference_checkpoint.json'
        if checkpoint_path.is_file():
            if json.loads(checkpoint_path.read_text(encoding='utf-8')) != checkpoint:
                raise ValueError('DeltaProof checkpoint changed while resuming inference.')
        else:
            temporary = checkpoint_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(checkpoint), encoding='utf-8')
            temporary.replace(checkpoint_path)
    inference_args = [
        '--config', Path(args.config).resolve(),
        '--input', pending_path,
        '--batch-id', f'round{round_id}',
        '--run-dir', run_dir,
        '--lean-project', deltaproof['lean_project'],
        '--num-simulations', rl['num_simulations'],
        '--num-sampled-actions', rl['num_sampled_actions'],
        '--tactic-timeout', rl['tactic_timeout'],
        '--final-check-timeout', rl['final_check_timeout'],
        '--parallel-searches', rl['num_actors'],
        '--max-concurrent-lean-imports', rl['max_concurrent_lean_imports'],
        '--inference-num-gpus', rl['inference_num_gpus'],
        '--inference-batch-size', rl['inference_batch_size'],
        '--inference-batch-timeout', rl['inference_batch_timeout'],
        '--seed', args.seed,
    ]
    for lean_import in deltaproof['imports']:
        inference_args.extend(['--import', lean_import])
    if unfinished:
        with timer('deltaproof_inference'):
            run_external_python(
                deltaproof['python'],
                'alphaproof.inference.infer',
                *inference_args,
                cwd=deltaproof['repo_dir'],
                stream=journal,
            )
    journal.finalize()

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
    with open(inference_metrics_path, encoding='utf-8') as metrics_file:
        progress.log_alphaproof([], json.load(metrics_file))
    progress.log(force=True)
    generated_proofs = apply_results(results, test_infos)

    init_ray_cluster()
    with timer('premise_collection'):
        generated_proofs = collect_premises(generated_proofs, deltaproof, rl)
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
        progress,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--seed', type=int, required=True)
    main(parser.parse_args())
