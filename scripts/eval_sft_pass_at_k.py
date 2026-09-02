import argparse
import gc
import json
import random
from pathlib import Path

import ray
import torch
from ray.util import ActorPool
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.model_utils import init_ray_cluster
from utils.prover.lean.verifier import create_ray_lean4_actors


def lean_code(prompt):
    return prompt.split('```lean4\n', 1)[1]


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.validation_data) as validation_file:
        validation = json.load(validation_file)

    rng = random.Random(args.seed)
    selected_indices = rng.sample(range(len(validation)), args.theorem_count)
    selected = [validation[index] for index in selected_indices]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.truncation_side = 'left'
    prompts = []
    for example in selected:
        tokens = tokenizer.encode(
            example['prompt'],
            max_length=args.prompt_tokens,
            truncation=True,
        )
        prompts.append(tokenizer.decode(tokens, skip_special_tokens=True))

    model = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.prompt_tokens + args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=args.samples_per_theorem,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )
    outputs = model.generate(prompts, sampling)

    samples = []
    for theorem_id, output in enumerate(outputs):
        for sample_id, completion in enumerate(output.outputs):
            generated = completion.text.split('\n```', 1)[0]
            samples.append({
                'theorem_id': theorem_id,
                'validation_index': selected_indices[theorem_id],
                'sample_id': sample_id,
                'header': lean_code(selected[theorem_id]['prompt']),
                'statement': '',
                'proof': generated,
            })

    del model
    gc.collect()
    torch.cuda.empty_cache()

    generations = [
        {
            'theorem_id': sample['theorem_id'],
            'validation_index': sample['validation_index'],
            'sample_id': sample['sample_id'],
            'generated': sample['proof'],
        }
        for sample in samples
    ]
    with open(output_dir / 'generations.json', 'w') as generations_file:
        json.dump(generations, generations_file, indent=2)

    if args.generate_only:
        return

    init_ray_cluster()
    actors = create_ray_lean4_actors(
        cpus_per_task=args.verifier_cpus_per_task,
        collect_premises=False,
    )
    pool = ActorPool(actors)
    batches = [
        samples[start:start + args.verifier_batch_size]
        for start in range(0, len(samples), args.verifier_batch_size)
    ]
    for batch in batches:
        pool.submit(lambda actor, items: actor.run.remote(items, batched=True), batch)

    results = []
    for _ in range(len(batches)):
        for result in pool.get_next_unordered():
            results.append({
                'theorem_id': result['theorem_id'],
                'validation_index': result['validation_index'],
                'sample_id': result['sample_id'],
                'generated': result['proof'],
                'complete': result.get('complete', False),
                'errors': result.get('errors', []),
                'system_messages': result.get('system_messages'),
                'verify_time': result.get('verify_time'),
            })

    for actor in actors:
        ray.kill(actor)
    ray.shutdown()

    results.sort(key=lambda result: (result['theorem_id'], result['sample_id']))
    solved = []
    for theorem_id in range(args.theorem_count):
        theorem_results = [result for result in results if result['theorem_id'] == theorem_id]
        solved.append(any(result['complete'] for result in theorem_results))

    complete_samples = sum(result['complete'] for result in results)
    summary = {
        'model': args.model,
        'validation_data': args.validation_data,
        'seed': args.seed,
        'temperature': args.temperature,
        'theorem_count': args.theorem_count,
        'samples_per_theorem': args.samples_per_theorem,
        'selected_validation_indices': selected_indices,
        'solved_theorems': sum(solved),
        'pass_at_k': sum(solved) / len(solved),
        'complete_samples': complete_samples,
        'sample_pass_rate': complete_samples / len(results),
    }
    with open(output_dir / 'results.json', 'w') as results_file:
        json.dump(results, results_file)
    with open(output_dir / 'summary.json', 'w') as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--validation-data', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--theorem-count', type=int, default=100)
    parser.add_argument('--samples-per-theorem', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--prompt-tokens', type=int, default=1024)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--dtype', default='float16')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--verifier-cpus-per-task', type=float, default=2)
    parser.add_argument('--verifier-batch-size', type=int, default=16)
    parser.add_argument('--generate-only', action='store_true')
    main(parser.parse_args())
