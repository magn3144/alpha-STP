"""Teacher-forced proof loss on verified validation data (no Lean execution)."""

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def encode_proofs(dataset, tokenizer):
    examples = []
    for row in dataset:
        proof = row['proof_target']
        assert proof and row['verified_proof'].endswith(proof)
        prefix = row['verified_proof'][:-len(proof)].rstrip()
        prompt = 'Complete the following Lean 4 code:\n\n```lean4\n' + prefix
        encoded = tokenizer(prompt + proof, return_offsets_mapping=True)
        first_target = next(i for i, (_, end) in enumerate(encoded['offset_mapping'])
                            if end > len(prompt))
        tokens = encoded['input_ids'] + [tokenizer.eos_token_id]
        examples.append((row['id'], tokens, first_target))
    return examples


@torch.inference_mode()
def evaluate(model, examples, context_length, stride):
    results = []
    for number, (theorem_id, tokens, first_target) in enumerate(examples, 1):
        total_nll = 0.0
        count = 0
        previous_end = 1
        end = min(context_length, len(tokens))
        while True:
            start = max(0, end - context_length)
            score_start = max(first_target, previous_end)
            if score_start < end:
                ids = torch.tensor([tokens[start:end]], device='cuda')
                logits = model(ids, use_cache=False).logits[0]
                prediction_start = score_start - start - 1
                assert prediction_start >= 0
                # Float32 cross entropy; only proof/EOS tokens contribute.
                total_nll += F.cross_entropy(
                    logits[prediction_start:-1].float(),
                    ids[0, prediction_start + 1:], reduction='sum').item()
                count += end - score_start
                del logits, ids
            previous_end = end
            if end == len(tokens):
                break
            end = min(end + stride, len(tokens))
        assert count == len(tokens) - first_target
        results.append({'id': theorem_id, 'nll_sum': total_nll,
                        'target_tokens': count, 'loss': total_nll / count})
        if number == 1 or number % 25 == 0:
            print(f'  {number}/{len(examples)} theorems; latest loss={total_nll/count:.6f}', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(os.path.expandvars(Path(args.config).read_text()))
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    raw = Path(config['validation_data']).read_bytes()
    dataset = [json.loads(line) for line in raw.splitlines()]
    assert len(dataset) == len({r['id'] for r in dataset}) == 400
    assert 0 < config['stride'] < config['context_length']
    checkpoints = [('starting_model', None, Path(config['starting_model']))]
    for name, run in config['checkpoint_runs'].items():
        models = sorted(Path(run).glob('round*/RL_model'),
                        key=lambda p: int(p.parent.name.removeprefix('round')))
        assert len(models) == config['expected_checkpoints'][name]
        checkpoints.extend((name, int(p.parent.name.removeprefix('round')), p) for p in models)
    metadata = dict(config, dataset_sha256=hashlib.sha256(raw).hexdigest(),
                    metric='proof and EOS token mean negative log likelihood, natural log',
                    precision='bfloat16 inference, float32 cross entropy',
                    long_proofs='overlapping windows; every target token scored exactly once',
                    torch_version=torch.__version__, started_at=time.time())
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    torch.set_num_threads(8)
    for name, round_number, path in checkpoints:
        started = time.time()
        print(f'Loading {name} round={round_number}: {path}', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        examples = encode_proofs(dataset, tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, local_files_only=True,
            attn_implementation='sdpa').to('cuda').eval()
        assert config['context_length'] <= model.config.max_position_embeddings
        records = evaluate(model, examples, config['context_length'], config['stride'])
        total_tokens = sum(r['target_tokens'] for r in records)
        result = {'run': name, 'round': round_number, 'checkpoint': str(path),
                  'loss': sum(r['nll_sum'] for r in records) / total_tokens,
                  'mean_theorem_loss': sum(r['loss'] for r in records) / len(records),
                  'target_tokens': total_tokens, 'theorem_count': len(records),
                  'seconds': time.time() - started}
        (output / f'{name}_round{round_number}.json').write_text(
            json.dumps(dict(result, theorems=records), indent=2))
        with (output / 'results.jsonl').open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        print(f'COMPLETED {json.dumps(result)}', flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (output / 'SUCCESS').write_text(f'{len(checkpoints)} models completed\n')


if __name__ == '__main__':
    main()
