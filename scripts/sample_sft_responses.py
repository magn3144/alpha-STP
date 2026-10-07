import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def main(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = json.loads(Path(args.test_data).read_text())
    rng = random.Random(args.seed)
    selected = []
    for task in ('solver', 'conjecturer'):
        indices = [
            index for index, example in enumerate(dataset)
            if ('<hard theorem>' in example['prompt']) == (task == 'conjecturer')
        ]
        rng.shuffle(indices)
        count = 0
        for index in indices:
            example = dataset[index]
            tokens = tokenizer.encode(example['prompt'], add_special_tokens=True, verbose=False)
            if len(tokens) > args.max_prompt_tokens:
                continue
            selected.append({'task': task, 'test_index': index, **example, 'prompt_tokens': tokens})
            count += 1
            if count == args.samples_per_task:
                break
        assert count == args.samples_per_task, f'Not enough complete {task} prompts'

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), attn_implementation='eager', local_files_only=True,
    ).to('cuda').eval()
    if args.dtype == 'bfloat16':
        assert torch.cuda.is_bf16_supported()
    assert model.config.eos_token_id == tokenizer.eos_token_id
    assert max(max(example['prompt_tokens']) for example in selected) < model.config.vocab_size
    assert max(len(example['prompt_tokens']) for example in selected) + args.max_new_tokens <= model.config.max_position_embeddings
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        do_sample=False,
        use_cache=True,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with torch.inference_mode():
        for example in selected:
            input_ids = torch.tensor([example['prompt_tokens']], device='cuda')
            attention_mask = torch.ones_like(input_ids)
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            assert torch.isfinite(model(**inputs).logits[:, -1]).all()
            cached = model.generate(**inputs, generation_config=generation, max_new_tokens=16)
            uncached = model.generate(**inputs, generation_config=generation, max_new_tokens=16, use_cache=False)
            assert torch.equal(cached, uncached), 'Cached and uncached generation disagree'
            responses = []
            for mode in ('greedy', 'sampled'):
                torch.manual_seed(args.seed + example['test_index'])
                kwargs = {} if mode == 'greedy' else {'do_sample': True, 'temperature': 0.7, 'top_p': 1.0, 'top_k': 0}
                output = model.generate(**inputs, generation_config=generation, **kwargs)
                assert torch.equal(output[:, :input_ids.shape[1]], input_ids)
                completion = output[0, input_ids.shape[1]:].tolist()
                responses.append({
                    'mode': mode,
                    'text': tokenizer.decode(completion, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    'token_ids': completion,
                    'finish_reason': 'eos' if completion[-1] == tokenizer.eos_token_id else 'length',
                })
            results.append({**example, 'responses': responses})
            (output_dir / 'responses.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
            print(json.dumps({'task': example['task'], 'test_index': example['test_index'], 'responses': responses}, ensure_ascii=False), flush=True)

    summary = {
        'model': args.model,
        'test_data': args.test_data,
        'seed': args.seed,
        'gpu': torch.cuda.get_device_name(),
        'dtype': args.dtype,
        'prompt_handling': 'Complete dataset prompts, no truncation, no chat template, direct token IDs',
        'cache_check': 'First 16 greedy tokens match with and without KV cache for every prompt',
        'generation_config': generation.to_dict(),
        'samples': len(results),
        'responses': sum(len(example['responses']) for example in results),
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    sections = ['# SFT response samples', f'Model: `{args.model}`', summary['prompt_handling']]
    for example in results:
        sections.extend([
            f"## {example['task']} — test index {example['test_index']}",
            f"Prompt length: {len(example['prompt_tokens'])} tokens",
            '### Dataset prompt', '````text\n' + example['prompt'] + '\n````',
            '### Dataset target', '````lean\n' + example['target'] + '\n````',
        ])
        for response in example['responses']:
            sections.extend([
                f"### {response['mode']} response ({response['finish_reason']}, {len(response['token_ids'])} tokens)",
                '````text\n' + response['text'] + '\n````',
            ])
    (output_dir / 'responses.md').write_text('\n\n'.join(sections) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect unmodified SFT test prompts and raw model responses.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--test-data', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--samples-per-task', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--max-prompt-tokens', type=int, default=1536)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    main(parser.parse_args())
