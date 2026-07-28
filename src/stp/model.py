import gc
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import torch
import torch.nn.functional as functional
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from stp.config import ModelSettings
from stp.records import TrainingExample
from stp.storage import write_json


@dataclass(frozen=True)
class GpuPolicy:
    """Deterministic mixed-precision policy selected from GPU capability."""

    name: str
    dtype: torch.dtype
    autocast_dtype: torch.dtype
    attention: str
    use_grad_scaler: bool


@dataclass
class ModelRuntime:
    """Loaded causal model resources owned by one pipeline phase."""

    model: Any
    tokenizer: Any
    policy: GpuPolicy


def select_gpu_policy() -> GpuPolicy:
    """Choose A100 BF16 or V100 FP16 behavior from CUDA capability."""

    if not torch.cuda.is_available():
        raise RuntimeError("STP requires one CUDA GPU.")
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return GpuPolicy(
            name="a100_bf16",
            dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
            attention="sdpa",
            use_grad_scaler=False,
        )
    if major == 7:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return GpuPolicy(
            name="v100_fp16",
            dtype=torch.float16,
            autocast_dtype=torch.float16,
            attention="eager",
            use_grad_scaler=True,
        )
    raise RuntimeError(f"Unsupported CUDA compute capability {major}.")


def load_runtime(
    model_path: str | Path,
    tokenizer_path: str | Path,
) -> ModelRuntime:
    """Load a causal model and tokenizer onto the single CUDA GPU."""

    policy = select_gpu_policy()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=policy.dtype,
        attn_implementation=policy.attention,
    )
    model.to("cuda")
    model.eval()
    return ModelRuntime(model=model, tokenizer=tokenizer, policy=policy)


def unload_runtime(runtime: ModelRuntime) -> None:
    """Release model memory before the next GPU-intensive phase."""

    runtime.model.to("cpu")
    del runtime.model
    del runtime.tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def right_truncate_prompt(
    prompt: str,
    tokenizer: Any,
    max_tokens: int,
) -> str:
    """Apply the original STP truncate-decode prompt preprocessing."""

    tokens = tokenizer.encode(
        prompt,
        return_tensors="pt",
        padding="longest",
        max_length=max_tokens,
        truncation=True,
    )[0]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def generate_texts(
    runtime: ModelRuntime,
    prompts: Sequence[str],
    seeds: Sequence[int],
    settings: ModelSettings,
    temperature: float,
    top_p: float,
) -> list[tuple[str, int, float]]:
    """Generate batched completions and return text, token count, and time."""

    results = []
    for start in tqdm(
        range(0, len(prompts), settings.generation_batch_size),
        desc="Generating",
    ):
        batch_prompts = [
            right_truncate_prompt(
                prompt,
                runtime.tokenizer,
                settings.max_sequence_length,
            )
            for prompt in prompts[
                start : start + settings.generation_batch_size
            ]
        ]
        batch_seeds = seeds[start : start + settings.generation_batch_size]
        torch.manual_seed(batch_seeds[0])
        encoded = runtime.tokenizer(
            list(batch_prompts),
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        input_length = encoded["input_ids"].shape[1]
        started = time.perf_counter()
        with torch.inference_mode():
            output = runtime.model.generate(
                **encoded,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                max_new_tokens=settings.max_new_tokens,
                pad_token_id=runtime.tokenizer.pad_token_id,
                eos_token_id=runtime.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        generated = output[:, input_length:]
        texts = runtime.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )
        for text, tokens in zip(texts, generated, strict=True):
            token_count = int((tokens != runtime.tokenizer.pad_token_id).sum())
            results.append((text, token_count, elapsed / len(texts)))
    return results


def embed_texts(
    runtime: ModelRuntime,
    texts: Sequence[str],
    settings: ModelSettings,
) -> np.ndarray:
    """Mean-pool final causal-model hidden states for curriculum matching."""

    embeddings = []
    for start in tqdm(
        range(0, len(texts), settings.embedding_batch_size),
        desc="Embedding",
    ):
        batch = texts[start : start + settings.embedding_batch_size]
        encoded = runtime.tokenizer(
            list(batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.max_sequence_length,
        ).to("cuda")
        with torch.inference_mode():
            output = runtime.model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = output.hidden_states[-1]
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings.append(pooled.float().cpu().numpy())
    return np.concatenate(embeddings) if embeddings else np.empty((0, 0))


def _encode_example(
    example: TrainingExample,
    tokenizer: Any,
    max_length: int,
) -> tuple[list[int], list[int], float]:
    """Tokenize one target-only training example."""

    prompt_ids = tokenizer.encode(example.prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(example.target, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        target_ids.append(tokenizer.eos_token_id)
    overflow = max(0, len(prompt_ids) + len(target_ids) - max_length)
    prompt_ids = prompt_ids[overflow:]
    if len(prompt_ids) + len(target_ids) > max_length:
        target_ids = target_ids[: max_length - len(prompt_ids)]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return input_ids, labels, example.weight


def _collate(
    examples: Sequence[TrainingExample],
    tokenizer: Any,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Pad encoded examples into one weighted training batch."""

    encoded = [
        _encode_example(example, tokenizer, max_length) for example in examples
    ]
    length = max(len(item[0]) for item in encoded)
    input_ids = []
    labels = []
    masks = []
    weights = []
    for ids, targets, weight in encoded:
        padding = length - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * padding)
        labels.append(targets + [-100] * padding)
        masks.append([1] * len(ids) + [0] * padding)
        weights.append(weight)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "weights": torch.tensor(weights, dtype=torch.float32),
    }


def _learning_rate_scale(step: int, warmup_steps: int, total_steps: int) -> float:
    """Return linear warmup followed by linear decay."""

    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def train_model(
    model_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: Path,
    examples: Sequence[TrainingExample],
    settings: ModelSettings,
    seed: int,
) -> dict[str, float | int | str]:
    """Fully train a causal model with weighted target-only loss."""

    random.seed(seed)
    torch.manual_seed(seed)
    runtime = load_runtime(model_path, tokenizer_path)
    model = runtime.model
    tokenizer = runtime.tokenizer
    model.train()
    model.config.use_cache = False
    if settings.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        cast(Any, list(examples)),
        batch_size=settings.train_microbatch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: _collate(
            batch,
            tokenizer,
            settings.max_sequence_length,
        ),
    )
    updates_per_epoch = math.ceil(
        len(loader) / settings.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * settings.epochs
    optimizer = AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _learning_rate_scale(
            step,
            settings.warmup_steps,
            total_updates,
        ),
    )
    scaler = torch.GradScaler(
        "cuda",
        enabled=runtime.policy.use_grad_scaler,
    )

    optimizer.zero_grad(set_to_none=True)
    losses = []
    update = 0
    for _ in range(settings.epochs):
        for batch_index, batch in enumerate(tqdm(loader, desc="Training")):
            input_ids = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            weights = batch["weights"].to("cuda")
            with torch.autocast(
                device_type="cuda",
                dtype=runtime.policy.autocast_dtype,
            ):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                token_losses = functional.cross_entropy(
                    logits[:, :-1].transpose(1, 2),
                    labels[:, 1:],
                    reduction="none",
                    ignore_index=-100,
                )
                target_mask = labels[:, 1:] != -100
                example_losses = (
                    (token_losses * target_mask).sum(dim=1)
                    / target_mask.sum(dim=1).clamp_min(1)
                )
                loss = (example_losses * weights).sum() / weights.sum()
                scaled_loss = loss / settings.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            losses.append(float(loss.detach().cpu()))

            final_batch = batch_index + 1 == len(loader)
            accumulation_end = (
                batch_index + 1
            ) % settings.gradient_accumulation_steps == 0
            if accumulation_end or final_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics: dict[str, float | int | str] = {
        "gpu_policy": runtime.policy.name,
        "examples": len(examples),
        "updates": update,
        "mean_loss": sum(losses) / len(losses),
    }
    write_json(output_dir / "training_metrics.json", metrics)
    unload_runtime(runtime)
    return metrics
