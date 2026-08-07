"""Fine-tune a causal model on complete Lean proofs with STP target loss."""

import argparse
import json
import math
import random
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import torch
import torch.nn.functional as functional
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from stp.core.records import TrainingExample
from stp.data.datasets import load_sft_examples
from stp.data.storage import read_json, write_json
from stp.finetuning.sft_logger import SFTLogger
from stp.inference.model import (
    ModelRuntime,
    collate_training_examples,
    load_runtime,
    load_tokenizer,
    unload_runtime,
)


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPOSITORY / "models/codegen2-1B_P"
DEFAULT_TRAIN_DATA = (
    REPOSITORY / "data/dataset/leantree_mathlib_sft/train.jsonl"
)
DEFAULT_VALIDATION_DATA = (
    REPOSITORY / "data/dataset/leantree_mathlib_sft/validation.jsonl"
)
DEFAULT_RUNS_DIR = REPOSITORY / "runs"
CONFIG_FILE = "config.json"
DATASET_STATS_FILE = "dataset_stats.json"
VALIDATION_SUBSET_FILE = "validation_subset.json"
METRICS_FILE = "metrics.jsonl"


@dataclass(frozen=True)
class DatasetStats:
    """Token-length filtering counts for one supervised dataset split."""

    records_read: int
    examples_kept: int
    proofs_too_long: int
    prompts_truncated: int
    target_tokens: int


@dataclass(frozen=True)
class BatchMetrics:
    """Loss and teacher-forced correctness values for one causal batch."""

    loss: torch.Tensor
    target_tokens: int
    correct_tokens: int
    exact_proofs: int


def positive_int(value: str) -> int:
    """Parse a positive command-line integer and return it."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert command-line paths to strings and return JSON-compatible values."""

    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def model_revision(model_path: Path) -> str:
    """Read and return the Hugging Face snapshot revision for local weights."""

    metadata = (
        model_path
        / ".cache/huggingface/download/pytorch_model.bin.metadata"
    )
    return metadata.read_text(encoding="utf-8").splitlines()[0]


def select_examples(
    examples: Sequence[TrainingExample],
    tokenizer: Any,
    max_sequence_length: int,
) -> tuple[list[TrainingExample], DatasetStats]:
    """Reject overlong proofs and return usable examples with length statistics."""

    selected = []
    proofs_too_long = 0
    prompts_truncated = 0
    target_tokens = 0
    for example in tqdm(examples, desc="Checking token lengths"):
        prompt_length = len(
            tokenizer.encode(example.prompt, add_special_tokens=False)
        )
        proof_length = len(
            tokenizer.encode(example.target, add_special_tokens=False)
        )
        if tokenizer.eos_token_id is not None:
            proof_length += 1
        if proof_length > max_sequence_length:
            proofs_too_long += 1
            continue
        if prompt_length + proof_length > max_sequence_length:
            prompts_truncated += 1
        selected.append(example)
        target_tokens += proof_length
    return selected, DatasetStats(
        records_read=len(examples),
        examples_kept=len(selected),
        proofs_too_long=proofs_too_long,
        prompts_truncated=prompts_truncated,
        target_tokens=target_tokens,
    )


def print_dataset_stats(name: str, stats: DatasetStats) -> None:
    """Print one split's kept, rejected, truncated, and token counts."""

    print(
        f"{name}: read {stats.records_read:,}, kept {stats.examples_kept:,}, "
        f"rejected {stats.proofs_too_long:,} long proofs, left-truncated "
        f"{stats.prompts_truncated:,} prompts, retained "
        f"{stats.target_tokens:,} target tokens",
        flush=True,
    )


def batch_metrics(
    runtime: ModelRuntime,
    batch: dict[str, torch.Tensor],
) -> BatchMetrics:
    """Compute weighted target-only loss and teacher-forced correctness."""

    input_ids = batch["input_ids"].to("cuda", non_blocking=True)
    labels = batch["labels"].to("cuda", non_blocking=True)
    attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
    weights = batch["weights"].to("cuda", non_blocking=True)
    autocast = (
        torch.autocast(
            device_type="cuda",
            dtype=runtime.policy.autocast_dtype,
        )
        if runtime.policy.autocast_dtype is not None
        else nullcontext()
    )
    with autocast:
        logits = runtime.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        shifted_labels = labels[:, 1:]
        token_losses = functional.cross_entropy(
            logits[:, :-1].transpose(1, 2),
            shifted_labels,
            reduction="none",
            ignore_index=-100,
        )
        target_mask = shifted_labels != -100
        example_losses = (
            (token_losses * target_mask).sum(dim=1)
            / target_mask.sum(dim=1)
        )
        loss = (example_losses * weights).sum() / weights.sum()

    predictions = logits[:, :-1].argmax(dim=-1)
    correct = predictions.eq(shifted_labels) & target_mask
    exact = (correct | ~target_mask).all(dim=1)
    return BatchMetrics(
        loss=loss,
        target_tokens=int(target_mask.sum().item()),
        correct_tokens=int(correct.sum().item()),
        exact_proofs=int(exact.sum().item()),
    )


def make_loader(
    examples: Sequence[TrainingExample],
    tokenizer: Any,
    max_sequence_length: int,
    batch_size: int,
) -> DataLoader[Any]:
    """Create an ordered dynamically padded loader and return its batches."""

    return DataLoader(
        cast(Any, list(examples)),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: collate_training_examples(
            batch,
            tokenizer,
            max_sequence_length,
        ),
    )


def validate(
    runtime: ModelRuntime,
    examples: Sequence[TrainingExample],
    args: argparse.Namespace,
) -> dict[str, float | int]:
    """Evaluate target loss and teacher-forced proof metrics without updates."""

    loader = make_loader(
        examples,
        runtime.tokenizer,
        args.max_sequence_length,
        args.validation_batch_size,
    )
    runtime.model.eval()
    loss_total = 0.0
    examples_seen = 0
    target_tokens = 0
    correct_tokens = 0
    exact_proofs = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Validating"):
            metrics = batch_metrics(runtime, batch)
            batch_examples = int(batch["input_ids"].shape[0])
            loss_total += float(metrics.loss.detach().cpu()) * batch_examples
            examples_seen += batch_examples
            target_tokens += metrics.target_tokens
            correct_tokens += metrics.correct_tokens
            exact_proofs += metrics.exact_proofs
    loss = loss_total / examples_seen
    runtime.model.train()
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "token_accuracy": correct_tokens / target_tokens,
        "teacher_forced_proof_accuracy": exact_proofs / examples_seen,
        "target_tokens": target_tokens,
    }


def learning_rate_scale(
    step: int,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Return linear warmup followed by linear decay for one update."""

    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    return max(
        0.0,
        (total_steps - step) / max(1, total_steps - warmup_steps),
    )


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the checkpoint with the greatest optimizer step, if present."""

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints = sorted(
        checkpoint
        for checkpoint in checkpoints_dir.glob("checkpoint_step_*")
        if (checkpoint / "trainer_state.pt").is_file()
    )
    return checkpoints[-1] if checkpoints else None


def trainer_state(
    epoch: int,
    next_batch: int,
    update: int,
    examples_seen: int,
    target_tokens_seen: int,
    epoch_loss_total: float,
    epoch_examples: int,
    epoch_target_tokens: int,
    optimizer: AdamW,
    scheduler: Any,
    scaler: torch.GradScaler,
) -> dict[str, Any]:
    """Collect resumable model-independent training state and return it."""

    return {
        "epoch": epoch,
        "next_batch": next_batch,
        "update": update,
        "examples_seen": examples_seen,
        "target_tokens_seen": target_tokens_seen,
        "epoch_loss_total": epoch_loss_total,
        "epoch_examples": epoch_examples,
        "epoch_target_tokens": epoch_target_tokens,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all(),
    }


def save_checkpoint(
    run_dir: Path,
    runtime: ModelRuntime,
    state: dict[str, Any],
) -> Path:
    """Atomically save Hugging Face weights and resumable trainer state."""

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    update = int(state["update"])
    checkpoint = checkpoints_dir / f"checkpoint_step_{update:08d}"
    temporary = checkpoints_dir / f"checkpoint_step_{update:08d}.tmp"
    temporary.mkdir()
    runtime.model.save_pretrained(temporary)
    runtime.tokenizer.save_pretrained(temporary)
    torch.save(state, temporary / "trainer_state.pt")
    temporary.rename(checkpoint)
    return checkpoint


def restore_trainer_state(
    checkpoint: Path,
    optimizer: AdamW,
    scheduler: Any,
    scaler: torch.GradScaler,
) -> dict[str, Any]:
    """Restore optimizer, scheduler, scaler, and random state from a checkpoint."""

    state: dict[str, Any] = torch.load(
        checkpoint / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    torch.cuda.set_rng_state_all(state["cuda_random_state"])
    return state


def append_metrics(path: Path, metrics: dict[str, Any]) -> None:
    """Append one completed epoch's local training and validation metrics."""

    with path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(metrics) + "\n")


def validation_subset_indices(
    run_dir: Path,
    validation_size: int,
    sample_size: int,
    seed: int,
    resume: bool,
) -> list[int]:
    """Create or restore fixed validation indices and return them."""

    path = run_dir / VALIDATION_SUBSET_FILE
    if resume:
        return [int(index) for index in read_json(path)]
    indices = random.Random(seed).sample(
        range(validation_size),
        k=min(sample_size, validation_size),
    )
    write_json(path, indices)
    return indices


def ordered_epoch_examples(
    examples: Sequence[TrainingExample],
    seed: int,
    epoch: int,
) -> list[TrainingExample]:
    """Return examples in the deterministic shuffled order for one epoch."""

    generator = torch.Generator().manual_seed(seed + epoch)
    indices = torch.randperm(len(examples), generator=generator).tolist()
    return [examples[index] for index in indices]


def train_epoch(
    runtime: ModelRuntime,
    train_examples: Sequence[TrainingExample],
    frequent_validation_examples: Sequence[TrainingExample],
    optimizer: AdamW,
    scheduler: Any,
    scaler: torch.GradScaler,
    logger: SFTLogger,
    run_dir: Path,
    args: argparse.Namespace,
    state: dict[str, Any],
    updates_per_epoch: int,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Train or resume one epoch and return aggregate metrics and next state."""

    epoch = int(state["epoch"])
    start_batch = int(state["next_batch"])
    ordered = ordered_epoch_examples(train_examples, args.seed, epoch)
    total_batches = math.ceil(len(ordered) / args.train_microbatch_size)
    remaining = ordered[start_batch * args.train_microbatch_size :]
    loader = make_loader(
        remaining,
        runtime.tokenizer,
        args.max_sequence_length,
        args.train_microbatch_size,
    )
    checkpoint_targets = {
        (epoch - 1) * updates_per_epoch
        + math.ceil(index * updates_per_epoch / args.checkpoints_per_epoch)
        for index in range(1, args.checkpoints_per_epoch)
        if math.ceil(index * updates_per_epoch / args.checkpoints_per_epoch)
        < updates_per_epoch
    }
    runtime.model.train()
    optimizer.zero_grad(set_to_none=True)
    group_loss_total = 0.0
    group_examples = 0

    for batch_index, batch in enumerate(
        tqdm(loader, desc=f"Training epoch {epoch}"),
        start=start_batch,
    ):
        metrics = batch_metrics(runtime, batch)
        batch_examples = int(batch["input_ids"].shape[0])
        group_start = (
            batch_index // args.gradient_accumulation_steps
        ) * args.gradient_accumulation_steps
        group_end = min(
            group_start + args.gradient_accumulation_steps,
            total_batches,
        )
        first_example = group_start * args.train_microbatch_size
        final_example = min(
            group_end * args.train_microbatch_size,
            len(ordered),
        )
        accumulation_examples = final_example - first_example
        scaled_loss = metrics.loss * batch_examples / accumulation_examples
        scaler.scale(scaled_loss).backward()

        detached_loss = float(metrics.loss.detach().cpu())
        group_loss_total += detached_loss * batch_examples
        group_examples += batch_examples
        state["epoch_loss_total"] += detached_loss * batch_examples
        state["epoch_examples"] += batch_examples
        state["epoch_target_tokens"] += metrics.target_tokens
        state["examples_seen"] += batch_examples
        state["target_tokens_seen"] += metrics.target_tokens
        state["next_batch"] = batch_index + 1

        if batch_index + 1 == group_end:
            learning_rate = float(optimizer.param_groups[0]["lr"])
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                runtime.model.parameters(),
                args.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["update"] += 1
            update = int(state["update"])

            if update % args.log_every == 0:
                logger.log_training(
                    update,
                    group_loss_total / group_examples,
                    learning_rate,
                    int(state["examples_seen"]),
                    int(state["target_tokens_seen"]),
                )
            group_loss_total = 0.0
            group_examples = 0

            if update in checkpoint_targets:
                checkpoint = save_checkpoint(
                    run_dir,
                    runtime,
                    trainer_state(
                        epoch,
                        int(state["next_batch"]),
                        update,
                        int(state["examples_seen"]),
                        int(state["target_tokens_seen"]),
                        float(state["epoch_loss_total"]),
                        int(state["epoch_examples"]),
                        int(state["epoch_target_tokens"]),
                        optimizer,
                        scheduler,
                        scaler,
                    ),
                )
                print(f"Saved {checkpoint}", flush=True)

            if (
                update % args.validation_interval == 0
                and int(state["next_batch"]) < total_batches
            ):
                validation_metrics = validate(
                    runtime,
                    frequent_validation_examples,
                    args,
                )
                logger.log_validation(
                    update,
                    validation_metrics,
                    len(frequent_validation_examples),
                )

    training_metrics: dict[str, float | int] = {
        "loss": float(state["epoch_loss_total"])
        / int(state["epoch_examples"]),
        "examples": int(state["epoch_examples"]),
        "target_tokens": int(state["epoch_target_tokens"]),
    }
    next_state = trainer_state(
        epoch + 1,
        0,
        int(state["update"]),
        int(state["examples_seen"]),
        int(state["target_tokens_seen"]),
        0.0,
        0,
        0,
        optimizer,
        scheduler,
        scaler,
    )
    return training_metrics, next_state


def save_final_model(run_dir: Path, runtime: ModelRuntime) -> Path:
    """Save the final Hugging Face model and tokenizer and return its directory."""

    output_dir = run_dir / "checkpoint"
    output_dir.mkdir(exist_ok=True)
    runtime.model.config.use_cache = True
    runtime.model.save_pretrained(output_dir)
    runtime.tokenizer.save_pretrained(output_dir)
    return output_dir


def train(args: argparse.Namespace) -> Path:
    """Run or resume whole-proof causal supervised fine-tuning."""

    run_dir = args.runs_dir / args.run_name
    config_path = run_dir / CONFIG_FILE
    if args.resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"SFT run does not exist: {run_dir}")
        saved_config = read_json(config_path)
        args.wandb_run_id = str(saved_config["wandb_run_id"])
    elif run_dir.exists():
        raise FileExistsError(f"SFT run already exists: {run_dir}")
    else:
        run_dir.mkdir(parents=True)
        config = serializable_args(args)
        config["model_revision"] = model_revision(args.model)
        write_json(config_path, config)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = load_tokenizer(args.tokenizer)
    raw_train_examples = load_sft_examples(args.train_data, "stp", tokenizer)
    raw_validation_examples = load_sft_examples(
        args.validation_data,
        "stp",
        tokenizer,
    )
    if args.num_examples is not None:
        raw_train_examples = raw_train_examples[: args.num_examples]
    if args.num_validation_examples is not None:
        raw_validation_examples = raw_validation_examples[
            : args.num_validation_examples
        ]
    train_examples, train_stats = select_examples(
        raw_train_examples,
        tokenizer,
        args.max_sequence_length,
    )
    validation_examples, validation_stats = select_examples(
        raw_validation_examples,
        tokenizer,
        args.max_sequence_length,
    )
    if not train_examples:
        raise ValueError("No training examples remain after length filtering.")
    if not validation_examples:
        raise ValueError("No validation examples remain after length filtering.")
    print_dataset_stats("Train", train_stats)
    print_dataset_stats("Validation", validation_stats)
    write_json(
        run_dir / DATASET_STATS_FILE,
        {
            "train": asdict(train_stats),
            "validation": asdict(validation_stats),
            "train_examples_used": len(train_examples),
            "validation_examples_used": len(validation_examples),
        },
    )
    subset_indices = validation_subset_indices(
        run_dir,
        len(validation_examples),
        args.validation_samples,
        args.seed,
        args.resume,
    )
    frequent_validation_examples = [
        validation_examples[index] for index in subset_indices
    ]

    checkpoint = latest_checkpoint(run_dir) if args.resume else None
    runtime = load_runtime(
        checkpoint or args.model,
        checkpoint or args.tokenizer,
    )
    del tokenizer
    runtime.model.train()
    runtime.model.config.use_cache = False
    if args.gradient_checkpointing:
        runtime.model.gradient_checkpointing_enable()

    train_batches = math.ceil(
        len(train_examples) / args.train_microbatch_size
    )
    updates_per_epoch = math.ceil(
        train_batches / args.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * args.epochs
    optimizer = AdamW(
        runtime.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_scale(
            step,
            args.warmup_steps,
            total_updates,
        ),
    )
    scaler = torch.GradScaler(
        "cuda",
        enabled=runtime.policy.use_grad_scaler,
    )
    state: dict[str, Any] = {
        "epoch": 1,
        "next_batch": 0,
        "update": 0,
        "examples_seen": 0,
        "target_tokens_seen": 0,
        "epoch_loss_total": 0.0,
        "epoch_examples": 0,
        "epoch_target_tokens": 0,
    }
    if checkpoint is not None:
        state = restore_trainer_state(
            checkpoint,
            optimizer,
            scheduler,
            scaler,
        )

    logger = SFTLogger(
        args.run_name,
        args.wandb_run_id,
        args.wandb_name,
        args.wandb_mode,
        args.resume,
        serializable_args(args),
    )
    metrics_path = run_dir / METRICS_FILE
    try:
        while int(state["epoch"]) <= args.epochs:
            epoch = int(state["epoch"])
            training_metrics, state = train_epoch(
                runtime,
                train_examples,
                frequent_validation_examples,
                optimizer,
                scheduler,
                scaler,
                logger,
                run_dir,
                args,
                state,
                updates_per_epoch,
            )
            validation_metrics = validate(
                runtime,
                validation_examples,
                args,
            )
            metrics = {
                "epoch": epoch,
                "train": training_metrics,
                "validation": validation_metrics,
            }
            append_metrics(metrics_path, metrics)
            logger.log_epoch(
                int(state["update"]),
                epoch,
                training_metrics,
                validation_metrics,
                len(validation_examples),
            )
            checkpoint = save_checkpoint(run_dir, runtime, state)
            print(
                f"Finished epoch {epoch}/{args.epochs}: train loss "
                f"{training_metrics['loss']:.4f}, validation loss "
                f"{validation_metrics['loss']:.4f}; saved {checkpoint}",
                flush=True,
            )
        output_dir = save_final_model(run_dir, runtime)
    except BaseException:
        logger.finish(exit_code=1)
        raise
    logger.finish(exit_code=0)
    unload_runtime(runtime)
    print(f"Saved final model to {output_dir}", flush=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    """Parse supervised whole-proof training arguments and return them."""

    parser = argparse.ArgumentParser(
        description="Fine-tune a causal model on complete Lean proofs."
    )
    parser.add_argument("run_name", help="Directory name under --runs-dir.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=DEFAULT_VALIDATION_DATA,
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--epochs", type=positive_int, required=True)
    parser.add_argument("--max-sequence-length", type=positive_int, default=2048)
    parser.add_argument("--train-microbatch-size", type=positive_int, default=1)
    parser.add_argument("--validation-batch-size", type=positive_int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=positive_int,
        default=32,
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoints-per-epoch", type=positive_int, default=1)
    parser.add_argument("--log-every", type=positive_int, default=100)
    parser.add_argument("--validation-interval", type=positive_int, default=500)
    parser.add_argument("--validation-samples", type=positive_int, default=512)
    parser.add_argument("--num-examples", type=positive_int)
    parser.add_argument("--num-validation-examples", type=positive_int)
    parser.add_argument("--wandb-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.tokenizer = args.tokenizer or args.model
    args.wandb_run_id = uuid.uuid4().hex

    if Path(args.run_name).name != args.run_name:
        parser.error("run_name must be a single directory name")
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if not args.tokenizer.is_dir():
        parser.error(f"tokenizer directory does not exist: {args.tokenizer}")
    if not args.train_data.is_file():
        parser.error(f"training JSONL does not exist: {args.train_data}")
    if not args.validation_data.is_file():
        parser.error(
            f"validation JSONL does not exist: {args.validation_data}"
        )
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")
    return args


def main() -> None:
    """Run supervised fine-tuning and print its output directory."""

    run_dir = train(parse_args())
    print(f"Training complete. Outputs saved under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
