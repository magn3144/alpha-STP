"""Prepare examples and train one STP checkpoint."""

from pathlib import Path
from typing import Sequence

from stp.core.config import Config
from stp.data.datasets import load_sft_examples
from stp.inference.model import load_tokenizer, train_model
from stp.core.records import (
    Conjecture,
    ConjectureAssessment,
    ConjectureFilterMetric,
    SolveAttempt,
    Statement,
    TrainingExample,
)
from stp.self_play.scoring import unproved_statements
from stp.data.storage import load_records, write_jsonl
from stp.self_play.training_data import (
    build_conjecturer_examples,
    deduplicate_examples,
    match_conjecturer_examples,
    replay_prover_examples,
)


def prepare_training_examples(
    config: Config,
    round_index: int,
    statements: Sequence[Statement],
    conjectures: Sequence[Conjecture],
    attempts: Sequence[SolveAttempt],
    scores: Sequence[ConjectureAssessment],
    previous_scores: Sequence[ConjectureAssessment],
    model_path: str | Path,
    tokenizer_path: str | Path,
    directory: Path,
) -> list[TrainingExample]:
    """Build or resume all examples used in this round."""

    training_path = directory / "training_examples.jsonl"
    if training_path.exists():
        return load_records(training_path, TrainingExample)

    tokenizer = load_tokenizer(tokenizer_path)

    conjecture_path = directory / "conjecture_training_examples.jsonl"
    metrics_path = directory / "conjecture_filter_metrics.jsonl"
    if conjecture_path.exists():
        conjecture_examples = load_records(conjecture_path, TrainingExample)
        load_records(metrics_path, ConjectureFilterMetric)
    else:
        unweighted, metrics = build_conjecturer_examples(
            conjectures,
            attempts,
            scores,
            round_index,
            config.run.conjecture_threshold,
            config.run.elegance_drop_fraction,
            config.run.unfocused_example_ratio,
            config.run.unfocused_example_minimum,
            config.run.seed + round_index,
        )
        targets = unproved_statements(
            statements,
            [*previous_scores, *scores],
        )
        conjecture_examples = match_conjecturer_examples(
            config,
            unweighted,
            targets,
            model_path,
            tokenizer_path,
        )
        write_jsonl(metrics_path, metrics)
        write_jsonl(conjecture_path, conjecture_examples)

    examples = deduplicate_examples(
        [
            *load_sft_examples(
                config.data.sft_dataset,
                config.model.prover_handler,
                tokenizer,
            ),
            *replay_prover_examples(config, round_index, tokenizer),
            *conjecture_examples,
        ]
    )
    write_jsonl(training_path, examples)
    return examples


def train_round(
    config: Config,
    round_index: int,
    examples: Sequence[TrainingExample],
    model_path: str | Path,
    tokenizer_path: str | Path,
    directory: Path,
) -> Path:
    """Train or resume this round's full-model checkpoint."""

    checkpoint = directory / "checkpoint"
    if (checkpoint / "config.json").exists():
        return checkpoint
    if not examples:
        raise ValueError("No SFT or self-play training examples are available.")
    train_model(
        model_path,
        tokenizer_path,
        checkpoint,
        examples,
        config.model,
        config.run.seed + round_index,
    )
    return checkpoint
