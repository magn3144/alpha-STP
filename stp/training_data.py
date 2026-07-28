"""Pure construction and matching of STP training examples."""

import math
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

from stp.artifacts import load_round_pool, round_dir
from stp.config import Config
from stp.model import (
    embed_texts,
    load_runtime,
    unload_runtime,
)
from stp.prompts import (
    END_THM,
    conjecturer_training_prompt,
    prover_prompt,
)
from stp.records import (
    Conjecture,
    ConjectureAssessment,
    ConjectureFilterMetric,
    SolveAttempt,
    Statement,
    TrainingExample,
)
from stp.scoring import wasserstein_matching
from stp.storage import load_records


def build_prover_examples(
    statements: Sequence[Statement | Conjecture],
    attempts: Sequence[SolveAttempt],
    scores: Sequence[ConjectureAssessment],
    round_index: int,
    proof_threshold: float,
) -> list[TrainingExample]:
    """Build weighted prover examples from verified results."""

    by_id = {statement.id: statement for statement in statements}
    rates = {score.statement_id: score.score for score in scores}
    success_counts: dict[str, int] = defaultdict(int)
    for attempt in attempts:
        if attempt.status == "proved":
            success_counts[attempt.statement_id] += attempt.multiplicity

    examples = []
    seen = set()
    for attempt in attempts:
        statement = by_id.get(attempt.statement_id)
        if (
            statement is None
            or attempt.status != "proved"
            or attempt.proof is None
            or rates[attempt.statement_id] > proof_threshold
        ):
            continue
        key = (attempt.statement_id, attempt.proof)
        if key in seen:
            continue
        seen.add(key)
        weight = math.exp(
            -0.001 * len(attempt.proof) - 0.01 * attempt.verify_seconds
        ) / success_counts[attempt.statement_id]
        examples.append(
            TrainingExample(
                prompt=prover_prompt(statement.statement, statement.header),
                target=attempt.proof,
                weight=weight,
                kind="proof",
                statement_id=statement.id,
                round=round_index,
            )
        )
    return examples


def build_conjecturer_examples(
    conjectures: Sequence[Conjecture],
    attempts: Sequence[SolveAttempt],
    scores: Sequence[ConjectureAssessment],
    round_index: int,
    pass_rate_threshold: float,
    elegance_drop_fraction: float,
    unfocused_ratio: int,
    unfocused_minimum: int,
    seed: int,
) -> tuple[list[TrainingExample], list[ConjectureFilterMetric]]:
    """Filter conjectures and build unweighted conjecturer examples."""

    rates = {score.statement_id: score.score for score in scores}
    grouped: dict[str, list[SolveAttempt]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt.statement_id].append(attempt)

    named = []
    trivial = []
    metrics: dict[str, ConjectureFilterMetric] = {}
    for conjecture in conjectures:
        successful = [
            attempt
            for attempt in grouped.get(conjecture.id, [])
            if attempt.status == "proved" and attempt.proof is not None
        ]
        successes = sum(attempt.multiplicity for attempt in successful)
        rate = rates.get(conjecture.id, 0.0)
        reused = any(
            conjecture.shared_lemma in attempt.invoked_lemmas
            for attempt in successful
        )
        minimum_length = (
            min(len(attempt.proof or "") for attempt in successful)
            if successful
            else None
        )
        if not 0.0 < rate <= pass_rate_threshold:
            reason = "pass_rate"
        elif not conjecture.shared_lemma and successful:
            reason = None
            trivial.append(conjecture)
        elif successful and reused:
            reason = None
            named.append(conjecture)
        else:
            reason = "lemma_not_reused"
        metrics[conjecture.id] = ConjectureFilterMetric(
            statement_id=conjecture.id,
            pass_rate=rate,
            successes=successes,
            trivial_lemma=conjecture.shared_lemma == "",
            named_lemma_reused=reused,
            minimum_proof_length=minimum_length,
            elegance_score=(
                minimum_length / len(conjecture.statement)
                if minimum_length is not None
                else None
            ),
            selected=False,
            rejection_reason=reason,
        )

    rng = random.Random(seed)
    rng.shuffle(named)
    rng.shuffle(trivial)
    trivial_limit = max(unfocused_ratio * len(named), unfocused_minimum)
    for conjecture in trivial[trivial_limit:]:
        metrics[conjecture.id] = replace(
            metrics[conjecture.id],
            rejection_reason="unfocused_cap",
        )
    candidates = [*named, *trivial[:trivial_limit]]

    elegance_scores = []
    for conjecture in candidates:
        score = metrics[conjecture.id].elegance_score
        assert score is not None
        elegance_scores.append(score)
    elegance_scores.sort()
    cutoff = (
        elegance_scores[
            min(
                int(len(elegance_scores) * elegance_drop_fraction),
                len(elegance_scores) - 1,
            )
        ]
        if elegance_scores
        else math.inf
    )
    selected = []
    for conjecture in candidates:
        metric = metrics[conjecture.id]
        assert metric.elegance_score is not None
        if metric.elegance_score < cutoff:
            metrics[conjecture.id] = replace(
                metric,
                rejection_reason="elegance",
            )
            continue
        metrics[conjecture.id] = replace(
            metric,
            selected=True,
            rejection_reason=None,
        )
        selected.append(
            TrainingExample(
                prompt=conjecturer_training_prompt(
                    conjecture.shared_lemma_statement,
                    conjecture.easy_statement,
                    conjecture.easy_proof,
                ),
                target=(
                    f"\n{conjecture.statement.removesuffix(':= by').strip()}\n"
                    f"{END_THM}"
                ),
                weight=1.0,
                kind="conjecture",
                statement_id=conjecture.id,
                round=round_index,
            )
        )
    return selected, [metrics[item.id] for item in conjectures]


def deduplicate_examples(
    examples: Iterable[TrainingExample],
) -> list[TrainingExample]:
    """Deduplicate prompt/target pairs while keeping the first."""

    result = []
    seen = set()
    for example in examples:
        key = (example.prompt, example.target)
        if key not in seen:
            seen.add(key)
            result.append(example)
    return result


def replay_prover_examples(
    config: Config,
    round_index: int,
) -> list[TrainingExample]:
    """Build prover examples from the configured replay window."""

    examples = []
    first = max(0, round_index - config.run.replay_rounds + 1)
    for replay_round in range(first, round_index + 1):
        directory = round_dir(config, replay_round)
        examples.extend(
            build_prover_examples(
                load_round_pool(directory),
                load_records(
                    directory / "solve_attempts.jsonl",
                    SolveAttempt,
                ),
                load_records(
                    directory / "assessments.jsonl",
                    ConjectureAssessment,
                ),
                replay_round,
                config.run.training_proof_threshold,
            )
        )
    return deduplicate_examples(examples)


def match_conjecturer_examples(
    config: Config,
    examples: Sequence[TrainingExample],
    statements: Sequence[Statement],
    model_path: str | Path,
    tokenizer_path: str | Path,
) -> list[TrainingExample]:
    """Match conjecture examples to the unproved source distribution."""

    if not examples or not statements:
        return []
    runtime = load_runtime(model_path, tokenizer_path)
    candidate_embeddings = embed_texts(
        runtime,
        [example.target for example in examples],
        config.model,
    )
    target_embeddings = embed_texts(
        runtime,
        [statement.statement for statement in statements],
        config.model,
    )
    unload_runtime(runtime)
    matched = wasserstein_matching(
        examples,
        candidate_embeddings,
        target_embeddings,
        [statement.matching_weight for statement in statements],
        config.run.wasserstein_max_weight,
    )
    return [replace(example, weight=weight) for example, weight in matched]
