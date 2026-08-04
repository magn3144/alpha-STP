"""Conjecture and proof generation for one STP round."""

import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from stp.core.config import Config
from stp.data.datasets import canonical_statement, stable_id
from stp.inference.model import ModelRuntime, generate_texts
from stp.inference.prompts import (
    conjecturer_generation_prompt,
    parse_conjecture,
)
from stp.core.records import (
    Conjecture,
    ConjectureAssessment,
    ConjectureInput,
    ProofRequest,
    SolveAttempt,
    Statement,
)
from stp.proving.solvers import solve_with_alphaproof, solve_with_llm

TRIVIAL_LEMMA = "theorem true : True"
CONJECTURE_SOURCE_THRESHOLD = 0.35


def prepare_conjecture_inputs(
    statements: Sequence[Statement],
    attempts: Sequence[SolveAttempt],
    scores: Sequence[ConjectureAssessment],
    declarations: dict[str, str],
    unproved_count: int,
    trivial_lemma_probability: float,
    lemma_cap_fraction: float,
    seed: int,
) -> list[ConjectureInput]:
    """Prepare paper-style theorem/proof/lemma inputs."""

    by_id = {statement.id: statement for statement in statements}
    rates = {score.statement_id: score.score for score in scores}
    if unproved_count == 0:
        return []

    proved = []
    seen_proofs = set()
    for attempt in attempts:
        key = (attempt.statement_id, attempt.proof)
        if (
            attempt.status == "proved"
            and attempt.proof is not None
            and attempt.statement_id in by_id
            and rates.get(attempt.statement_id, 0.0)
            > CONJECTURE_SOURCE_THRESHOLD
            and key not in seen_proofs
        ):
            seen_proofs.add(key)
            proved.append(attempt)

    rng = random.Random(seed)
    candidates = []
    for attempt in proved:
        lemma_names = [
            name for name in attempt.invoked_lemmas if name in declarations
        ]
        if rng.random() < trivial_lemma_probability:
            lemma_names.append("")
        for name in lemma_names:
            candidates.append((attempt, name))
    rng.shuffle(candidates)

    cap = math.ceil(lemma_cap_fraction * unproved_count)
    lemma_counts: dict[str, int] = defaultdict(int)
    used_pairs = set()
    retained = []
    for attempt, name in candidates:
        pair = (attempt.statement_id, name)
        if pair in used_pairs or lemma_counts[name] >= cap:
            continue
        used_pairs.add(pair)
        lemma_counts[name] += 1
        retained.append((attempt, name))

    weighted = []
    for attempt, name in retained:
        statement = by_id[attempt.statement_id]
        priority = rng.random() ** (1.0 / statement.matching_weight)
        weighted.append((priority, attempt, name))
    weighted.sort(key=lambda item: item[0], reverse=True)

    inputs = []
    for index, (_, attempt, name) in enumerate(weighted[:unproved_count]):
        statement = by_id[attempt.statement_id]
        identity = f"{attempt.request_id}\n{name}"
        input_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        inputs.append(
            ConjectureInput(
                id=input_id,
                seed_statement_id=statement.id,
                seed_statement=statement.statement,
                seed_proof=attempt.proof or "",
                source_attempt_id=attempt.request_id,
                shared_lemma=name,
                shared_lemma_statement=(
                    TRIVIAL_LEMMA if name == "" else declarations[name]
                ),
                trivial_lemma=name == "",
                header=statement.header,
                labels=statement.labels,
                matching_weight=statement.matching_weight,
                generation_seed=seed + index,
            )
        )
    return inputs


def screen_conjectures(conjectures: Sequence[Conjecture]) -> list[Conjecture]:
    """Hook for the future disproof screen."""

    return list(conjectures)


def rank_conjectures(conjectures: Sequence[Conjecture]) -> list[Conjecture]:
    """Hook for the future conjecturer value head."""

    return list(conjectures)


def generate_conjectures(
    config: Config,
    round_index: int,
    inputs: Sequence[ConjectureInput],
    statements: Sequence[Statement],
    maximum: int,
    runtime: ModelRuntime,
) -> list[Conjecture]:
    """Generate, deduplicate, screen, rank, and budget conjectures."""

    generation_inputs = [
        conjecture_input
        for _ in range(config.run.conjecture_multiplier)
        for conjecture_input in inputs
    ]
    prompts = [
        conjecturer_generation_prompt(
            item.shared_lemma_statement,
            item.seed_statement,
            item.seed_proof,
        )
        for item in generation_inputs
    ]
    outputs = generate_texts(
        runtime,
        prompts,
        [
            item.generation_seed + sample * len(inputs)
            for sample in range(config.run.conjecture_multiplier)
            for item in inputs
        ],
        config.model,
        config.solver.temperature,
        config.solver.top_p,
    )
    existing = {statement.id for statement in statements}
    conjectures = []
    for conjecture_input, (text, _, _) in zip(
        generation_inputs,
        outputs,
        strict=True,
    ):
        statement = canonical_statement(parse_conjecture(text))
        conjecture_id = stable_id((conjecture_input.header or "") + statement)
        if conjecture_id in existing:
            continue
        existing.add(conjecture_id)
        conjectures.append(
            Conjecture(
                id=conjecture_id,
                conjecture_input_id=conjecture_input.id,
                statement=statement,
                easy_statement=conjecture_input.seed_statement,
                easy_proof=conjecture_input.seed_proof,
                shared_lemma=conjecture_input.shared_lemma,
                shared_lemma_statement=conjecture_input.shared_lemma_statement,
                header=conjecture_input.header,
                labels=tuple(
                    label
                    for label in conjecture_input.labels
                    if not label.startswith("conjecture ")
                )
                + ("conjecture 1",),
            )
        )
    rng = random.Random(config.run.seed + round_index)
    rng.shuffle(conjectures)
    return rank_conjectures(screen_conjectures(conjectures))[:maximum]


def make_proof_requests(
    statements: Sequence[Statement | Conjecture],
    attempts: int,
    seed: int,
) -> list[ProofRequest]:
    """Expand statements into deterministically seeded requests."""

    requests = []
    for statement_index, statement in enumerate(statements):
        source = "conjecture" if isinstance(statement, Conjecture) else "dataset"
        for attempt in range(attempts):
            requests.append(
                ProofRequest(
                    id=f"{statement.id}:{attempt}",
                    statement_id=statement.id,
                    statement=statement.statement,
                    header=statement.header,
                    attempt=attempt,
                    seed=seed + statement_index * attempts + attempt,
                    source=source,
                )
            )
    return requests


def generate_proofs(
    requests: Sequence[ProofRequest],
    runtime: ModelRuntime | None,
    config: Config,
    artifact_dir: Path,
) -> list[SolveAttempt]:
    """Solve requests with the configured LLM or AlphaProof backend."""

    if config.solver.kind == "llm":
        assert runtime is not None
        return solve_with_llm(requests, runtime, config)
    return solve_with_alphaproof(requests, config, artifact_dir)
