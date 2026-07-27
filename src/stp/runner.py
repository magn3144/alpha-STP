import random
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from stp.algorithm import (
    assess_conjectures,
    build_training_examples,
    conjecture_prompt,
    deduplicate_training_examples,
    make_requests,
    parse_conjecture,
    rank_conjectures,
    screen_conjectures,
    select_conjecture_sources,
    select_dataset_statements,
    wasserstein_matching,
)
from stp.config import Config
from stp.data import (
    canonical_statement,
    load_sft_examples,
    load_statements,
    stable_id,
)
from stp.model import (
    ModelRuntime,
    generate_texts,
    embed_texts,
    load_runtime,
    train_model,
    unload_runtime,
)
from stp.records import (
    Conjecture,
    ConjectureAssessment,
    RoundState,
    SolveAttempt,
    Statement,
    TrainingExample,
)
from stp.solvers import solve_with_alphaproof, solve_with_llm
from stp.storage import load_records, read_json, write_json, write_jsonl


def _round_dir(config: Config, round_index: int) -> Path:
    """Return the artifact directory for one round."""

    return config.run.output_dir / f"round_{round_index:03d}"


def _checkpoint(config: Config, round_index: int) -> str | Path:
    """Return the causal checkpoint used at the start of a round."""

    if round_index == 0:
        return config.model.name
    return _round_dir(config, round_index - 1) / "checkpoint"


def _load_pool(round_dir: Path) -> list[Statement | Conjecture]:
    """Load selected dataset statements and generated conjectures."""

    statements = load_records(round_dir / "selected.jsonl", Statement)
    conjectures = load_records(round_dir / "conjectures.jsonl", Conjecture)
    return [*statements, *conjectures]


def _previous_assessments(
    config: Config,
    round_index: int,
) -> list[ConjectureAssessment]:
    """Return maximum historical solve rates for curriculum selection."""

    best: dict[str, ConjectureAssessment] = {}
    for previous in range(round_index):
        path = _round_dir(config, previous) / "assessments.jsonl"
        if not path.exists():
            continue
        for assessment in load_records(path, ConjectureAssessment):
            if (
                assessment.statement_id not in best
                or assessment.score > best[assessment.statement_id].score
            ):
                best[assessment.statement_id] = assessment
    return list(best.values())


def _conjecture_level(labels: Sequence[str]) -> int:
    """Read the current generated-conjecture depth from labels."""

    levels = [
        int(label.split()[-1])
        for label in labels
        if label.startswith("conjecture ")
    ]
    return max(levels, default=0)


def _generate_conjectures(
    config: Config,
    round_index: int,
    selected: Sequence[Statement],
    runtime: ModelRuntime,
) -> list[Conjecture]:
    """Generate new hard conjectures from the previous round's easy proofs."""

    if round_index == 0:
        return []
    previous_dir = _round_dir(config, round_index - 1)
    previous_pool = _load_pool(previous_dir)
    previous_attempts = load_records(
        previous_dir / "solve_attempts.jsonl",
        SolveAttempt,
    )
    previous_assessments = load_records(
        previous_dir / "assessments.jsonl",
        ConjectureAssessment,
    )
    sources = select_conjecture_sources(
        previous_pool,
        previous_attempts,
        previous_assessments,
        len(selected),
        config.run.seed + round_index,
    )
    sources = sources * config.run.conjecture_multiplier
    shared_name = ""
    shared_statement = "theorem true : True"
    prompts = [
        conjecture_prompt(source.statement, proof, shared_statement)
        for source, proof in sources
    ]
    outputs = generate_texts(
        runtime,
        prompts,
        [
            config.run.seed + round_index * 1_000_000 + index
            for index in range(len(prompts))
        ],
        config.model,
        config.solver.temperature,
        config.solver.top_p,
    )
    existing = {statement.id for statement in selected}
    conjectures = []
    for (source, proof), (text, _, _) in zip(sources, outputs, strict=True):
        statement = canonical_statement(parse_conjecture(text))
        conjecture_id = stable_id((source.header or "") + statement)
        if conjecture_id in existing:
            continue
        existing.add(conjecture_id)
        level = _conjecture_level(source.labels) + 1
        conjectures.append(
            Conjecture(
                id=conjecture_id,
                statement=statement,
                easy_statement=source.statement,
                easy_proof=proof,
                shared_lemma=shared_name,
                shared_lemma_statement=shared_statement,
                header=source.header,
                labels=tuple(
                    label
                    for label in source.labels
                    if not label.startswith("conjecture ")
                )
                + (f"conjecture {level}",),
            )
        )
    return rank_conjectures(screen_conjectures(conjectures))


def _replay_training_examples(
    config: Config,
    round_index: int,
) -> list[TrainingExample]:
    """Build new training examples from the configured replay window."""

    examples = load_sft_examples(config.data.sft_dataset)
    first = max(0, round_index - config.run.replay_rounds + 1)
    for replay_round in range(first, round_index + 1):
        round_dir = _round_dir(config, replay_round)
        pool = _load_pool(round_dir)
        attempts = load_records(
            round_dir / "solve_attempts.jsonl",
            SolveAttempt,
        )
        assessments = load_records(
            round_dir / "assessments.jsonl",
            ConjectureAssessment,
        )
        examples.extend(
            build_training_examples(
                pool,
                attempts,
                assessments,
                replay_round,
                config.run.training_proof_threshold,
                config.run.conjecture_threshold,
            )
        )
    return deduplicate_training_examples(examples)


def _match_conjecture_examples(
    config: Config,
    examples: Sequence[TrainingExample],
    statements: Sequence[Statement],
    model_path: str | Path,
    tokenizer_path: str | Path,
) -> list[TrainingExample]:
    """Project hard-conjecture examples toward the source distribution."""

    candidates = [example for example in examples if example.kind == "conjecture"]
    if not candidates:
        return list(examples)
    runtime = load_runtime(model_path, tokenizer_path)
    candidate_embeddings = embed_texts(
        runtime,
        [example.target for example in candidates],
        config.model,
    )
    target_embeddings = embed_texts(
        runtime,
        [statement.statement for statement in statements],
        config.model,
    )
    unload_runtime(runtime)
    matched = wasserstein_matching(
        candidates,
        candidate_embeddings,
        target_embeddings,
        [statement.matching_weight for statement in statements],
    )
    retained = [example for example in examples if example.kind != "conjecture"]
    retained.extend(replace(example, weight=weight) for example, weight in matched)
    return retained


def _git_revision(repository: Path) -> dict[str, str | bool]:
    """Read a repository's commit and dirty status for provenance."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status)}


def write_manifest(config: Config) -> None:
    """Record code, Lean, model, and configuration provenance once."""

    path = config.run.output_dir / "manifest.json"
    if path.exists():
        return
    repository = Path(__file__).resolve().parents[2]
    alphaproof = repository.parent / "AlphaProof"
    lean_version = subprocess.run(
        ["lake", "env", "lean", "--version"],
        cwd=config.lean.project_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    write_json(
        path,
        {
            "stp": _git_revision(repository),
            "alphaproof": _git_revision(alphaproof),
            "leantree": _git_revision(alphaproof / "vendor/leantree"),
            "lean_version": lean_version,
            "lean_project": str(config.lean.project_dir),
            "base_model": config.model.name,
            "solver": config.solver.kind,
        },
    )
    shutil.copy2(config.path, config.run.output_dir / "config.toml")


def run_round(config: Config, round_index: int) -> RoundState:
    """Execute or resume one complete STP self-play round."""

    config.run.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(config)
    round_dir = _round_dir(config, round_index)
    round_dir.mkdir(parents=True, exist_ok=True)
    state_path = round_dir / "state.json"
    if state_path.exists():
        values = read_json(state_path)
        return RoundState(**values)

    statements = load_statements(config.data.dataset_config)
    selected_path = round_dir / "selected.jsonl"
    if selected_path.exists():
        selected = load_records(selected_path, Statement)
    else:
        selected = select_dataset_statements(
            statements,
            _previous_assessments(config, round_index),
            config.run.statements_per_round,
            config.run.seed + round_index,
        )
        write_jsonl(selected_path, selected)

    model_path = _checkpoint(config, round_index)
    tokenizer_path = (
        config.model.tokenizer if round_index == 0 else model_path
    )
    runtime: ModelRuntime | None = None
    conjecture_path = round_dir / "conjectures.jsonl"
    if conjecture_path.exists():
        conjectures = load_records(conjecture_path, Conjecture)
    else:
        if round_index > 0:
            runtime = load_runtime(model_path, tokenizer_path)
            conjectures = _generate_conjectures(
                config,
                round_index,
                selected,
                runtime,
            )
        else:
            conjectures = []
        write_jsonl(conjecture_path, conjectures)

    pool: list[Statement | Conjecture] = [*selected, *conjectures]
    requests = make_requests(
        pool,
        config.solver.attempts_per_statement,
        config.run.seed + round_index * 10_000_000,
    )
    request_path = round_dir / "proof_requests.jsonl"
    if not request_path.exists():
        write_jsonl(request_path, requests)

    attempts_path = round_dir / "solve_attempts.jsonl"
    if attempts_path.exists():
        attempts = load_records(attempts_path, SolveAttempt)
    elif config.solver.kind == "llm":
        if runtime is None:
            runtime = load_runtime(model_path, tokenizer_path)
        attempts = solve_with_llm(requests, runtime, config)
        write_jsonl(attempts_path, attempts)
    else:
        if runtime is not None:
            unload_runtime(runtime)
            runtime = None
        attempts = solve_with_alphaproof(requests, config, round_dir)
        write_jsonl(attempts_path, attempts)
    if runtime is not None:
        unload_runtime(runtime)

    assessment_path = round_dir / "assessments.jsonl"
    if assessment_path.exists():
        assessments = load_records(
            assessment_path,
            ConjectureAssessment,
        )
    else:
        assessments = assess_conjectures(attempts)
        write_jsonl(assessment_path, assessments)

    training_path = round_dir / "training_examples.jsonl"
    if training_path.exists():
        training_examples = load_records(training_path, TrainingExample)
    else:
        training_examples = _replay_training_examples(config, round_index)
        training_examples = _match_conjecture_examples(
            config,
            training_examples,
            statements,
            model_path,
            tokenizer_path,
        )
        write_jsonl(training_path, training_examples)

    checkpoint = round_dir / "checkpoint"
    if not (checkpoint / "config.json").exists():
        if not training_examples:
            raise ValueError("No SFT or self-play training examples are available.")
        train_model(
            model_path,
            tokenizer_path,
            checkpoint,
            training_examples,
            config.model,
            config.run.seed + round_index,
        )

    solved = sum(assessment.successes > 0 for assessment in assessments)
    state = RoundState(
        round=round_index,
        checkpoint=str(checkpoint),
        selected_statements=len(selected),
        conjectures=len(conjectures),
        solver_attempts=sum(item.multiplicity for item in attempts),
        solved_statements=solved,
        training_examples=len(training_examples),
    )
    write_json(state_path, state.__dict__)
    write_json(
        round_dir / "metrics.json",
        {
            **state.__dict__,
            "generated_tokens": sum(item.generated_tokens for item in attempts),
            "solver_seconds": sum(item.duration_seconds for item in attempts),
            "verification_seconds": sum(
                item.verify_seconds for item in attempts
            ),
        },
    )
    return state


def run(config: Config) -> list[RoundState]:
    """Execute all configured rounds, resuming completed artifacts."""

    config.run.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(config)
    return [run_round(config, index) for index in range(config.run.rounds)]


def evaluate(config: Config, checkpoint: Path) -> dict[str, int | float | str]:
    """Evaluate one checkpoint with the configured fixed solver budget."""

    stamp = time.strftime("%Y%m%d-%H%M%S")
    artifact_dir = config.run.output_dir / "evaluation" / stamp
    artifact_dir.mkdir(parents=True)
    statements = load_statements(config.data.dataset_config)
    requests = make_requests(
        statements,
        config.solver.attempts_per_statement,
        config.run.seed,
    )
    write_jsonl(artifact_dir / "proof_requests.jsonl", requests)
    if config.solver.kind == "llm":
        runtime = load_runtime(checkpoint, checkpoint)
        attempts = solve_with_llm(requests, runtime, config)
        unload_runtime(runtime)
    else:
        attempts = solve_with_alphaproof(requests, config, artifact_dir)
    write_jsonl(artifact_dir / "solve_attempts.jsonl", attempts)
    assessments = assess_conjectures(attempts)
    write_jsonl(artifact_dir / "assessments.jsonl", assessments)
    solved = sum(item.successes > 0 for item in assessments)
    metrics: dict[str, int | float | str] = {
        "checkpoint": str(checkpoint),
        "solver": config.solver.kind,
        "statements": len(statements),
        "solved": solved,
        "pass_at_k": solved / len(statements),
        "attempts_per_statement": config.solver.attempts_per_statement,
    }
    write_json(artifact_dir / "metrics.json", metrics)
    return metrics
