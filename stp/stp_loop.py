"""The explicit outer STP self-play loop."""

import time
from pathlib import Path

from stp.artifacts import (
    historical_attempts,
    input_checkpoint,
    previous_scores,
    round_dir,
    write_manifest,
)
from stp.config import Config
from stp.data import load_statements
from stp.declarations import load_declaration_map
from stp.generate import (
    generate_conjectures,
    generate_proofs,
    make_proof_requests,
    prepare_conjecture_inputs,
)
from stp.model import ModelRuntime, load_runtime, unload_runtime
from stp.records import (
    Conjecture,
    ConjectureAssessment,
    ConjectureInput,
    RoundState,
    SolveAttempt,
    Statement,
)
from stp.scoring import (
    calculate_scores,
    select_dataset_statements,
    unproved_statements,
)
from stp.storage import load_records, read_json, write_json, write_jsonl
from stp.train import prepare_training_examples, train_round


def run_round(config: Config, round_index: int) -> RoundState:
    """Execute or resume one self-play round."""

    config.run.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(config)
    directory = round_dir(config, round_index)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.json"
    if state_path.exists():
        return RoundState(**read_json(state_path))

    # 1. Choose source problems.
    statements = load_statements(config.data.dataset_config)
    old_scores = previous_scores(config, round_index)
    selected_path = directory / "selected.jsonl"
    if selected_path.exists():
        selected = load_records(selected_path, Statement)
    else:
        selected = select_dataset_statements(
            statements,
            old_scores,
            config.run.statements_per_round,
            config.run.seed + round_index,
        )
        write_jsonl(selected_path, selected)

    model_path = input_checkpoint(config, round_index)
    tokenizer_path = config.model.tokenizer if round_index == 0 else model_path
    runtime: ModelRuntime | None = None

    # 2. Build conjecturer inputs and generate conjectures.
    selected_unproved = unproved_statements(selected, old_scores)
    input_path = directory / "conjecture_inputs.jsonl"
    if input_path.exists():
        conjecture_inputs = load_records(input_path, ConjectureInput)
    elif round_index == 0:
        conjecture_inputs = []
        write_jsonl(input_path, conjecture_inputs)
    else:
        declarations = load_declaration_map(config)
        conjecture_inputs = prepare_conjecture_inputs(
            statements,
            historical_attempts(config, round_index),
            old_scores,
            declarations,
            len(selected_unproved),
            config.run.trivial_lemma_probability,
            config.run.lemma_input_cap_fraction,
            config.run.seed + round_index,
        )
        del declarations
        write_jsonl(input_path, conjecture_inputs)

    conjecture_path = directory / "conjectures.jsonl"
    if conjecture_path.exists():
        conjectures = load_records(conjecture_path, Conjecture)
    elif conjecture_inputs:
        runtime = load_runtime(model_path, tokenizer_path)
        conjectures = generate_conjectures(
            config,
            round_index,
            conjecture_inputs,
            statements,
            len(selected_unproved),
            runtime,
        )
        write_jsonl(conjecture_path, conjectures)
    else:
        conjectures = []
        write_jsonl(conjecture_path, conjectures)

    # 3. Ask the configured solver to prove data and conjectures.
    pool: list[Statement | Conjecture] = [*selected, *conjectures]
    requests = make_proof_requests(
        pool,
        config.solver.attempts_per_statement,
        config.run.seed + round_index * 10_000_000,
    )
    request_path = directory / "proof_requests.jsonl"
    if not request_path.exists():
        write_jsonl(request_path, requests)

    attempts_path = directory / "solve_attempts.jsonl"
    if attempts_path.exists():
        attempts = load_records(attempts_path, SolveAttempt)
    else:
        if config.solver.kind == "llm" and runtime is None:
            runtime = load_runtime(model_path, tokenizer_path)
        if config.solver.kind == "alphaproof" and runtime is not None:
            unload_runtime(runtime)
            runtime = None
        attempts = generate_proofs(requests, runtime, config, directory)
        write_jsonl(attempts_path, attempts)
    if runtime is not None:
        unload_runtime(runtime)

    # 4. Score results, construct examples, and train the next checkpoint.
    score_path = directory / "assessments.jsonl"
    if score_path.exists():
        scores = load_records(score_path, ConjectureAssessment)
    else:
        scores = calculate_scores(attempts)
        write_jsonl(score_path, scores)

    examples = prepare_training_examples(
        config,
        round_index,
        statements,
        conjectures,
        attempts,
        scores,
        old_scores,
        model_path,
        tokenizer_path,
        directory,
    )
    checkpoint = train_round(
        config,
        round_index,
        examples,
        model_path,
        tokenizer_path,
        directory,
    )

    state = RoundState(
        round=round_index,
        checkpoint=str(checkpoint),
        selected_statements=len(selected),
        conjectures=len(conjectures),
        solver_attempts=sum(item.multiplicity for item in attempts),
        solved_statements=sum(score.successes > 0 for score in scores),
        training_examples=len(examples),
    )
    write_json(state_path, state.__dict__)
    write_json(
        directory / "metrics.json",
        {
            **state.__dict__,
            "generated_tokens": sum(item.generated_tokens for item in attempts),
            "solver_seconds": sum(item.duration_seconds for item in attempts),
            "verification_seconds": sum(
                item.verify_seconds for item in attempts
            ),
            "conjecture_inputs": len(conjecture_inputs),
            "conjecture_examples": sum(
                example.kind == "conjecture" for example in examples
            ),
        },
    )
    return state


def run(config: Config) -> list[RoundState]:
    """Execute all configured rounds."""

    return [run_round(config, index) for index in range(config.run.rounds)]


def evaluate(config: Config, checkpoint: Path) -> dict[str, int | float | str]:
    """Evaluate a checkpoint with the configured solver budget."""

    stamp = time.strftime("%Y%m%d-%H%M%S")
    directory = config.run.output_dir / "evaluation" / stamp
    directory.mkdir(parents=True)
    statements = load_statements(config.data.dataset_config)
    requests = make_proof_requests(
        statements,
        config.solver.attempts_per_statement,
        config.run.seed,
    )
    write_jsonl(directory / "proof_requests.jsonl", requests)
    runtime = (
        load_runtime(checkpoint, checkpoint)
        if config.solver.kind == "llm"
        else None
    )
    attempts = generate_proofs(requests, runtime, config, directory)
    if runtime is not None:
        unload_runtime(runtime)
    write_jsonl(directory / "solve_attempts.jsonl", attempts)
    scores = calculate_scores(attempts)
    write_jsonl(directory / "assessments.jsonl", scores)
    solved = sum(score.successes > 0 for score in scores)
    metrics: dict[str, int | float | str] = {
        "checkpoint": str(checkpoint),
        "solver": config.solver.kind,
        "statements": len(statements),
        "solved": solved,
        "pass_at_k": solved / len(statements),
        "attempts_per_statement": config.solver.attempts_per_statement,
    }
    write_json(directory / "metrics.json", metrics)
    return metrics
