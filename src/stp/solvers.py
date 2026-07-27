import subprocess
from pathlib import Path
from typing import Any, Sequence, cast

from stp.algorithm import deduplicate_attempts, parse_proof, proof_prompt
from stp.config import Config
from stp.data import alphaproof_theorem
from stp.lean import verify_attempts
from stp.model import ModelRuntime, generate_texts
from stp.records import ProofRequest, SolveAttempt, SolveStatus
from stp.storage import read_jsonl, write_jsonl


def solve_with_llm(
    requests: Sequence[ProofRequest],
    runtime: ModelRuntime,
    config: Config,
) -> list[SolveAttempt]:
    """Generate whole proofs and verify unique completions with LeanTree."""

    prompts = [proof_prompt(request.statement, request.header) for request in requests]
    outputs = generate_texts(
        runtime,
        prompts,
        [request.seed for request in requests],
        config.model,
        config.solver.temperature,
        config.solver.top_p,
    )
    attempts = [
        SolveAttempt(
            request_id=request.id,
            statement_id=request.statement_id,
            attempt=request.attempt,
            solver="llm",
            seed=request.seed,
            status="failed",
            proof=parse_proof(text),
            duration_seconds=duration,
            generated_tokens=tokens,
            verify_seconds=0.0,
            metrics={},
        )
        for request, (text, tokens, duration) in zip(
            requests,
            outputs,
            strict=True,
        )
    ]
    unique = deduplicate_attempts(attempts)
    return verify_attempts(unique, requests, config.lean)


def solve_with_alphaproof(
    requests: Sequence[ProofRequest],
    config: Config,
    artifact_dir: Path,
) -> list[SolveAttempt]:
    """Run one AlphaProof batch CLI and normalize its JSONL output."""

    input_path = artifact_dir / "alphaproof_requests.jsonl"
    output_path = artifact_dir / "alphaproof_results.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "request_id": request.id,
                "theorem": alphaproof_theorem(request.statement),
                "header": request.header,
                "seed": request.seed,
            }
            for request in requests
        ],
    )
    command = [
        *config.solver.alphaproof_command,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--run-dir",
        str(config.solver.alphaproof_run_dir),
        "--lean-project",
        str(config.lean.project_dir),
        "--num-simulations",
        str(config.solver.alphaproof_num_simulations),
        "--num-sampled-actions",
        str(config.solver.alphaproof_num_sampled_actions),
    ]
    subprocess.run(
        command,
        check=True,
        timeout=config.solver.alphaproof_timeout_seconds,
    )
    values = read_jsonl(output_path)
    by_id: dict[str, dict[str, Any]] = {
        str(value["request_id"]): value for value in values
    }
    attempts = []
    for request in requests:
        value = by_id[request.id]
        status = cast(SolveStatus, value["status"])
        attempts.append(
            SolveAttempt(
                request_id=request.id,
                statement_id=request.statement_id,
                attempt=request.attempt,
                solver="alphaproof",
                seed=request.seed,
                status=status,
                proof=value.get("proof"),
                duration_seconds=float(value["duration_seconds"]),
                generated_tokens=int(value.get("generated_tokens", 0)),
                verify_seconds=float(value.get("verify_seconds", 0.0)),
                metrics=dict(value.get("metrics", {})),
            )
        )
    return deduplicate_attempts(attempts)
