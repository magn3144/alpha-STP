import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any, Sequence

from leantree import LeanProject
from leantree.repl_adapter.interaction import (
    LeanInteractionException,
    LeanProcessException,
)

from stp.config import LeanSettings
from stp.records import ProofRequest, SolveAttempt


def _verify_chunk(
    items: list[tuple[SolveAttempt, ProofRequest]],
    settings: LeanSettings,
) -> list[tuple[bool, float, dict[str, Any]]]:
    """Verify a chunk while reusing one LeanTree process."""

    results = []
    with LeanProject(str(settings.project_dir)).environment() as environment:
        for module in settings.imports:
            environment.send_command(
                f"import {module}",
                timeout=settings.timeout_seconds,
            )
        checkpoint = environment.checkpoint()
        for attempt, request in items:
            started = time.perf_counter()
            code = (
                (request.header or "")
                + request.statement
                + "\n"
                + (attempt.proof or "")
            )
            try:
                response = environment.send_command(
                    code,
                    timeout=settings.timeout_seconds,
                )
            except (LeanInteractionException, LeanProcessException) as error:
                results.append(
                    (
                        False,
                        time.perf_counter() - started,
                        {"error": str(error)},
                    )
                )
                environment.rollback_to(checkpoint)
                continue
            messages = response.get("messages", [])
            errors = [
                message
                for message in messages
                if message.get("severity") == "error"
            ]
            uses_sorry = any(
                "sorryAx" in str(message.get("data", ""))
                or "declaration uses 'sorry'" in str(message.get("data", ""))
                for message in messages
            )
            results.append(
                (
                    not errors and not uses_sorry,
                    time.perf_counter() - started,
                    {"messages": messages},
                )
            )
            environment.rollback_to(checkpoint)
    return results


def verify_attempts(
    attempts: Sequence[SolveAttempt],
    requests: Sequence[ProofRequest],
    settings: LeanSettings,
) -> list[SolveAttempt]:
    """Verify generated LLM proofs concurrently on CPU."""

    requests_by_id = {request.id: request for request in requests}
    pairs = [(attempt, requests_by_id[attempt.request_id]) for attempt in attempts]
    chunks = [
        pairs[index :: settings.workers] for index in range(settings.workers)
    ]
    chunks = [chunk for chunk in chunks if chunk]
    with ProcessPoolExecutor(max_workers=settings.workers) as executor:
        futures = [
            executor.submit(_verify_chunk, chunk, settings) for chunk in chunks
        ]
        verified_chunks = [future.result() for future in futures]

    verified_by_request = {
        attempt.request_id: result
        for chunk, results in zip(chunks, verified_chunks, strict=True)
        for (attempt, _), result in zip(chunk, results, strict=True)
    }
    verified = []
    for attempt in attempts:
        complete, duration, metrics = verified_by_request[attempt.request_id]
        verified.append(
            replace(
                attempt,
                status="proved" if complete else "failed",
                verify_seconds=duration,
                metrics=attempt.metrics | metrics,
            )
        )
    return verified
