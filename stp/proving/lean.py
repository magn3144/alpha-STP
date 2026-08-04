"""Lean proof verification and dependency extraction."""

import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any, Collection, Sequence

from leantree import LeanProject
from leantree.repl_adapter.interaction import (
    LeanInteractionException,
    LeanProcessException,
)

from stp.core.config import LeanSettings
from stp.core.records import ProofRequest, SolveAttempt

DECLARATION_NAME = re.compile(
    r"\b(?:theorem|lemma)\s+([^\s({:]+)"
)


def _declaration_name(statement: str) -> str:
    """Extract the declared Lean name from a canonical theorem statement."""

    match = DECLARATION_NAME.search(statement)
    if match is None:
        raise ValueError("Statement does not contain a theorem or lemma name.")
    return match.group(1)


def _dependency_query(name: str) -> str:
    """Build a Lean command returning proof-only direct constants as JSON."""

    return f"""run_cmd Lean.Elab.Command.liftTermElabM do
  let info ← Lean.getConstInfo ``{name}
  let proofConstants := info.value!.getUsedConstants
  let typeConstants := info.type.getUsedConstants
  let used := proofConstants.filter fun constant =>
    !typeConstants.contains constant
  Lean.logInfo (
    Lean.Json.arr (
      used.map fun constant => Lean.Json.str constant.toString
    )
  ).compress
"""


def _invoked_lemmas(
    response: dict[str, Any],
) -> tuple[str, ...]:
    """Read direct proof constants returned by Lean."""

    messages = response.get("messages", [])
    values = json.loads(str(messages[-1]["data"]))
    return tuple(sorted(values))


def _verify_chunk(
    items: list[tuple[SolveAttempt, ProofRequest]],
    settings: LeanSettings,
) -> list[tuple[bool, float, dict[str, Any], tuple[str, ...]]]:
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
                dependency_response = environment.send_command(
                    _dependency_query(_declaration_name(request.statement)),
                    timeout=settings.timeout_seconds,
                )
            except (LeanInteractionException, LeanProcessException) as error:
                results.append(
                    (
                        False,
                        time.perf_counter() - started,
                        {"error": str(error)},
                        (),
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
                    _invoked_lemmas(dependency_response),
                )
            )
            environment.rollback_to(checkpoint)
    return results


def verify_attempts(
    attempts: Sequence[SolveAttempt],
    requests: Sequence[ProofRequest],
    settings: LeanSettings,
    declaration_names: Collection[str],
) -> list[SolveAttempt]:
    """Verify proofs and record direct theorem dependencies on CPU."""

    requests_by_id = {request.id: request for request in requests}
    pairs = [(attempt, requests_by_id[attempt.request_id]) for attempt in attempts]
    chunks = [
        pairs[index :: settings.workers] for index in range(settings.workers)
    ]
    chunks = [chunk for chunk in chunks if chunk]
    with ProcessPoolExecutor(max_workers=settings.workers) as executor:
        futures = [
            executor.submit(
                _verify_chunk,
                chunk,
                settings,
            )
            for chunk in chunks
        ]
        verified_chunks = [future.result() for future in futures]

    verified_by_request = {
        attempt.request_id: result
        for chunk, results in zip(chunks, verified_chunks, strict=True)
        for (attempt, _), result in zip(chunk, results, strict=True)
    }
    verified = []
    for attempt in attempts:
        complete, duration, metrics, used_constants = verified_by_request[
            attempt.request_id
        ]
        invoked_lemmas = tuple(
            name for name in used_constants if name in declaration_names
        )
        verified.append(
            replace(
                attempt,
                status="proved" if complete else "failed",
                verify_seconds=duration,
                invoked_lemmas=invoked_lemmas,
                metrics=attempt.metrics | metrics,
            )
        )
    return verified
