"""Shared immutable values returned by prover model handlers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratedProof:
    """One extracted proof plus its raw generation metadata."""

    proof: str | None
    raw_output: str
    generated_tokens: int
    duration_seconds: float
