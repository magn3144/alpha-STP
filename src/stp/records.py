from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SolveStatus = Literal["proved", "failed", "timeout", "error"]


@dataclass(frozen=True)
class Statement:
    """A canonical Lean statement and its dataset metadata."""

    id: str
    statement: str
    header: str | None
    labels: tuple[str, ...]
    matching_weight: float
    source: str


@dataclass(frozen=True)
class Conjecture:
    """A generated hard statement paired with its easy source problem."""

    id: str
    statement: str
    easy_statement: str
    easy_proof: str
    shared_lemma: str
    shared_lemma_statement: str
    header: str | None
    labels: tuple[str, ...]


@dataclass(frozen=True)
class ProofRequest:
    """One seeded solver attempt for a statement."""

    id: str
    statement_id: str
    statement: str
    header: str | None
    attempt: int
    seed: int
    source: str


@dataclass(frozen=True)
class SolveAttempt:
    """The normalized result of one or more identical solver attempts."""

    request_id: str
    statement_id: str
    attempt: int
    solver: str
    seed: int
    status: SolveStatus
    proof: str | None
    duration_seconds: float
    generated_tokens: int
    verify_seconds: float
    multiplicity: int = 1
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConjectureAssessment:
    """Difficulty and success statistics for one statement."""

    statement_id: str
    method: str
    score: float
    attempts: int
    successes: int
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingExample:
    """One weighted target-only causal language-model example."""

    prompt: str
    target: str
    weight: float
    kind: Literal["proof", "conjecture", "sft"]
    statement_id: str
    round: int


@dataclass(frozen=True)
class RoundState:
    """Small resumable summary written after a completed STP round."""

    round: int
    checkpoint: str
    selected_statements: int
    conjectures: int
    solver_attempts: int
    solved_statements: int
    training_examples: int


def record_to_dict(record: Any) -> dict[str, Any]:
    """Convert a dataclass record to JSON-compatible data."""

    return asdict(record)
