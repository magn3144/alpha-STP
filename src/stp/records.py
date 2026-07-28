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
class TheoremDeclaration:
    """A fully qualified theorem name and its proof-free declaration."""

    full_name: str
    statement: str


@dataclass(frozen=True)
class ConjectureInput:
    """One paper-style lemma-guided input to the conjecturer."""

    id: str
    seed_statement_id: str
    seed_statement: str
    seed_proof: str
    source_attempt_id: str
    shared_lemma: str
    shared_lemma_statement: str
    trivial_lemma: bool
    header: str | None
    labels: tuple[str, ...]
    matching_weight: float
    generation_seed: int


@dataclass(frozen=True)
class Conjecture:
    """A generated hard statement paired with its easy source problem."""

    id: str
    conjecture_input_id: str
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
    invoked_lemmas: tuple[str, ...] = ()
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
class ConjectureFilterMetric:
    """Paper-style filtering diagnostics for one generated conjecture."""

    statement_id: str
    pass_rate: float
    successes: int
    trivial_lemma: bool
    named_lemma_reused: bool
    minimum_proof_length: int | None
    elegance_score: float | None
    selected: bool
    rejection_reason: str | None


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
