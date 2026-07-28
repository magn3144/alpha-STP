"""Original STP prover and conjecturer prompts."""

START_LEMMA_STMT = "<easy theorem>"
START_THM = "<hard theorem>"
END_THM = "</hard theorem>"
INVOKED_LEMMA = "<lemma>"
LEAN_CODE_PROMPT = "Complete the following Lean 4 code:\n\n```lean4\n"
PROVER_PROMPT = (
    LEAN_CODE_PROMPT
    + "import Mathlib\n"
    "import Aesop\n"
    "set_option maxHeartbeats 0\n"
    "open BigOperators Real Nat Topology Rat\n"
)


def prover_prompt(statement: str, header: str | None) -> str:
    """Build the original STP whole-proof prompt."""

    if header is not None:
        return LEAN_CODE_PROMPT + header + statement.strip()
    return f"{PROVER_PROMPT}\n{statement.strip()}"


def _conjecturer_prompt(
    shared_lemma_statement: str,
    seed_statement: str,
    seed_proof: str,
) -> str:
    """Build the paper-style conjecturer prompt prefix."""

    easy_theorem = seed_statement + seed_proof
    return (
        LEAN_CODE_PROMPT
        + f"{INVOKED_LEMMA}\n{shared_lemma_statement.strip()}\n"
        + f"{START_LEMMA_STMT}\n{easy_theorem.strip()}\n"
        + START_THM
    )


def conjecturer_generation_prompt(
    shared_lemma_statement: str,
    seed_statement: str,
    seed_proof: str,
) -> str:
    """Build the prompt used for conjecture generation."""

    return (
        _conjecturer_prompt(
            shared_lemma_statement,
            seed_statement,
            seed_proof,
        )
        + "\n theorem"
    )


def conjecturer_training_prompt(
    shared_lemma_statement: str,
    seed_statement: str,
    seed_proof: str,
) -> str:
    """Build the prompt used for conjecturer training."""

    return _conjecturer_prompt(
        shared_lemma_statement,
        seed_statement,
        seed_proof,
    )


def parse_proof(text: str) -> str:
    """Extract a tactic proof from a model completion."""

    return text.split("\n```", 1)[0]


def parse_conjecture(text: str) -> str:
    """Extract a theorem declaration and append the empty proof."""

    statement = "theorem " + text.split(END_THM, 1)[0].strip()
    if ":=" in statement:
        statement = statement.split(":=", 1)[0]
    return statement + ":= by"
