"""Extract whole-proof STP examples from AlphaProof's LeanTree dataset.

Each prompt contains the original Mathlib source preceding the declaration, so
imports, namespaces, variables, definitions, and earlier lemmas remain in scope.
The target is the complete outer tactic proof following ``:= by``.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, TextIO


REPOSITORY = Path(__file__).resolve().parents[1]
DELTA_PROOF = REPOSITORY.parent / "delta-proof"
DEFAULT_INPUT = DELTA_PROOF / "data/dataset/leantree_mathlib.jsonl"
DEFAULT_MATHLIB_REPOSITORY = (
    DELTA_PROOF / "lean_project/.lake/packages/mathlib"
)
DEFAULT_OUTPUT_DIR = REPOSITORY / "data/dataset/leantree_mathlib_sft"
DEFAULT_MATHLIB_REVISION = "v4.19.0"
VALIDATION_FRACTION = 0.1
SPLITS = ("train", "validation", "test")


def count_lines(path: Path) -> int:
    """Count source-file records in a JSONL input and return the total."""

    with path.open("rb") as input_file:
        return sum(1 for _ in input_file)


def file_split(file_index: int, total_files: int) -> str:
    """Map a zero-based file index to AlphaProof's train or validation split."""

    validation_files = int(total_files * VALIDATION_FRACTION)
    validation_start = total_files - validation_files
    return "train" if file_index < validation_start else "validation"


def repository_path(record: dict[str, Any]) -> str:
    """Convert a LeanTree package path and return its Mathlib repository path."""

    package, path = str(record["path"]).split("/", 1)
    if package != "mathlib":
        raise ValueError(f"Expected a mathlib package path, got {record['path']!r}.")
    return path


def read_source(
    record: dict[str, Any],
    mathlib_repository: Path,
    mathlib_revision: str,
) -> str:
    """Read one source file at the dataset's pinned Mathlib Git revision."""

    result = subprocess.run(
        [
            "git",
            "-C",
            str(mathlib_repository),
            "show",
            f"{mathlib_revision}:{repository_path(record)}",
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8")


def successful_outer_block(
    theorem: dict[str, Any],
    source: str,
) -> dict[str, Any] | None:
    """Find the successful top-level ``:= by`` block, or return no block."""

    theorem_span = theorem["span"]
    theorem_start = int(theorem_span["start"])
    theorem_finish = int(theorem_span["finish"])
    candidates = []
    for block in theorem["by_blocks"]:
        block_span = block.get("span")
        tree = block.get("tree")
        if block_span is None or not isinstance(tree, dict):
            continue
        if tree.get("error") is not None or not tree.get("nodes"):
            continue
        block_start = int(block_span["start"])
        block_finish = int(block_span["finish"])
        statement = source[theorem_start:block_start].rstrip()
        trailing_source = source[block_finish:theorem_finish]
        if statement.endswith(":= by") and not trailing_source.strip():
            candidates.append(block)

    if len(candidates) > 1:
        raise ValueError("A theorem has multiple outer ':= by' proof blocks.")
    return candidates[0] if candidates else None


def theorem_example(
    theorem: dict[str, Any],
    source: str,
    relative_path: str,
) -> dict[str, Any] | None:
    """Convert one serialized theorem into an STP example, or return no example."""

    if theorem.get("error") is not None or theorem.get("span") is None:
        return None
    block = successful_outer_block(theorem, source)
    if block is None:
        return None

    theorem_start = int(theorem["span"]["start"])
    block_start = int(block["span"]["start"])
    block_finish = int(block["span"]["finish"])
    formal_statement = source[theorem_start:block_start].strip()
    proof = source[block_start:block_finish].strip()
    if "sorry" in proof or "admit" in proof:
        return None
    return {
        "formal_statement": formal_statement,
        "proof": proof,
        "header": source[:theorem_start].rstrip() + "\n",
        "source": relative_path,
    }


def extract_file_examples(
    record: dict[str, Any],
    mathlib_repository: Path,
    mathlib_revision: str,
) -> list[dict[str, Any]]:
    """Read one source file and return all supported whole-proof examples."""

    source = read_source(record, mathlib_repository, mathlib_revision)
    return [
        example
        for theorem in record["theorems"]
        if (example := theorem_example(theorem, source, str(record["path"])))
        is not None
    ]


def open_outputs(output_dir: Path) -> dict[str, TextIO]:
    """Create the split output files and return their writable handles."""

    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        split: (output_dir / f"{split}.jsonl").open("x", encoding="utf-8")
        for split in SPLITS
    }


def extract_dataset(
    input_path: Path,
    mathlib_repository: Path,
    mathlib_revision: str,
    output_dir: Path,
) -> dict[str, int]:
    """Extract all source records and return per-split example counts."""

    total_files = count_lines(input_path)
    validation_files = int(total_files * VALIDATION_FRACTION)
    counts = {split: 0 for split in SPLITS}
    outputs = open_outputs(output_dir)
    try:
        with input_path.open(encoding="utf-8") as input_file:
            for file_index, line in enumerate(input_file):
                record = json.loads(line)
                split = file_split(file_index, total_files)
                examples = extract_file_examples(
                    record,
                    mathlib_repository,
                    mathlib_revision,
                )
                for example in examples:
                    outputs[split].write(
                        json.dumps(example, ensure_ascii=False) + "\n"
                    )
                counts[split] += len(examples)
                if (file_index + 1) % 500 == 0:
                    print(
                        f"Processed {file_index + 1:,}/{total_files:,} files; "
                        f"wrote {sum(counts.values()):,} examples",
                        flush=True,
                    )
    finally:
        for output in outputs.values():
            output.close()

    print(
        f"Split {total_files - validation_files:,} train files and "
        f"{validation_files:,} validation files exactly as AlphaProof; "
        "AlphaProof defines no test files.",
        flush=True,
    )
    return counts


def parse_args() -> argparse.Namespace:
    """Parse source dataset, Mathlib revision, and output CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Extract full Mathlib proofs for STP supervised training."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--mathlib-repository",
        type=Path,
        default=DEFAULT_MATHLIB_REPOSITORY,
    )
    parser.add_argument(
        "--mathlib-revision",
        default=DEFAULT_MATHLIB_REVISION,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"LeanTree JSONL does not exist: {args.input}")
    if not args.mathlib_repository.is_dir():
        parser.error(
            f"Mathlib repository does not exist: {args.mathlib_repository}"
        )
    existing = [
        args.output_dir / f"{split}.jsonl"
        for split in SPLITS
        if (args.output_dir / f"{split}.jsonl").exists()
    ]
    if existing:
        parser.error(
            "Output files already exist: "
            + ", ".join(str(path) for path in existing)
        )
    return args


def main() -> None:
    """Extract the dataset and print the resulting split sizes."""

    args = parse_args()
    counts = extract_dataset(
        args.input,
        args.mathlib_repository,
        args.mathlib_revision,
        args.output_dir,
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
