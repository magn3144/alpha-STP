This fork is a local, single-GPU PyTorch implementation of
[Self-play LLM Theorem Provers with Iterative Conjecturing and Proving](https://arxiv.org/abs/2502.00212).

It is used as part of my master thesis, where I experiment with potential performance optimizations of STP.

## Code structure

The outer algorithm is intentionally visible in three files:

- `stp/stp_loop.py` selects problems and runs the conjecture, proof, scoring,
  and training phases.
- `stp/generate.py` prepares conjecturer inputs and calls either the LLM or
  AlphaProof solver.
- `stp/train.py` assembles the round's training data and trains the next
  full-model checkpoint.

Prompt text, scoring, immutable records, artifact I/O, Lean verification,
solver adapters, and detailed training-example transformations live in
focused supporting modules under `stp/`.

The LLM solver uses the configured number of independent proof attempts.
AlphaProof instead performs one fixed-budget tree search per theorem, stores
the complete AND-OR tree, and scores its hardest-subproblem OR projection by
the fraction of proven frontier nodes. This transformation and metric live in
`stp/search_metrics.py`; AlphaProof only returns the raw search tree.

## Running

Create the version-coupled theorem declaration artifact before starting a
run:

```bash
uv run stp declarations --config configs/example.toml
```

Then run or resume the configured self-play loop:

```bash
uv run stp run --config configs/example.toml
```

Each round stores its conjecturer inputs, generated conjectures, proof
attempts with invoked lemmas, filter diagnostics, weighted conjecturer
examples, complete training set, and next Hugging Face checkpoint under the
configured output directory. The declaration command must be rerun after the
Lean project, Mathlib revision, or imported modules change.
