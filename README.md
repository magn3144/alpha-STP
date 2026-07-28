This fork is a local, single-GPU PyTorch implementation of
[Self-play LLM Theorem Provers with Iterative Conjecturing and Proving](https://arxiv.org/abs/2502.00212).

It is used as part of my master thesis, where I experiment with potential performance optimizations of STP.

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
