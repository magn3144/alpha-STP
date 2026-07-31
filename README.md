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

## Numina evaluation

Evaluate both a Hugging Face causal LLM and AlphaProof on the problems in
`data/dataset/numina_sft_evaluation/test.jsonl` with:

```bash
uv run python scripts/evaluate_numina.py \
  --config configs/example.toml \
  --llm-model YOUR_MODEL_OR_CHECKPOINT
```

Use `--llm-tokenizer` when the tokenizer lives at a different path. The script
always samples 32 LLM proofs per problem and performs one AlphaProof search
using the budget in the config. It creates a timestamped directory under
`data/evaluations` containing `llm_proof_attempts.jsonl`,
`alphaproof_search_trees.jsonl`, and `difficulty_scores.jsonl`. Repeated LLM
outputs are stored once with their count in the `multiplicity` field.
