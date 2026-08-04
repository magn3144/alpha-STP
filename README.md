This fork is a local, single-GPU PyTorch implementation of
[Self-play LLM Theorem Provers with Iterative Conjecturing and Proving](https://arxiv.org/abs/2502.00212).

It is used as part of my master thesis, where I experiment with potential performance optimizations of STP.

## Code structure

The outer algorithm is intentionally visible in three files under
`stp/self_play/`:

- `stp/self_play/stp_loop.py` selects problems and runs the conjecture, proof,
  scoring, and training phases.
- `stp/self_play/generate.py` prepares conjecturer inputs and calls either the
  LLM or AlphaProof solver.
- `stp/self_play/train.py` assembles the round's training data and trains the
  next full-model checkpoint.

Shared configuration and immutable records live in `stp/core/`. Dataset and
artifact I/O live in `stp/data/`, model runtime code and prompt formats in
`stp/inference/`, and Lean verification and solver adapters in `stp/proving/`.

Proof-model prompting, sampling, answer parsing, and training formatting are
selected by `model.prover_handler`. The available handlers are `stp`, which
uses the original completion format, and `kimina_numina`, which uses the
Kimina-Prover chat template and extracts the final exact-theorem `lean4` block.
Both expose the same normalized proof-generation interface under
`stp/inference/prover_models/`.

The LLM solver uses the configured number of independent proof attempts.
Its output budget is configured independently from conjecture generation with
`solver.prover_max_new_tokens`; the example configuration uses 8196 tokens.
AlphaProof instead performs one fixed-budget tree search per theorem, stores
the complete AND-OR tree, and scores its hardest-subproblem OR projection by
the fraction of proven frontier nodes. This transformation and metric live in
`stp/proving/search_metrics.py`; AlphaProof only returns the raw search tree.

## Running

Create the version-coupled theorem declaration artifact before starting a
run:

```bash
uv run stp declarations --config configs/alpha_stp.toml
```

Then run or resume the configured self-play loop:

```bash
uv run stp run --config configs/alpha_stp.toml
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
uv run python -m stp.evaluate_difficulty_score.evaluate_numina \
  --config configs/alpha_stp.toml \
  --llm-model models/Kimina-Prover-Preview-Distill-1.5B \
  --llm-prover-handler kimina_numina
```

Use `--llm-tokenizer` when the tokenizer lives at a different path. The script
samples `LLM_ATTEMPTS` proofs per problem and performs one AlphaProof search
with `ALPHAPROOF_ROLLOUTS` rollouts. Both parameters are defined at the top of
the script. It creates a timestamped directory under
`data/evaluations` containing `llm_generations.jsonl`,
`alphaproof_search_trees.jsonl`, and `difficulty_scores.jsonl`. Repeated LLM
outputs are stored once with their count in the `multiplicity` field. The
script completes both solvers for one problem before moving to the next, saves
each solver result immediately, and resumes an existing evaluation when given
the same `--name`. Each difficulty record contains one problem and solver score
with separate solver, Lean verification, and total time in seconds.

To compare the proof-tree difficulty metric for one problem at AlphaProof
budgets from 8 through 512 simulations, run:

```bash
uv run python scripts/evaluate_alphaproof_simulations.py \
  --config configs/alpha_stp.toml \
  --problem-index 0
```

This creates a timestamped directory under `data/evaluations` with a JSONL
summary, a line plot, and the raw and projected search-tree artifacts for each
budget. Use `--name` to choose the directory name.
