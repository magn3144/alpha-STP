# Thesis workspace conventions

These are observations from this user's setup as of 2026-10-11, not platform guarantees. Verify them before operating a new job. Do not preserve historical SSH ports as defaults.

## Persistent layout

The mounted folder has been `/work/master-thesis`, with:

- `repos/alpha-STP` and `repos/delta-proof`: persistent Git checkouts.
- `env.sh`: shared environment sourced by UCloud experiment launchers in both repositories.
- `wandb.env`: private shell-format credentials file, mode `600`, outside both repositories. It exports `WANDB_API_KEY` and is sourced by the shared `env.sh`. Inspect whether a key is populated without displaying it.
- `tools/bin`, `tools/python`, `tools/elan`, and `cache`: persistent tools and caches. `env.sh` exports `THESIS`, `ALPHA_STP`, `DELTA_PROOF`, `ELAN_HOME`, and the tool paths. It does not set Alpha-STP's `DATA`; experiment launchers set `DATA="$ALPHA_STP/data"`.

The local repositories have been under `/Users/magn3144/Documents/GitHub/`. The UCloud identity has been `~/.ssh/id_ed25519_ucloud`, with username `ucloud` on `ssh.cloud.sdu.dk`; obtain the current job's port.

Alpha-STP launchers are under `jobs/bash/UCloud` and YAML under `jobs/yaml/UCloud`. Delta-Proof launchers are under `scripts/batch/UCloud`. Both use isolated `.venv-ucloud-b200` environments. Their original `.venv`, dependency definitions, lockfiles, and non-UCloud jobs must remain unchanged unless explicitly authorized. Check local and remote configurations independently.

## Observed pitfalls

- Mounted `/work` storage rejected Ray UNIX sockets. Ray's runtime files therefore needed `RAY_TMPDIR=/tmp`. Persistent results stayed in the mounted folder.
- The completed 7B STP SFT workload on one B200 used FP32 parameters and BF16 compute. Its validated batch-16 configuration used `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`, `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=128`, and CUDA cuBLAS GEMMs through `--xla_gpu_enable_triton_gemm=false`. These are workload-specific settings, not defaults for every B200 application. Inspect its current launcher and benchmark records before reusing them.
- GPU imports and offline SFT smoke tests passed while the first fresh online batch launch failed at W&B setup. Shared persistent authentication fixed subsequent launches. Verify online tracking separately from offline tests.
- An SFT run can complete successfully while the RL loop still has blockers. Inspect the configured starting checkpoint, `LSB_DJOB_NUMPROC` assumptions, Ray paths, Lean workspace, `lake` executable, REPL build, and full multi-stage smoke coverage. Do not treat successful SFT as proof that generation and Lean verification work.
- `/work/master-thesis/env.sh` is outside the repositories. A change there affects launchers that source it but does not appear in repository Git diffs. Explain this when reporting changes.

Read actual run metadata to find the desired checkpoint. Do not automatically substitute a historical model, dataset, run ID, or hyperparameter from this reference.
