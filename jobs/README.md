# DTU LSF jobs

All jobs request one A100 from the `gpua100` queue in exclusive-process mode. Generation jobs request 12 CPU slots and 8 GB of system memory per slot because Lean verification runs alongside the GPU worker. Training-only jobs request 8 CPU slots and 16 GB per slot for dataset processing and checkpoint conversion.

The DTU queue currently contains both 40 GB and 80 GB A100 cards. DTU does not document an LSF selector for the 80 GB cards, so these jobs do not guarantee a particular GPU-memory size. The model must fit on the card assigned by LSF.

From the repository root on DTU, create the environment using the Python and CUDA modules selected for this project:

```sh
module load python3/3.10.18 cuda/12.1.1
uv sync --locked --python "$(command -v python3)"
```

If the uv cache should live on DTU scratch storage, add this once to `~/.bashrc` and start a new shell:

```sh
export UV_CACHE_DIR=/work3/USERID/uv-cache
```

Replace `USERID` with your DTU user ID. `/work3` is scratch storage and is not backed up; that is appropriate for a reproducible package cache.

Before submitting, create the ignored local configuration:

```sh
cp jobs/config.sh.example jobs/config.sh
```

Set the W&B entity and any model overrides in `jobs/config.sh`. By default, the environment and runtime storage directories are `.venv` and `storage` inside the repository. The `storage` directory is ignored by Git. Confirm queue access with `bqueues -l gpua100` on DTU.

Submit from the repository root, for example:

```sh
bsub < jobs/rl_steps.sh
bstat
```

The tracked `jobs/logs/` directory receives separate standard-output and standard-error files. GPU jobs currently have a maximum walltime of 24 hours; change the round interval in a job script when one invocation cannot finish all rounds within that limit.
