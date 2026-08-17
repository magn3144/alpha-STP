# DTU LSF jobs

All jobs request one A100 from the `gpua100` queue in exclusive-process mode. Generation jobs request 12 CPU slots and 8 GB of system memory per slot because Lean verification runs alongside the GPU worker. Training-only jobs request 8 CPU slots and 16 GB per slot for dataset processing and checkpoint conversion.

The DTU queue currently contains both 40 GB and 80 GB A100 cards. DTU does not document an LSF selector for the 80 GB cards, so these jobs do not guarantee a particular GPU-memory size. The model must fit on the card assigned by LSF.

Before submitting, create the local configuration:

```sh
cp jobs/config.sh.example jobs/config.sh
```

Set the virtual-environment path, modules, storage path, and models in `jobs/config.sh`. Confirm the module names with `module avail` and queue access with `bqueues -l gpua100` on DTU.

Submit from the repository root, for example:

```sh
bsub < jobs/rl_steps.sh
bstat
```

The tracked `jobs/logs/` directory receives separate standard-output and standard-error files. GPU jobs currently have a maximum walltime of 24 hours; change the round interval in a job script when one invocation cannot finish all rounds within that limit.
