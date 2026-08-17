# Unofficial implementation of STP

This implementation of STP has been adapted for a single DTU LSF node with one or more NVIDIA GPUs and local filesystem storage.

## Running on DTU HPC

Experiment orchestration is implemented as Python CLIs under `RL/`. Scheduler resource requests and DTU environment setup are kept separately under `jobs/`.

All included job scripts request one A100 from the `gpua100` queue. See [`jobs/README.md`](jobs/README.md) for configuration and submission instructions. In short:

```sh
cp jobs/config.sh.example jobs/config.sh
# Edit jobs/config.sh after checking module avail and your DTU paths.
bsub < jobs/rl_steps.sh
```

The generation jobs use the allocated GPU for vLLM or embeddings and the allocated CPU slots for Lean verification. Training starts after generation and uses the same GPU through JAX/Levanter. The inference model must fit on one GPU, and single-GPU full-model training requires enough memory on the A100 assigned by LSF.

This implementation of "STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving" has been modified to run on GPUs. It also supports using my unofficial implementation of AlphaProof as the solver instead of an LLM.
