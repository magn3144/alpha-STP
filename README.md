# Unofficial implementation of STP

This implementation of STP has been adapted for a single DTU LSF node with one or more NVIDIA GPUs and local filesystem storage.

## Running on DTU HPC

Experiment orchestration is implemented as Python CLIs under `RL/`. Scheduler resource requests and DTU environment setup are kept separately under `jobs/`.

Create the local macOS environment from the repository root:

```sh
uv sync --locked
```

On DTU, load Python and create a separate Linux environment from the same lockfile:

```sh
module load python3/3.10.18
uv sync --locked --python "$(command -v python3)"
```

The lockfile selects CPU JAX on macOS and CUDA JAX plus vLLM on Linux. The Linux wheels include their CUDA libraries, so loading a different CUDA module can cause library conflicts. Do not copy `.venv` between the two systems; run `uv sync --locked` in each clone.

All included job scripts request one A100 from the `gpua100` queue. See [`jobs/README.md`](jobs/README.md) for configuration and submission instructions. In short:

```sh
cp jobs/config.sh.example jobs/config.sh
# Edit jobs/config.sh after checking module avail and your DTU paths.
bsub < jobs/rl_steps.sh
```

The generation jobs use the allocated GPU for vLLM or embeddings and the allocated CPU slots for Lean verification. Training starts after generation and uses the same GPU through JAX/Levanter. The inference model must fit on one GPU, and single-GPU full-model training requires enough memory on the A100 assigned by LSF.

## Debugging the STP loop

Download the small debug model once:

```sh
source .venv/bin/activate
huggingface-cli download amd/AMD-Llama-135m \
    --local-dir storage/models/AMD-Llama-135m
```

Connect VS Code to `hpclogin1`, open a terminal, and enter an interactive A100 shell:

```sh
a100sh
nvidia-smi
```

Choose a GPU with no listed processes and almost no allocated memory, then start the debug launcher with that GPU index:

```sh
DEBUG_GPU=1 jobs/debug_rl.sh
```

Enter your DTU password when prompted. When the launcher says it is waiting for VS Code, open **Run and Debug**, select **Attach: STP tiny-model GPU debug**, and press F5. The profile enables subprocess debugging and runs one round with one statement and one sample. Each run writes to a new directory under `storage/STP_debug_amd135m/`. Press Ctrl+C in the terminal after stopping the debugger.

## Weights & Biases

Training metrics are logged to Weights & Biases. Create a W&B project, then set its entity (your username or team) and project name in the ignored `jobs/config.sh`:

```sh
WANDB_ENTITY=your-wandb-username
WANDB_PROJECT=your-project-name
```

Authenticate once on DTU using the same virtual environment configured by `STP_VENV`:

```sh
source .venv/bin/activate
wandb login --relogin --verify
```

Paste the API key from your W&B user settings when prompted. The login is stored in your home directory and is available to jobs submitted under your account. Do not add the API key to `jobs/config.sh` or commit it to the repository.

Both the data metrics and the Levanter training metrics use `WANDB_ENTITY` and `WANDB_PROJECT` and resume under the same W&B run.

This implementation of "STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving" has been modified to run on GPUs. It also supports using my unofficial implementation of AlphaProof as the solver instead of an LLM.
