---
name: ucloud
description: "Understand and operate SDU UCloud: prepare and monitor CPU/GPU jobs, configure batch execution and SSH, mount persistent storage, transfer files, manage job environments, and diagnose failed runs using the official user guide. Use for UCloud operations and adapting projects to UCloud; not for DTU LSF jobs."
---

# UCloud

Use the [official UCloud guide](https://docs.cloud.sdu.dk/) as the platform reference. Check the relevant current page when app options, products, quotas, or behavior matter. Different providers and app versions can expose different features; do not extrapolate one app's interface to every app.

For the user's alpha-STP/delta-proof setup, read [references/thesis-workspace.md](references/thesis-workspace.md). Those paths and conventions are local choices, not UCloud requirements.

## Choose the operational route

- For submission, consult [Submit a Job](https://docs.cloud.sdu.dk/guide/submitting.html) and the selected app's page. Identify the project/provider, app version, machine size, CPU/RAM/GPU allocation, wall time, storage mounts, execution mode, and any startup arguments. Use existing session information before asking for missing details.
- For resource or storage questions, start with [Resource Allocation](https://docs.cloud.sdu.dk/guide/resources-intro.html) or [File Management](https://docs.cloud.sdu.dk/guide/file-management.html). Inspect current grants and available products rather than inventing availability or prices. Provider capabilities can differ.
- For running or completed jobs, use [Monitor a Job](https://docs.cloud.sdu.dk/guide/monitoring.html). Prefer authorized SSH for commands and file inspection, and browser tools for portal actions when available. Do not invent a submission CLI or API.
- For platform faults beyond the application, use [Troubleshooting](https://docs.cloud.sdu.dk/help/troubleshooting.html). Prepare a concise report with job ID, app/version, resources, relevant logs, and reproduction steps; sending it is a separate action governed by the user's authorization.

Operate within the requested scope. Preparing or smoke-testing a job does not authorize launching a full experiment, stopping other jobs, or changing unrelated environments. Honor explicit authorization already given; this skill adds no blanket approval gate.

## Batch execution and job lifetime

For a workload that should stop its allocation on completion, select **Batch mode** in [PyTorch](https://docs.cloud.sdu.dk/Apps/pytorch.html#batch-mode), or **Batch processing** in [Terminal](https://docs.cloud.sdu.dk/Apps/terminal.html#run-a-script-in-batch-mode). These accept a Bash script that starts with the job; the job ends after it returns. Terminal also documents an **Extra options** field for script arguments. Verify argument support for other apps instead of assuming it.

Keep the workload in the foreground and propagate its exit status. If a script launches children, wait for them, including evaluation and final exports. A detached process, `nohup`, `&`, or `tmux` is useful for some interactive workflows but is not the mechanism for a batch allocation to track completion.

**Initialization** configures the startup environment; it is not a substitute for selecting batch mode. Manually running the training script inside an interactive app does not make that app stop afterward. SSH access itself does not change a foreground batch script's completion condition.

Distinguish script completion from workload success. An immediate completed job may be a startup failure. Choose wall time from measured runtime, including compilation, evaluation, and checkpoint export; credits or time limits can terminate a job before the workload finishes.

## Storage and reproducible environments

Selected data volumes appear beneath `/work`; inspect actual mount names and permissions before assuming a path. Persistent application inputs, code, environments, models, credentials, and outputs must be in explicitly mounted storage or the job output area. Do not rely on a previous container's home directory, `/tmp`, activated shell, installed system packages, or login state surviving a new job.

Mount the complete dependency tree, not only the batch script. Check absolute virtualenv interpreter links and paths to toolchains, datasets, and caches. Shared mounted storage can be accessed by multiple jobs; avoid modifying code or environments underneath active workloads.

Record whether changes were made to local Git files, the remote checkout, or an untracked shared environment script. Remote changes outside repositories will not appear in a local Git diff. Sync only reviewed task files and preserve unrelated remote changes.

Use the environment defined for the selected app/platform. When adding an isolated GPU environment, keep the original virtualenv and dependency files intact if required by the user. Avoid implicit environment selection such as an unqualified `uv run` that can pick a project's original `.venv`.

## SSH and file transfer

Obtain the connection command from the current job's SSH panel or the user. SSH-enabled apps require a registered public key; the assigned port can change between jobs. Do not reuse a historical port as evidence of current access.

Use the existing authorized identity, for example:

```bash
ssh -i ~/.ssh/id_ed25519_ucloud -o IdentitiesOnly=yes ucloud@ssh.cloud.sdu.dk -p PORT
scp -i ~/.ssh/id_ed25519_ucloud -o IdentitiesOnly=yes -P PORT local-file ucloud@ssh.cloud.sdu.dk:/work/project/
```

Treat these as examples; substitute the current endpoint and actual mount. Keep host-key checking enabled. Diagnose connection failures before retrying:

- Timeout/refusal: verify the job is running and the endpoint is current.
- Public-key rejection: inspect `ssh -v` and `ssh-add -l`. If the server accepts the public key but the client cannot sign, the encrypted private key may need unlocking into the agent. Ask the user to run `ssh-add ~/.ssh/id_ed25519_ucloud` locally and enter the passphrase there. Never ask for the passphrase in chat.
- Missing files after login: inspect `/work`; SSH access does not mean the necessary storage was mounted.

After a GPU job ends, a small authorized CPU monitoring job can read its persistent artifacts if the relevant folders are mounted. Its own output folder does not automatically contain the ended job's logs. Use the completed-job folder in UCloud Files, or ask for it to be copied/mounted when necessary. Do not launch a GPU workload on the monitoring CPU job.

## Fresh-job authentication and preflight

Store service credentials on private persistent storage outside Git, or in the user's existing secret mechanism. A `.gitignore` entry is not access control. Restrict file permissions and ensure the shared startup environment exports credentials to child processes. Verify presence/export without printing values; do not place secrets in logs, command-line arguments, committed YAML, or saved job parameters.

For W&B, an interactive login in an ephemeral home may not survive the next job. An explicitly loaded private file exporting `WANDB_API_KEY` can provide persistent authentication. Offline tracking is a valid chosen mode, but an offline smoke test does not validate an online production run. Verify authentication and access to the configured project using the intended production mode before declaring the launch ready; don't silently substitute offline mode.

Check the allocated hardware and the selected interpreter, imports, GPU kernel execution, actual data/model paths, writable outputs, verification toolchains, and scheduler assumptions. UCloud jobs do not inherently supply DTU LSF variables such as `LSB_DJOB_NUMPROC`; provide CPU resources explicitly or obtain them from the actual allocation.

A useful smoke run exercises the real workload path with production precision, sequence length, kernels, optimizer, and sharding. Check finite results, checkpoint/export completion, and any next-stage reload. For multi-stage jobs, verify generation, external verification, data selection, and training handoffs, not merely imports. Keep smoke artifacts separate from production and report exactly what was and was not verified.

## Diagnose and verify completion

Read the job's standard output and `JobParameters.json` before attributing a failure. The guide documents `stdout.txt`; some observed app versions use `stdout-0.log`, so inspect actual filenames. `job-0.sh` shows the backend startup command. A configured `job-report.csv` provides sampled resource usage. Completed-job outputs are reachable from the completion page or `Jobs` directory. [Monitoring reference](https://docs.cloud.sdu.dk/guide/monitoring.html#job-completed)

Separate fatal errors from warnings. Compiler memory estimates and JAX reserved pools are not proof of OOM; a completed optimizer step with finite loss is stronger evidence. For capacity planning, measure live peak allocations as well as reserved device memory. Investigate recognized allocation errors, host-memory limits, resource starvation, missing paths, or authentication failures according to the actual log.

Report success only after checking the expected final step/iteration, saved metadata, referenced model shards and their structural completeness, and completion logs or process exit status when available. State limitations if only an early log or an inaccessible tracking page is available. Do not infer theorem-solving quality from falling training loss alone.
