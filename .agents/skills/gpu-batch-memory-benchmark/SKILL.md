---
name: gpu-batch-memory-benchmark
description: Create GPU batch-job benchmarks that find the largest non-OOM training or inference batch size and measure steady-state memory and token throughput for every tested size. Use when writing or revising batch-memory, batch-size, or throughput experiments.
---

# GPU Batch Memory Benchmark

Create a scheduler batch job that runs the real training or inference workload on the GPU model and GPU count specified by the user. Follow the repository's existing job conventions. On DTU, use LSF directives, select the matching GPU queue, request GPUs explicitly, and use `mode=exclusive_process`.

## Preserve the real workload

Make candidate batch size the only performance-relevant variable. Record the mode, workload settings, GPU name and count, GPU memory capacity, software versions, and job ID with the results.

For training, match the intended model, sequence length, precision, optimizer, gradient checkpointing, attention implementation, sharding, and data pipeline.

For inference, match the intended model, precision or quantization, attention implementation, compilation mode, prompt length, requested output length, decoding parameters, KV-cache settings, and serving concurrency. Use representative inputs of controlled lengths. State whether the benchmark measures prefill, decode, or end-to-end generation; do not combine results with different definitions.

Validate the allocated GPU name and count before loading the model. Fail if they do not exactly match the requested hardware. Do not silently run on another accelerator.

## Measure each candidate

Run every candidate in a clean subprocess so an OOM or retained allocator state cannot contaminate later trials.

For each batch size:

1. Load the model and data, then run at least one unmeasured warm-up iteration of the selected workload. For training this is a complete optimizer step; for inference it is a complete request or generation pass. Warm-up must include compilation, kernel selection, and initialization. Use additional warm-up iterations when timings are not yet stable.
2. Run multiple measured iterations. Prefer at least five when job time permits.
3. Synchronize the GPU before starting and stopping each timer because accelerator execution is asynchronous.
4. Measure peak GPU memory across warm-up and measured workload iterations at a sufficiently short sampling interval, such as 100 ms, or use an accurate framework peak-memory counter.
5. Compute tokens/s only from measured iterations, never from startup, compilation, checkpoint loading, evaluation, or warm-up. Report aggregate steady-state throughput, preferably the median, and retain individual timings so variability is visible.
6. Verify that the workload produced valid results. Training loss must be finite; inference must return the requested number of valid outputs. Treat invalid results as errors, not successes.

For fixed-length packed training, use:

```text
tokens/s = global_batch_size * sequence_length * measured_steps / measured_elapsed_seconds
```

For variable-length unpadded training, count the tokens actually processed instead.

For inference, report the token counts and rates appropriate to the selected workload:

- Prefill: input tokens/s.
- Decode: generated tokens/s, measured after prefill.
- End-to-end generation: input tokens/s, generated tokens/s, and total tokens/s separately. Also report request latency and requests/s.

State the token-counting and timing boundaries used. Do not infer throughput from a progress bar that includes compilation or startup.

Disable unrelated work such as evaluation, checkpoint export, and online experiment tracking unless it is part of the intended workload. Do not change performance-relevant kernels or settings merely to make a candidate fit.

## Find the OOM boundary

Use an exponential search followed by an integer binary search:

1. Start from a known-safe batch size, or 1 if none is known.
2. Double the candidate until it OOMs or reaches a user-specified upper bound.
3. Binary-search between the largest success and smallest OOM until adjacent valid batch sizes are identified.

If divisibility or sharding constraints restrict valid sizes, search only valid candidates and report the constraint. Treat only recognized GPU allocation failures as OOM. Abort and report other errors instead of misclassifying them. Apply a reasonable per-trial timeout so hangs do not consume the entire allocation.

The largest batch size that completes warm-up and every measured iteration with valid results is the maximum successful batch. For inference, it must complete the configured prompt and output lengths because KV-cache memory grows with sequence length. If no OOM boundary is reached, say so and report the tested upper bound; do not claim it is the maximum.

## Results

Write a machine-readable table and print a concise summary in the batch-job output. Include every attempted batch size, including failures, with at least:

```text
mode  status  batch_size  measured_iterations  median_iteration_s  tokens_per_s  peak_gpu_mib  total_gpu_mib  memory_percent
```

Add mode-specific fields rather than hiding important distinctions. Training results should include sequence length and optimizer-step throughput. Inference results should include prompt length, output length, measured phase, latency, requests/s, and separate input/generated/total token rates as applicable.

Use statuses such as `success`, `oom`, `error`, and `timeout`. For failed trials, leave unavailable performance fields empty and preserve the trial log. Finish by reporting the largest successful batch, the first OOM batch, and paths to the table and per-trial logs.

Do not select the production batch solely because it is the absolute maximum. Clearly distinguish the measured maximum from any recommended batch with headroom, and base a recommendation on measured memory margin and throughput rather than an arbitrary batch reduction.
