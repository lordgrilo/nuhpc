---
name: nuhpc
description: Run, monitor and collect experiments on Northeastern's Explorer HPC cluster (Slurm, GPUs) via the `nuhpc` CLI. Use when a task needs GPU compute, LLM/VLM inference or evals at scale, parameter sweeps, or training that can't run locally.
---

# Running jobs on Explorer with nuhpc

You reach the cluster only through `nuhpc`. Never call `ssh`, `scp`, `rsync` or `sbatch` directly. Always pass `--json` and parse the output; errors come back as `{"error": "..."}` with exit code 1.

## Workflow

1. **Orient.** Run `nuhpc --json templates` to see the templates and their params (read the descriptions; don't guess). Run `nuhpc --json check` if you don't yet know the partitions or whether SSH works.
2. **Make code fit the contract.** The entry script takes `--params <json> --out <dir>` and writes every result into `--out`. Keep data and weights out of the code dir; put large inputs on the cluster with `nuhpc push` and reference them by remote path.
3. **Dry-run first** for any new template or sweep shape: `nuhpc --json --dry-run submit ...`. Then read `local_stage/job.sbatch` and check the resources, array size, and `worst_case_gpu_hours`.
4. **Submit.** `nuhpc --json submit TEMPLATE --name <short-descriptive> --code <dir> -p k=v ... [--sweep f | --grid k=a,b] --profile <p>`. Record the `run_id`.
5. **Poll with backoff.** `nuhpc --json status <run_id>`. Check every 2–5 min early, then every 10–15 min. GPU jobs can sit PENDING for a long time. That is normal and not an error, so do not resubmit because of it.
6. **On failure,** use `nuhpc logs <run_id> --task <i>` for the failed tasks. Fix the root cause and resubmit only the failed configurations (a new `--sweep` file containing just those rows), not the whole sweep.
7. **Collect.** `nuhpc --json fetch <run_id> --include '*.json' --include '*.jsonl' ...` Fetch only what the analysis needs; never fetch checkpoints unless asked.
8. **Report** the run_ids, what ran, what failed and why, and where the results are locally.

## Hard rules

- **Never pass `--confirm-big`.** If a submit is rejected by limits, stop and ask the human, giving the worst-case GPU-hours and task count. Alternatively, shrink the job (fewer tasks, shorter `--time`, smaller GPU) when that still answers the question.
- Use the smallest resources that fit. Request `--time` near the realistic runtime plus margin, not the partition maximum, because shorter jobs schedule sooner and count less against limits. Use `--max-parallel` so you don't flood the shared queue.
- Before a long sweep, run one task as a pilot (`--grid` with a single value, or a 1-row sweep). Check its runtime and output, then launch the rest.
- Weights must already be cached on the cluster (`hf_download`) unless the smoke test showed that compute nodes have internet.
- Don't cancel runs you didn't submit in this session unless asked.
- If SSH fails with a hint to run `nuhpc connect`, stop and tell the human. Connecting is interactive (Duo/password) and is theirs to do.
