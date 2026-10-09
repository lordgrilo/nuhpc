---
name: nuhpc
description: Run, monitor and collect experiments on Northeastern's Explorer HPC cluster (Slurm, GPUs) via the `nuhpc` CLI. Use when a task needs GPU compute, LLM/VLM inference or evals at scale, logit or hidden-state readouts from open-weight models, parameter sweeps, or training that can't run locally.
---

# Running jobs on Explorer with nuhpc

You reach the cluster only through `nuhpc`. Never call `ssh`, `scp`, `rsync` or `sbatch` directly. Always pass `--json` and parse the output; errors come back as `{"error": "..."}` with exit code 1.

## Workflow

1. **Orient.** Run `nuhpc --json templates` to see the templates and their params (read the descriptions; don't guess). Run `nuhpc --json check` if you don't yet know whether SSH works.
2. **Pick the mode by what you need.** The README's "Which mode?" table has the measured costs.
   - **Quick check** (does it import, does CUDA work, how long does one batch take):
     `nuhpc --json run --timeout 540 --code <dir> --profile a100x1 --partition gpu-short -- python probe.py`.
     It runs any command with no contract, 30 minutes unless `--time`, and returns the log and fetched `$HPC_OUT`. Exit 2 means the command failed (the log says why); exit 3 means it's still queued or running, so follow it with `nuhpc wait <run_id>`.
   - **Logits, log-probs, hidden states, forced prefixes:** `python`. Your script loads the model with transformers. Run one task per model and loop over every condition inside it. Worked example: `run_vlm.py` in `miller-science/code/stroop_vlm/`, which took about 15 GPU-minutes for 9,500 trials on a 7B VLM.
   - **Generated text at scale** (evals, prompt or sampling sweeps): `vllm_eval` with `--pack`. Rows that share a model then share one vLLM server, and the client runs once per row (`outputs/task_K/row_R/`).
   - **Weights:** run `hf_download` first, because jobs run with `HF_HUB_OFFLINE=1`.
3. **Make code fit the contract.** The entry script takes `--params <json> --out <dir>` and writes everything it keeps into `--out`.
   - Make it resumable: skip rows already in the output file, so a timeout or requeue costs nothing.
   - Put large inputs and assets on the cluster with `nuhpc push` and reference them by remote path. The cluster has no macOS fonts: push the `.ttf` and pass its path as a param.
4. **Dry-run first** for any new template or sweep shape: `nuhpc --json --dry-run submit ...`. It changes nothing on the cluster, and only writes the local stage folder (`local_stage`) so you can inspect it. Read `local_stage/job.sbatch` and check the resources, array size, `n_rows`, and `worst_case_gpu_hours`.
5. **Pilot, then submit.** Run one small task first (a `limit` param, or a 1-row sweep) and check its runtime and output. Then `nuhpc --json submit TEMPLATE --name <short-descriptive> --project <project> --code <dir> -p k=v ... [--sweep f | --grid k=a,b] [--pack] --profile <p>`. Record the `run_id`. `--project` is a short, stable tag for the work the run serves (e.g. `miller`); reuse the tag earlier runs of that work used (`nuhpc --json runs`), and pass it to `run` too.
6. **Wait with `nuhpc --json wait <run_id>... --timeout 540`.** Keep `--timeout` under your shell tool's time limit. Exit 3 means it timed out and the current states are included, so call it again, or run it in the background. It rides out network drops and returns final states plus the log tail of the first failed task. Don't hand-write polling loops. For progress mid-run, use `nuhpc --json logs <run_id> --task i --grep '^\[p|s/trial|Traceback|Error'`.
7. **On failure,** read `failed_log_tail` or the logs. Fix the root cause, then resubmit only the failed configurations (a sweep file holding just those rows), not the whole sweep.
8. **Collect.** `nuhpc --json fetch <run_id> --include '*.json' --include '*.jsonl' ...` Fetch only what the analysis needs; never fetch checkpoints unless asked.
9. **Report** the run_ids, what ran, what failed and why, and where the results are locally. Add the compute from `nuhpc --json usage <run_id>...`: GPU-hours used against `req_gpu_h`. If `time_%` or `gpu_util_%` is low, say so and size the next submit from it (a shorter `--time`, a bigger batch, or a smaller GPU).

## Normal on Explorer (don't "fix" these)

- GPU jobs sit `PENDING` for minutes to an hour. Never resubmit because of it.
- vLLM startup takes 2–19 minutes, because weights and imports come over network storage. `vllm_eval` allows 45 minutes. Don't cancel a run that is still starting.
- `cuda available: True` can still fail at the first real kernel when a wheel's CUDA is newer than the driver (570.86, CUDA ≤12.8). Trust the smoke template's `cuda kernel: ok` line instead.
- Jobs inherit `http_proxy`, which is how compute nodes reach the internet. A server you start on the node must be reached with `no_proxy=127.0.0.1,localhost`; `vllm_eval` sets this.
- The run ledger is local: runs submitted from another machine show in `queue`, but not in `status` or `wait`.

## Hard rules

- **Never pass `--confirm-big`.** If a submit is rejected by limits, stop and ask the human, giving the worst-case GPU-hours and task count. Alternatively, shrink the job (fewer tasks, shorter `--time`, smaller GPU) when that still answers the question.
- **Use `a100x1` or `h200x1`**, adding `--partition gpu-short` (≤ 2 h, starts sooner) for pilots. Don't use `gpu1`: it can land on a 16 GB T4, or a V100 that this environment hasn't been tested on. Ask before multi-GPU: the `multigpu` partition needs RC access.
- Request `--time` near the realistic runtime plus margin, not the partition maximum. For `vllm_eval`, that is up to 20 minutes of server startup plus the client's runtime for every row of a pack; 45 minutes covers a small pilot. Use `--max-parallel` so you don't flood the shared queue.
- Weights must already be cached (`hf_download`). Gated models (Llama, Gemma) need an `HF_TOKEN`; ask the human.
- The Python environment (`/projects/nplab/gio/envs/llm`) is the human's to change. The README records how it was built.
- Don't cancel runs you didn't submit in this session unless asked.
- If SSH fails with a hint to run `nuhpc connect`, stop and tell the human. Connecting is interactive and is theirs to do.
