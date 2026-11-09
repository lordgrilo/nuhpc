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
   - **Logits, log-probs, hidden states, forced prefixes:** `python`. Your script loads the model with transformers. Run one task per model and loop over every condition inside it. Worked example: [`examples/logprobs.py`](https://github.com/lordgrilo/nuhpc/blob/main/examples/logprobs.py), which scores candidate answers by log-probability and resumes. A script built this way ran 9,500 trials of a 7B VLM in about 15 GPU-minutes.
   - **Generated text at scale** (evals, prompt or sampling sweeps): `vllm_eval` with `--pack`. Rows that share a model then share one vLLM server, and the client runs once per row (`outputs/task_K/row_R/`).
   - **Weights:** run `hf_download` first, because jobs run with `HF_HUB_OFFLINE=1`.
3. **Make code fit the contract.** The entry script takes `--params <json> --out <dir>` and writes everything it keeps into `--out`.
   - Make it resumable: skip rows already in the output file, so a timeout or requeue costs nothing.
   - Put large inputs and assets on the cluster with `nuhpc push` and reference them by remote path. The cluster has no macOS fonts: push the `.ttf` and pass its path as a param.
4. **Dry-run first** for any new template or sweep shape: `nuhpc --json --dry-run submit ...`. It changes nothing on the cluster, and only writes the local stage folder (`local_stage`) so you can inspect it. Read `local_stage/job.sbatch` and check the resources, array size, `n_rows`, and `worst_case_gpu_hours`.
5. **Pilot, then submit.** Run one small task first (a `limit` param, or a 1-row sweep) and check its runtime and output. Then `nuhpc --json submit TEMPLATE --name <short-descriptive> --project <project> --code <dir> -p k=v ... [--sweep f | --grid k=a,b] [--pack] --profile <p>`. Record the `run_id`. `--project` is a short, stable tag for the work the run serves (e.g. `miller`); reuse the tag earlier runs of that work used (`nuhpc --json runs`), and pass it to `run` too. If the human declared that project's directory in the config, the tag also decides where the run's `runs/` and `data/` go, so give `push`, `pull` and `ls` the same `--project`; `nuhpc --json check` lists the declared projects. You cannot choose a directory yourself: if work needs a new one, ask the human to add a `[projects.NAME]` entry.
6. **Wait with `nuhpc --json wait <run_id>... --timeout 540`.** Keep `--timeout` under your shell tool's time limit. Exit 3 means it timed out and the current states are included, so call it again, or run it in the background. It rides out network drops and returns final states plus the log tail of the first failed task. Don't hand-write polling loops. For progress mid-run, use `nuhpc --json logs <run_id> --task i --grep '^\[p|s/trial|Traceback|Error'`.
7. **On failure,** read `failed_log_tail` or the logs. Fix the root cause, then resubmit only the failed configurations (a sweep file holding just those rows), not the whole sweep.
8. **Collect.** `nuhpc --json fetch <run_id> --include '*.json' --include '*.jsonl' ...` Fetch only what the analysis needs; never fetch checkpoints unless asked.
9. **Space.** To see where disk space goes, run `nuhpc --json disk --scan` (a quick `du` on the login node, about a minute, no job; ask the human first for `--lab`), then `nuhpc --json disk [PATH]` reads the cached scan. Only if file counts or the largest files are needed, and the human agrees, add `--job` (a short compute-node job). Report the biggest folders and how old they are, and say when the scan is partial; never delete anything, there is no delete verb and clearing out is the human's to do.
10. **Report** the run_ids, what ran, what failed and why, and where the results are locally. Add the compute from `nuhpc --json usage <run_id>...`: GPU-hours used against `req_gpu_h`. If `time_%` or `gpu_util_%` is low, say so and size the next submit from it (a shorter `--time`, a bigger batch, or a smaller GPU).

## Size jobs before you submit

Explorer limits how many jobs one user can run at once, and **every array task counts as a job**: 50 on `short`, 2 on `gpu-short`, 4 on `gpu` (live caps, 2026-10-15). Many tiny tasks therefore queue behind your own cap while the cluster is idle, whereas one wide job starts in about a minute. nuhpc refuses more than 16 array tasks of at most 2 CPUs unless they are packed, and warns from 9.

1. **Pack CPU rows:** `--rows-per-job N` runs N rows side by side in one job. `--cpus` and `--mem` describe ONE row; nuhpc asks Slurm for N times that. Keep a job within a typical node (about 20 CPUs and 60 GB on `short`, roughly 3 GB per CPU): rows of 1 CPU and 6 GB fit as `--rows-per-job 10`. Take each row's memory from a pilot's peak in `nuhpc --json usage` plus 30%, not from a guess.
2. **`--time` covers the whole job** and is a cap that kills the task: set it to about 1.5 times the slowest measured row times the number of waves (`rows-per-job` divided by `--parallel`, which is 1 wave by default). A timeout loses the work, so make rows write results as they go. Don't ask for far more than needed either: it delays the start.
3. **Before any submit of more than a few jobs, run `nuhpc --json standing`** (one read-only call, about 15 s). If `queue.label` is `POOR`, a cap is holding your jobs back (`QOSMaxJobsPerUserLimit`): submit nothing more, and wait or pack what is queued; `nuhpc --json queue` lists the jobs. `JobArrayTaskLimit` is your own `%N` throttle and needs no action. Then read `advice`, as the next section says: a poor standing changes how you submit.
4. **GPUs:** rows share a GPU job only one after another. `gpu-short` runs 2 jobs per user and `gpu` 4, so never launch GPU pilots in parallel: put all conditions for one model in one job, and use `gpu` for batches.
5. **Read `warnings` in every submit result.** A warning means the shape is wrong (too many tiny tasks, a job bigger than a typical node): fix it, or stop and tell the human. Never use `--confirm-big` to get past a refusal, and never route work through another partition to get around a cap.

## When standing is poor: bundle and cut waste, then ask

`nuhpc --json standing` returns three things to act on:

- `advice`: a list of `{code, kind, text}`. Kind `plan` (`caps`, `fairshare`, `busy_queue`) means how you use the cluster has to change. Kind `note` (`gpu_full`) only affects which GPU type to ask for.
- `cluster.partitions[]`: how busy each queue is for everyone (`label`, `median_wait_min`, `waiting`, free `gpus` by type). It describes the cluster, not you, and never changes `overall`.
- `overall` and `fairshare.label`, which describe your account.

**Trigger:** `overall` is not `GOOD`, or any `advice` entry has `kind: "plan"`. Before you submit anything beyond a pilot or a handful of jobs:

1. **Measure the waste.** Run `nuhpc --json usage --since 7d`. In `totals`, read the `all` row and the row of the project you are working for: `cpu_idle_%` (core-hours that sat idle, CPU-only tasks), `gpu_idle_%` (GPU-hours that sat idle) and `failed_%` (billed hours spent on tasks that did not complete). Poor means `cpu_idle_%` above 25, `gpu_idle_%` above 50, or `failed_%` above 5. The per-run rows in `runs` (`cpu_idle_%`, `gpu_util_%`, `time_%`, `mem_%`) show which runs did it; name the worst by `cpu_h` times `cpu_idle_%`.
2. **Write a short cost plan** for the human. Name the actual runs, and for each poor marker give the fix:
   - `cpu_idle_%` high: ask for the cores the work kept busy (CPUs requested times `1 - cpu_idle_%/100`), measured by a pilot, and bundle small tasks with `--rows-per-job` so one wide job replaces many narrow ones.
   - `gpu_idle_%` high: one task per model with every condition looped inside it, a bigger batch, a smaller GPU type, or a shorter `--time`.
   - `failed_%` high: pilot first, `--time` at about 1.5 times the slowest measured task, memory at the pilot's peak plus 30%, rows that write as they go, and resubmit only the failed rows.
   - `caps`: pack rows; do not add jobs. `busy_queue`: expect the stated wait, ask for a shorter `--time` so the job can backfill, and never cancel and resubmit a job that is only waiting.
   Give the saving as hours: idle core-hours (`cpu_h` times `cpu_idle_%`) and idle GPU-hours at about 100 core-hours each. That weight is an estimate; say so.
3. **Ask before the heavy submit.** A pilot or a handful of jobs may go ahead. Anything larger waits until the human accepts the plan or says to go anyway. A poor fair share never blocks a submit by itself; the plan is what makes the human's call informed.
4. **Report against the plan.** After the run, check `nuhpc --json usage <run_id>` and say whether the idle and failed shares fell.

Do not propose these, because they do not lower the bill: bundling jobs to save billing (packing changes queue waits and the job cap, not the hours billed; only fewer idle or lost hours do), submitting at quiet times (fair share ignores when a job runs), moving work to another partition to get round a cap, or cancelling and resubmitting a queued job. If `gpu_full` says the type you need has no free card, the job will wait and that is normal; ask the human before switching GPU type to chase free cards, because it changes cost and may change results.

## Normal on Explorer (don't "fix" these)

- GPU jobs sit `PENDING` for minutes to an hour. Never resubmit because of it.
- vLLM startup takes 2–19 minutes, because weights and imports come over network storage. `vllm_eval` allows 45 minutes. Don't cancel a run that is still starting.
- `cuda available: True` can still fail at the first real kernel when a wheel's CUDA is newer than the driver (570.86, CUDA ≤12.8). Trust the smoke template's `cuda kernel: ok` line instead.
- Jobs inherit `http_proxy`, which is how compute nodes reach the internet. A server you start on the node must be reached with `no_proxy=127.0.0.1,localhost`; `vllm_eval` sets this.
- The run ledger is local: runs submitted from another machine show in `queue`, but not in `status`, `wait` or `usage`.

## Hard rules

- **Never pass `--confirm-big`.** If a submit is rejected by limits, stop and ask the human, giving the worst-case GPU-hours and task count. Alternatively, shrink the job (fewer tasks, shorter `--time`, smaller GPU) when that still answers the question.
- **No heavy submit while `advice` holds a `plan` entry** (`caps`, `fairshare`, `busy_queue`) until the human has accepted your cost plan (see "When standing is poor"). Pilots and a handful of jobs are fine.
- **Use `a100x1` or `h200x1`**, adding `--partition gpu-short` (≤ 2 h, starts sooner) for pilots. Don't use `gpu1`: it can land on a 16 GB T4, or a V100 that this environment hasn't been tested on. Ask before multi-GPU: the `multigpu` partition needs RC access.
- Request `--time` near the realistic runtime plus margin, not the partition maximum. For `vllm_eval`, that is up to 20 minutes of server startup plus the client's runtime for every row of a pack; 45 minutes covers a small pilot. Use `--max-parallel` so you don't flood the shared queue.
- Weights must already be cached (`hf_download`). Gated models (Llama, Gemma) need an `HF_TOKEN`; ask the human.
- The Python environment (whatever `env_setup` activates) is the human's to change. If imports fail because it is missing or lacks a package, stop and give the human the command (`nuhpc submit env_build --profile cpu`, adding `-p extra="..."` for packages); don't run `env_build` yourself.
- Don't cancel runs you didn't submit in this session unless asked.
- If SSH fails with a hint to run `nuhpc connect`, or a hint that the server refused the SSH key, stop and tell the human. Both are theirs to do: connecting is interactive, and getting the key means signing in to Open OnDemand with Duo.
