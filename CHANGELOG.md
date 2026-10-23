# Changelog

## 0.7.0 (2026-10-23)

- **`nuhpc disk` opens at once and refreshes live.** In a terminal the explorer shows the last scan immediately (an empty one the first time) while the new data is collected on a thread. A line under the title says what is happening, with a spinner and the time: "connecting…", then "collecting folder sizes · N measured" for the quick scan (the remote output is streamed), or "queued as job N…", "running on a compute node…", "fetching the result…" for `--job`. The new scan is swapped in place, keeping your folder and row, and the line becomes "updated just now" (or the reason it failed, with the old data kept). `r` collects again, `--no-refresh` skips it, quitting stops the collection, and stderr output during the refresh is held back so it cannot tear the screen. Piped, `--print` and `--json` behave as before: nothing is refreshed unless `--scan` or `--job` is given.
- **A scan job never runs from inside the folder it scans.** `disk --project NAME --job` used to submit under that project's tag, putting its run folder inside the project directory; it now always runs from your own work root.
- **`submit` no longer double-queues after an sbatch error.** Slurm sometimes accepts a job and still reports "Unexpected message received"; `nuhpc submit` now looks for a job already queued from that run's folder and adopts it, so a retry does not queue it twice.
- New `Remote.stream` (lines as they arrive, never retried) and `--no-refresh`.

## 0.6.1 (2026-10-23)

- **`nuhpc disk` shows how full the disk is.** The header gets the filesystem's size, use and free space with a gauge, and this folder's share of it. Where `df` reports a directory quota (as it does inside a group's folder on Explorer), the gauge is labelled `quota` and a second line shows how full the whole filesystem beneath it is. Both scanners record it (`fs` in the JSON, with `parent` for the pool under a quota): the quick scan from `df`, the full scan from `statvfs`.

## 0.6.0 (2026-10-23)

- **`nuhpc disk`: see where your space goes.** `nuhpc disk --scan` measures a folder tree and opens an interactive explorer in the terminal in the style of `mo analyze`: bars sorted by size with percentages, sizes in the `du -h` convention, ages, arrow keys to drill down and back, `s` to sort by size or age, `l` for the largest files. Pastel colours cycle by rank and ages fade from mint to rose; `NO_COLOR` and a non-UTF-8 locale are honoured, and rows are cut to the terminal width. Piped, `--print` or `--json` gives a static report or the data; the result is cached under `~/.local/share/nuhpc/disk/`, and `--from FILE` opens a saved scan.
- **The default scan is not a job.** It runs `du` on the login node: a capped (`--budget`, 60 s), lowest-priority script sent over ssh on stdin, six `du` at a time over the folders two levels down, nothing written on the cluster. Folders it cannot finish are shown as `not read` or as lower bounds (`>=`) and the scan is marked partial. It gives sizes and ages, not file counts or names. (A multi-threaded Python walker was tried first and was killed on the login node about half the time: login nodes allow each user a fraction of a core.)
- **`--job` for the full scan.** The `disk_scan` template and `diskscan.py` (stdlib, Python 3.9-safe for the compute nodes) walk the tree with parallel readers on a compute node and add file counts, the largest files per folder and overall, unreadable-folder counts and a time budget with a partial result; `--collect RUN` picks up a scan that outlived the wait.
- **Scopes:** your `user_dir`, `--project NAME` (declared projects only), or `--lab` (the group folder: one total per member, depth 1, no file names).

## 0.5.0 (2026-10-20)

- **`nuhpc standing`:** one read-only login-node call (no job) that reports queue pressure against the live per-partition caps and your account's fair-share standing, with a GOOD / OK / POOR verdict. Fair share: the factor, how the account's use compares with its equal share and its rank among all accounts, how its use splits between you and the rest (names only with `--users`), the priority points it is worth, and an estimate of the recovery time from the decay half-life. `--brief` prints one line, and the JSON has a short `brief` string for status lines. Cut-offs go in an optional `[standing]` config table. A failed query reads as `unknown`, never as good news. The report is styled like `npl` (a title, centred rules, gauges, aligned rows cut to the terminal width), coloured only on a terminal; `NO_COLOR` and `FORCE_COLOR` are honoured, and a non-UTF-8 locale or stream gets an ASCII form.
- **`standing` shows every queue and the whole cluster.** The Queue section lists every partition you can use (idle ones too) with its cap and time limit. A new Cluster section reports, per queue and for everyone: busy cores, jobs waiting on the scheduler versus held (dependency, hold, cap, start time), the median and slowest-tenth wait of jobs that started in the last 6 hours (from when each became eligible; shown only with 5 or more starts), the oldest job still waiting, and free GPUs by type. Each queue is labelled GOOD, OK or POOR from its median wait; the label never changes the overall verdict, and other users' jobs are counted but never named. If the scheduler shows only your own jobs, the section reads `unknown` instead of guessing. New `[standing]` keys: `wait_window_h`, `cluster_wait_ok_min`, `cluster_wait_poor_min`.
- **Advice.** `standing` returns `advice`, a list of `{code, kind, text}`: `caps`, `fairshare` and `busy_queue` (kind `plan`: change how you use the cluster) and `gpu_full` (kind `note`). The report prints it under an Advice rule, and the JSON carries it for agents.
- **`usage` reports waste.** Per run and per project: `cpu_idle_%` (core-hours idle, CPU-only tasks), `gpu_idle_%` (sampled GPU-hours idle) and `failed_%` (billed hours lost to tasks that did not complete; a GPU-hour counts as about 100 CPU-hours, an estimate), all weighted by hours.
- **Skill:** a "When standing is poor" section makes an agent measure the waste with `usage`, write a cost plan (right-size, bundle with `--rows-per-job`, pilot first, `--time` from measurements) and get the human's acceptance before any heavy submit; it also lists what does not lower the bill. A matching hard rule was added.
- **Fix:** a job submitted to several partitions at once no longer shows as a stray `a,b` row in your queue; it counts in each of them.
- **README correction:** it said GPU time dominates fair share. Billing counts a GPU-hour as about 100 CPU-hours, so a large CPU burst counts as much (11,000 CPU-hours in one day were worth about 110 GPU-hours). Fair share follows total billed use; packing rows fixes the running-job cap but does not change it.

## 0.4.1 (2026-10-16)

- **Fix:** the "bigger than most nodes" warning (`limits.big_job_cpus` and `big_job_mem`) no longer applies to GPU jobs. Those thresholds describe CPU nodes, so the warning fired on every ordinary GPU job (an A100 job of 12 CPUs and 120 GB, an H200 job of 16 CPUs and 200 GB). The README's wording about fair share now says it belongs to the Slurm account, which a lab shares.

## 0.4.0 (2026-10-15)

- **CPU packing.** `submit --rows-per-job N [--parallel P]` puts N rows into each job and runs them side by side. `--cpus` and `--mem` describe one row; the job asks for that times P (the submit result shows `per_row_resources`, `resources` and `waves`). Each row gets its own `outputs/task_K/row_R/` folder, and `HPC_OUT` and `HPC_PARAMS` name it inside the row, so scripts need no change. Rows get `OMP_NUM_THREADS` and the other BLAS thread variables set to their CPU count, their output lines are prefixed `[row R]`, and a failing row leaves its exit code in `row_R/exit_code` without stopping the others. GPU jobs run their rows one at a time.
- The `python` and `cmd` templates now run through `hpc_each_row`, so they support packing; `vllm_eval --pack` is unchanged. Packed rows now run in their own subshell, and every packed row writes `exit_code`.
- **Submit-time checks.** More than `limits.small_task_max` (default 16) array tasks of at most `small_task_cpus` (2) CPUs and no GPU are refused unless packed or given `--confirm-big`; above `small_task_warn` (8) they warn. The message names the `--rows-per-job` to use. `limits.big_job_cpus` and `big_job_mem` warn when one job is bigger than a typical node. Warnings go to stderr and to `warnings` in the JSON result.
- Jobs export `HPC_PACK_PARALLEL` and `HPC_ROW_CPUS`.
- README: a "Sizing jobs on Explorer" section with the live limits, node sizes and rules, and the packing contract. `SKILL.md`: a sizing checklist for agents.

## 0.3.0 (2026-10-13)

- **`user_dir` and project directories.** Installs (`envs/`, `hf_cache/`) live in `user_dir`, your folder on project storage, shared by all your projects. Runs and data live in a `nuhpc/` folder inside `user_dir`, or inside a project directory declared as `[projects.NAME] dir = "..."` and chosen with `--project NAME` on `submit`, `run`, `push`, `pull`, `ls` and `check`. Only declared directories can be used, so an agent cannot write elsewhere.
- Jobs export `HPC_USER_DIR`; templates get `{{USER_DIR}}`, and `env_build` builds under it. `HPC_REMOTE_ROOT` is now the run's work root.
- `nuhpc init --user-dir DIR` writes the config with your folder filled in, and a config that still has the placeholder is refused with a message. `nuhpc check` verifies `user_dir` and creates the work root only when `user_dir` exists, so a typo does not create a folder.
- Configs with only `remote_root` behave exactly as before. With both set, `user_dir` holds the installs and `remote_root` replaces `<user_dir>/nuhpc` as the default work root.

## 0.2.0 (2026-10-12)

- **Usage accounting.** Every finished run gets a usage record: what it requested beside what Slurm accounted per task, plus GPU utilisation from 30 s `nvidia-smi` samples. Records are kept locally, beside fetched results and in `~/.nuhpc/usage/` on the cluster. `submit`/`run --project` tag runs, and `nuhpc usage` reports per run and per project.
- **Per-person environments.** The `env_build` template builds `$remote_root/envs/llm` in a job (uv, Python 3.12, torch, vLLM). Templates can declare `# env-setup: none`, and jobs export `HPC_REMOTE_ROOT`, which the example config's `env_setup` uses.
- **Direct-mode example.** `examples/logprobs.py` scores candidate answers by log-probability.
- **Example config.** `remote_root` is on project storage, and `default_profile` is `a100x1` (`gpu1` can land on a T4).
- MIT licence.

## 0.1.0 (2026-10-07/08)

- Verbs `check`, `run`, `submit`, `wait`, `status`, `logs`, `fetch`, `cancel`, `runs`, `queue`, `push`, `pull`, `ls`, with `--json` and `--dry-run`.
- Templates `smoke`, `cmd`, `hf_download`, `python`, `vllm_eval` (with `--pack`) and `train_ddp`.
- GPU-hour and task limits behind `--confirm-big`, and retries for repeatable transport calls.
