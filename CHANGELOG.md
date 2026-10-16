# Changelog

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
