# nuhpc

A small CLI for running Slurm jobs on Northeastern's **Explorer** cluster (the successor of Discovery), from your laptop or from an agent.

It is built for LLM/VLM experiments first, with GPU training later. It is pure Python standard library (3.11+), and it relies on `ssh` and `rsync` on the local machine.

```
laptop / agent ──nuhpc──► ssh+rsync ──► login.explorer.northeastern.edu ──sbatch──► GPU nodes
                 ledger                    $remote_root/runs/<run_id>/{job.sbatch, sweep.jsonl, code/, outputs/, logs/}
```

## Design choices

- **Verbs, not a shell.** The agent can only call `submit / status / logs / fetch / cancel / push / pull / ls`. There is no `exec`, and remote paths are confined to `remote_root`. This is the main safety and audit boundary.
- **Every job is a run.** Each run is a frozen snapshot of the rendered sbatch, the parameters, and the code. That makes runs reproducible and easy to inspect: everything lives in `runs/<run_id>/`.
- **One contract for user code.** Your script takes `--params <params.json> --out <dir>` and writes what it wants kept into `--out`. Sweeps, evals, and training all use this same shape.
- **Sweeps are job arrays.** One task per row or grid point, throttled with `%max_parallel`. This is how you run models or configurations in parallel without monopolising the GPU queue.
- **Guardrails for agents.** Every submit computes its worst-case GPU-hours and task count. Anything above the limits in your config needs `--confirm-big`, and the agent instructions say that flag requires a human.

## Install

```bash
uv tool install git+https://github.com/lordgrilo/nuhpc    # or, from a clone: uv tool install --editable .
nuhpc init                     # writes ~/.config/nuhpc/config.toml
```

`uv` fetches a Python ≥3.11 if your system one is older. With `--editable`, edits to the clone take effect immediately; otherwise upgrade with `uv tool upgrade nuhpc`. Tests run offline: `uv run --no-project --with pytest --with-editable . pytest`.

Edit the config: set `remote_root`, `env_setup`, and the profiles. Every line marked `VERIFY` is an assumption about Explorer that you should check.

### SSH (the part that matters for unattended use)

Add this to `~/.ssh/config`:

```
Host explorer
    HostName login.explorer.northeastern.edu
    User YOUR_NU_USERNAME
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 12h
    ServerAliveInterval 60
```

Then install your key on the cluster with `ssh-copy-id explorer`, which RC documents. If logins also require Duo or a password, run `nuhpc connect` once and approve it. For the next 12 hours every `nuhpc` call reuses that socket.

All of nuhpc's own connections use `BatchMode=yes`. An expired session therefore fails immediately with a hint, rather than hanging an agent at a password prompt.

### Python environment

Jobs activate whatever `env_setup` names, so build that environment once, **inside a job**. Explorer kills conda on login nodes (even `conda --version`). Project storage costs about 0.1 s per file written, so a serial `conda`/`pip` install of vLLM plus torch runs for hours. `uv` installs in parallel and finished in 19 minutes. Compute nodes have outbound internet. The shared Anaconda Python has pip; the system `python3` does not.

```bash
# in an sbatch script (-p short -c 8 --mem=32G -t 06:00:00), or under `srun ... --pty bash`
export UV_CACHE_DIR=/tmp/$USER-uv UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1   # precompile: first imports otherwise
                                               # write each .pyc serially to /projects (stalled vLLM for 17 min)
export UV_PYTHON_INSTALL_DIR=/projects/YOUR_GROUP/envs/.python   # the venv's interpreter must exist on every node
/shared/EL9/explorer/anaconda3/2024.06/bin/python3 -m pip install --target /tmp/$USER-uvbin uv
/tmp/$USER-uvbin/bin/uv venv --python 3.12 /projects/YOUR_GROUP/envs/llm
/tmp/$USER-uvbin/bin/uv pip install --python /projects/YOUR_GROUP/envs/llm/bin/python --torch-backend=cu129 \
  "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  huggingface_hub openai
```

**Match CUDA to the driver.** A plain `pip install vllm` gets CUDA 13.0 wheels. Explorer's GPU driver (570.86 on 2026-10-07, CUDA ≤12.8) is too old for them: torch still reports `cuda available: True`, but the first kernel fails with "driver too old". CUDA 12.9 builds run on it, because NVIDIA keeps runtimes compatible within 12.x. Check `nvidia-smi` in the smoke log after any driver or version change; the smoke template launches a real kernel for this reason.

`env_setup` then needs only `source /projects/YOUR_GROUP/envs/llm/bin/activate`. Put the environment on project storage rather than `$HOME`: it is 11 GB, and a lab can share one.

### First run

```bash
nuhpc check                                  # SSH ok? lists partitions, time limits, GPU types
nuhpc check --smoke --profile gpu1           # submits a job that tests GPU, torch/vllm, internet, disk
nuhpc logs last                              # read the smoke results
```

The smoke test answers the question that decides your whole workflow: **do compute nodes have outbound internet?** If they don't, download weights ahead of time (see below) and keep `HF_HUB_OFFLINE=1`.

Use the `check` output to fix the partition names and GPU types in your profiles.

## Commands

| command | what it does |
|---|---|
| `nuhpc templates` | list job templates and their parameters |
| `nuhpc submit TEMPLATE [opts]` | render, upload code, `sbatch`; prints `run_id` and `job_id` |
| `nuhpc status [RUN...]` | per-run state with a per-task breakdown (default: last 10 runs) |
| `nuhpc wait RUN... [--timeout S]` | block until the runs finish, polling with backoff and riding out network drops; prints final states and the log tail of the first failed task (exit 3 on timeout) |
| `nuhpc logs RUN [--task i] [-n N] [--grep RE]` | tail a task's log, or only the lines matching RE across the whole log |
| `nuhpc fetch RUN [--include '*.json'] [--max-size 500M]` | rsync `outputs/` and `logs/` to `local_results/<run_id>` |
| `nuhpc cancel RUN [--task i]` | `scancel` a whole run or one array task |
| `nuhpc runs` / `nuhpc queue` | local ledger / your jobs in `squeue` |
| `nuhpc push LOCAL [REL] [--globus]` | upload data to `remote_root/data/REL` |
| `nuhpc pull REL [LOCAL] [--globus]` | download a path relative to `remote_root` |
| `nuhpc ls [REL]` | list a directory under `remote_root` |
| `nuhpc connect` | open the persistent SSH session (interactive) |

Global flags: `--json` gives machine-readable output, including errors as `{"error": ...}`. `--dry-run` prints every remote command and leaves the rendered `job.sbatch` in `~/.local/share/nuhpc/runs/<run_id>/` for inspection.

RUN can be a full `run_id`, a unique prefix, a Slurm job id, or `last`.

### Submit options

```
--code DIR            snapshot this project into the run (excludes .git, venvs, data/, weights; add .nuhpcignore)
-p key=value          parameter; values parse as JSON (so 0.1, true, [1,2] are typed)
--sweep FILE          .jsonl / .json list / .csv, one task per row (merged over -p defaults)
--grid key=a,b,c      cartesian product, repeatable; combines with --sweep
--max-parallel N      array throttle
--profile NAME        resource preset; override with --partition --time --gpus --mem --cpus --nodes --gres
--pack                one array task per group of rows sharing the template's `# pack-by:` keys
                      (vllm_eval: one server per model; the client runs once per row)
--confirm-big         bypass limits (humans only)
```

## Templates

| template | use |
|---|---|
| `smoke` | environment check; run first and after any env change |
| `hf_download` | cache an HF model or dataset into `HF_HOME` on cluster storage |
| `python` | generic: `python <entry> --params --out [args]` |
| `vllm_eval` | start a vLLM OpenAI-compatible server on the node, wait for health, run your client, shut down. Tensor parallel size = GPUs per node |
| `train_ddp` | `srun torchrun` across 1..N nodes with c10d rendezvous |

In a template, `{{CODE}} {{RUN_DIR}} {{DATA}} {{REMOTE_ROOT}} {{NODES}} {{GPUS_PER_NODE}} {{GPUS}} {{CPUS}}` are filled at render time. Per-task values come from `hpc_param KEY [DEFAULT]` at run time.

Read parameters into variables first, as in `X="$(hpc_param key)"`. Under `set -e`, a missing parameter aborts the job only inside an assignment, not when the substitution is inline in a command.

To add your own templates, drop `*.sbatch` files into `templates_dir`. Lines starting with `# doc:` show up in `nuhpc templates`, which is also how an agent learns what a template expects. A template supports `--pack` by declaring `# pack-by: key ...` (the params that must be equal within one task) and running its per-row work through `hpc_each_row CMD`, which sets `HPC_ROW_PARAMS` and `HPC_ROW_OUT` for each row.

## Workflow: LLM/VLM experiments

```bash
# 1. Cache weights once (CPU node; needs internet on that node. Otherwise download elsewhere and nuhpc push)
nuhpc submit hf_download --profile cpu -p repo=Qwen/Qwen2.5-VL-7B-Instruct

# 2. Upload the eval data
nuhpc push ./prompts.jsonl prompts.jsonl

# 3. Sweep: 2 models x 2 temperatures from the file, x 2 max_tokens = 8 array tasks, 4 at a time
nuhpc submit vllm_eval --name vlm-eval --code . --profile a100x1 --time 03:00:00 \
  -p entry=eval_client.py -p prompts=/scratch/USER/nuhpc/data/prompts.jsonl \
  --sweep examples/sweep.jsonl --grid max_tokens=256,1024 --max-parallel 4

nuhpc status vlm-eval
nuhpc fetch vlm-eval --include '*.json' --include '*.jsonl'
```

`examples/eval_client.py` is a minimal text client that follows the contract. `examples/vlm_client.py` sends images: put a `prompts.jsonl` (`{"id", "prompt", "image"}` per line) and the images in one folder, `nuhpc push` it, and pass `-p prompts=<remote path>/prompts.jsonl`. Verified on 2026-10-08 with Qwen2.5-VL-3B on an A100.

For 70B-class models, use a multi-GPU profile (`--gpus 4`). vLLM picks the tensor-parallel size up automatically from the template.

Three practical points:

- **Each task restarts vLLM**, which took 2–19 minutes on Explorer: fast on a node that has read the weights recently, slow on a cold one. Pass `--pack` so the rows that share a model share one server: the client then runs once per row, writing to `outputs/task_K/row_R/`. Size `--time` for all rows of a pack.

- Model size drives your queue time more than anything else: one H200 or A100-80GB job usually starts faster than a four-GPU job.
- Keep `max_parallel` modest so you don't monopolise the shared GPU queue.

## Later: training

`train_ddp` already handles single-node multi-GPU and multi-node runs (`--nodes 2 --gpus 4`). Your script initialises with `torch.distributed.init_process_group("nccl")` and only rank 0 writes to `--out`.

Things to add when you get there:
- **Checkpoint/resume.** Write checkpoints to `$HPC_OUT/ckpt`, and add `#SBATCH --requeue` plus a resume-from-latest path. Wall-clock limits will cut long runs.
- **Fetch selectively.** `fetch` skips files above `fetch_max_file_size` so checkpoints don't fill your laptop. Pass `--max-size` explicitly when you want them.
- **Experiment tracking.** For W&B, use offline mode if nodes have no internet and sync from the login node.

## Agents

### Mode 1: agent outside, cluster runs plain batch (recommended)

1. Link `skill/nuhpc/` into both agents' user skill folders. It tells the agent how to use the CLI and what it must not do. Claude Code reads `~/.claude/skills/`; Codex (the ChatGPT-account agent) reads `~/.agents/skills/`. One copy serves both, so updating the repo updates both:
   ```bash
   mkdir -p ~/.agents/skills ~/.claude/skills
   ln -s "$PWD/skill/nuhpc" ~/.agents/skills/nuhpc
   ln -s ~/.agents/skills/nuhpc ~/.claude/skills/nuhpc
   ```
2. Restrict the tools. For Codex: its default sandbox has no network, so it asks before running `nuhpc` outside the sandbox. Answering "always" records `prefix_rule(pattern=["nuhpc"], decision="allow")` in `~/.codex/rules/default.rules`. For Claude Code, use the project's `.claude/settings.json` or flags:
   ```json
   { "permissions": {
       "allow": ["Bash(nuhpc:*)"],
       "deny":  ["Bash(ssh:*)", "Bash(scp:*)", "Bash(rsync:*)", "Bash(sftp:*)"] } }
   ```
   Honest caveat: prefix rules can't block `--confirm-big` appearing in the middle of a command. The limits are a guardrail backed by the skill's instructions, not a security boundary. For a hard boundary, give the agent a config with stricter limits and remove the flag from your installed copy.
3. **Delayed or unattended runs.** Schedule a headless run on a machine that stays on:
   ```bash
   echo 'cd ~/proj && claude -p "Execute the plan in PLAN.md with nuhpc. Poll status every 10 min until all runs finish, fetch results, write REPORT.md." --allowedTools "Bash(nuhpc:*)" Read Write Edit' | at 02:00
   ```
   With Codex, `codex exec "..."` plays the same role. The SSH ControlMaster session must still be alive at that time. If Duo is required and the session has expired, the run fails immediately and cleanly; it does not hang.

### Mode 2: agent on the cluster

Only reach for this when the decide-inspect-resubmit loop has to sit next to data too large to move. Before doing it:
- **No agents on login nodes.** RC's policy says jobs on login nodes get terminated, and a watchdog kills heavy processes there. The agent has to run inside an allocation, and you pay for idle compute while it thinks. Use a cheap CPU allocation that submits GPU jobs rather than an agent sitting on a GPU.
- **It needs outbound HTTPS** to the API from the node it runs on. Check the smoke test result.
- **Protect the API key.** Keep it in a `chmod 600` file under `$HOME` and never put it in the run snapshot. Its scope is your whole account, so prefer a separate, spend-limited key.
- It's worth a short email to rchelp@northeastern.edu first. Long-lived agent processes are a gray area under shared-cluster policies.

Recipe, if you go ahead: install Node and Claude Code in `$HOME`, and write a `python`-style template whose body runs `claude -p "..." --allowedTools "Bash(sbatch:*)" "Bash(squeue:*)" Read Write` from `{{CODE}}`. Submit it on the `cpu` profile with a bounded `--time`.

## Known gaps / VERIFY

- Partition names, GPU type strings, time limits, `module` names: the example config reflects what one account saw on 2026-10-07. Confirm yours with `nuhpc check` and `module avail`.
- `/scratch/$USER` can exist but be root-owned and unwritable. Check before pointing `HF_HOME` or `remote_root` at it.
- Scratch purge policy: check RC's current rules. Treat `runs/` as temporary and `fetch` what you keep.
- Globus support is a thin wrapper over the `globus` CLI. It needs `globus login` once plus both endpoint UUIDs in the config.
- `status` uses `sacct`. If accounting lags, a brand-new run shows `SUBMITTED` for a few seconds.
- Jobs inherit the cluster's `http_proxy` (the compute nodes' route to the internet). Anything that talks to a server on the node itself must bypass it with `no_proxy`; `vllm_eval` sets this for 127.0.0.1.
- The ledger is local to each machine. Runs submitted from another laptop are visible in `queue`, but not in `runs` or `status`.
