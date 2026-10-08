# nuhpc

A small CLI for running GPU work on Northeastern's **Explorer** cluster (the successor of Discovery), from your laptop or from an agent (Claude Code, Codex).

It covers quick one-off tests, larger batch jobs that run models directly, and high-throughput vLLM serving. It is pure Python standard library (3.11+), and it relies on `ssh` and `rsync` on the local machine.

```
laptop / agent ──nuhpc──► ssh+rsync ──► login.explorer.northeastern.edu ──sbatch──► GPU nodes
                 ledger                    $remote_root/runs/<run_id>/{job.sbatch, sweep.jsonl, code/, outputs/, logs/}
```

## Which mode?

| mode | use it for | you write | overhead per job | measured on Explorer |
|---|---|---|---|---|
| **Local** (Mac, MPS) | prompt and stimulus debugging, tiny pilots | anything | none | 7B VLM: about 1 s/trial, 3 s under memory pressure; no CUDA |
| **Quick test**: `nuhpc run -- CMD` | "does this run on a GPU?", timing one batch, one-off scripts | any command | queue wait + ~1 min | — |
| **Direct batch job**: `python` template | full experiments that need log-probs, hidden states, forced prefixes or custom forward passes; sweeps over models and conditions | a script taking `--params --out` | queue wait + model load (minutes) | VLM Stroop: 9,556 trials of a 7B VLM in ≈ 15 GPU-min (0.09 s/trial, batch 24, transformers) |
| **vLLM at scale**: `vllm_eval --pack` | lots of generated text: evals, prompt and sampling sweeps | an OpenAI-API client | queue wait + **2–19 min server startup per model** | Qwen2.5-VL-3B: 2 image prompts in 0.75 s once up |
| **Training**: `train_ddp` | training on 1..N GPUs or nodes | a `torch.distributed` script | queue wait | not yet run |

How to choose:

- **Need anything from inside the forward pass?** That means full-vocabulary log-probs, hidden states, logit lens, or a forced answer prefix. Run the model **directly** (transformers in a `python` job). vLLM returns sampled text and only the top-k log-probs (20 by default).
- **Need many generated tokens across many prompts?** Use **vLLM**: continuous batching makes it much faster per token. It pays off only when there is enough generation per model to amortise the 2–19 minute startup, roughly 20 minutes or more. Always pass `--pack`, so the rows that share a model share one server.
- **Starting something new?** Begin with a quick `nuhpc run`, or a 1-row pilot, before any sweep. GPU queue waits range from minutes to an hour, so a failed sweep costs far more than a pilot.
- **Small and interactive?** Stay on the laptop. Explorer earns its keep once a job needs CUDA, more than about 30 GB of GPU memory, or more than an hour of compute.

## Quick start

```bash
uv tool install git+https://github.com/lordgrilo/nuhpc    # or, from a clone: uv tool install --editable .
nuhpc init                     # writes ~/.config/nuhpc/config.toml
```

`uv` fetches a Python ≥3.11 if your system one is older. With `--editable`, edits to the clone take effect immediately; otherwise upgrade with `uv tool upgrade nuhpc`. Tests run offline: `uv run --no-project --with pytest --with-editable . pytest`.

Edit the config: set `remote_root`, `env_setup`, and the profiles. `nuhpc/example_config.toml` holds the values that worked on 2026-10-07; lines marked `VERIFY` still need checking for your account.

### SSH (the part that matters for unattended use)

Install your key on the cluster once with `ssh-copy-id YOUR_NU_USERNAME@login.explorer.northeastern.edu`, which RC documents. Key logins need no Duo prompt, so `nuhpc` and agents can run unattended. For speed, or if your logins do ask for Duo, also add this to `~/.ssh/config`:

```
Host explorer login.explorer.northeastern.edu
    HostName login.explorer.northeastern.edu
    User YOUR_NU_USERNAME
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 12h
    ServerAliveInterval 60
```

With this block, `nuhpc connect` opens one authenticated session that every call reuses for 12 hours. All of nuhpc's own connections use `BatchMode=yes`, so an expired session fails at once with a hint, instead of hanging an agent at a password prompt. Calls that are safe to repeat (uploads, status, logs) are retried automatically after a dropped connection; `sbatch` never is.

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

`env_setup` then needs only `source /projects/YOUR_GROUP/envs/llm/bin/activate`. Put the environment on project storage rather than `$HOME`: it is 11 GB, and a lab can share one. Add what your own experiments need (e.g. `pip install` into the same venv, in a job).

### First run

```bash
nuhpc check                                                   # SSH ok? lists partitions, time limits, GPU types
nuhpc check --smoke --profile a100x1 --partition gpu-short    # a job testing GPU, a real CUDA kernel, imports, internet
nuhpc wait last && nuhpc logs last                            # read the smoke results
```

## Day-to-day on Explorer

```bash
nuhpc queue                                    # my jobs in Slurm, with the reason a job is still pending
nuhpc run --code . --profile a100x1 --partition gpu-short -- python probe.py    # quick test, blocks, prints output
nuhpc submit python --code . --sweep rows.jsonl --profile a100x1 --time 02:00:00   # batch job (one task per row)
nuhpc wait RUN                                 # block until done; survives laptop sleep; shows failed logs
nuhpc logs RUN --task 3 --grep 'Traceback|Error|s/trial'                      # progress or errors only
nuhpc fetch RUN --include '*.json' --include '*.jsonl'                        # results to ~/nuhpc-results/RUN
nuhpc push ./stimuli stimuli                   # inputs to $remote_root/data/stimuli
nuhpc cancel RUN                               # scancel
```

RUN can be a full `run_id`, a unique prefix, a Slurm job id, or `last`.

**Where things live.** On the cluster, under `remote_root`:

- `runs/<run_id>/`: `job.sbatch`, `sweep.jsonl`, `packs.json`, `code/` (the snapshot), `outputs/task_K/` (`row_R/` when packed), and `logs/`.
- `data/`: whatever you `push`.

The environment and the HF cache sit wherever your `env_setup` points; ours are `envs/llm` and `hf_cache`. Locally: the config is `~/.config/nuhpc/config.toml`, the ledger and staged jobs are in `~/.local/share/nuhpc/`, and fetched results go to `~/nuhpc-results/<run_id>/`.

**Partitions seen on 2026-10-07.**

| partition | time limit | what for |
|---|---|---|
| `gpu` | 8 h | A100 (40 and 80 GB), H200, T4 and V100 nodes |
| `gpu-short` | 2 h | the same nodes; pilots and quick tests often start sooner here |
| `gpu-interactive` | 2 h | interactive sessions (`srun --pty`), outside nuhpc |
| `short` | 2 days | CPU (the default) |
| `sharing` | 1 h | mixed GPUs |

`multigpu` is invisible until RC grants access.

**Why a job is pending** (`nuhpc queue` shows the reason):

| reason | meaning | what to do |
|---|---|---|
| `Priority` | other jobs are ahead of you | wait |
| `Resources` | no free GPU of that type right now | wait, or ask for another type |
| `Nodes required for job are DOWN, DRAINED or reserved…` | that partition's nodes are taken by a higher-priority partition | resubmit on the other partition (`gpu` ↔ `gpu-short`) |
| `QOSMaxGRESPerUser`, `AssocGrpGRES…` | you've hit a per-user GPU cap | wait for your own jobs, or use fewer GPUs |

**Costs.** Every submit reports `worst_case_gpu_hours` (GPUs × nodes × walltime × tasks). Anything above the limits in your config needs `--confirm-big`. Keep `--time` near the real runtime: shorter jobs schedule sooner.

## Workflows

### Quick tests: `nuhpc run`

```bash
nuhpc run --code . --profile a100x1 --partition gpu-short -- python probe.py --n 8
```

This submits the command on a compute node with a 30-minute limit unless you pass `--time`, waits, prints its output, and fetches anything it wrote to `$HPC_OUT`. No `--params/--out` contract is needed. It exits 0 if the command succeeded, 2 if it failed (the log is printed), and 3 if `--timeout` passed first; the job keeps running, so `nuhpc wait` it. It is the `cmd` template underneath: `nuhpc submit cmd -p cmd="..."` does the same without blocking.

### Direct batch jobs: `python` template

Your script loads the model itself (transformers, or plain torch) and follows one contract: it takes `--params <params.json> --out <dir>` and writes what it keeps into `--out`. Make it **resumable**, skipping rows already in the output file, so a timeout or requeue costs nothing. Put one model in each task and loop over all conditions inside it; spread models across tasks with `--sweep`.

```bash
# rows.jsonl: one line per model, e.g. {"entry": "run_vlm.py", "model": "Qwen/Qwen2.5-VL-7B-Instruct", "batch": 24}
nuhpc submit hf_download --profile cpu -p repo=Qwen/Qwen2.5-VL-7B-Instruct        # weights first (jobs run offline)
nuhpc submit python --name stroop --code . --sweep rows.jsonl --profile a100x1 --time 02:00:00
nuhpc wait stroop && nuhpc fetch stroop --include '*.jsonl' --include '*.json'
```

Worked example: `miller-science/code/stroop_vlm/run_vlm.py` (VLM Stroop: next-token log-probs with an fp32 head, forced answer prefixes, and resume). The cluster has no macOS fonts, so it pushes Arial Bold to `data/fonts/` and passes the path as a param.

### vLLM at scale: `vllm_eval --pack`

```bash
nuhpc push ./my-eval my-eval                   # prompts.jsonl ({"id", "prompt", "image"} per line) + images
nuhpc submit vllm_eval --name vlm-eval --pack --code examples --profile a100x1 --time 01:00:00 \
  -p entry=vlm_client.py -p prompts=/projects/YOUR_GROUP/nuhpc/data/my-eval/prompts.jsonl \
  --grid model=Qwen/Qwen2.5-VL-3B-Instruct,Qwen/Qwen2.5-VL-7B-Instruct --grid temperature=0,0.7
nuhpc wait vlm-eval && nuhpc fetch vlm-eval --include '*.jsonl' --include '*.json'
```

The template starts an OpenAI-compatible vLLM server on the node, waits until it answers (allowing up to 45 minutes), runs your client against `$OPENAI_BASE_URL`, and shuts down. Tensor-parallel size equals the GPUs per node.

With `--pack`, rows that share a model (and its `vllm_args`) run in one task against one server, so the grid above is 2 tasks, not 4. Results land in `outputs/task_K/row_R/`. Size `--time` for all rows of a pack: up to 20 minutes of startup plus the client's work.

`examples/vlm_client.py` sends images: local files go inline as base64, and URLs pass through. `examples/eval_client.py` is the text-only version. Both resume. Verified on 2026-10-08 with Qwen2.5-VL-3B on an A100, including `--pack`: one server (up in 100 s) served a temperature 0 row and a temperature 0.7 row.

Model size drives queue time more than anything else: one A100-80GB or H200 job usually starts before a four-GPU job. Keep `--max-parallel` modest on the shared queue.

### Training: `train_ddp`

`train_ddp` handles single-node multi-GPU and multi-node runs (`--nodes 2 --gpus 4`). Your script initialises with `torch.distributed.init_process_group("nccl")`, and only rank 0 writes to `--out`. Not yet exercised on Explorer. Things to add when you get there:

- **Checkpoint and resume.** Write checkpoints to `$HPC_OUT/ckpt`, and add `#SBATCH --requeue` plus a resume-from-latest path. Wall-clock limits will cut long runs.
- **Fetch selectively.** `fetch` skips files above `fetch_max_file_size`, so checkpoints don't fill your laptop. Pass `--max-size` when you want them.
- **Experiment tracking.** For W&B, use offline mode and sync afterwards.

## Command reference

| command | what it does |
|---|---|
| `nuhpc templates` | list job templates and their parameters |
| `nuhpc run [opts] -- CMD...` | quick test: submit CMD, wait, print its output, fetch `$HPC_OUT` (exit 2 if it failed, 3 on timeout) |
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
| `nuhpc check [--smoke]` / `nuhpc connect` | verify SSH and list partitions / open the persistent SSH session (interactive) |

Global flags: `--json` gives machine-readable output, including errors as `{"error": ...}`. `--dry-run` changes nothing on the cluster: it prints every remote command and leaves the rendered `job.sbatch` in `~/.local/share/nuhpc/runs/<run_id>/` for inspection.

### Submit options

```
--code DIR            snapshot this project into the run (excludes .git, venvs, data/, weights; add .nuhpcignore)
-p key=value          parameter; values parse as JSON (so 0.1, true, [1,2] are typed)
--sweep FILE          .jsonl / .json list / .csv, one task per row (merged over -p defaults)
--grid key=a,b,c      cartesian product, repeatable; combines with --sweep
--pack                one array task per group of rows sharing the template's `# pack-by:` keys
                      (vllm_eval: one server per model; the client runs once per row)
--max-parallel N      array throttle
--profile NAME        resource preset; override with --partition --time --gpus --mem --cpus --nodes --gres
--confirm-big         bypass limits (humans only)
```

## Templates

| template | use |
|---|---|
| `smoke` | environment check (GPU, a real CUDA kernel, imports, internet, disk); run first and after any env change |
| `cmd` | any shell command, no contract (`nuhpc run` wraps it) |
| `hf_download` | cache an HF model or dataset into `HF_HOME` on cluster storage |
| `python` | your script: `python <entry> --params --out [args]` |
| `vllm_eval` | vLLM OpenAI-compatible server + your client; supports `--pack` |
| `train_ddp` | `srun torchrun` across 1..N nodes with c10d rendezvous |

In a template, `{{CODE}} {{RUN_DIR}} {{DATA}} {{REMOTE_ROOT}} {{NODES}} {{GPUS_PER_NODE}} {{GPUS}} {{CPUS}}` are filled at render time. Per-task values come from `hpc_param KEY [DEFAULT]` at run time.

Read parameters into variables first, as in `X="$(hpc_param key)"`. Under `set -e`, a missing parameter aborts the job only inside an assignment, not when the substitution is inline in a command.

To add your own templates, drop `*.sbatch` files into `templates_dir`. Lines starting with `# doc:` show up in `nuhpc templates`, which is also how an agent learns what a template expects. A template supports `--pack` by declaring `# pack-by: key ...` (the params that must be equal within one task) and running its per-row work through `hpc_each_row CMD`, which sets `HPC_ROW_PARAMS` and `HPC_ROW_OUT` for each row.

## Agents

### Mode 1: agent outside, cluster runs plain batch (recommended)

1. Link `skill/nuhpc/` into both agents' user skill folders. It tells the agent how to use the CLI and what it must not do. Claude Code reads `~/.claude/skills/`; Codex (the ChatGPT-account agent) reads `~/.agents/skills/`. One copy serves both, so updating the repo updates both:
   ```bash
   mkdir -p ~/.agents/skills ~/.claude/skills
   ln -s "$PWD/skill/nuhpc" ~/.agents/skills/nuhpc
   ln -s ~/.agents/skills/nuhpc ~/.claude/skills/nuhpc
   ```
   Both were checked on 2026-10-08. A Claude Code session ran the VLM Stroop experiments through the skill, and Codex 0.154 found the skill unprompted, dry-ran a packed sweep and used `wait` as instructed.
2. Restrict the tools. For Codex: its default sandbox has no network, so it asks before running `nuhpc` outside the sandbox. Answering "always" records `prefix_rule(pattern=["nuhpc"], decision="allow")` in `~/.codex/rules/default.rules`. For Claude Code, use the project's `.claude/settings.json` or flags:
   ```json
   { "permissions": {
       "allow": ["Bash(nuhpc:*)"],
       "deny":  ["Bash(ssh:*)", "Bash(scp:*)", "Bash(rsync:*)", "Bash(sftp:*)"] } }
   ```
   Honest caveat: prefix rules can't block `--confirm-big` appearing in the middle of a command. The limits are a guardrail backed by the skill's instructions, not a security boundary. For a hard boundary, give the agent a config with stricter limits and remove the flag from your installed copy.
3. **Delayed or unattended runs.** Schedule a headless run on a machine that stays on, since a sleeping laptop pauses everything:
   ```bash
   echo 'cd ~/proj && claude -p "Execute the plan in PLAN.md with nuhpc; use nuhpc wait, fetch results, write REPORT.md." --allowedTools "Bash(nuhpc:*)" Read Write Edit' | at 02:00
   ```
   With Codex, `codex exec "..."` plays the same role. The SSH key, or a live ControlMaster session, must work at that time. If it doesn't, the run fails immediately and cleanly; it does not hang.

### Mode 2: agent on the cluster

Only reach for this when the decide-inspect-resubmit loop has to sit next to data too large to move. Before doing it:

- **No agents on login nodes.** RC's policy says jobs on login nodes get terminated, and a watchdog kills heavy processes there. The agent has to run inside an allocation, and you pay for idle compute while it thinks. Use a cheap CPU allocation that submits GPU jobs rather than an agent sitting on a GPU.
- **It needs outbound HTTPS** to the API from the node it runs on. Compute nodes have it, through the cluster proxy.
- **Protect the API key.** Keep it in a `chmod 600` file under `$HOME` and never put it in the run snapshot. Its scope is your whole account, so prefer a separate, spend-limited key.
- It's worth a short email to rchelp@northeastern.edu first. Long-lived agent processes are a gray area under shared-cluster policies.

Recipe, if you go ahead: install Node and Claude Code in `$HOME`, and write a `python`-style template whose body runs `claude -p "..." --allowedTools "Bash(sbatch:*)" "Bash(squeue:*)" Read Write` from `{{CODE}}`. Submit it on the `cpu` profile with a bounded `--time`.

## Troubleshooting on Explorer

Every row below happened while setting this up (2026-10-07/08).

| symptom | cause | fix |
|---|---|---|
| `Permission denied (publickey)` | your key isn't in the cluster's `authorized_keys` | `ssh-copy-id` once (see Quick start) |
| `conda … Killed` on the login node | RC's watchdog kills heavy processes there | do installs inside a job |
| torch says `cuda available: True`, then "NVIDIA driver … too old" | the wheel's CUDA is newer than the driver (570.86, CUDA ≤12.8) | CUDA 12.9 builds (see Python environment); trust the smoke test's `cuda kernel: ok` |
| vLLM log says "Application startup complete", but the job reports "not healthy" | jobs inherit `http_proxy`, so requests to `127.0.0.1` went to the proxy | `vllm_eval` sets `no_proxy`; do the same for any server you start yourself |
| the first vLLM start takes 15+ min, later ones are faster | first imports write each `.pyc` to `/projects` (~0.3 s per file) | build with `UV_COMPILE_BYTECODE=1`, or run `python -m compileall -j 16` on the env in a job |
| vLLM startup takes 2–19 min | weights (7 GB at ~30 MB/s on a cold node) and imports come over network storage | normal: use `--pack`, and give `--time` room |
| `rsync: unexpected end of file`, `ssh … exit 255` | a dropped or throttled connection | repeatable calls retry automatically; if a submit failed before `sbatch`, just submit again |
| `wait` prints transport errors, then carries on | laptop asleep, or network changed | normal: it gives up only after 10 failures in a row |
| a job pending for a long time | see "Why a job is pending" | — |
| `status` says `SUBMITTED` | `sacct` hasn't caught up yet | wait a few seconds |

## Design notes

- **Verbs, not a shell.** An agent can only call `run / submit / wait / status / logs / fetch / cancel / push / pull / ls`. There is no `exec` on the login node, and remote paths are confined to `remote_root`. Code you upload does run on compute nodes; that is the point, and it stays inside Slurm's accounting.
- **Every job is a run.** Each run is a frozen snapshot of the rendered sbatch, the parameters, and the code. That makes runs reproducible and easy to inspect: everything lives in `runs/<run_id>/`.
- **One contract for batch code.** Your script takes `--params <params.json> --out <dir>` and writes what it wants kept into `--out`. Sweeps, evals and training all use this shape. Quick tests (`run`/`cmd`) skip it.
- **Sweeps are job arrays.** There is one task per row, or per pack, throttled with `%max_parallel`. This is how you run models or configurations in parallel without monopolising the GPU queue.
- **Guardrails for agents.** Every submit computes its worst-case GPU-hours and task count. Anything above the limits in your config needs `--confirm-big`, and the agent instructions say that flag requires a human.

## Known gaps

- Partition names, GPU type strings, time limits and `module` names are what one account saw on 2026-10-07. Confirm yours with `nuhpc check` and `module avail`.
- `/scratch/$USER` can exist but be root-owned and unwritable. Check before pointing `HF_HOME` or `remote_root` at it. Check RC's scratch purge policy before relying on scratch at all.
- Globus support is a thin wrapper over the `globus` CLI. It needs `globus login` once plus both endpoint UUIDs in the config.
- The ledger is local to each machine. Runs submitted from another laptop are visible in `queue`, but not in `runs`, `status` or `wait`.
- `pull` of a directory nests it one level deeper (`pulled/x/x`), because of rsync's trailing-slash rule.
