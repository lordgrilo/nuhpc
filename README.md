# nuhpc

A small CLI for running GPU work on Northeastern's **Explorer** cluster (the successor of Discovery), from your laptop or from an agent (Claude Code, Codex).

It covers quick one-off tests, larger batch jobs that run models directly, and high-throughput vLLM serving. It is pure Python standard library (3.11+), and it relies on `ssh` and `rsync` on the local machine. Underneath it is plain Slurm; see "On another Slurm cluster" for what is Explorer's.

```
laptop / agent ──nuhpc──► ssh+rsync ──► login.explorer.northeastern.edu ──sbatch──► GPU nodes
                 ledger                    <work root>/runs/<run_id>/{job.sbatch, sweep.jsonl, code/, outputs/, logs/, usage/}
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

Each person sets nuhpc up once, with their own cluster account, config and Python environment. It takes about half an hour, most of it the environment building on the cluster.

```bash
uv tool install git+https://github.com/lordgrilo/nuhpc    # or, from a clone: uv tool install --editable .
nuhpc init                     # writes ~/.config/nuhpc/config.toml
```

`uv` fetches a Python ≥3.11 if your system one is older. With `--editable`, edits to the clone take effect immediately; otherwise upgrade with `uv tool upgrade nuhpc`. Tests run offline: `uv run --no-project --with pytest --with-editable . pytest`.

Then, in order:

1. **Config.** In `~/.config/nuhpc/config.toml`, set `user_dir` to your own folder on project storage, such as `/projects/<group>/<you>` (`nuhpc init --user-dir` writes it for you), and `ssh_host` (next section). Environments and the model cache go there; runs and data go to a `nuhpc/` folder inside it, or inside a project directory you declare (see Where things live). The rest works as written: `env_setup` activates the environment that step 3 builds, and the profiles and limits are values that worked on 2026-10-07 (lines marked `VERIFY` are unchecked for other accounts).
2. **SSH**, below; `nuhpc check` confirms it.
3. **Python environment**, below: one `env_build` job.
4. **First run**, below: the smoke test.
5. **Agents**, optionally: link the skill (see Agents).

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

Each person builds their own environment, once, in a job:

```bash
nuhpc submit env_build --profile cpu      # about 20 minutes; builds <user_dir>/envs/llm (11 GB)
nuhpc wait last && nuhpc logs last        # ends with the installed versions
```

It installs `uv`, Python 3.12, torch and vLLM 0.31 as CUDA 12.9 builds, plus `huggingface_hub` and `openai`, into `<user_dir>/envs/llm`. The example config's `env_setup` activates exactly that path: inside every job, `$HPC_USER_DIR` is your `user_dir`. Environments live there, not in a project directory, so all your projects share them. Params change what it builds (`nuhpc templates` lists them):

- `-p extra="huggingface_hub openai pandas"` sets the extra packages; keep the first two if you want them.
- `-p vllm=none` skips vLLM, and `-p env=envs/other` builds a second environment beside the first.
- On an existing environment it installs into it, which is how you add packages later. To start clean, build at a new path: deleting an environment is slow (see Cleaning up).

The environment is per person on purpose: whoever builds it owns it, so nobody's running experiment changes when someone else adds a package.

Why a job, and why built this way (all found on Explorer, 2026-10-07):

- Explorer kills conda on login nodes, even `conda --version`. Compute nodes have outbound internet.
- Project storage costs about 0.1 s per file written, so a serial `conda` or `pip` install of vLLM and torch runs for hours. `uv` installs in parallel and finished in 19 minutes. Its cache stays on the node's `/tmp`, files are copied rather than hardlinked across filesystems, and bytecode is compiled during the build; without that, first imports wrote each `.pyc` to `/projects` one by one and stalled vLLM for 17 minutes.
- The interpreter goes to `envs/.python` on project storage, so every node can run it.
- `uv` itself comes from PyPI through the shared Anaconda Python, which has pip; the system `python3` has none. `-p bootstrap_python=` names another.

**Match CUDA to the driver.** A plain `pip install vllm` gets CUDA 13.0 wheels. Explorer's GPU driver (570.86 on 2026-10-07, CUDA ≤12.8) is too old for them: torch still reports `cuda available: True`, but the first kernel fails with "driver too old". CUDA 12.9 builds run on it, because NVIDIA keeps runtimes compatible within 12.x. Check `nvidia-smi` in the smoke log after any driver or version change; the smoke template launches a real kernel for this reason.

### First run

```bash
nuhpc check                                                   # SSH ok? lists partitions, time limits, GPU types
nuhpc check --smoke --profile a100x1 --partition gpu-short    # a job testing GPU, a real CUDA kernel, imports, internet
nuhpc wait last && nuhpc logs last                            # read the smoke results
```

The smoke log should show `cuda kernel: ok` and the torch and vLLM versions that `env_build` printed.

## Day-to-day on Explorer

```bash
nuhpc queue                                    # my jobs in Slurm, with the reason a job is still pending
nuhpc run --code . --profile a100x1 --partition gpu-short -- python probe.py    # quick test, blocks, prints output
nuhpc submit python --code . --sweep rows.jsonl --profile a100x1 --time 02:00:00 --project miller   # batch job, one task per row
nuhpc wait RUN                                 # block until done; survives laptop sleep; shows failed logs
nuhpc logs RUN --task 3 --grep 'Traceback|Error|s/trial'                      # progress or errors only
nuhpc fetch RUN --include '*.json' --include '*.jsonl'                        # results to ~/nuhpc-results/RUN
nuhpc usage --since 7d                         # compute used vs requested per run, totals per project
nuhpc push ./stimuli stimuli                   # inputs to <work root>/data/stimuli (add --project NAME for a project's)
nuhpc cancel RUN                               # scancel
```

RUN can be a full `run_id`, a unique prefix, a Slurm job id, or `last`.

**Where things live.** On the cluster, installs and work are kept apart:

```
<user_dir>/                      your folder, e.g. /projects/<group>/<you>
├── envs/                        environments (llm/, .python/): shared by all your projects
├── hf_cache/                    model weights
├── nuhpc/                       default work root
│   ├── runs/<run_id>/           job.sbatch, sweep.jsonl, packs.json, code/ (the snapshot),
│   │                            outputs/task_K/ (row_R/ when packed), logs/, usage/ (GPU samples)
│   └── data/                    whatever you `push`
└── <project dir>/nuhpc/         the same runs/ and data/, for a declared project
```

A project directory is declared in the config and chosen with `--project NAME` on `submit`, `run`, `push`, `pull`, `ls` and `check`:

```toml
[projects.miller]
dir = "/projects/<group>/<you>/miller"       # may be anywhere you can write
```

`nuhpc submit python --project miller ...` then keeps that run in `<dir>/nuhpc/runs/`, and `nuhpc push ./stimuli --project miller` puts the data in `<dir>/nuhpc/data/`. Inside the job, `$HPC_REMOTE_ROOT` is that work root. Everything nuhpc writes sits in a `nuhpc/` folder, so it never mixes with your own project files. A project name with no entry in the config is only a tag for `nuhpc usage`, and the run uses the default location; the `remote_dir` in the submit output shows where it went. Only directories listed in the config can be used, so an agent cannot write anywhere else. `nuhpc check [--project NAME]` verifies `user_dir` and creates the work root.

Older configs that set only `remote_root` keep working unchanged: it is the work root, and `envs/` and `hf_cache/` sit beside `runs/` in it. If a config sets both, `user_dir` holds the installs and `remote_root` replaces `<user_dir>/nuhpc` as the default work root.

Locally: the config is `~/.config/nuhpc/config.toml`, the ledger, staged jobs and usage records are in `~/.local/share/nuhpc/`, and fetched results go to `~/nuhpc-results/<run_id>/`. Usage records are also copied to `~/.nuhpc/usage/` in your cluster login home (not `user_dir`; see Usage).

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
| `QOSMaxJobsPerUserLimit` | you already have the maximum number of jobs running in that partition (every array task counts) | wait, or [pack rows into fewer jobs](#sizing-jobs-on-explorer); submitting more only lengthens the queue |
| `JobArrayTaskLimit` | your own `%N` throttle on that array (`--max-parallel`) | wait, or raise the throttle if the caps below allow it |

**Costs.** Every submit reports `worst_case_gpu_hours` (GPUs × nodes × walltime × tasks). Anything above the limits in your config needs `--confirm-big`. Keep `--time` near the real runtime: shorter jobs schedule sooner.

**Usage.** Each run gets a usage record, `<run_id>.json`, the first time `wait`, `run`, `fetch` or `usage` sees it finished. It holds what was requested (from `meta.json`) next to what Slurm accounted, per task: state and exit code, node, the GPU type of that node (Explorer's accounting counts GPUs without naming them, so the type comes from `sinfo`), queue wait, elapsed time, GPU-hours and CPU-hours (allocated × elapsed), CPU and memory efficiency, and time used as a share of `--time`. GPU jobs also sample `nvidia-smi` every 30 s into `runs/<run_id>/usage/task_K_gpu.csv`, which adds mean and peak GPU utilisation and peak GPU memory (first node only on multi-node runs). A record is written once, since Slurm's accounting of a finished job never changes, and kept in three places:

- `~/.local/share/nuhpc/usage/`: every run submitted from this machine; what `nuhpc usage` reads.
- `~/nuhpc-results/<run_id>/usage.json`: beside the results, once fetched (the CSVs come along in `usage/`).
- `~/.nuhpc/usage/` in your cluster login home: outside every work root, so cleaning `runs/` keeps it, and shared by every machine that submits as you.

`nuhpc usage [RUN...] [--since 7d|all] [--project P]` prints one row per finished run and totals per project, with the share of what was allocated that went unused: `cpu_idle_%` (core-hours that sat idle, on CPU-only tasks, since a GPU job's cores are incidental), `gpu_idle_%` (sampled GPU-hours that sat idle) and `failed_%` (billed hours spent on tasks that did not complete, a GPU-hour counted as about 100 CPU-hours, which is an estimate). All are weighted by hours; the first call also records any older run that lacks a record (the GPU columns stay empty for runs from before the sampler). `mem_%` is Slurm's peak memory over the request, and can pass 100 % without an out-of-memory kill: a model download on Explorer showed 168 %. Tag runs with `submit --project P` (or `run --project P`) to make the per-project totals mean something. If the cluster can't be reached when a run is first seen finished, the command warns `usage not recorded` and carries on; the next `wait`, `fetch` or `usage` records the run from Slurm's accounting, so nothing is lost.

**Cleaning up.** Run folders stay in `runs/` until you delete them: a few KB of logs for a test, MB of outputs for an experiment. `fetch` what you keep, then delete old runs yourself:

```bash
ssh YOUR_NU_USERNAME@login.explorer.northeastern.edu 'cd <work root>/runs && rm -rf -- <run_id> <run_id>'
```

There is deliberately no `nuhpc` delete command, so an agent allowed to run `nuhpc` cannot destroy results. Deleting a run folder leaves alone its ledger entry (history), its usage records, its fetched copy in `~/nuhpc-results/`, and any weights it cached in `HF_HOME`; `logs` and `fetch` on that run fail afterwards. The local staging copies in `~/.local/share/nuhpc/runs/` can go at any time. To delete something with thousands of files, such as an old Python environment, do it inside a job with parallel deletes (`find DIR -type f -print0 | xargs -0 -P 16 rm -f`): file operations on `/projects` cost about 0.1 s each. Environments built in the same folder share its `.python` interpreter store (`envs/.python` beside `envs/llm`), so deleting an environment means deleting its own folder only, never `.python`, which the others run from. The `home =` line of each environment's `pyvenv.cfg` names the interpreter it uses; that one can go too once no other environment's `pyvenv.cfg` names it.

## Sizing jobs on Explorer

Most waits and failures on Explorer come from how a submit is shaped, not from the cluster being full. Limits, as the scheduler reported them on 2026-10-15 (`sacctmgr show qos`; per user, per partition) next to what Research Computing [publishes](https://rc.northeastern.edu/partitions/):

| partition | running jobs | queued jobs | GPUs | time (default / max) |
|---|---|---|---|---|
| `short` (CPU) | 50 | 1000 | – | 4 h / 48 h; 1024 cores, 25 TB RAM in all |
| `gpu-short` | 2 | 8 (docs: 4) | 2 (1 per job in the docs) | 1 h / 2 h |
| `gpu` | 4 | 8 | 4, one per job | 4 h / 8 h |
| `sharing` | 2 | 4 | 2 | 1 h |

**Every array task counts as a job**, in both the running and the queued column. So 48 one-CPU tasks hold 48 of your 50 `short` slots while using about 5% of your core allowance, and everything else you submit waits behind them. A job also waits for a node that fits it: most `short` nodes have 20 cores and about 62 GB (roughly 3 GB per core), and only about a dozen have more (28 cores with 256 GB, 48 with 126 GB, one with 128 and 512 GB), so a job that asks for more than a typical node can only run on those. And the scheduler's biggest priority term is fair share, computed from the recent use of your Slurm account (half-life 7 days). A PI's account is usually shared by the whole lab, and every member of it gets the same fair-share factor, so a heavy burst by anyone in the account slows what everyone in it submits next. Billing weighs a GPU-hour at roughly 100 CPU-hours, so 11,000 CPU-hours in a day count like about 110 GPU-hours: fair share follows the total billed use, whichever kind it is, and packing rows fixes the running-job cap without changing it (the same hours are used). Only using less moves it. `nuhpc standing` shows where you are.

Rules that follow:

- **Pack small CPU rows into one job** (next section) instead of an array of one-CPU tasks. Size a job to stay within a typical node: about 20 CPUs and 60 GB.
- **Time is a cap, not a hint.** A task that reaches `--time` is killed and its work is lost, so request about 1.5 times the slowest task you have measured (`nuhpc usage` after a pilot), and write results as you go. Asking for far more only delays the start, and fair share is charged on use.
- **Memory:** size it from the pilot's peak (`mem_%` in `nuhpc usage`) plus about 30%. Memory over 100% in `nuhpc usage` is Slurm's accounting; a task is only out of memory when it ends `OUT_OF_MEMORY`.
- **GPUs:** `gpu-short` runs 2 jobs per user and `gpu` 4, so don't launch GPU pilots in parallel. Put all conditions for one model into one job, and use `gpu` for batches.
- **Idle GPUs are reclaimed.** Research Computing cancels GPU jobs whose compute utilisation stays under 10% for an hour; `sacct` shows `CANCELLED by 0`.
- **Home** has a 75 GB quota; keep environments and caches on project storage (`nuhpc` builds them there).

nuhpc enforces the first rule. A submit of more than `limits.small_task_max` (default 16) array tasks of at most `small_task_cpus` (2) CPUs and no GPU is refused unless it is packed or given `--confirm-big`, and from `small_task_warn` (8) tasks it warns. `limits.big_job_cpus` and `big_job_mem` (20 and "60G" in the example config) make it warn when one CPU job is bigger than a typical CPU node; GPU jobs are not judged by them, as GPU nodes are much bigger. Warnings are printed to stderr and listed under `warnings` in the JSON result.

### Where you stand: `nuhpc standing`

One read-only call to the login node (no job, about 15 seconds) that reports your queues, how busy the whole cluster is, and your account's fair share, with a verdict about you. On a terminal it is coloured (GOOD green, OK yellow, POOR red) with gauges, in the look of `npl`:

```
Explorer standing · cluster time 2026-10-16 10:00

  [OK]  fair share ok 0.31 · queue good
  me.pi · account lab.pi · 4 users

──────────────────────────────── Queue ─────────────────────────────────
  gpu              ░░░░░░░░  0 of 4 running   0 pending · 8 h max
  gpu-interactive  ░░░░░░░░  0 of 2 running   0 pending · 2 h max
  gpu-short        ░░░░░░░░  0 of 2 running   0 pending · 2 h max
  sharing          ░░░░░░░░  0 of 2 running   0 pending · 1 h max
  short            █░░░░░░░  6 of 50 running  0 pending · 2 d max
  status           GOOD · no cap pressure

─────────────────────────────── Cluster ────────────────────────────────
  gpu              ████░░░░  52% busy  GOOD    2 waiting
                   median wait 4 min · slowest tenth 5 min
                   oldest waiting 2 h
  gpu-interactive  ████░░░░  52% busy  GOOD    0 waiting
                   no job started in the last 6 h
  gpu-short        ████░░░░  52% busy  unknown 2 waiting
                   only 3 jobs started in the last 6 h
                   oldest waiting 1 h
  sharing          ████░░░░  55% busy  GOOD    0 waiting
                   no job started in the last 6 h
  short            ███████░  88% busy  OK      7 waiting · 4 held
                   median wait 1.5 h · slowest tenth 5 h
                   oldest waiting 12 h
  free GPUs        gpu · gpu-interactive · gpu-short
                   a100 0 of 7 · h200 2 of 8 · t4 4 of 4
  free GPUs        sharing
                   l40s 2 of 3
  status           OK · short median wait 1.5 h

────────────────────────────── Fair share ──────────────────────────────
  factor     ██████░░░░░░░░░░░░░░ 0.312   good ≥ 0.5 · ok ≥ 0.25
  usage      3.8x its equal share · 2 of 5 accounts use more
  recovery   about 13 days of no use to reach 1x (half-life 7 d)
  account    you █████░░░░░░░░░░░░░░░ 23% · the other 3 users 77%
  priority   fair share gives a new job 6,240 of 20,000 points

──────────────────────────────── Advice ────────────────────────────────
  - Fair share is OK (0.31) and jobs are billed on what they allocate:
    size CPUs and memory from a pilot's measured use, set --time near
    the measured runtime, and pilot before any large submit.
  - No free a100 GPU on gpu, gpu-interactive, gpu-short: a job asking
    for it will wait. Free now: t4 4, h200 2.

  --users names the others · --brief one line · --json for scripts
```

- **Queue** lists every partition you can use, idle ones too, with your running jobs against its live cap (read from the scheduler, not hard-coded) and its time limit. `POOR` means a cap is holding jobs back (`QOSMax...` reasons; your own `--max-parallel` throttle does not count), `OK` means you are near a cap (80%) or jobs have waited over 6 h, `GOOD` means neither.
- **Cluster** shows how loaded each queue is for everyone, not only you: the share of usable cores that are busy, the jobs waiting on the scheduler (as opposed to held by a dependency, a hold, a cap or a start time), the median and slowest-tenth wait of the jobs that started in the last 6 hours (counted from when each became eligible), the oldest job still waiting, and the free GPUs by type. A queue is `GOOD` under 30 minutes of median wait, `OK` under 2 hours and `POOR` beyond, and `unknown` when fewer than 5 jobs started. It never changes the overall verdict, since you cannot fix it; it changes the advice. Other people's jobs are counted, never named, and if the scheduler hides them (only your own jobs show) the section says so instead of guessing. The waits cover only jobs that have started, so they understate the worst case; the oldest-waiting figure is there to correct for that, and array tasks held back by their own `%N` limit count as waiting.
- **Fair share** is the account's, so it is the lab's. `GOOD` is a factor of 0.5 or more, `OK` 0.25 or more, `POOR` below that. The report shows how your account's use compares with its equal share, its rank among all accounts, and how the account's use splits between you and the rest (names only with `--users`).
- The overall verdict is the worse of queue and fair share. A `POOR` fair share costs nothing while the cluster has free resources; it shows as longer waits for scarce GPUs.
- **Advice** appears when something should change: a cap holding jobs back (pack rows), a `POOR` or `OK` fair share (right-size, pilot, set `--time` from measurements), a busy queue (ask for a shorter `--time`; do not cancel and resubmit) and GPU types with no free card. In the JSON each item is `{code, kind, text}`; `kind` is `plan` when how you use the cluster has to change and `note` when it only affects what to ask for.
- `--brief` prints just the verdict line; `--json` has everything, including a short `brief` string for status lines. The cut-offs are in an optional `[standing]` table of the config: `wait_hours`, `near_cap`, `fairshare_good`, `fairshare_low`, `wait_window_h`, `cluster_wait_ok_min`, `cluster_wait_poor_min`. Piped, redirected or with `NO_COLOR` set it is plain text, `FORCE_COLOR=1` colours it anyway (for a pager), and a non-UTF-8 locale gets an ASCII form.

**Improving fair share.** It is computed from what your account's jobs *allocate*, in billed CPU-hours (a GPU-hour counts about 100), and the usage halves every 7 days. So:

- **Right-size the CPUs and GPUs you ask for.** Cores that sit idle are billed like busy ones: `nuhpc usage` shows `cpu_idle_%` per run and per project (and `cpu_eff` per task in the record), and a job that asks for 16 CPUs and uses 30% of them is billed three times what it needed. Pilot with fewer cores and compare the runtime.
- **Don't pay twice for the same result.** Failed, timed-out and cancelled tasks are billed for the hours they ran: pilot first, set `--time` at about 1.5 times the slowest measured task, and write results as you go.
- **Let it decay, and spread bursts.** A burst keeps the account above its share for weeks; nothing you do afterwards removes it earlier, but not repeating it lets it halve each week. Packing rows and moving jobs to quiet hours do not change the bill (they do avoid waiting).
- **It is shared.** Everyone in the account has the same factor, so a heavy member sets it for the lab; `nuhpc standing --users` shows who. Whether a larger share or a group allocation is available is a question for Research Computing.

### Where your space goes: `nuhpc disk`

An explorer for the space your folders use, in the spirit of `mo analyze`: one bar per folder, sorted by size, with its share of the folder above it, its size and how long ago it last changed. Pastel colours cycle down the list and fade from mint (changed recently) to rose (untouched for over a year).

```bash
nuhpc disk --scan          # measure (about a minute, no job), then open the explorer
nuhpc disk                 # open the cached scan again, instantly
nuhpc disk hf_cache/hub    # start inside a folder
nuhpc disk --scan --job    # the full scan as a short compute-node job: adds file counts and the largest files
```

```
 nuhpc disk · /projects/lab/you
 scanned 2 min ago · ≥ 682 G · quick scan · partial (1 folder not fully read)
 you

❯  1  ████████████▎░   87.9%  models/                                           600 G   11 d
   2  █▋░░░░░░░░░░░░   11.7%  runs/                                              80 G  today
   3  ▏░░░░░░░░░░░░░    0.3%  envs/                                             ≥ 2 G    2 d
   4  ▏░░░░░░░░░░░░░    0.1%  notes/                                            400 M    2 y
   5  ▏░░░░░░░░░░░░░    0.0%  (files in this folder)                              4 K

 ↑↓ move · ⏎ open · ⌫ back · s sort: size · l largest · q quit                           1/5
```

Keys: up and down (or `j` and `k`) move, Enter or right opens a folder, Backspace or left goes back to where you were, `s` sorts by size or by age (the stalest first, for clearing out), `l` lists the largest files (full scans only; Enter shows the folder one lives in), `q` quits. Piped, or with `--print`, it prints the report once; `--json` prints the scan.

- **The quick scan is not a job.** `--scan` runs `du` on the login node: a small script goes over ssh on stdin and runs six `du` at a time, one per folder two levels down, each at the lowest priority and all inside a time cap (`--budget`, 60 s by default). Nothing is written on the cluster. Folders `du` does not finish inside the cap show as `not read`, and the folders above them as lower bounds (`≥ 7.84 G`, as in the sample, where `envs` was cut off), with a note saying so; raise the cap, or use `--job`. It gives sizes and ages, not file counts or file names.
- **Why `du`, and why capped.** On project storage a single reader gets through about 800 files a second, so a folder of hundreds of thousands of small files (a Python environment) takes minutes. Login nodes allow each user a fraction of one core and kill processes that use more: a multi-threaded scanner written in Python was killed about half the time there, while `du`, which mostly waits on the file server, was not. Hence several `du` in parallel and a hard cap, rather than one clever walker.
- **The full scan: `--job`.** The `disk_scan` template runs a read-only walk with 16 parallel readers on a compute node (4 CPUs, 30 minutes at most, so under 2 CPU-hours, and it appears in `nuhpc usage`); it queues like any job. It adds file counts, the 8 largest files of each folder and the 25 largest overall, counts unreadable folders, and reports a partial result if it runs out of time. If `--job` times out while still queued it exits with status 3 and the run id; `nuhpc wait RUN` and then `nuhpc disk --collect RUN` bring the result in.
- **Scopes.** By default your `user_dir`; `--project NAME` for a declared project's folder; `--lab` for your group's folder (the parent of `user_dir`), which keeps one total per member and no file names at all (a `--job` lab scan gets 2 hours). `--depth` sets how many folder levels are listed (default 4; deeper folders are only totalled). The result is cached per scope in `~/.local/share/nuhpc/disk/` and its age is on screen; `nuhpc disk --from scan.json` opens a saved scan, for example one a colleague sent you.
- **Sizes** are apparent sizes, what `ls -l` reports, in powers of 1024 with the letters `du -h` and `df -h` use (`631 G`), not blocks allocated on disk. Symlinks are never followed, `.snapshot` folders are skipped, and with `--job` folders you cannot read are counted and flagged.
- **Read-only.** The explorer cannot delete anything, and neither can nuhpc: clean up in a shell, after reading the list (see "Cleaning up").

## Workflows

### Quick tests: `nuhpc run`

```bash
nuhpc run --code . --profile a100x1 --partition gpu-short -- python probe.py --n 8
```

This submits the command on a compute node with a 30-minute limit unless you pass `--time`, waits, prints its output, and fetches anything it wrote to `$HPC_OUT`. No `--params/--out` contract is needed. It exits 0 if the command succeeded, 2 if it failed (the log is printed), and 3 if `--timeout` passed first; the job keeps running, so `nuhpc wait` it. It is the `cmd` template underneath: `nuhpc submit cmd -p cmd="..."` does the same without blocking.

### Direct batch jobs: `python` template

Your script loads the model itself (transformers, or plain torch) and follows one contract: it takes `--params <params.json> --out <dir>` and writes what it keeps into `--out`. Make it **resumable**, skipping rows already in the output file, so a timeout or requeue costs nothing. Put one model in each task and loop over all conditions inside it; spread models across tasks with `--sweep`.

```bash
# rows.jsonl: one line per model, e.g. {"model": "Qwen/Qwen2.5-7B-Instruct"}; -p gives what every row shares
nuhpc push ./questions.jsonl questions.jsonl                                     # rows of {"id", "prompt", "choices"}
nuhpc submit hf_download --profile cpu -p repo=Qwen/Qwen2.5-7B-Instruct          # weights first (jobs run offline)
nuhpc submit python --name scores --code examples --sweep rows.jsonl --profile a100x1 --time 01:00:00 \
  -p entry=logprobs.py -p chat=true -p prompts=<work root>/data/questions.jsonl
nuhpc wait scores && nuhpc fetch scores --include '*.jsonl' --include '*.json'
```

The worked example is `examples/logprobs.py`. It scores candidate answers by the log-probability of each forced continuation, with all of a prompt's choices in one padded batch and the log-softmax in fp32, and it resumes. A research script built the same way ran 9,556 trials of a 7B VLM in about 15 GPU-minutes. If your script draws text into images, push the font and pass its path as a param: the cluster has no macOS fonts.

### Many small CPU rows: `--rows-per-job`

A sweep of many short or single-CPU rows should not be an array with one task per row (see Sizing jobs). `--rows-per-job N` puts N rows into each job and runs them side by side:

```bash
# 48 fits, 1 CPU and about 6 GB each, one script per row (rows.jsonl: {"entry": "fit.py", "seed": 0, ...})
nuhpc submit python --name fits --code . --sweep rows.jsonl --profile cpu --cpus 1 --mem 6G --time 04:00:00 \
  --rows-per-job 8
```

This submits 6 jobs, each with 8 CPUs and 48 GB, and each job runs its 8 rows at once. The rules:

- **`--cpus` and `--mem` describe one row.** nuhpc asks Slurm for that times the number of rows run at once, and prints both (`per_row_resources`, `resources`).
- **`--parallel P` runs only P rows at once** (default: all N). The job then takes `ceil(N / P)` waves, so `--time` is that many row runtimes. `--time` always covers the whole job.
- **Rows are independent processes.** Your script needs no change: each row runs with its own `--params` and its own folder `outputs/task_K/row_R/`, and `HPC_OUT` and `HPC_PARAMS` point at that folder inside the row, so rows cannot overwrite each other. Rows may differ in `entry` and `args`.
- **Threads:** `OMP_NUM_THREADS` and the other BLAS thread variables are set to the row's `--cpus`, so a row's numpy does not grab the whole node.
- **A failing row doesn't stop the others.** Its exit code is left in `row_R/exit_code`, its output lines in the log start with `[row R]`, and the job fails at the end if any row did. Read one row with `nuhpc logs RUN --task K --grep '\[row 3\]'`.
- **Memory is shared by the job:** a row that exceeds its share can take the whole job down with it, so size `--mem` from measured peaks.
- **GPU jobs** run their rows one after another (`--parallel` above 1 is refused), which still saves the per-job start; for several models use `--pack` below.
- It works with the `python` and `cmd` templates, and with any template of your own that runs its work through `hpc_each_row`.

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
| `nuhpc fetch RUN [--include '*.json'] [--max-size 500M]` | rsync `outputs/` and `logs/` (and the GPU samples in `usage/`) to `local_results/<run_id>` |
| `nuhpc usage [RUN...] [--since 7d\|all] [--project P]` | compute used vs requested per finished run, totals per project, with idle and failed shares (see Usage) |
| `nuhpc cancel RUN [--task i]` | `scancel` a whole run or one array task |
| `nuhpc runs` / `nuhpc queue` | local ledger / your jobs in `squeue` |
| `nuhpc standing [--users] [--brief]` | your queues against the live caps, how busy the cluster is (waits, load, free GPUs), and your account's fair share, with a GOOD / OK / POOR verdict and advice (see Where you stand) |
| `nuhpc push LOCAL [REL] [--project P] [--globus]` | upload data to `<work root>/data/REL` |
| `nuhpc pull REL [LOCAL] [--project P] [--globus]` | download a path relative to the work root |
| `nuhpc ls [REL] [--project P]` | list a directory under the work root |
| `nuhpc disk [PATH] [--scan] [--project P \| --lab] [--depth N]` | where your space goes: a quick `du` scan (no job; `--job` for a full one), then an interactive terminal explorer (see Where your space goes) |
| `nuhpc check [--smoke] [--project P]` / `nuhpc connect` | verify SSH and `user_dir`, create the work root, list partitions / open the persistent SSH session (interactive) |
| `nuhpc init [--user-dir DIR]` | write the example config, with your `user_dir` filled in |

Global flags: `--json` gives machine-readable output, including errors as `{"error": ...}`. `--dry-run` changes nothing on the cluster: it prints every remote command and leaves the rendered `job.sbatch` in `~/.local/share/nuhpc/runs/<run_id>/` for inspection.

### Submit options

```
--code DIR            snapshot this project into the run (excludes .git, venvs, data/, weights; add .nuhpcignore)
-p key=value          parameter; values parse as JSON (so 0.1, true, [1,2] are typed)
--sweep FILE          .jsonl / .json list / .csv, one task per row (merged over -p defaults)
--grid key=a,b,c      cartesian product, repeatable; combines with --sweep
--pack                one array task per group of rows sharing the template's `# pack-by:` keys
                      (vllm_eval: one server per model; the client runs once per row)
--rows-per-job N      N rows per job, run side by side on CPUs; --cpus/--mem describe ONE row
--parallel P          with --rows-per-job: rows running at once (default all; 1 on GPU jobs)
--max-parallel N      array throttle
--project NAME        tag for usage accounting (nuhpc usage --project NAME); a project declared in the
                      config also selects that project's directory for the run's runs/ and data/
--profile NAME        resource preset; override with --partition --time --gpus --mem --cpus --nodes --gres
--confirm-big         bypass limits (humans only)
```

## Templates

| template | use |
|---|---|
| `smoke` | environment check (GPU, a real CUDA kernel, imports, internet, disk); run first and after any env change |
| `env_build` | your Python environment: `uv`, Python 3.12, torch, vLLM; once per person, on a CPU profile |
| `cmd` | any shell command, no contract (`nuhpc run` wraps it) |
| `hf_download` | cache an HF model or dataset into `HF_HOME` on cluster storage |
| `python` | your script: `python <entry> --params --out [args]` |
| `vllm_eval` | vLLM OpenAI-compatible server + your client; supports `--pack` |
| `train_ddp` | `srun torchrun` across 1..N nodes with c10d rendezvous |

In a template, `{{CODE}} {{RUN_DIR}} {{DATA}} {{REMOTE_ROOT}} {{USER_DIR}} {{NODES}} {{GPUS_PER_NODE}} {{GPUS}} {{CPUS}}` are filled at render time. Per-task values come from `hpc_param KEY [DEFAULT]` at run time.

Read parameters into variables first, as in `X="$(hpc_param key)"`. Under `set -e`, a missing parameter aborts the job only inside an assignment, not when the substitution is inline in a command.

To add your own templates, drop `*.sbatch` files into `templates_dir`. Lines starting with `# doc:` show up in `nuhpc templates`, which is also how an agent learns what a template expects. A template that must run without `env_setup`, as `env_build` does, declares `# env-setup: none`. Every job exports `HPC_USER_DIR`, `HPC_REMOTE_ROOT`, `HPC_RUN_ID`, `HPC_RUN_DIR`, `HPC_TASK_ID`, `HPC_OUT` and `HPC_PARAMS`. A template supports `--pack` by declaring `# pack-by: key ...` (the params that must be equal within one task) and running its per-row work through `hpc_each_row CMD`, which sets `HPC_ROW_PARAMS` and `HPC_ROW_OUT` for each row.

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
   The skill also tells the agent what to do when `nuhpc standing` reports a poor standing: measure the waste with `nuhpc usage`, write a cost plan (right-size, bundle, pilot first) and get your acceptance before any heavy submit. That is an instruction to the agent; no `nuhpc` command refuses a submit because of your standing.
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

Every row below happened while setting this up (2026-10-07/09).

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
| `nuhpc usage` shows `mem_%` above 100, with no out-of-memory kill | Slurm's peak-memory figure on Explorer can exceed the request (a model download showed 168 %) | nothing: it is Slurm's number, not a warning |
| a job pending for a long time | see "Why a job is pending" | — |
| `submit` refuses: "N array tasks of 1 CPU each would use N of your per-user running-job slots" | an array of tiny tasks holds the per-user running-job cap, and everything else waits behind it | `--rows-per-job` (the message gives a number); see Sizing jobs |
| tasks end `TIMEOUT` after exactly the `--time` you gave | `--time` is a cap you set, not Explorer's maximum | measure a pilot, request about 1.5 times the slowest task, write results as you go |
| `status` says `SUBMITTED` | `sacct` hasn't caught up yet | wait a few seconds |

## On another Slurm cluster

The code is plain Slurm: `ssh` and `rsync` locally; `sbatch`, `sacct`, `sinfo`, `squeue` and `scancel` on the cluster. What is Explorer's:

- `example_config.toml`: the partitions, the GPU types in the profiles, and the limits.
- The `env_build` defaults: `bootstrap_python` (Explorer's shared Anaconda; elsewhere, any Python with pip), and `torch_backend=cu129` with the cu129 vLLM wheel. Match these to your driver (`nvidia-smi`).
- `vllm_eval` sets `no_proxy` for localhost, which is harmless where there is no proxy.
- This README's measured times, driver notes and troubleshooting table.

Everything else, the usage accounting included, reads standard Slurm output.

## Design notes

- **Verbs, not a shell.** An agent can only call `run / submit / wait / status / logs / fetch / usage / cancel / push / pull / ls`. There is no `exec` on the login node, and remote paths are confined to the work root of a directory you listed in the config; the one exception is the fixed `~/.nuhpc/usage/` folder that nuhpc itself writes usage records to. Code you upload does run on compute nodes; that is the point, and it stays inside Slurm's accounting.
- **Every job is a run.** Each run is a frozen snapshot of the rendered sbatch, the parameters, and the code. That makes runs reproducible and easy to inspect: everything lives in `runs/<run_id>/`.
- **One contract for batch code.** Your script takes `--params <params.json> --out <dir>` and writes what it wants kept into `--out`. Sweeps, evals and training all use this shape. Quick tests (`run`/`cmd`) skip it.
- **Sweeps are job arrays.** There is one task per row, or per pack, throttled with `%max_parallel`. This is how you run models or configurations in parallel without monopolising the GPU queue.
- **Guardrails for agents.** Every submit computes its worst-case GPU-hours and task count. Anything above the limits in your config needs `--confirm-big`, and the agent instructions say that flag requires a human. So does a large array of tiny tasks, which would otherwise fill the user's running-job cap; packing is the way out.

## Known gaps

- Partition names, GPU type strings, time limits and `module` names are what one account saw on 2026-10-07. Confirm yours with `nuhpc check` and `module avail`.
- `/scratch/$USER` can exist but be root-owned and unwritable. Check before pointing `HF_HOME`, `user_dir` or a project directory at it. Check RC's scratch purge policy before relying on scratch at all.
- Globus support is a thin wrapper over the `globus` CLI. It needs `globus login` once plus both endpoint UUIDs in the config.
- The ledger is local to each machine. Runs submitted from another laptop are visible in `queue`, but not in `runs`, `status`, `wait` or `usage`. Their usage records do reach `~/.nuhpc/usage/` on the cluster, which `nuhpc usage` doesn't read yet.
- `pull` of a directory nests it one level deeper (`pulled/x/x`), because of rsync's trailing-slash rule.
- The local side has been run on macOS only. Linux should work as it is (it needs `ssh` and `rsync`); Windows needs WSL.

## License

MIT; see [LICENSE](LICENSE). Changes are listed in [CHANGELOG.md](CHANGELOG.md).
