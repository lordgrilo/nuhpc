"""nuhpc: an agent-friendly CLI for running Slurm jobs on Northeastern's Explorer cluster.

Design rules:
  * Installs (envs/, hf_cache/) live under `user_dir`; work (runs/, data/) under a nuhpc/ folder in the project's
    directory, or in user_dir. Remote paths given on the CLI are relative to that work root.
  * Every job is a *run*: rendered sbatch + params + code snapshot in runs/<run_id>/ on the cluster.
  * Every command supports --json so an agent can parse results; humans get tables.
  * No arbitrary remote exec. The agent gets verbs, not a shell.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import itertools
import json
import math
import os
import posixpath
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import tomllib
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path

from nuhpc import diskquick, diskview

CONFIG_PATH = Path(os.environ.get("NUHPC_CONFIG", "~/.config/nuhpc/config.toml")).expanduser()
STATE_DIR = Path(os.environ.get("NUHPC_STATE", "~/.local/share/nuhpc")).expanduser()
LEDGER = STATE_DIR / "ledger.jsonl"

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
DONE = {"COMPLETED", "FINISHED_WITH_FAILURES"}   # overall run states that `wait` stops on
TRANSIENT_EXIT = {255, 10, 12, 30, 35}            # ssh connection failure; rsync socket/protocol/timeout errors
RETRY_DELAYS = (5, 15)                            # seconds between attempts at a repeatable remote call
DEFAULT_EXCLUDES = [".git", "__pycache__", ".venv", "venv", "*.pyc", ".ipynb_checkpoints",
                    "wandb", "outputs", "data", "*.ckpt", "*.safetensors", "*.pt", "*.bin"]
RES_KEYS = ("partition", "time", "cpus", "mem", "gres", "nodes", "account", "constraint")
# Usage records: kept locally in STATE_DIR/usage/, beside the results, and on the cluster in $HOME (relative path),
# outside remote_root so cleaning runs/ keeps them, and shared by every machine that submits as this user.
REMOTE_USAGE = ".nuhpc/usage"
USAGE_FIELDS = "JobID,State,ExitCode,Submit,Start,End,ElapsedRaw,TimelimitRaw,NodeList,AllocTRES,TotalCPU,MaxRSS,ReqMem"
SECTION = "__nuhpc_usage_section__"


class NuhpcError(Exception):
    pass


# --------------------------------------------------------------------------- config

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise NuhpcError(f"No config at {CONFIG_PATH}. Run `nuhpc init` and edit it.")
    cfg = tomllib.loads(CONFIG_PATH.read_text())
    if "ssh_host" not in cfg:
        raise NuhpcError("config is missing required key 'ssh_host'")
    if not (cfg.get("user_dir") or cfg.get("remote_root")):
        raise NuhpcError("config needs user_dir: your folder on project storage, e.g. /projects/<group>/<you>")
    for key in ("user_dir", "remote_root"):
        if key in cfg:
            if not str(cfg[key]).startswith("/"):
                raise NuhpcError(f"{key} must be an absolute path on the cluster, got '{cfg[key]}'")
            if "YOUR_" in cfg[key]:
                raise NuhpcError(f"{key} still has the placeholder '{cfg[key]}': edit {CONFIG_PATH}")
    cfg.setdefault("local_results", "~/nuhpc-results")
    cfg.setdefault("env_setup", "")
    cfg.setdefault("profiles", {})
    cfg.setdefault("limits", {})
    cfg.setdefault("templates_dir", "")
    cfg.setdefault("globus", {})
    cfg.setdefault("projects", {})
    cfg.setdefault("standing", {})
    for name, proj in cfg["projects"].items():
        if not (isinstance(proj, dict) and str(proj.get("dir", "")).startswith("/")):
            raise NuhpcError(f"[projects.{name}] needs dir = an absolute path on the cluster")
    return cfg


def user_dir(cfg: dict) -> str:
    """Where installs live (envs/, hf_cache/): shared by all the user's projects. Legacy configs: remote_root."""
    return (cfg.get("user_dir") or cfg["remote_root"]).rstrip("/")


def work_root(cfg: dict, project: str | None = None) -> str:
    """Where runs/ and data/ live: nuhpc/ inside the declared project's dir, else the configured default.
    The default is remote_root if set (older configs), else nuhpc/ inside user_dir. An undeclared project
    name is only a tag."""
    proj = cfg["projects"].get(project) if project else None
    if proj:
        return proj["dir"].rstrip("/") + "/nuhpc"
    return (cfg.get("remote_root") or user_dir(cfg) + "/nuhpc").rstrip("/")


def remote_path(cfg: dict, rel: str = "", project: str | None = None) -> str:
    rel = (rel or "").strip()
    if rel.startswith("/") or ".." in Path(rel).parts:
        raise NuhpcError("remote paths are relative to the work root (no absolute paths, no '..')")
    root = work_root(cfg, project)
    return f"{root}/{rel}" if rel else root


# --------------------------------------------------------------------------- transport

class Remote:
    def __init__(self, cfg: dict, dry: bool):
        self.cfg, self.host, self.dry = cfg, cfg["ssh_host"], dry

    def _exec(self, argv: list[str], capture=True, retry=True, input: str | None = None) -> str:
        if self.dry:
            print("[dry-run] " + " ".join(shlex.quote(a) for a in argv), file=sys.stderr)
            return ""
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                p = subprocess.run(argv, text=True, capture_output=capture, input=input)
            except FileNotFoundError:
                raise NuhpcError(f"'{argv[0]}' not found on this machine (install it)")
            if p.returncode == 0:
                return p.stdout if capture else ""
            err = (p.stderr or "").strip()
            auth = "Permission denied" in err or "Host key" in err or "BatchMode" in err
            # ssh 255 / rsync 10, 12, 30, 35: dropped or throttled connections. Retried only when the
            # caller says the command is safe to repeat (never sbatch: a retry could submit twice).
            if not (retry and not auth and p.returncode in TRANSIENT_EXIT and attempt < len(RETRY_DELAYS)):
                break
            print(f"[nuhpc] {argv[0]} connection failed (exit {p.returncode}); retrying in "
                  f"{RETRY_DELAYS[attempt]}s", file=sys.stderr)
            time.sleep(RETRY_DELAYS[attempt])
        if auth:
            err += "\n(hint: run `nuhpc connect` to open an authenticated SSH session first)"
        raise NuhpcError(f"command failed ({p.returncode}): {' '.join(argv[:3])} ...\n{err}")

    def run(self, cmd: str, retry=True, input: str | None = None) -> str:
        # Login shell so `module`, `sbatch`, etc. are on PATH. `input` is fed to the command's stdin (a script for `python3 -`).
        return self._exec(["ssh", "-o", "BatchMode=yes", self.host, "bash -lc " + shlex.quote(cmd)], retry=retry, input=input)

    def stream(self, cmd: str, input: str | None = None, on_line=None, on_start=None) -> str:
        """Like run(), but hands each output line to `on_line` as it arrives, and the process to `on_start` so that a caller can stop it.
        Never retried: a capped, long-running command is not silently repeated."""
        argv = ["ssh", "-o", "BatchMode=yes", self.host, "bash -lc " + shlex.quote(cmd)]
        if self.dry:
            print("[dry-run] " + " ".join(shlex.quote(a) for a in argv), file=sys.stderr)
            return ""
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError:
            raise NuhpcError("'ssh' not found on this machine (install it)")
        if on_start:
            on_start(proc)
        err: list[str] = []
        reader = threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)   # drained so a chatty remote cannot block
        reader.start()
        try:
            proc.stdin.write(input or "")
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        lines = []
        for line in proc.stdout:
            lines.append(line)
            if on_line:
                on_line(line.rstrip("\n"))
        code = proc.wait()
        reader.join(2)
        if code != 0:
            text = "".join(err).strip()
            hint = "\n(hint: run `nuhpc connect` to open an authenticated SSH session first)" if "Permission denied" in text else ""
            raise NuhpcError(f"command failed ({code}): {' '.join(argv[:3])} ...\n{text}{hint}")
        return "".join(lines)

    def rsync(self, src: str, dst: str, extra: list[str] | None = None) -> None:
        argv = ["rsync", "-az", "--partial", "-e", "ssh -o BatchMode=yes", *(extra or []), src, dst]
        self._exec(argv, capture=False)


# --------------------------------------------------------------------------- helpers

def parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def parse_kv(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise NuhpcError(f"expected key=value, got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = parse_value(v)
    return out


def build_tasks(sweep: str | None, grid: list[str] | None, base: dict) -> list[dict]:
    tasks = [dict(base)]
    if sweep:
        p = Path(sweep)
        if not p.exists():
            raise NuhpcError(f"sweep file not found: {sweep}")
        if p.suffix == ".jsonl":
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        elif p.suffix == ".json":
            rows = json.loads(p.read_text())
        elif p.suffix == ".csv":
            rows = [{k: parse_value(v) for k, v in r.items()} for r in csv.DictReader(p.open())]
        else:
            raise NuhpcError("sweep file must be .jsonl, .json (list) or .csv")
        if not rows:
            raise NuhpcError("sweep file is empty")
        tasks = [{**base, **r} for r in rows]
    if grid:
        axes = []
        for g in grid:
            k, _, vals = g.partition("=")
            axes.append([(k, parse_value(v)) for v in vals.split(",")])
        tasks = [{**t, **dict(combo)} for t in tasks for combo in itertools.product(*axes)]
    return tasks


def resolve_resources(cfg: dict, args) -> tuple[str | None, dict]:
    name = args.profile or cfg.get("default_profile")
    if name and name not in cfg["profiles"]:
        raise NuhpcError(f"unknown profile '{name}'. Known: {', '.join(cfg['profiles']) or 'none'}")
    res = dict(cfg["profiles"].get(name, {})) if name else {}
    for k in RES_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            res[k] = v
    if args.gpus is not None:
        parts = str(res.get("gres", "gpu")).split(":")
        res["gres"] = f"gpu:{parts[1]}:{args.gpus}" if len(parts) == 3 else f"gpu:{args.gpus}"
    res.setdefault("nodes", 1)
    res.setdefault("cpus", 4)
    res.setdefault("mem", "16G")
    res.setdefault("time", "01:00:00")
    if "partition" not in res:
        raise NuhpcError("no partition: pass --partition or use a --profile that sets one")
    return name, res


def mem_to_gb(s) -> float:
    """A Slurm memory request ('6G', '1500M', '2T'; a bare number is MB) in GB."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*", str(s).upper())
    if not m:
        raise NuhpcError(f"can't read memory '{s}': use a number with a unit, like 6G or 1500M")
    return float(m.group(1)) * 1024.0 ** ("KMGT".index(m.group(2) or "M") - 2)


def scale_mem(s, k: int) -> str:
    m = re.fullmatch(r"\s*(\d+)\s*([KMGT]?)B?\s*", str(s).upper())
    if not m:
        raise NuhpcError(f"can't multiply --mem '{s}' by {k} rows: use an integer with a unit, like 6G or 1500M")
    return f"{int(m.group(1)) * k}{m.group(2)}"


def gpus_per_node(res: dict) -> int:
    g = str(res.get("gres", "") or "")
    if not g.startswith("gpu"):
        return 0
    m = re.search(r":(\d+)$", g)
    return int(m.group(1)) if m else 1


def time_hours(t: str) -> float:
    days = 0
    t = str(t)
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
        p = [int(x) for x in t.split(":")] + [0, 0]
        return days * 24 + p[0] + p[1] / 60 + p[2] / 3600
    p = [int(x) for x in t.split(":")]
    if len(p) == 1:
        return p[0] / 60
    if len(p) == 2:
        return p[0] / 60 + p[1] / 3600
    return p[0] + p[1] / 60 + p[2] / 3600


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-")[:40] or "run"


def find_template(cfg: dict, name: str) -> Path:
    cands = [Path(name)]
    if cfg["templates_dir"]:
        td = Path(cfg["templates_dir"]).expanduser()
        cands += [td / name, td / f"{name}.sbatch"]
    pkg = Path(str(resources.files("nuhpc") / "templates"))
    cands += [pkg / name, pkg / f"{name}.sbatch"]
    for c in cands:
        if c.is_file():
            return c
    raise NuhpcError(f"template '{name}' not found (try `nuhpc templates`)")


def pack_keys(template: Path) -> list[str]:
    """Params a template needs uniform within one array task, from its '# pack-by:' line."""
    for line in template.read_text().splitlines():
        if line.startswith("# pack-by:"):
            return line[len("# pack-by:"):].split()
    return []


def pack_rows(tasks: list[dict], keys: list[str]) -> list[list[int]]:
    """Group row indices by their values on `keys`, in order of first appearance."""
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(tasks):
        groups.setdefault(json.dumps([t.get(k) for k in keys], sort_keys=True), []).append(i)
    return list(groups.values())


def suggest_rows_per_job(cfg: dict, res: dict, n_rows: int) -> int:
    """How many one-row jobs fit side by side in a job that stays within a typical node."""
    lim, k = cfg["limits"], n_rows
    k = min(k, int(lim.get("big_job_cpus") or 16) // max(1, int(res["cpus"])))
    if lim.get("big_job_mem"):
        k = min(k, int(mem_to_gb(lim["big_job_mem"]) // mem_to_gb(res["mem"])))
    return k


def check_job_shape(cfg: dict, res: dict, job_res: dict, n: int, n_rows: int, confirm_big: bool) -> list[str]:
    """What a submit asks of the scheduler. Many tiny array tasks are refused (they use up the per-user running-job
    cap, which counts every array task); smaller excesses and jobs bigger than most nodes return warnings."""
    lim, warnings = cfg["limits"], []
    cpus, small_max, small_warn = int(job_res["cpus"]), lim.get("small_task_max", 16), lim.get("small_task_warn", 8)
    if gpus_per_node(job_res) == 0 and cpus <= lim.get("small_task_cpus", 2) and (
            (small_max and n > small_max) or (small_warn and n > small_warn)):
        k = suggest_rows_per_job(cfg, res, n_rows)
        what = (f"{n} array tasks of {cpus} CPU{'s' if cpus > 1 else ''} each would use {n} of your per-user "
                "running-job slots: the scheduler caps how many jobs one user can run at once, and every array task counts.")
        fix = (f"Pack them: --rows-per-job {k} runs {k} rows side by side in one job ({k * int(res['cpus'])} CPUs, "
               f"{scale_mem(res['mem'], k)}), {-(-n_rows // k)} jobs in all, and --time then covers one row's runtime."
               if k >= 2 else "Rows this large cannot be packed within a typical node: submit fewer at a time.")
        if small_max and n > small_max:
            if not confirm_big:
                raise NuhpcError(f"{what} {fix} Submitting them unpacked needs --confirm-big (human approval).")
            warnings.append(f"{what} Allowed by --confirm-big. {fix}")
        else:
            warnings.append(f"{what} {fix}")
    bc, bm = lim.get("big_job_cpus"), lim.get("big_job_mem")   # CPU-node sizes: GPU nodes are far bigger, so GPU jobs skip this
    if gpus_per_node(job_res) == 0 and ((bc and cpus > int(bc)) or (bm and mem_to_gb(job_res["mem"]) > mem_to_gb(bm))):
        typical = " and ".join(x for x in (f"{bc} CPUs" if bc else "", str(bm) if bm else "") if x)
        warnings.append(f"each job asks for {cpus} CPUs and {job_res['mem']}, bigger than most nodes (about {typical}): "
                        "it can only run on the few large nodes, so expect long waits. "
                        "Use fewer rows per job or less memory per row.")
    return warnings


def all_templates(cfg: dict) -> list[Path]:
    dirs = [Path(str(resources.files("nuhpc") / "templates"))]
    if cfg["templates_dir"]:
        dirs.insert(0, Path(cfg["templates_dir"]).expanduser())
    seen, out = set(), []
    for d in dirs:
        for p in sorted(d.glob("*.sbatch")) if d.is_dir() else []:
            if p.stem not in seen:
                seen.add(p.stem)
                out.append(p)
    return out


PREAMBLE = r'''set -euo pipefail
export HPC_USER_DIR="{user_dir}"
export HPC_REMOTE_ROOT="{remote_root}"
export HPC_PACK_PARALLEL="{parallel}"
export HPC_ROW_CPUS="{row_cpus}"
export HPC_RUN_ID="{run_id}"
export HPC_RUN_DIR="{run_dir}"
export HPC_TASK_ID="${{SLURM_ARRAY_TASK_ID:-0}}"
export HPC_OUT="$HPC_RUN_DIR/outputs/task_$HPC_TASK_ID"
export HPC_PARAMS="$HPC_OUT/params.json"
mkdir -p "$HPC_OUT"
# Rows (sweep.jsonl line indices) this task runs: its own, or a group under --pack (packs.json).
if [ -f "$HPC_RUN_DIR/packs.json" ]; then
  HPC_ROWS="$(python3 -c 'import json,sys; print(*json.load(open(sys.argv[1]))[int(sys.argv[2])])' "$HPC_RUN_DIR/packs.json" "$HPC_TASK_ID")"
else
  HPC_ROWS="$HPC_TASK_ID"
fi
export HPC_ROWS
sed -n "$((${{HPC_ROWS%% *}} + 1))p" "$HPC_RUN_DIR/sweep.jsonl" > "$HPC_PARAMS"

# hpc_each_row CMD...: run CMD once per row of this task, with HPC_ROW, HPC_ROW_PARAMS, HPC_ROW_OUT set.
# Unpacked runs keep the flat layout (outputs/task_K). Packed rows (--pack, --rows-per-job) write to
# outputs/task_K/row_R, and inside the row HPC_OUT and HPC_PARAMS name that row's own folder and params.
# With --rows-per-job on CPUs, HPC_PACK_PARALLEL rows run at once: their output lines are prefixed "[row R]"
# and each gets HPC_ROW_CPUS threads. A failing row doesn't stop the others; its exit code is left in
# row_R/exit_code, and the task fails at the end if any row did.
_hpc_row() {{
  local r="$1" rrc=0; shift
  export HPC_ROW="$r" HPC_ROW_OUT="$HPC_OUT/row_$r" HPC_ROW_PARAMS="$HPC_OUT/row_$r/params.json"
  export HPC_OUT="$HPC_ROW_OUT" HPC_PARAMS="$HPC_ROW_PARAMS"
  mkdir -p "$HPC_ROW_OUT"
  sed -n "$((r + 1))p" "$HPC_RUN_DIR/sweep.jsonl" > "$HPC_ROW_PARAMS"
  echo "[nuhpc] row $r: $(cat "$HPC_ROW_PARAMS")"
  "$@" || rrc=$?
  [ "$rrc" -eq 0 ] || echo "[nuhpc] row $r failed (exit $rrc)"
  echo "$rrc" > "$HPC_ROW_OUT/exit_code"
  return "$rrc"
}}
hpc_each_row() {{
  if [ ! -f "$HPC_RUN_DIR/packs.json" ]; then
    HPC_ROW="$HPC_TASK_ID" HPC_ROW_PARAMS="$HPC_PARAMS" HPC_ROW_OUT="$HPC_OUT" "$@"; return
  fi
  local par="${{HPC_PACK_PARALLEL:-1}}" base r p pids="" rc=0
  base="$(jobs -rp | wc -l)"   # background jobs that aren't rows (a GPU sampler, a server)
  for r in $HPC_ROWS; do
    if [ "$par" -le 1 ]; then
      ( _hpc_row "$r" "$@" ) || rc=1
      continue
    fi
    while [ "$(( $(jobs -rp | wc -l) - base ))" -ge "$par" ]; do sleep 0.2; done
    (
      export OMP_NUM_THREADS="$HPC_ROW_CPUS" MKL_NUM_THREADS="$HPC_ROW_CPUS"
      export OPENBLAS_NUM_THREADS="$HPC_ROW_CPUS" NUMEXPR_NUM_THREADS="$HPC_ROW_CPUS"
      _hpc_row "$r" "$@" || true
    ) 2>&1 | while IFS= read -r line || [ -n "$line" ]; do echo "[row $r] $line"; done &
    pids="$pids $!"
  done
  for p in $pids; do wait "$p" || true; done
  for r in $HPC_ROWS; do
    [ "$par" -le 1 ] || [ "$(cat "$HPC_OUT/row_$r/exit_code" 2>/dev/null || echo 1)" -eq 0 ] || rc=1
  done
  return $rc
}}

# hpc_param KEY [DEFAULT]: read a parameter of this task (strings raw, others as JSON).
hpc_param() {{
python3 - "$1" "${{2-__NUHPC_REQUIRED__}}" <<'PY'
import json, os, sys
d = json.load(open(os.environ["HPC_PARAMS"]))
k, default = sys.argv[1], sys.argv[2]
if k in d:
    v = d[k]; print(v if isinstance(v, str) else json.dumps(v))
elif default != "__NUHPC_REQUIRED__":
    print(default)
else:
    sys.exit(f"nuhpc: missing required param '{{k}}'")
PY
}}

# ---- environment (from config.env_setup) ----
{env_setup}
# ---- provenance ----
echo "[nuhpc] run=$HPC_RUN_ID task=$HPC_TASK_ID host=$(hostname) start=$(date -Is)"
command -v nvidia-smi >/dev/null && nvidia-smi -L || true
cat "$HPC_PARAMS"; echo
# ---- usage: GPU samples every 30 s, summarised into usage.json once the run ends ----
# No EXIT trap (templates own that); Slurm kills the sampler with the job. Multi-node: first node only.
if [ -n "${{CUDA_VISIBLE_DEVICES:-}}" ] && command -v nvidia-smi >/dev/null; then
  mkdir -p "$HPC_RUN_DIR/usage"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 30 \
    > "$HPC_RUN_DIR/usage/task_${{HPC_TASK_ID}}_gpu.csv" 2>/dev/null &
fi
'''


def render(cfg, template: Path, res: dict, n_tasks: int, max_par: int | None,
           run_id: str, run_dir: str, job_name: str, project: str | None = None,
           parallel: int = 1, row_cpus: int | None = None) -> str:
    gpn = gpus_per_node(res)
    hdr = ["#!/bin/bash",
           f"#SBATCH --job-name={job_name}",
           f"#SBATCH --partition={res['partition']}",
           f"#SBATCH --time={res['time']}",
           f"#SBATCH --nodes={res['nodes']}",
           "#SBATCH --ntasks-per-node=1",
           f"#SBATCH --cpus-per-task={res['cpus']}",
           f"#SBATCH --mem={res['mem']}"]
    if res.get("gres"):
        hdr.append(f"#SBATCH --gres={res['gres']}")
    for k in ("account", "constraint"):
        if res.get(k):
            hdr.append(f"#SBATCH --{k}={res[k]}")
    if n_tasks > 1:
        hdr.append(f"#SBATCH --array=0-{n_tasks - 1}" + (f"%{max_par}" if max_par else ""))
        hdr.append(f"#SBATCH --output={run_dir}/logs/%x-%A_%a.out")
    else:
        hdr.append(f"#SBATCH --output={run_dir}/logs/%x-%j.out")
    ctx = {"RUN_ID": run_id, "RUN_DIR": run_dir, "CODE": f"{run_dir}/code",
           "DATA": remote_path(cfg, "data", project), "REMOTE_ROOT": remote_path(cfg, project=project),
           "USER_DIR": user_dir(cfg),
           "NODES": str(res["nodes"]), "GPUS_PER_NODE": str(gpn),
           "GPUS": str(gpn * int(res["nodes"])), "CPUS": str(res["cpus"])}
    body = template.read_text()
    body = re.sub(r"\{\{(\w+)\}\}", lambda m: ctx.get(m.group(1), m.group(0)), body)
    left = re.findall(r"\{\{(\w+)\}\}", body)
    if left:
        raise NuhpcError(f"template uses unknown placeholders: {sorted(set(left))}. "
                         f"Per-task values belong in params: use $(hpc_param key).")
    # A template that builds the environment can't run the env_setup that activates it.
    env_setup = ("# skipped: this template declares '# env-setup: none'" if "# env-setup: none" in body.splitlines()
                 else cfg["env_setup"].strip())
    pre = PREAMBLE.format(run_id=run_id, run_dir=run_dir, user_dir=user_dir(cfg),
                          remote_root=remote_path(cfg, project=project), env_setup=env_setup,
                          parallel=parallel, row_cpus=row_cpus or res["cpus"])
    return "\n".join(hdr) + "\n\n" + pre + "\n# ---- template: " + template.stem + " ----\n" + body


def ledger_read() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def ledger_append(entry: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def find_run(run: str) -> dict:
    entries = ledger_read()
    if run == "last" and entries:
        return entries[-1]
    hits = [e for e in entries if e["run_id"] == run or e["run_id"].startswith(run) or e["job_id"] == run]
    if not hits:
        raise NuhpcError(f"no run matching '{run}' (see `nuhpc runs`)")
    if len(hits) > 1 and not any(e["run_id"] == run for e in hits):
        raise NuhpcError(f"'{run}' is ambiguous: {[e['run_id'] for e in hits]}")
    return next((e for e in hits if e["run_id"] == run), hits[-1])


def array_count(jobid: str) -> int:
    m = re.search(r"_\[([^\]%]+)", jobid)
    if not m:
        return 1
    n = 0
    for part in m.group(1).split(","):
        a, _, b = part.partition("-")
        n += (int(b) - int(a) + 1) if b else 1
    return n


def emit(data, args) -> None:
    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return
    if isinstance(data, list):
        if not data:
            print("(nothing)")
            return
        cols = list(data[0].keys())
        rows = [[str(d.get(c, "")) for c in cols] for d in data]
        w = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
        for r in rows:
            print("  ".join(v.ljust(w[i]) for i, v in enumerate(r)))
    elif isinstance(data, dict):
        for k, v in data.items():
            print(f"{k}: {v}")
    else:
        print(data)


# --------------------------------------------------------------------------- usage

def clock_seconds(s: str) -> float:
    """Slurm [D-][HH:]MM:SS[.mmm] (TotalCPU) in seconds."""
    if not s:
        return 0.0
    days, s = s.split("-", 1) if "-" in s else ("0", s)
    p = [float(x) for x in s.split(":")]
    p = [0.0] * (3 - len(p)) + p
    return int(days) * 86400 + p[0] * 3600 + p[1] * 60 + p[2]


def size_gb(s: str) -> float | None:
    """Slurm sizes (MaxRSS '16777216K', ReqMem '64G' or legacy '64Gn') in GB."""
    m = re.match(r"([\d.]+)([KMGT]?)", s or "")
    return float(m.group(1)) * 1024.0 ** ("KMGT".index(m.group(2)) - 2 if m.group(2) else -3) if m else None


def ratio(a, b):
    return round(a / b, 3) if a is not None and b else None


def gpu_summary_cmd(entries: list[dict]) -> str:
    """Remote shell: per task, the sample count, mean and max GPU utilisation (%) and max memory (MiB)."""
    csvs = " ".join(f"{shlex.quote(e['remote_dir'])}/usage/task_*_gpu.csv" for e in entries)
    awk = ("awk -F', *' -v f=\"$f\" '$3 ~ /^[0-9.]+$/ {n++; s+=$3; if ($3>u) u=$3; if ($4>m) m=$4} "
           "END {if (n) printf \"%s|%d|%.1f|%d|%d\\n\", f, n, s/n, u, m}' \"$f\"")
    return f'for f in {csvs}; do if [ -f "$f" ]; then {awk}; fi; done'


def usage_query(entries: list[dict]) -> str:
    """Remote shell, one call (each ssh is a fresh login): make the cluster-home copy dir, then print three
    sections: accounting, the GPU type of each node used (Explorer's sacct doesn't name it), GPU samples."""
    ids = ",".join(e["job_id"] for e in entries)
    nodes = (r"""nodes=$(printf '%s\n' "$acct" | awk -F'|' '$1 !~ /[.]/ && $9 !~ /^None/ {print $9}' """
             r"""| sort -u | paste -sd, -); [ -z "$nodes" ] || sinfo -h -N -n "$nodes" -o '%N|%G' || true""")
    return (f"mkdir -p ~/{REMOTE_USAGE}; acct=$(sacct -n -P --format={USAGE_FIELDS} -j {ids}) || exit 1; "
            f"printf '%s\\n' \"$acct\"; echo {SECTION}; {nodes}; echo {SECTION}; " + gpu_summary_cmd(entries))


def parse_usage(out: str, entries: list[dict]) -> dict[str, dict]:
    """Usage records of the runs whose tasks have all ended (until then the numbers still change)."""
    acct, node_gres, samples = (out.split(SECTION) + ["", ""])[:3]
    alloc: dict[tuple, list[str]] = {}
    steps: dict[tuple, list[list[str]]] = {}
    for line in acct.splitlines():
        p = line.split("|")
        if len(p) != 13:
            continue
        jid, _, step = p[0].partition(".")
        base, _, idx = jid.partition("_")
        if idx.startswith("["):        # array tasks that haven't started
            continue
        key = (base, int(idx or 0))
        if step:
            steps.setdefault(key, []).append(p)
        else:
            alloc[key] = p
    node_gpu = {}
    for line in node_gres.splitlines():   # 'd1026|gpu:a100:3(S:0-1)'
        node, _, gres = line.partition("|")
        if m := re.search(r"gpu:([\w.-]+):\d", gres):
            node_gpu[node] = m.group(1)
    gpu = {}
    for line in samples.splitlines():
        m = re.match(r"(.*)/usage/task_(\d+)_gpu\.csv\|(\d+)\|([\d.]+)\|(\d+)\|(\d+)$", line)
        if m:
            gpu[(m.group(1), int(m.group(2)))] = m.groups()[2:]
    recs = {}
    for e in entries:
        keys = sorted(k for k in alloc if k[0] == e["job_id"])
        if len(keys) < e["n_tasks"] or any(alloc[k][1].split()[0] not in TERMINAL for k in keys):
            continue
        tasks = []
        for k in keys:
            p, st = alloc[k], steps.get(k, [])
            tres = dict(kv.split("=", 1) for kv in p[9].split(",") if "=" in kv)
            gpus, cpus, elapsed = int(tres.get("gres/gpu", 0)), int(tres.get("cpu", 0)), int(p[6] or 0)
            limit = int(p[7]) * 60 if p[7].isdigit() else None
            try:
                wait = int((datetime.fromisoformat(p[4]) - datetime.fromisoformat(p[3])).total_seconds())
            except ValueError:         # never started: Start is "None" or "Unknown"
                wait = None
            rss = max((size_gb(s[11]) or 0 for s in st), default=None)
            g = gpu.get((e["remote_dir"], k[1]))
            tasks.append({
                "task": k[1], "state": p[1].split()[0], "exit_code": p[2], "node": p[8],
                "gpu_type": next((t.split(":", 1)[1] for t in tres if t.startswith("gres/gpu:")),
                                 node_gpu.get(p[8]) if gpus else None),
                "gpus": gpus, "cpus": cpus, "submit": p[3], "start": p[4], "end": p[5],
                "queue_wait_s": wait, "elapsed_s": elapsed, "time_limit_s": limit,
                "gpu_hours": round(gpus * elapsed / 3600, 3), "cpu_hours": round(cpus * elapsed / 3600, 3),
                "cpu_eff": ratio(max(clock_seconds(p[10]), sum(clock_seconds(s[10]) for s in st)), cpus * elapsed),
                "max_rss_gb": round(rss, 2) if rss is not None else None, "mem_eff": ratio(rss, size_gb(p[12])),
                "time_eff": ratio(elapsed, limit),
                "gpu_samples": int(g[0]) if g else None, "gpu_util_mean": float(g[1]) if g else None,
                "gpu_util_max": int(g[2]) if g else None,
                "gpu_mem_max_gb": round(int(g[3]) / 1024, 2) if g else None})
        stage = STATE_DIR / "runs" / e["run_id"] / "meta.json"
        requested = json.loads(stage.read_text())["resources"] if stage.exists() else {}
        states: dict[str, int] = {}
        for t in tasks:
            states[t["state"]] = states.get(t["state"], 0) + 1
        waits = [t["queue_wait_s"] for t in tasks if t["queue_wait_s"] is not None]
        recs[e["run_id"]] = {
            "run_id": e["run_id"], "job_id": e["job_id"], "project": e.get("project"), "template": e["template"],
            "profile": e["profile"], "created": e["created"], "recorded": datetime.now().isoformat(timespec="seconds"),
            "requested": {**requested, "n_tasks": e["n_tasks"], "worst_case_gpu_hours": e["worst_case_gpu_hours"]},
            "tasks": tasks,
            "totals": {"gpu_hours": round(sum(t["gpus"] * t["elapsed_s"] for t in tasks) / 3600, 3),
                       "cpu_hours": round(sum(t["cpus"] * t["elapsed_s"] for t in tasks) / 3600, 3),
                       "worst_case_gpu_hours": e["worst_case_gpu_hours"],
                       "queue_wait_s_max": max(waits, default=None), "states": states}}
    return recs


def ensure_usage(cfg: dict, R, entries: list[dict]) -> dict[str, dict]:
    """Usage records for these runs. Saved ones are read back; finished runs without one are recorded now
    and the local records copied to the cluster home. Fails soft: a transport error only warns."""
    d = STATE_DIR / "usage"
    recs = {e["run_id"]: json.loads((d / f"{e['run_id']}.json").read_text())
            for e in entries if (d / f"{e['run_id']}.json").exists()}
    todo = [e for e in entries if e["run_id"] not in recs]
    if todo and not R.dry:
        try:
            new = parse_usage(R.run(usage_query(todo)), todo)
            if new:
                d.mkdir(parents=True, exist_ok=True)
                for r in new.values():
                    (d / f"{r['run_id']}.json").write_text(json.dumps(r, indent=2))
                recs.update(new)
                R.rsync(f"{d}/", f"{cfg['ssh_host']}:{REMOTE_USAGE}/")   # whole dir: retries any failed copy
        except NuhpcError as ex:
            print(f"[nuhpc] usage not recorded: {str(ex).splitlines()[0]}", file=sys.stderr)
    for r in recs.values():           # beside the results, once fetched
        res = Path(cfg["local_results"]).expanduser() / r["run_id"]
        if res.is_dir():
            (res / "usage.json").write_text(json.dumps(r, indent=2))
    return recs


# --------------------------------------------------------------------------- standing

STANDING_MARK = "__nuhpc_standing__"
RANK = {"GOOD": 0, "OK": 1, "POOR": 2}
CAP_REASONS = ("QOSMax", "AssocMax", "AssocGrp")   # a limit blocks the job; JobArrayTaskLimit is your own %N throttle
STANDING_DEFAULTS = {"wait_hours": 6, "near_cap": 0.8, "fairshare_good": 0.5, "fairshare_low": 0.25,
                     "wait_window_h": 6, "cluster_wait_ok_min": 30, "cluster_wait_poor_min": 120}
MIN_STARTS = 5                 # a queue's median wait needs at least this many jobs that started inside the window
RUN_STATES = ("RUNNING", "COMPLETING", "CONFIGURING")
WAITING = ("Priority", "Resources")   # pending on the scheduler's account; any other reason (dependency, hold, cap, start time) is held
DOWN = ("down", "drain", "drng", "inval", "maint", "resv", "reserved", "fail", "unk", "future")   # node states that are not capacity


def standing_thresholds(cfg: dict) -> dict:
    return {**STANDING_DEFAULTS, **cfg.get("standing", {})}


def fmt_dur(minutes: float) -> str:
    """A wait in the unit that fits: minutes under an hour, hours under two days, else days."""
    if minutes < 60:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.3g} h" if minutes < 2880 else f"{minutes / 1440:.3g} d"


def fmt_limit(hours: float) -> str:
    return f"{hours:g} h" if hours < 24 else f"{hours / 24:g} d"


def _hours(t):
    try:
        return time_hours(t)
    except (ValueError, TypeError):
        return None                                       # UNLIMITED, N/A or absent


def _try_time(text):
    try:
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None                                       # "Unknown" while a job has no eligible or start time


def _gres(text: str) -> dict:
    """GPUs by type from a Slurm gres string such as gpu:a100:4(S:0-1) or gpu:a100:3(IDX:0-2),gpu:t4:2."""
    out = {}
    for item in re.sub(r"\([^)]*\)", "", text).split(","):
        p = item.split(":")
        if p[0] == "gpu" and p[-1].isdigit():
            t = p[1] if len(p) == 3 else "gpu"
            out[t] = out.get(t, 0) + int(p[-1])
    return out


def standing_query(window_h: float = 6) -> str:
    """Remote shell, one login-node call: who I am, the cluster clock, the per-partition caps, my queue, the fair-share
    tree, the priority of my pending jobs, the scheduler weights, and the cluster's own load (cores, GPUs, everyone's
    queue, and when recent jobs started). Read-only, and it never fails as a whole: a section the cluster won't give
    simply comes back empty."""
    m = STANDING_MARK
    mins = int(round(window_h * 60))
    return "; ".join([
        f'echo "{m} me"; echo "$USER"; sacctmgr -nP show assoc where user=$USER format=Account 2>/dev/null | head -1',
        f'echo "{m} now"; date +%Y-%m-%dT%H:%M:%S',
        f'echo "{m} qos"; sacctmgr -nP show qos format=Name,MaxJobsPU,MaxSubmitPU 2>/dev/null',
        f'echo "{m} part"; scontrol show partition -o 2>/dev/null | '
        "awk '{for (i = 1; i <= NF; i++) if ($i ~ /^(PartitionName|QoS|MaxTime)=/) printf \"%s \", $i; print \"\"}'",
        f'echo "{m} queue"; squeue --me -h -o "%P|%T|%r|%V|%i|%j" 2>/dev/null',
        f'echo "{m} share"; sshare -a -n -P -o Account,User,RawShares,NormShares,RawUsage,NormUsage,EffectvUsage,'
        'FairShare,LevelFS 2>/dev/null',
        f'echo "{m} prio"; sprio -h -u $USER -o "%i %Y %A %F %J %Q" 2>/dev/null',
        f'echo "{m} config"; scontrol show config 2>/dev/null | grep -E "^(PriorityDecayHalfLife|PriorityWeight[A-Za-z]*) "',
        # the cluster itself: cores allocated/idle per queue, GPUs per node, everyone's jobs (counted, never named),
        # and when the jobs that started in the window had been waiting
        f'echo "{m} cores"; sinfo -h -o "%R|%C" 2>/dev/null',
        f'echo "{m} gpus"; sinfo -h -N -O "Partition:24,NodeList:16,StateLong:16,Gres:48,GresUsed:64" 2>/dev/null',
        f'echo "{m} cqueue"; squeue -h -a -o "%P|%T|%r|%V|%i|%u" 2>/dev/null',
        f'echo "{m} waits"; sacct -a -X -n -P -S now-{mins}minutes -E now -s R,CD,F,TO '
        "-o Partition,Submit,Eligible,Start 2>/dev/null",
        "true"])


def _num(x):
    try:
        return float(str(x).strip())
    except ValueError:
        return None


def parse_standing(out: str, cfg: dict, users: bool = False) -> dict:
    """The queue's pressure against the live caps, and the account's fair-share standing, each with a label
    (GOOD, OK, POOR) and an overall verdict. Colleagues are named only with users=True."""
    sec, cur = {}, None
    for line in out.splitlines():
        if line.startswith(STANDING_MARK + " "):
            cur = line[len(STANDING_MARK) + 1:].strip()
            sec[cur] = []
        elif cur and line.strip():
            sec[cur].append(line.rstrip())
    th = standing_thresholds(cfg)
    me = sec.get("me", [])
    user, account = (me[0].strip() if me else ""), (me[1].strip() if len(me) > 1 else "")
    try:
        now = datetime.fromisoformat(sec["now"][0].strip())
    except (KeyError, IndexError, ValueError):
        now = None

    # ---- queue
    caps = {}
    for line in sec.get("qos", []):
        f = line.split("|")
        if len(f) >= 3:
            caps[f[0]] = {"max_jobs": int(f[1]) if f[1].isdigit() else None, "max_submit": int(f[2]) if f[2].isdigit() else None}
    part_qos, part_limit = {}, {}
    for line in sec.get("part", []):
        d = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
        if d.get("PartitionName"):
            part_qos[d["PartitionName"]] = None if d.get("QoS") in (None, "N/A") else d["QoS"]
            part_limit[d["PartitionName"]] = _hours(d.get("MaxTime"))
    parts, waiting_long = {}, 0
    for name in part_qos:                    # every queue is listed, the idle ones too
        parts[name] = {"partition": name, "running": 0, "pending": 0, "reasons": {}, "oldest_wait_h": None}
    for line in sec.get("queue", []):
        f = line.split("|")
        if len(f) < 6:
            continue
        names, state, reason, submit, jobid = f[0].split(","), f[1], f[2].strip().strip("()"), f[3], f[4]
        age = None
        if state == "PENDING":
            try:
                age = (now - datetime.fromisoformat(submit)).total_seconds() / 3600
            except (TypeError, ValueError):
                pass
            waiting_long += age is not None and age >= th["wait_hours"]
        for part in names:                    # a job submitted to several queues waits in each of them
            q = parts.setdefault(part, {"partition": part, "running": 0, "pending": 0, "reasons": {}, "oldest_wait_h": None})
            if state in RUN_STATES:
                q["running"] += 1
            elif state == "PENDING":
                n = array_count(jobid)
                q["pending"] += n
                q["reasons"][reason] = q["reasons"].get(reason, 0) + n
                if age is not None:
                    q["oldest_wait_h"] = round(max(age, q["oldest_wait_h"] or 0), 1)
    blocked = 0
    near = []
    for q in parts.values():
        lim = caps.get(part_qos.get(q["partition"]) or q["partition"], {})
        q["cap"], q["submit_cap"], q["limit_h"] = lim.get("max_jobs"), lim.get("max_submit"), part_limit.get(q["partition"])
        blocked += sum(n for r, n in q["reasons"].items() if r.startswith(CAP_REASONS))
        if q["cap"] and q["running"] >= th["near_cap"] * q["cap"]:
            near.append(f"{q['partition']} {q['running']}/{q['cap']}")
    if "queue" not in sec:       # no section at all: the query failed (an empty section means no jobs)
        qlabel, qnote = "unknown", "queue unavailable"
    elif blocked:
        qlabel, qnote = "POOR", f"{blocked} job{'s' if blocked > 1 else ''} held back by a cap"
    elif near or waiting_long:
        qlabel = "OK"
        qnote = ", ".join(([f"near the running-job cap: {', '.join(near)}"] if near else [])
                          + ([f"{waiting_long} waiting over {th['wait_hours']:g} h"] if waiting_long else []))
    else:
        qlabel, qnote = "GOOD", "no cap pressure"
    queue = {"label": qlabel, "note": qnote, "blocked_by_caps": blocked, "waiting_long": int(waiting_long),
             "wait_hours": th["wait_hours"], "partitions": sorted(parts.values(), key=lambda q: q["partition"])}

    # ---- fair share
    conf = {k.strip(): v.strip() for k, v in (l.split("=", 1) for l in sec.get("config", []) if "=" in l)}
    weights = {k[len("PriorityWeight"):].lower(): int(v) for k, v in conf.items() if k.startswith("PriorityWeight") and v.isdigit()}
    half = time_hours(conf["PriorityDecayHalfLife"]) / 24 if "PriorityDecayHalfLife" in conf else None
    rows = [l.split("|") for l in sec.get("share", []) if l.count("|") >= 8]
    if not account:
        account = next((r[0].strip() for r in rows if r[1] == user), "")
    accts = [r for r in rows if r[1] == "" and r[0].strip() not in ("", "root")]
    mine_a = next((r for r in accts if r[0].strip() == account), None)
    members = [r for r in rows if r[0].strip() == account and r[1] != ""]
    mine_u = next((r for r in members if r[1] == user), None)
    prio = [r for r in (l.split() for l in sec.get("prio", [])) if len(r) >= 6 and r[1].isdigit()]
    fair = {"label": "unknown"}
    if mine_a and mine_u:
        share, eff, level, factor = _num(mine_a[3]), _num(mine_a[6]), _num(mine_a[8]), _num(mine_u[7])
        total_usage = sum(_num(r[4]) or 0 for r in members)
        mine_usage = _num(mine_u[4]) or 0
        label = ("unknown" if factor is None else "GOOD" if factor >= th["fairshare_good"]
                 else "OK" if factor >= th["fairshare_low"] else "POOR")
        fair = {"label": label, "factor": factor, "level": level,
                "thresholds": {"good": th["fairshare_good"], "low": th["fairshare_low"]},
                "ratio": eff / share if share and eff is not None else None,
                "accounts_above": sum(1 for r in accts if (_num(r[6]) or 0) > (eff or 0)), "accounts_total": len(accts),
                "account_usage": _num(mine_a[4]), "n_users": len(members),
                "your_usage_pct": 100 * mine_usage / total_usage if total_usage else 0.0,
                "others_pct": 100 - 100 * mine_usage / total_usage if total_usage else 0.0,
                "half_life_days": half, "weights": weights,
                "max_fairshare_points": weights.get("fairshare"),
                "fairshare_points": round(factor * weights["fairshare"]) if factor is not None and "fairshare" in weights else None,
                "max_priority": sum(weights.get(k, 0) for k in ("age", "fairshare", "jobsize", "partition", "qos")) or None,
                "recovery_days": half * math.log2(1 / level) if half and level and 0 < level < 1 else None,
                "priority": ({k: round(sum(float(r[i]) for r in prio) / len(prio))
                              for k, i in (("priority", 1), ("age", 2), ("fairshare", 3), ("jobsize", 4), ("qos", 5))}
                             if prio else None)}
        if users:
            fair["users"] = [{"user": r[1], "usage_pct": round(100 * (_num(r[4]) or 0) / total_usage, 1) if total_usage else 0.0,
                              "factor": _num(r[7])} for r in sorted(members, key=lambda r: -(_num(r[4]) or 0))]

    # ---- cluster: everyone's load on each queue you can use (other people's jobs are counted, never named)
    cl = {}
    for name in list(part_qos) or [l.split("|")[0].strip() for l in sec.get("cores", [])]:
        cl[name.rstrip("*")] = {"partition": name.rstrip("*"), "running": 0, "waiting": 0, "held": 0, "oldest_waiting_h": None,
                                "started": 0, "median_wait_min": None, "p90_wait_min": None, "cores_busy_pct": None, "gpus": {}}
    users = set()
    for line in sec.get("cqueue", []):
        f = line.split("|")
        if len(f) < 6:
            continue
        users.add(f[5].strip())
        state, reason, n = f[1], f[2].strip().strip("()"), array_count(f[4])
        waits_on_scheduler = reason in WAITING or reason.startswith("Nodes required")
        age = None
        if state == "PENDING" and waits_on_scheduler:
            try:
                age = (now - datetime.fromisoformat(f[3])).total_seconds() / 3600
            except (TypeError, ValueError):
                pass
        for name in f[0].split(","):
            p = cl.get(name.strip().rstrip("*"))
            if p is None:                      # a queue you can't use
                continue
            if state in RUN_STATES:
                p["running"] += 1
            elif state == "PENDING":
                p["waiting" if waits_on_scheduler else "held"] += n
                if age is not None:
                    p["oldest_waiting_h"] = round(max(age, p["oldest_waiting_h"] or 0), 1)
    starts = {}
    for line in sec.get("waits", []):          # Partition|Submit|Eligible|Start; only starts inside the window say how long jobs wait now
        f = line.split("|")
        try:
            start = datetime.fromisoformat(f[3])
            if now is None or start < now - timedelta(hours=th["wait_window_h"]):
                continue
        except (IndexError, ValueError):
            continue                           # no start time yet: still pending
        ready = next((t for t in (_try_time(f[2]), _try_time(f[1])) if t), None)     # when it became eligible, else when it was submitted
        if ready:
            starts.setdefault(f[0].strip().rstrip("*"), []).append(max((start - ready).total_seconds() / 60, 0))
    for name, v in starts.items():
        if name in cl:
            v.sort()
            cl[name]["started"] = len(v)
            if len(v) >= MIN_STARTS:
                cl[name]["median_wait_min"], cl[name]["p90_wait_min"] = statistics.median(v), v[int(0.9 * (len(v) - 1))]
    for line in sec.get("cores", []):          # allocated/idle/other/total: cores on down or drained nodes ("other") are not capacity
        name, _, c = line.partition("|")
        try:
            alloc, idle = (int(x) for x in c.split("/")[:2])
        except ValueError:
            continue
        if name.strip().rstrip("*") in cl:
            cl[name.strip().rstrip("*")]["cores_busy_pct"] = 100 * alloc / (alloc + idle) if alloc + idle else None
    for line in sec.get("gpus", []):           # Partition NodeList State Gres GresUsed, one line per node and queue
        f = line.split()
        if len(f) < 5 or f[2].lower().startswith(DOWN) or f[0].rstrip("*") not in cl:
            continue
        used = _gres(f[4])
        for t, n in _gres(f[3]).items():
            g = cl[f[0].rstrip("*")]["gpus"].setdefault(t, {"total": 0, "used": 0, "free": 0})
            g["total"] += n
            g["used"] += min(used.get(t, 0), n)
            g["free"] = g["total"] - g["used"]
    loaded = "cqueue" in sec and "waits" in sec
    hidden = bool(user) and users == {user}     # every job in the scheduler's view is mine: other people's are not shown
    for p in cl.values():
        med = p["median_wait_min"]
        p["label"] = ("unknown" if not loaded or hidden else
                      "unknown" if med is None and p["waiting"] else "GOOD" if med is None else
                      "GOOD" if med < th["cluster_wait_ok_min"] else "OK" if med < th["cluster_wait_poor_min"] else "POOR")
    known_c = [p for p in cl.values() if p["label"] in RANK]
    if not loaded or not cl:
        clabel, cnote = "unknown", "cluster load unavailable"
    elif hidden:
        clabel, cnote = "unknown", "the scheduler shows only your own jobs"
    elif not known_c:
        clabel, cnote = "unknown", "too few recent starts to judge the waits"
    else:
        clabel = max((p["label"] for p in known_c), key=RANK.get)
        cnote = ", ".join(f"{p['partition']} median wait {fmt_dur(p['median_wait_min'])}" for p in sorted(known_c, key=lambda p: p["partition"])
                          if p["label"] != "GOOD") or "queues start jobs quickly"
    cluster = {"label": clabel, "note": cnote, "usable": loaded and not hidden and bool(cl), "window_h": th["wait_window_h"],
               "thresholds": {"ok_min": th["cluster_wait_ok_min"], "poor_min": th["cluster_wait_poor_min"]},
               "partitions": sorted(cl.values(), key=lambda p: p["partition"])}

    # ---- advice: what an agent (or you) should change, given the readings above; empty when all is well
    advice = []
    if blocked:
        advice.append({"code": "caps", "kind": "plan", "text": f"{blocked} job{'s' if blocked > 1 else ''} held back by a per-user cap: pack rows with "
                                               "--rows-per-job instead of adding jobs, and let the queue drain before submitting more."})
    if fair["label"] in ("OK", "POOR"):
        advice.append({"code": "fairshare", "kind": "plan", "text": f"Fair share is {fair['label']} ({fair['factor']:.2f}) and jobs are billed on what they "
                       "allocate: size CPUs and memory from a pilot's measured use, set --time near the measured runtime, and pilot "
                       "before any large submit."})
    busy = [p for p in cluster["partitions"] if p["label"] == "POOR"]
    if busy:
        advice.append({"code": "busy_queue", "kind": "plan", "text": "Busy right now: " + ", ".join(f"{p['partition']} (median wait {fmt_dur(p['median_wait_min'])})"
                       for p in busy) + ". Ask for a shorter --time so a job can backfill, and never cancel and resubmit a job that is only waiting."})
    groups = {}
    for p in cluster["partitions"]:            # queues that share nodes show the same GPUs: report them once
        if p["gpus"]:
            groups.setdefault(tuple(sorted((t, g["total"], g["free"]) for t, g in p["gpus"].items())), []).append(p["partition"])
    def few(items, n):                         # at most n names, then how many were left out
        return ", ".join(items[:n]) + (f" (+{len(items) - n} more)" if len(items) > n else "")
    full = [f"No free {few([t for t, total, free in key if total and not free], 3)} GPU on {', '.join(names)}: a job asking for it will wait. "
            f"Free now: {few([f'{t} {free}' for t, total, free in sorted(key, key=lambda k: -k[2]) if free], 4) or 'nothing'}."
            for key, names in groups.items() if any(total and not free for t, total, free in key)]
    if full:
        advice.append({"code": "gpu_full", "kind": "note", "text": " ".join(full)})

    # ---- verdict
    known = [x["label"] for x in (queue, fair) if x["label"] in RANK]
    overall = max(known, key=RANK.get) if known else "unknown"
    if fair["label"] == "unknown":
        fpart, fbrief = "fair share unavailable", "fair share unknown"
    else:
        ratio = f"account at {fair['ratio']:.1f}x its share, " if fair["ratio"] is not None else ""
        fpart = (f"fair share {fair['label']} ({fair['factor']:.2f}: {ratio}"
                 f"{fair['accounts_above']} of {fair['accounts_total']} accounts use more)")
        fbrief = f"fair share {fair['label'].lower()} {fair['factor']:.2f}"
    return {"user": user, "account": account, "cluster_time": now.isoformat(timespec="minutes") if now else None,
            "overall": overall, "queue": queue, "fairshare": fair, "cluster": cluster, "advice": advice,
            "verdict": f"{overall} — {fpart} · queue {qlabel} ({qnote})",
            "brief": f"{fbrief}, queue {qlabel.lower()}"}


STYLE = {"red": "31", "green": "32", "yellow": "33", "cyan": "36", "bold": "1", "dim": "2", "rev": "7"}
LABEL_STYLE = {"GOOD": "green", "OK": "yellow", "POOR": "red", "unknown": "dim"}
ASCII = str.maketrans({"─": "-", "█": "#", "░": ".", "·": "|", "—": "-", "≥": ">", "×": "x", "…": "~"})


def use_color(tty: bool, env) -> bool:
    """Colour on a terminal, unless NO_COLOR is set; FORCE_COLOR turns it on elsewhere (a pager, a test)."""
    if env.get("NO_COLOR"):
        return False
    return bool(env.get("FORCE_COLOR")) or tty


def unicode_ok(env, encoding) -> bool:
    """Unicode needs both a UTF-8 locale (LC_ALL, else LC_CTYPE, else LANG; none set counts as UTF-8) and a stream that can
    encode it. Python coerces a C locale to UTF-8 for its streams, so the stream alone would never say no."""
    locale = next((env[k] for k in ("LC_ALL", "LC_CTYPE", "LANG") if env.get(k)), "")
    return bool(encoding) and encoding.lower().replace("-", "") .startswith("utf") and (not locale or "utf" in locale.lower())


def bar(frac, width: int, uni: bool = True) -> str:
    """A gauge of `width` cells filled to `frac` (0 to 1); a nonzero fraction always shows at least one cell."""
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))
    n = max(1, round(frac * width)) if frac > 0 else 0
    full, empty = ("█", "░") if uni else ("#", ".")
    return full * n + empty * (width - n)


def render_standing(s: dict, color: bool = False, width: int = 80, uni: bool = True) -> str:
    """The report, in the look of `npl`: a title line, centred rules, indented rows, gauges. Every row is cut to the
    terminal width, and colour only wraps text in escape codes, so stripping them leaves exactly the plain report."""
    def paint(text, *styles):
        return f"\033[{';'.join(STYLE[x] for x in styles)}m{text}\033[0m" if color and styles else text

    def row(*parts):          # parts: "text" or ("text", "style", ...); the row never passes `width` columns
        room, out = width, []
        for part in parts:
            text, styles = (part, ()) if isinstance(part, str) else (part[0], part[1:])
            if room <= 0:
                break
            if len(text) > room:
                text = text[:room - 1] + "…"
            room -= len(text)
            out.append(paint(text, *styles))
        return "".join(out)

    def rule(title):
        left = (width - len(title) - 2) // 2
        return paint("─" * left, "dim") + f" {paint(title, 'bold')} " + paint("─" * (width - left - len(title) - 2), "dim")

    f, q = s["fairshare"], s["queue"]
    gw = 20 if width >= 72 else 12 if width >= 60 else 8        # gauge width
    nw = max([10] + [len(p["partition"]) + 1 for p in q["partitions"] + s["cluster"]["partitions"]])   # name column
    gq = max(8, min(gw, width - nw - 52))                        # gauge width on the queue and cluster rows
    users = f" · {f['n_users']} users" if f.get("n_users") else ""
    clock = f" · cluster time {s['cluster_time'].replace('T', ' ')}" if s["cluster_time"] else ""
    out = ["", row(("Explorer standing", "bold"), (clock, "dim")), "",
           row("  ", (f"[{s['overall'].upper()}]", LABEL_STYLE.get(s["overall"], "dim"), "bold", "rev"), "  " + s["brief"].replace(", ", " · ")),
           row("  ", (f"{s['user'] or '?'} · account {s['account'] or '?'}{users}", "dim")), "", rule("Queue")]
    for p in q["partitions"]:
        cap, hot = p["cap"], bool(p["cap"] and p["running"] >= 0.8 * p["cap"])
        jobs = f"{p['running']} of {cap} running" if cap else f"{p['running']} running"
        out.append(row(f"  {p['partition']:<{nw}} ", (bar(p["running"] / cap if cap else None, gq, uni), "red" if hot else "green" if p["running"] else "dim"),
                       f"  {jobs:<17}", (f"{p['pending']} pending", "yellow" if p["pending"] else "dim"),
                       (f" · {fmt_limit(p['limit_h'])} max" if p.get("limit_h") else "", "dim")))
        if p["reasons"]:         # why they wait, and for how long, on a line of their own so it is never cut off
            why = ", ".join("nodes down or reserved" if r.startswith("Nodes required for job") else r for r in p["reasons"])
            out.append(row(" " * (nw + 3), (why + (f" · oldest {p['oldest_wait_h']:g} h" if p["oldest_wait_h"] is not None else ""), "yellow")))
    if not q["partitions"]:
        out.append(row("  ", ("no jobs in the queue", "dim")))
    out += [row(f"  {'status':<{nw}} ", (q["label"], LABEL_STYLE.get(q["label"], "dim"), "bold"), (f" · {q['note']}", "dim")), "", rule("Cluster")]
    c = s["cluster"]
    if c["usable"]:
        for p in c["partitions"]:
            lab, busy, n = LABEL_STYLE.get(p["label"], "dim"), p["cores_busy_pct"], p["started"]
            out.append(row(f"  {p['partition']:<{nw}} ", (bar(None if busy is None else busy / 100, gq, uni), lab),
                           f"  {f'{busy:.0f}% busy' if busy is not None else 'load unknown':<9}", (f" {p['label']:<7}", lab, "bold"),
                           f" {p['waiting']} waiting", (f" · {p['held']} held" if p["held"] else "", "dim")))
            bits = [f"median wait {fmt_dur(p['median_wait_min'])} · slowest tenth {fmt_dur(p['p90_wait_min'])}" if p["median_wait_min"] is not None
                    else f"only {n} job{'s' if n != 1 else ''} started in the last {c['window_h']:g} h" if n
                    else f"no job started in the last {c['window_h']:g} h"]
            if p["oldest_waiting_h"] is not None:
                bits.append(f"oldest waiting {fmt_dur(p['oldest_waiting_h'] * 60)}")
            if len(" · ".join(bits)) > width - nw - 3:        # too long for one line: one fact per line, none cut off
                out += [row(" " * (nw + 3), (b, "dim")) for b in bits]
            else:
                out.append(row(" " * (nw + 3), (" · ".join(bits), "dim")))
    else:
        out.append(row("  ", (c["note"], "dim")))
    groups = {}
    for p in c["partitions"]:                  # queues that share nodes share their GPUs: list them once
        if p["gpus"]:
            groups.setdefault(tuple(sorted((t, g["total"], g["free"]) for t, g in p["gpus"].items())), []).append(p["partition"])
    for key, names in groups.items():
        out.append(row(f"  {'free GPUs':<{nw}} ", (" · ".join(names), "dim")))
        line, used = [], 0
        for t, total, free in key:             # a type with none free is red, a quarter or less yellow
            item = f"{t} {free} of {total}"
            if line and used + 3 + len(item) > width - nw - 3:
                out.append(row(" " * (nw + 3), *line))
                line, used = [], 0
            used += len(item) + (3 if line else 0)
            line += ([(" · ", "dim")] if line else []) + [(item, "red" if not free else "yellow" if free <= total / 4 else "green")]
        out.append(row(" " * (nw + 3), *line))
    out += [row(f"  {'status':<{nw}} ", (c["label"], LABEL_STYLE.get(c["label"], "dim"), "bold"), (f" · {c['note']}", "dim")), "", rule("Fair share")]
    if f["label"] == "unknown":
        out.append(row("  ", ("the cluster did not return the fair-share tree (sshare)", "dim")))
    else:
        th, lab = f["thresholds"], LABEL_STYLE[f["label"]]
        out.append(row(f"  {'factor':<9}  ", (bar(f["factor"], gw, uni), lab), " ", (f"{f['factor']:.3f}", lab, "bold"),
                       (f"   good ≥ {th['good']:g} · ok ≥ {th['low']:g}", "dim")))
        if f["ratio"] is not None:
            out.append(row(f"  {'usage':<9}  ", (f"{f['ratio']:.1f}x", "bold"),
                           f" its equal share · {f['accounts_above']} of {f['accounts_total']} accounts use more"))
            if f["recovery_days"]:
                out.append(row(f"  {'recovery':<9}  ", (f"about {f['recovery_days']:.0f} days", "bold"),
                               f" of no use to reach 1x (half-life {f['half_life_days']:g} d)"))
        others = max(f["n_users"] - 1, 0)
        out.append(row(f"  {'account':<9}  you ", (bar(f["your_usage_pct"] / 100, gw, uni), "cyan"), f" {f['your_usage_pct']:.0f}%",
                       (f" · the other {others} user{'' if others == 1 else 's'} {f['others_pct']:.0f}%", "dim")))
        for u in f.get("users", []):
            out.append(row((f"             {u['user']:<18} {u['usage_pct']:>5.1f}%  factor {u['factor'] if u['factor'] is not None else '-'}", "dim")))
        if f["fairshare_points"] is not None:
            out.append(row(f"  {'priority':<9}  fair share gives a new job {f['fairshare_points']:,} of {f['max_fairshare_points']:,} points"))
        if f["priority"]:
            pr, n = f["priority"], sum(x["pending"] for x in q["partitions"])
            out.append(row((f"             {n} pending avg {pr['priority']:,} of {f['max_priority']:,} "
                            f"(fair share {pr['fairshare']:,}, age {pr['age']:,})", "dim")))
    if s["advice"]:
        out += ["", rule("Advice")]
        for a in s["advice"]:
            for i, text in enumerate(textwrap.wrap(a["text"], width - 4, break_on_hyphens=False, break_long_words=False)):
                out.append(row(("  - " if i == 0 else "    ") + text))
    out += ["", row(("  --users names the others · --brief one line · --json for scripts", "dim")), ""]
    text = "\n".join(out)
    return text if uni else text.translate(ASCII)


# --------------------------------------------------------------------------- commands

EXAMPLE_CONFIG = (Path(__file__).parent / "example_config.toml")


def cmd_init(args, cfg, R):
    if CONFIG_PATH.exists() and not args.force:
        raise NuhpcError(f"{CONFIG_PATH} exists (use --force to overwrite)")
    text = EXAMPLE_CONFIG.read_text()
    if args.user_dir:
        if not args.user_dir.startswith("/"):
            raise NuhpcError("--user-dir must be an absolute path on the cluster, e.g. /projects/<group>/<you>")
        text = text.replace("/projects/YOUR_GROUP/YOUR_USER", args.user_dir.rstrip("/"))
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(text)
    emit({"config": str(CONFIG_PATH), "next": ("add the ~/.ssh/config block from the README, then `nuhpc connect` "
          "and `nuhpc check --smoke`" if args.user_dir else "set user_dir (your folder on project storage) and "
          "ssh_host, add the ~/.ssh/config block from the README, then `nuhpc connect` and `nuhpc check --smoke`")}, args)


def cmd_connect(args, cfg, R):
    # Interactive on purpose: this is where a human types a password / approves Duo.
    # With ControlMaster+ControlPersist in ~/.ssh/config, later BatchMode calls reuse it.
    argv = ["ssh", "-fN", cfg["ssh_host"]]
    if args.dry_run:
        print("[dry-run] " + " ".join(argv), file=sys.stderr)
        return
    rc = subprocess.call(argv)
    if rc != 0:
        raise NuhpcError("ssh connect failed")
    emit({"connected": cfg["ssh_host"]}, args)


def cmd_check(args, cfg, R):
    project = getattr(args, "project", None)
    home, root = shlex.quote(user_dir(cfg)), shlex.quote(remote_path(cfg, project=project))
    # The work root is created only when user_dir exists and is writable: a typo must not become a new folder.
    out = R.run(f"echo ok; hostname; "
                f"if [ ! -d {home} ]; then echo 'user_dir: MISSING'; elif [ ! -w {home} ]; then echo 'user_dir: not writable'; "
                f"else echo 'user_dir: ok'; fi; "
                f"if [ -d {home} ] && [ -w {home} ]; then mkdir -p {root} && df -h {root} | tail -1; else echo '(not checked)'; fi; "
                "sinfo -h -o '%P|%l|%G|%a' | sort -u")
    lines = out.splitlines()
    status = lines[2].split(": ", 1)[-1] if len(lines) > 2 else ""
    if lines and status != "ok":
        raise NuhpcError(f"user_dir {user_dir(cfg)} is {status} on the cluster: fix user_dir in {CONFIG_PATH}")
    info = {"ssh": "ok" if lines[:1] == ["ok"] else "dry-run" if args.dry_run else "FAILED",
            "login_node": lines[1] if len(lines) > 1 else "", "user_dir": status,
            "work_root": remote_path(cfg, project=project), "remote_root_disk": lines[3] if len(lines) > 3 else "",
            **({"projects": {n: pr["dir"] for n, pr in cfg["projects"].items()}} if cfg["projects"] else {}),
            "partitions": [dict(zip(("partition", "timelimit", "gres", "avail"), l.split("|")))
                           for l in lines[4:] if "|" in l]}
    if args.smoke:
        args.template, args.name, args.param, args.sweep, args.grid = "smoke", "smoke", [], None, None
        args.code, args.max_parallel, args.confirm_big = None, None, False
        info["smoke_submit"] = cmd_submit(args, cfg, R, quiet=True)
    if args.json or not info["partitions"]:
        emit(info, args)
    else:
        emit({k: v for k, v in info.items() if k != "partitions"}, args)
        print()
        emit(info["partitions"], args)


def cmd_templates(args, cfg, R):
    rows = []
    for p in all_templates(cfg):
        doc = " ".join(l[len("# doc:"):].strip() for l in p.read_text().splitlines() if l.startswith("# doc:"))
        rows.append({"template": p.stem, "description": doc})
    emit(rows, args)


def cmd_submit(args, cfg, R, quiet=False, tasks=None):
    template = find_template(cfg, args.template)
    if tasks is None:
        tasks = build_tasks(args.sweep, args.grid, parse_kv(args.param))
    rpj, par = getattr(args, "rows_per_job", None), getattr(args, "parallel", None)
    if par and not rpj:
        raise NuhpcError("--parallel needs --rows-per-job (how many rows share a job)")
    keys, packs = [], None
    if rpj is not None:
        body = template.read_text()
        if getattr(args, "pack", False):
            raise NuhpcError("--rows-per-job and --pack both group rows; use one (--pack is for templates that share a server)")
        if rpj < 1:
            raise NuhpcError("--rows-per-job must be at least 1")
        if "# pack-by:" in body:
            raise NuhpcError(f"template '{template.stem}' groups rows by model or server: use --pack, not --rows-per-job")
        if "hpc_each_row" not in body:
            raise NuhpcError(f"template '{template.stem}' does not support --rows-per-job (it does not run its work per row)")
        packs = [list(range(i, min(i + rpj, len(tasks)))) for i in range(0, len(tasks), rpj)]
    elif getattr(args, "pack", False):
        keys = pack_keys(template)
        if not keys:
            raise NuhpcError(f"template '{template.stem}' does not support --pack (no '# pack-by:' line)")
        packs = pack_rows(tasks, keys)
    n = len(packs) if packs else len(tasks)   # array tasks: what reaches the queue and the limits
    prof, res = resolve_resources(cfg, args)
    job_res, waves = res, None                # res is what one row asks for; job_res is what sbatch gets
    if rpj:
        if par is None:
            par = 1 if gpus_per_node(res) else rpj
        if par < 1:
            raise NuhpcError("--parallel must be at least 1")
        if par > rpj:
            raise NuhpcError(f"--parallel {par} cannot exceed --rows-per-job {rpj}")
        if par > 1:
            if gpus_per_node(res):
                raise NuhpcError("--parallel above 1 runs rows side by side on CPUs. A GPU job runs its rows one at a "
                                 "time: drop --parallel, or loop over the rows inside your script.")
            if int(res["nodes"]) != 1:
                raise NuhpcError("rows run side by side on one node: use --nodes 1 with --rows-per-job")
            job_res = {**res, "cpus": int(res["cpus"]) * par, "mem": scale_mem(res["mem"], par)}
        waves = -(-rpj // par)
    gpu_h = gpus_per_node(job_res) * int(job_res["nodes"]) * time_hours(job_res["time"]) * n
    lim = cfg["limits"]
    if lim.get("max_tasks") and n > lim["max_tasks"] and not args.confirm_big:
        raise NuhpcError(f"{n} tasks exceeds limits.max_tasks={lim['max_tasks']}. Needs --confirm-big (human approval).")
    if lim.get("max_gpu_hours_per_submit") and gpu_h > lim["max_gpu_hours_per_submit"] and not args.confirm_big:
        raise NuhpcError(f"worst-case {gpu_h:.1f} GPU-hours exceeds limits.max_gpu_hours_per_submit="
                         f"{lim['max_gpu_hours_per_submit']}. Needs --confirm-big (human approval).")
    warnings = check_job_shape(cfg, res, job_res, n, len(tasks), args.confirm_big)
    for w in warnings:
        print(f"[nuhpc] warning: {w}", file=sys.stderr)
    max_par = args.max_parallel or lim.get("default_max_parallel")

    run_id = f"{slug(args.name or template.stem)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    project = getattr(args, "project", None)
    run_dir = remote_path(cfg, f"runs/{run_id}", project)
    stage = STATE_DIR / "runs" / run_id
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "job.sbatch").write_text(render(cfg, template, job_res, n, max_par, run_id, run_dir,
                                             slug(args.name or template.stem), project,
                                             parallel=par if rpj else 1, row_cpus=int(res["cpus"])))
    (stage / "sweep.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    if packs:
        (stage / "packs.json").write_text(json.dumps(packs))
    meta = {"run_id": run_id, "template": template.stem, "project": project,
            "profile": prof, "resources": job_res, "n_tasks": n,
            "n_rows": len(tasks), "pack_by": keys or None,
            "rows_per_job": rpj, "parallel": par if rpj else None,
            "max_parallel": max_par, "worst_case_gpu_hours": round(gpu_h, 2), "code": args.code,
            "created": datetime.now().isoformat(timespec="seconds")}
    (stage / "meta.json").write_text(json.dumps(meta, indent=2))

    host = cfg["ssh_host"]
    if args.code:
        code = Path(args.code).expanduser().resolve()
        if not code.is_dir():
            raise NuhpcError(f"--code must be a directory: {code}")
    R.run(f"mkdir -p {shlex.quote(run_dir)}/logs {shlex.quote(run_dir)}/outputs {shlex.quote(run_dir)}/code "
          f"{shlex.quote(run_dir)}/usage")
    R.rsync(f"{stage}/", f"{host}:{run_dir}/")
    if args.code:
        ex = [f"--exclude={e}" for e in DEFAULT_EXCLUDES]
        if (code / ".nuhpcignore").exists():
            ex.append(f"--exclude-from={code / '.nuhpcignore'}")
        R.rsync(f"{code}/", f"{host}:{run_dir}/code/", ex)
    try:
        out = R.run(f"cd {shlex.quote(run_dir)} && sbatch --parsable job.sbatch", retry=False)
    except NuhpcError as first:
        # Slurm sometimes accepts a job and still reports an error ("Unexpected message received"). Look for a job already queued from
        # this run's folder before the caller retries: a repeat would queue it twice.
        try:
            out = R.run(f"squeue --me -h -o '%i|%Z' | grep -F -- {shlex.quote('|' + run_dir)} | head -1 | cut -d'|' -f1")
        except NuhpcError:
            raise first
        if not out.strip():
            raise first
        print(f"[nuhpc] warning: sbatch reported an error, but job {out.strip()} exists for this run; using it", file=sys.stderr)
    job_id = out.strip().split(";")[0] if out.strip() else "DRY-RUN"
    result = {**meta, "per_row_resources": res if rpj else None, "waves": waves, "warnings": warnings,
              "job_id": job_id, "remote_dir": run_dir, "local_stage": str(stage)}
    if not args.dry_run:
        ledger_append({k: result[k] for k in ("run_id", "job_id", "template", "project", "profile", "n_tasks",
                                             "remote_dir", "created", "worst_case_gpu_hours")})
    if quiet:
        return {"run_id": run_id, "job_id": job_id, "local_stage": str(stage)}
    emit(result, args)


def run_states(entries: list[dict], R) -> tuple[list[dict], dict[str, list[int]]]:
    """Overall state per run from sacct, plus the array task ids that ended badly, per job id."""
    ids = ",".join(e["job_id"] for e in entries)
    out = R.run(f"sacct -X -n -P --format=JobID,State,Elapsed,ExitCode -j {ids}")
    counts: dict[str, dict[str, int]] = {}
    failed: dict[str, list[int]] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        base, _, idx = parts[0].partition("_")
        state = parts[1].split()[0]
        counts.setdefault(base, {}).setdefault(state, 0)
        counts[base][state] += array_count(parts[0])
        if state in TERMINAL and state != "COMPLETED" and not idx.startswith("["):
            failed.setdefault(base, []).append(int(idx or 0))
    rows = []
    for e in entries:
        c = counts.get(e["job_id"], {})
        done = sum(v for k, v in c.items() if k in TERMINAL)
        if not c:
            overall = "SUBMITTED"
        elif c.get("RUNNING"):
            overall = "RUNNING"
        elif c.get("PENDING"):
            overall = "PENDING"
        elif set(c) == {"COMPLETED"}:
            overall = "COMPLETED"
        elif done >= e["n_tasks"]:
            overall = "FINISHED_WITH_FAILURES"
        else:
            overall = "MIXED"
        rows.append({"run_id": e["run_id"], "job_id": e["job_id"], "state": overall,
                     "done": f"{done}/{e['n_tasks']}", "breakdown": ",".join(f"{k}={v}" for k, v in sorted(c.items()))})
    return rows, failed


def cmd_status(args, cfg, R):
    entries = [find_run(r) for r in args.run] if args.run else ledger_read()[-args.last:]
    if not entries:
        return emit([], args)
    emit(run_states(entries, R)[0], args)


def wait_for(entries: list[dict], R, timeout: float | None, interval: float, dry: bool):
    """Poll until every run is done or the timeout passes. Returns (rows, failed, timed_out)."""
    deadline = None if timeout is None else time.monotonic() + timeout
    delay, errors, rows, failed, last = interval, 0, [], {}, None
    while True:
        try:
            rows, failed = run_states(entries, R)
            errors = 0
        except NuhpcError as ex:   # laptop asleep, network blip: keep waiting, but not forever
            errors += 1
            if errors >= 10:
                raise
            print(f"[wait] transport error {errors}/10: {str(ex).splitlines()[-1]}", file=sys.stderr)
        states = [r["state"] for r in rows]
        if states and states != last:
            print(f"[wait] {datetime.now():%H:%M:%S} "
                  + " ".join(f"{r['run_id']}={r['state']}({r['done']})" for r in rows), file=sys.stderr)
            last = states
        if dry or (rows and all(s in DONE for s in states)):
            return rows, failed, False
        if deadline is not None and time.monotonic() >= deadline:
            return rows, failed, True
        time.sleep(delay if deadline is None else max(0.0, min(delay, deadline - time.monotonic())))
        delay = min(delay * 1.5, 300)


def cmd_wait(args, cfg, R):
    entries = [find_run(r) for r in args.run]
    rows, failed, timed_out = wait_for(entries, R, args.timeout, args.interval, args.dry_run)
    if timed_out:
        emit({"timed_out": True, "runs": rows} if args.json else rows, args)
        sys.exit(3)
    ensure_usage(cfg, R, entries)
    for r, e in zip(rows, entries):
        bad = sorted(failed.get(e["job_id"], []))
        if bad:
            r["failed_tasks"] = bad
            try:
                r["failed_log_tail"] = log_tail(R, e, bad[0], args.lines)
            except NuhpcError as ex:
                r["failed_log_tail"] = f"(could not read the log: {ex})"
    if args.json:
        return emit({"timed_out": False, "runs": rows}, args)
    emit([{k: v for k, v in r.items() if k != "failed_log_tail"} for r in rows], args)
    for r in rows:
        if r.get("failed_log_tail"):
            print(f"\n== {r['run_id']}: log of task {r['failed_tasks'][0]} (first failed) ==\n{r['failed_log_tail']}", end="")


def cmd_run(args, cfg, R):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise NuhpcError("nothing to run: nuhpc run [options] -- COMMAND [ARGS...]")
    if args.time is None:
        args.time = "00:30:00"   # a quick test, not the profile's batch walltime
    args.template, args.name = "cmd", args.name or "run"
    args.sweep = args.grid = args.param = args.max_parallel = None
    args.confirm_big = args.pack = False
    sub = cmd_submit(args, cfg, R, quiet=True, tasks=[{"cmd": shlex.join(command)}])
    if args.dry_run:
        return emit(sub, args)
    e = find_run(sub["run_id"])
    rows, _, timed_out = wait_for([e], R, args.timeout, args.interval, False)
    state = rows[0]["state"] if rows else "SUBMITTED"
    if timed_out:
        emit({"run_id": e["run_id"], "job_id": e["job_id"], "state": state, "timed_out": True,
              "next": f"nuhpc wait {e['run_id']}"}, args)
        sys.exit(3)
    log = log_tail(R, e, 0, args.lines)
    dest, _ = fetch_run(cfg, R, e)
    ensure_usage(cfg, R, [e])
    if args.json:
        emit({"run_id": e["run_id"], "job_id": e["job_id"], "state": state, "log": log, "local": str(dest)}, args)
    else:
        print(log, end="")
        print(f"[nuhpc run] {e['run_id']}: {state}; outputs in {dest}", file=sys.stderr)
    if state != "COMPLETED":
        sys.exit(2)


def log_tail(R, e: dict, task: int, n: int, grep: str | None = None) -> str:
    files = f"{shlex.quote(e['remote_dir'])}/logs/*" + (f"_{int(task)}.out" if e["n_tasks"] > 1 else ".out")
    if grep:   # whole log searched, last n matches kept; the pattern is quoted for the remote shell
        return R.run(f"grep -h -E -e {shlex.quote(grep)} {files} | tail -n {int(n)}")
    return R.run(f"tail -n {int(n)} {files}")


def cmd_logs(args, cfg, R):
    e = find_run(args.run)
    out = log_tail(R, e, args.task, args.lines, args.grep)
    if args.json:
        emit({"run_id": e["run_id"], "task": args.task, "log": out}, args)
    else:
        print(out, end="")


def fetch_run(cfg, R, e: dict, include=None, max_size=None) -> tuple[Path, str | None]:
    dest = Path(cfg["local_results"]).expanduser() / e["run_id"]
    dest.mkdir(parents=True, exist_ok=True)
    max_size = max_size or cfg["limits"].get("fetch_max_file_size")
    size = [f"--max-size={max_size}"] if max_size else []
    extra = size + (["-m", "--include=*/", *[f"--include={p}" for p in include], "--exclude=*"] if include else [])
    for sub in ("outputs", "logs"):
        R.rsync(f"{cfg['ssh_host']}:{e['remote_dir']}/{sub}/", f"{dest}/{sub}/", extra)
    R.rsync(f"{cfg['ssh_host']}:{e['remote_dir']}/meta.json", f"{dest}/meta.json")
    try:   # GPU samples always come along; runs from before usage accounting have no usage/ dir
        R.rsync(f"{cfg['ssh_host']}:{e['remote_dir']}/usage/", f"{dest}/usage/", size)
    except NuhpcError:
        pass
    return dest, max_size


def cmd_fetch(args, cfg, R):
    e = find_run(args.run)
    dest, max_size = fetch_run(cfg, R, e, args.include, args.max_size)
    ensure_usage(cfg, R, [e])
    emit({"run_id": e["run_id"], "local": str(dest), "max_file_size": max_size or "none"}, args)


def cmd_cancel(args, cfg, R):
    e = find_run(args.run)
    target = f"{e['job_id']}_{args.task}" if args.task is not None else e["job_id"]
    R.run(f"scancel {shlex.quote(target)}")
    emit({"cancelled": target, "run_id": e["run_id"]}, args)


def cmd_runs(args, cfg, R):
    rows = [{**{k: e[k] for k in ("run_id", "job_id", "template")}, "project": e.get("project") or "-",
             **{k: e[k] for k in ("profile", "n_tasks", "created")}} for e in ledger_read()[-args.last:]]
    emit(rows, args)


GPU_HOUR_COST = 100       # about how many CPU-hours the scheduler bills for one GPU-hour on Explorer (an estimate, not a published rate)


def waste(tasks: list) -> dict:
    """What a set of finished tasks allocated and did not use, weighted by hours: the share of core-hours that sat idle (CPU-only
    tasks: a GPU job's cores are incidental next to its GPU), of sampled GPU-hours that sat idle, and of billed hours spent on tasks
    that did not complete."""
    cpu_known = cpu_used = gpu_known = gpu_used = billed = lost = 0.0
    for x in tasks:
        if x["cpu_eff"] is not None and not x["gpus"]:
            cpu_known += x["cpu_hours"]
            cpu_used += x["cpu_eff"] * x["cpu_hours"]
        if x["gpu_util_mean"] is not None:
            gpu_known += x["gpu_hours"]
            gpu_used += x["gpu_hours"] * x["gpu_util_mean"] / 100
        hours = x["cpu_hours"] + GPU_HOUR_COST * x["gpu_hours"]
        billed += hours
        lost += hours if x["state"] != "COMPLETED" else 0
    idle = lambda used, known: round(100 * (1 - used / known), 1) if known else "-"
    return {"cpu_idle_%": idle(cpu_used, cpu_known), "gpu_idle_%": idle(gpu_used, gpu_known),
            "failed_%": round(100 * lost / billed, 1) if billed else "-"}


def cmd_usage(args, cfg, R):
    if args.run:
        entries = [find_run(r) for r in args.run]
    else:
        entries = ledger_read()
        if args.since != "all":
            m = re.fullmatch(r"(\d+)([hd])", args.since)
            if not m:
                raise NuhpcError("--since takes a duration such as 48h or 7d, or 'all'")
            cutoff = datetime.now() - timedelta(hours=int(m.group(1)) * (24 if m.group(2) == "d" else 1))
            entries = [e for e in entries if datetime.fromisoformat(e["created"]) >= cutoff]
    if args.project:
        entries = [e for e in entries if e.get("project") == args.project]
    recs = ensure_usage(cfg, R, entries)
    rows, totals, pooled = [], {}, {}
    for e in entries:
        u = recs.get(e["run_id"])
        if not u:
            continue
        t, tot = u["tasks"], u["totals"]
        util = [x["gpu_util_mean"] for x in t if x["gpu_util_mean"] is not None]
        mem = [x["mem_eff"] for x in t if x["mem_eff"] is not None]
        time_eff = ratio(sum(x["elapsed_s"] for x in t), sum(x["time_limit_s"] or 0 for x in t))
        row = {"run_id": u["run_id"], "project": u["project"] or "-",
               "gpu": next((f"{x['gpus']}x{x['gpu_type'] or 'gpu'}" for x in t if x["gpus"]), "-"),
               "states": ",".join(f"{k}={v}" for k, v in tot["states"].items()),
               "wait_min": round(tot["queue_wait_s_max"] / 60, 1) if tot["queue_wait_s_max"] is not None else "-",
               "gpu_h": tot["gpu_hours"], "req_gpu_h": tot["worst_case_gpu_hours"], "cpu_h": tot["cpu_hours"],
               "time_%": round(100 * time_eff) if time_eff is not None else "-",
               "gpu_util_%": round(sum(util) / len(util)) if util else "-",
               "mem_%": round(100 * max(mem)) if mem else "-", "cpu_idle_%": waste(t)["cpu_idle_%"]}
        rows.append(row)
        for key in (row["project"], "all"):
            g = totals.setdefault(key, {"project": key, "runs": 0, "gpu_h": 0.0, "req_gpu_h": 0.0, "cpu_h": 0.0})
            g["runs"] += 1
            for f in ("gpu_h", "req_gpu_h", "cpu_h"):
                g[f] = round(g[f] + row[f], 2)
            pooled.setdefault(key, []).extend(t)
    for key, g in totals.items():
        g.update(waste(pooled[key]))
    totals = sorted(totals.values(), key=lambda g: g["project"] == "all")
    unfinished = [e["run_id"] for e in entries if e["run_id"] not in recs]
    if args.json:
        return emit({"runs": rows, "totals": totals, "unfinished": unfinished,
                     "records": str(STATE_DIR / "usage")}, args)
    emit(rows, args)
    if rows:
        print()
        emit(totals, args)
    if unfinished:
        print(f"[nuhpc] {len(unfinished)} run(s) not finished yet, not counted", file=sys.stderr)


def cmd_standing(args, cfg, R):
    s = parse_standing(R.run(standing_query(standing_thresholds(cfg)["wait_window_h"])), cfg, users=args.users)
    if args.json:
        return emit(s, args)
    color = use_color(sys.stdout.isatty(), os.environ)
    uni = unicode_ok(os.environ, getattr(sys.stdout, "encoding", None))
    if args.brief:
        head, _, rest = s["verdict"].partition(" ")
        line = (f"\033[1;{STYLE[LABEL_STYLE.get(head, 'dim')]}m{head}\033[0m {rest}" if color else s["verdict"])
        return print(line if uni else line.translate(ASCII))
    width = min(100, max(50, shutil.get_terminal_size((80, 24)).columns))
    print(render_standing(s, color=color, width=width, uni=uni))


def cmd_queue(args, cfg, R):
    out = R.run("squeue --me -h -o '%i|%j|%P|%T|%M|%l|%R'")
    rows = [dict(zip(("job_id", "name", "partition", "state", "elapsed", "limit", "reason"), l.split("|")))
            for l in out.splitlines() if l.strip()]
    emit(rows, args)


def cmd_push(args, cfg, R):
    src = Path(args.local).expanduser().resolve()
    if not src.exists():
        raise NuhpcError(f"not found: {src}")
    dst = remote_path(cfg, f"data/{args.remote or src.name}", args.project)
    if args.globus:
        return _globus(cfg, R, args, str(src), dst, to_remote=True)
    R.run(f"mkdir -p {shlex.quote(dst if src.is_dir() else str(Path(dst).parent))}")
    R.rsync(f"{src}/" if src.is_dir() else str(src), f"{cfg['ssh_host']}:{dst}{'/' if src.is_dir() else ''}")
    emit({"pushed": str(src), "remote": dst}, args)


def cmd_pull(args, cfg, R):
    src = remote_path(cfg, args.remote, args.project)
    dst = Path(args.local or Path(cfg["local_results"]) / "pulled" / Path(args.remote).name).expanduser()
    if args.globus:
        return _globus(cfg, R, args, str(dst.resolve()), src, to_remote=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    R.rsync(f"{cfg['ssh_host']}:{src}", str(dst))
    emit({"pulled": src, "local": str(dst)}, args)


def _globus(cfg, R, args, local, remote, to_remote):
    g = cfg["globus"]
    if not (g.get("local_endpoint") and g.get("remote_endpoint")):
        raise NuhpcError("set [globus] local_endpoint and remote_endpoint in config")
    a, b = f"{g['local_endpoint']}:{local}", f"{g['remote_endpoint']}:{remote}"
    argv = ["globus", "transfer", "--recursive", "--sync-level", "checksum", "--label", "nuhpc",
            "--jmespath", "task_id", "--format", "unix", *((a, b) if to_remote else (b, a))]
    out = R._exec(argv, retry=False)   # a repeated transfer request would duplicate it
    emit({"globus_task": out.strip() or "DRY-RUN", "wait_with": f"globus task wait {out.strip()}"}, args)


def cmd_ls(args, cfg, R):
    out = R.run(f"ls -la {shlex.quote(remote_path(cfg, args.path or '', args.project))}")
    print(out, end="") if not args.json else emit({"listing": out}, args)


# --------------------------------------------------------------------------- disk

def disk_scope(cfg: dict, args) -> dict:
    """Which folder to measure and how. By default the whole lab: your group's folder (the parent of user_dir), every member's folders and
    the shared ones, listed to depth 4 with folder names but no file names. `--mine` is just your user_dir, and `--project NAME` a declared
    project's folder. Nothing outside these can be scanned. Without a group folder above user_dir, the default is your own folder."""
    flags = [bool(getattr(args, "project", None)), bool(getattr(args, "lab", False)), bool(getattr(args, "mine", False))]
    if sum(flags) > 1:
        raise NuhpcError("use either --project, --lab or --mine, not more than one")
    depth = getattr(args, "depth", None) or 4
    if getattr(args, "project", None):
        proj = cfg.get("projects", {}).get(args.project)
        if not proj:
            raise NuhpcError(f"no folder is declared for project '{args.project}': add [projects.{args.project}] dir = ... to the config")
        return {"root": proj["dir"].rstrip("/"), "depth": depth, "names": True, "time": "00:30:00", "project": args.project}
    parent = posixpath.dirname(user_dir(cfg))
    has_group = parent.count("/") >= 2
    if getattr(args, "lab", False) and not has_group:
        raise NuhpcError(f"--lab needs user_dir inside a group folder such as /projects/<group>/<you>, not {user_dir(cfg)}")
    if has_group and not getattr(args, "mine", False):
        return {"root": parent, "depth": depth, "names": False, "time": "02:00:00", "project": None}
    return {"root": user_dir(cfg), "depth": depth, "names": True, "time": "00:30:00", "project": None}


def scope_key(cfg: dict, root: str) -> str:
    """The cache name of a scan, from the folder it measured."""
    root = root.rstrip("/")
    if root == user_dir(cfg):
        return "user"
    if root == posixpath.dirname(user_dir(cfg)):
        return "lab"
    for name, proj in cfg.get("projects", {}).items():
        if root == proj["dir"].rstrip("/"):
            return f"project-{slug(name)}"
    return "other-" + slug(root)


def check_scan(scan, source) -> dict:
    if not (isinstance(scan, dict) and scan.get("version") == 1 and isinstance(scan.get("tree"), dict)):
        raise NuhpcError(f"{source} is not a nuhpc disk scan")
    return scan


def load_scan(path) -> dict:
    try:
        return check_scan(json.loads(Path(path).read_text()), path)
    except (OSError, ValueError) as ex:
        raise NuhpcError(f"cannot read {path}: {ex}")


def finish_quick(out: str, scope: dict, cfg: dict, started: float) -> dict:
    """Assemble the quick scan from the remote script's records and cache it. An empty reply is an error, not an empty picture."""
    if not any(line[:2] in ("D\t", "F\t") or line[:1].isdigit() for line in out.splitlines()):
        raise NuhpcError(f"nothing came back for {scope['root']}: does it exist, and can you read it?")
    scan = diskquick.tree_from_quick(out, scope["root"], scope["depth"], elapsed=time.monotonic() - started, host=cfg["ssh_host"])
    cache = STATE_DIR / "disk"
    cache.mkdir(parents=True, exist_ok=True)
    try:                                                          # folders this run could not finish keep what the last scan knew
        scan = diskquick.merge_scans(scan, load_scan(cache / f"{scope_key(cfg, scan['root'])}.json"))
    except NuhpcError:
        pass
    tmp = cache / ".incoming.json"
    tmp.write_text(json.dumps(scan, separators=(",", ":")))
    os.replace(tmp, cache / f"{scope_key(cfg, scan['root'])}.json")
    return scan


def scan_here(args, cfg: dict, R):
    """The quick scan: folder sizes from `du` on the login node, no job. A capped, lowest-priority script goes over ssh on stdin
    (nothing is written there), six `du` run at a time, and what finished inside the cap is assembled here; the rest is flagged."""
    scope = disk_scope(cfg, args)
    budget = 60 if args.budget is None else args.budget
    script = diskquick.quick_script(scope["root"], scope["depth"], budget)
    if args.dry_run:
        R.run("bash -s", retry=False, input=script)
        return None
    print(f"[nuhpc] measuring {scope['root']} with du on the login node (at most {budget:g} s, lowest priority; nothing is written there)",
          file=sys.stderr)
    t0 = time.monotonic()
    out = R.run("bash -s", retry=False, input=script)             # a capped run is never silently repeated
    scan = finish_quick(out, scope, cfg, t0)
    if scan["partial"]:
        print(f"[nuhpc] partial: {scan['unscanned']} folders were not fully read before the {budget:g} s cap, so the sizes above them are "
              f"lower bounds (marked >=). Raise the cap with --budget SECONDS, or use --job for the whole thing, with file counts and the "
              f"largest files, on a compute node.", file=sys.stderr)
    return scan


def quick_work(args, cfg: dict, R):
    """The quick scan as work for the explorer's refresher: the same script, streamed, so the status line can say what is happening."""
    scope = disk_scope(cfg, args)
    budget = 60 if args.budget is None else args.budget
    script = diskquick.quick_script(scope["root"], scope["depth"], budget)

    def work(progress, track):
        started, measured = time.monotonic(), [0]
        progress("connecting…")

        def on_line(line):
            measured[0] += line[:1].isdigit()                     # du's own records: one per second-level folder finished
            progress(f"collecting folder sizes · {measured[0]} measured" if measured[0] else "collecting folder sizes…")
        out = R.stream("bash -s", input=script, on_line=on_line, on_start=track)
        return finish_quick(out, scope, cfg, started)
    return work


def collect_scan(cfg: dict, R, e: dict) -> dict:
    """Fetch a finished scan run's result into the local cache, under the scope of the folder it measured."""
    cache = STATE_DIR / "disk"
    cache.mkdir(parents=True, exist_ok=True)
    incoming = cache / f".{e['run_id']}.json"
    try:
        R.rsync(f"{cfg['ssh_host']}:{e['remote_dir']}/outputs/task_0/disk.json", str(incoming))
        if not incoming.exists():
            raise NuhpcError(f"no scan result for {e['run_id']}: has it finished? (nuhpc status {e['run_id']})")
        scan = load_scan(incoming)
        os.replace(incoming, cache / f"{scope_key(cfg, scan['root'])}.json")
    finally:
        incoming.unlink(missing_ok=True)
    try:
        ensure_usage(cfg, R, [e])
    except NuhpcError:
        pass
    return scan


def scan_job(args, cfg: dict, scope: dict, code: str):
    """The submit arguments of a scan job. It always runs from your own work root, never from inside the folder it scans."""
    cpu = cfg.get("profiles", {}).get("cpu", {})
    budget = int(time_hours(scope["time"]) * 3600) - 180
    return argparse.Namespace(
        json=args.json, dry_run=args.dry_run, template="disk_scan", name="disk-scan", sweep=None, grid=None,
        param=[f"root={scope['root']}", f"depth={scope['depth']}", f"names={'true' if scope['names'] else 'false'}", f"budget={budget}"],
        code=code, max_parallel=None, confirm_big=False, pack=False, rows_per_job=None, parallel=None,
        project=None, profile=None, partition=cpu.get("partition", "short"), time=scope["time"], cpus=4,
        mem="8G", gres=None, nodes=1, account=None, constraint=None, gpus=None)


def run_scan(args, cfg: dict, R):
    """Submit the scan job (the scanner travels with it), wait for it, and collect the result. None for a dry run."""
    scope = disk_scope(cfg, args)
    with tempfile.TemporaryDirectory(prefix="nuhpc-diskscan-") as code:
        shutil.copy(Path(__file__).with_name("diskscan.py"), Path(code) / "diskscan.py")
        job = scan_job(args, cfg, scope, code)
        if args.dry_run:
            cmd_submit(job, cfg, R)
            return None
        sub = cmd_submit(job, cfg, R, quiet=True)
    e = find_run(sub["run_id"])
    print(f"[nuhpc] scanning {scope['root']} as run {e['run_id']} (job {e['job_id']}); a few minutes, depending on the file count",
          file=sys.stderr)
    timeout = time_hours(scope["time"]) * 3600 + 600 if args.timeout is None else args.timeout
    rows, _, timed_out = wait_for([e], R, timeout, args.interval, False)
    state = rows[0]["state"] if rows else "SUBMITTED"
    if timed_out:
        emit({"run_id": e["run_id"], "job_id": e["job_id"], "state": state, "timed_out": True,
              "next": f"nuhpc wait {e['run_id']}, then nuhpc disk --collect {e['run_id']}"}, args)
        sys.exit(3)
    if state != "COMPLETED":
        raise NuhpcError(f"the scan job {e['run_id']} ended {state}: read its log with `nuhpc logs {e['run_id']}`")
    return collect_scan(cfg, R, e)


def job_work(args, cfg: dict, R):
    """The job scan as work for the explorer's refresher: submit, watch the job (queued, running), fetch the result."""
    scope = disk_scope(cfg, args)

    def work(progress, track):
        progress("submitting the scan job…")
        with tempfile.TemporaryDirectory(prefix="nuhpc-diskscan-") as code:
            shutil.copy(Path(__file__).with_name("diskscan.py"), Path(code) / "diskscan.py")
            sub = cmd_submit(scan_job(args, cfg, scope, code), cfg, R, quiet=True)
        e = find_run(sub["run_id"])
        deadline = time.monotonic() + time_hours(scope["time"]) * 3600 + 600
        while True:
            rows, _ = run_states([e], R)
            state = rows[0]["state"] if rows else "SUBMITTED"
            if state == "COMPLETED":
                break
            if state in DONE:
                raise NuhpcError(f"the scan job ended {state}: nuhpc logs {e['run_id']}")
            progress(f"queued as job {e['job_id']}…" if state in ("SUBMITTED", "PENDING") else f"running on a compute node (job {e['job_id']})")
            if time.monotonic() > deadline:
                raise NuhpcError(f"the scan job is still {state.lower()}: nuhpc wait {e['run_id']}, then nuhpc disk --collect {e['run_id']}")
            time.sleep(5)
        progress("fetching the result…")
        return collect_scan(cfg, R, e)
    return work


def disk_live(args, cfg: dict, R) -> None:
    """The explorer opens at once on the cached scan (or on an empty one, the first time) while a refresher collects fresh data on a
    thread; a status line says what it is doing, and the new scan is swapped in when it arrives."""
    scope = disk_scope(cfg, args)
    cached = STATE_DIR / "disk" / f"{scope_key(cfg, scope['root'])}.json"
    scan = load_scan(cached) if cached.exists() else diskview.placeholder_scan(scope["root"])
    ex = diskview.Explorer(scan)
    if args.path:
        try:
            ex.goto(args.path)
        except KeyError:
            if not scan.get("placeholder"):
                raise NuhpcError(f"'{args.path}' is not in this scan (it lists folders to depth {scan.get('depth')}; rescan deeper with --depth)")
    ex.can_refresh = True
    refresher = diskview.Refresher((job_work if args.job else quick_work)(args, cfg, R))
    errors = io.StringIO()                                        # anything printed to stderr meanwhile would tear the screen
    with contextlib.redirect_stderr(errors):
        diskview.run_explorer(ex, color=use_color(True, os.environ), uni=unicode_ok(os.environ, getattr(sys.stdout, "encoding", None)),
                              refresher=refresher)
    if errors.getvalue():
        sys.stderr.write(errors.getvalue())


def cmd_disk(args, cfg, R):
    if (args.scan or args.job) and args.collect:
        raise NuhpcError("use either --scan (or --job) or --collect, not both")
    if args.no_refresh and (args.scan or args.job):
        raise NuhpcError("--no-refresh and --scan/--job contradict each other")
    interactive = sys.stdout.isatty() and sys.stdin.isatty() and not args.print and not args.json
    if interactive and not (args.from_file or args.collect or args.no_refresh or args.dry_run):
        return disk_live(args, cfg, R)
    if args.from_file:
        scan = load_scan(Path(args.from_file).expanduser())
    elif args.collect:
        scan = collect_scan(cfg, R, find_run(args.collect))
    elif args.scan or args.job:
        scan = (run_scan if args.job else scan_here)(args, cfg, R)
        if scan is None:
            return
    else:
        scope = disk_scope(cfg, args)
        cached = STATE_DIR / "disk" / f"{scope_key(cfg, scope['root'])}.json"
        if not cached.exists():
            raise NuhpcError(f"no scan of {scope['root']} yet: run `nuhpc disk --scan` (about a minute, no job)")
        scan = load_scan(cached)
    ex = diskview.Explorer(scan)
    if args.path:
        try:
            ex.goto(args.path)
        except KeyError:
            raise NuhpcError(f"'{args.path}' is not in this scan (it lists folders to depth {scan.get('depth')}; "
                             f"rescan deeper with --depth)")
    if args.json:
        return emit(ex.node() if args.path else scan, args)
    uni = unicode_ok(os.environ, getattr(sys.stdout, "encoding", None))
    if sys.stdout.isatty() and sys.stdin.isatty() and not args.print:
        return diskview.run_explorer(ex, color=use_color(True, os.environ), uni=uni)
    width = min(100, max(50, shutil.get_terminal_size((80, 24)).columns))
    print(ex.frame(width=width, color=use_color(sys.stdout.isatty(), os.environ), uni=uni, footer=False))


# --------------------------------------------------------------------------- argparse

def add_resource_flags(p):
    g = p.add_argument_group("resources (override the profile)")
    g.add_argument("--profile")
    g.add_argument("--partition")
    g.add_argument("--time", help="Slurm time, e.g. 04:00:00 or 1-00:00:00")
    g.add_argument("--cpus", type=int)
    g.add_argument("--mem")
    g.add_argument("--gpus", type=int, help="GPUs per node (keeps the GPU type from the profile)")
    g.add_argument("--gres")
    g.add_argument("--nodes", type=int)
    g.add_argument("--account")
    g.add_argument("--constraint")


def build_parser() -> argparse.ArgumentParser:
    # Global flags work before or after the subcommand. Sub-level copies use SUPPRESS so they
    # don't overwrite a value given at top level.
    def flags(p, default):
        p.add_argument("--json", action="store_true", default=default, help="machine-readable output")
        p.add_argument("--dry-run", action="store_true", default=default,
                       help="print remote commands, change nothing remotely")
    common = argparse.ArgumentParser(add_help=False)
    flags(common, argparse.SUPPRESS)

    ap = argparse.ArgumentParser(prog="nuhpc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    flags(ap, False)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def sub(name, fn, help):
        p = sp.add_parser(name, parents=[common], help=help, description=help)
        p.set_defaults(fn=fn)
        return p

    p = sub("init", cmd_init, "write an example config to ~/.config/nuhpc/config.toml")
    p.add_argument("--force", action="store_true")
    p.add_argument("--user-dir", help="your folder on the cluster's project storage, e.g. /projects/<group>/<you>")
    sub("connect", cmd_connect, "open a persistent SSH session (interactive: password/Duo happens here)")
    p = sub("check", cmd_check, "verify SSH and user_dir, create the work root, list partitions; --smoke also submits an env test job")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--project", help="check this declared project's work root instead of the default")
    add_resource_flags(p)
    sub("templates", cmd_templates, "list job templates")

    p = sub("submit", cmd_submit, "render a template into a run, upload code, sbatch it")
    p.add_argument("template")
    p.add_argument("--name")
    p.add_argument("--project", help="tag for usage accounting (e.g. miller, runes); see `nuhpc usage`")
    p.add_argument("--code", help="local project dir to snapshot into the run (respects .nuhpcignore)")
    p.add_argument("-p", "--param", action="append", help="key=value (value parsed as JSON if possible); repeatable")
    p.add_argument("--sweep", help=".jsonl/.json/.csv: one task (array element) per row")
    p.add_argument("--grid", action="append", help="key=v1,v2,... cartesian product; repeatable")
    p.add_argument("--max-parallel", type=int, help="max array tasks running at once")
    p.add_argument("--confirm-big", action="store_true", help="bypass configured limits (human approval only)")
    p.add_argument("--pack", action="store_true",
                   help="one array task per group of rows sharing the template's pack-by keys "
                        "(vllm_eval: one server per model, the client runs once per row)")
    p.add_argument("--rows-per-job", type=int,
                   help="pack this many rows into each job (array task). CPU rows run side by side: --cpus and "
                        "--mem then describe ONE row, and the job asks for that times --parallel")
    p.add_argument("--parallel", type=int,
                   help="with --rows-per-job: rows running at once (default all of them; 1 on GPU jobs)")
    add_resource_flags(p)

    p = sub("run", cmd_run, "quick test: run COMMAND on a compute node (30 min unless --time), wait, "
                            "print its output, fetch $HPC_OUT")
    p.add_argument("--name")
    p.add_argument("--project", help="tag for usage accounting; see `nuhpc usage`")
    p.add_argument("--code", help="local project dir to snapshot; COMMAND runs inside it")
    p.add_argument("--timeout", type=float, help="stop waiting after this many seconds: exit 3, the job keeps going")
    p.add_argument("--interval", type=float, default=15, help="first poll interval in s")
    p.add_argument("-n", "--lines", type=int, default=200, help="log lines printed at the end")
    add_resource_flags(p)
    p.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND [ARGS...]")

    p = sub("status", cmd_status, "state of runs (default: last 10)")
    p.add_argument("run", nargs="*")
    p.add_argument("--last", type=int, default=10)
    p = sub("wait", cmd_wait, "block until runs finish (polls with backoff); prints final states and "
                              "the log tail of the first failed task")
    p.add_argument("run", nargs="+")
    p.add_argument("--timeout", type=float, help="stop after this many seconds: exit 3, current states included")
    p.add_argument("--interval", type=float, default=30, help="first poll interval in s; grows x1.5 up to 300")
    p.add_argument("-n", "--lines", type=int, default=40, help="log lines shown for a failed task")
    p = sub("logs", cmd_logs, "tail a run's log (per array task)")
    p.add_argument("run")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("-n", "--lines", type=int, default=80)
    p.add_argument("--grep", help="only lines matching this extended regex, searched over the whole log "
                                  "(e.g. '^\\[p|s/trial|Traceback|Error'); -n keeps the last matches")
    p = sub("fetch", cmd_fetch, "rsync a run's outputs/ and logs/ to local_results")
    p.add_argument("run")
    p.add_argument("--include", action="append", help="glob(s) to fetch only, e.g. '*.json'")
    p.add_argument("--max-size", help="skip files larger than this, e.g. 500M")
    p = sub("cancel", cmd_cancel, "scancel a run (or one array task)")
    p.add_argument("run")
    p.add_argument("--task", type=int)
    p = sub("runs", cmd_runs, "list runs submitted from this machine")
    p.add_argument("--last", type=int, default=20)
    p = sub("usage", cmd_usage, "compute used vs requested per finished run, with totals per project "
                                "(records each run once, locally and in the cluster home)")
    p.add_argument("run", nargs="*")
    p.add_argument("--since", default="7d", help="runs submitted within e.g. 48h or 7d (default), or 'all'")
    p.add_argument("--project", help="only runs tagged with this project")
    sub("queue", cmd_queue, "your jobs in the Slurm queue")
    p = sub("standing", cmd_standing, "queue pressure against the live caps and your account's fair-share standing, "
                                      "with a verdict (read-only, one login-node call)")
    p.add_argument("--users", action="store_true", help="name the other users in your account and their share of its usage")
    p.add_argument("--brief", action="store_true", help="print only the one-line verdict")
    p = sub("push", cmd_push, "upload data to the work root's data/ folder")
    p.add_argument("local")
    p.add_argument("--project", help="use this declared project's work root")
    p.add_argument("remote", nargs="?", help="destination relative to data/ (default: same name)")
    p.add_argument("--globus", action="store_true")
    p = sub("pull", cmd_pull, "download a path (relative to the work root)")
    p.add_argument("remote")
    p.add_argument("--project", help="use this declared project's work root")
    p.add_argument("local", nargs="?")
    p.add_argument("--globus", action="store_true")
    p = sub("ls", cmd_ls, "list a directory under the work root")
    p.add_argument("path", nargs="?")
    p.add_argument("--project", help="use this declared project's work root")
    p = sub("disk", cmd_disk, "see where your space goes: a read-only scan job on the cluster, then an interactive "
                              "explorer in the terminal (bars per folder, drill down)")
    p.add_argument("path", nargs="?", help="start inside this folder of the scan, e.g. hf_cache/hub")
    p.add_argument("--scan", action="store_true", help="measure first: folder sizes from du on the login node, no job "
                                                      "(capped by --budget; folders it cannot finish are flagged)")
    p.add_argument("--budget", type=float, help="seconds the quick scan may take (default 60)")
    p.add_argument("--job", action="store_true", help="measure with a short read-only compute-node job instead: complete, with file counts "
                                                    "and the largest files, but it queues")
    p.add_argument("--collect", metavar="RUN", help="fetch the result of a --job scan that finished after its wait timed out")
    p.add_argument("--from", dest="from_file", metavar="FILE", help="explore a saved scan.json instead")
    p.add_argument("--project", help="scan or show this declared project's folder")
    p.add_argument("--lab", action="store_true", help="the whole lab: your group's folder, every member's folders and the shared ones "
                                                   "(the default; folder names to --depth, no file names)")
    p.add_argument("--mine", action="store_true", help="only your own folder (user_dir), instead of the whole lab")
    p.add_argument("--depth", type=int, help="levels of folders the scan lists (default 4); deeper ones are only totalled")
    p.add_argument("--print", action="store_true", help="print the report once instead of opening the explorer")
    p.add_argument("--no-refresh", action="store_true", help="in a terminal, open the cached scan without collecting fresh data "
                                                          "(by default the explorer opens at once and refreshes in the background)")
    p.add_argument("--timeout", type=float, help="with --job: seconds to wait for the scan (default: its time limit plus 10 minutes)")
    p.add_argument("--interval", type=float, default=15, help="first poll interval in s while waiting; grows x1.5 up to 300")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cfg = {} if args.cmd == "init" else load_config()
        R = Remote(cfg, args.dry_run) if cfg else None
        args.fn(args, cfg, R)
    except NuhpcError as e:
        if getattr(args, "json", False):
            print(json.dumps({"error": str(e)}))
        else:
            print(f"nuhpc: error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
