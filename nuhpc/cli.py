"""nuhpc: an agent-friendly CLI for running Slurm jobs on Northeastern's Explorer cluster.

Design rules:
  * Everything remote lives under `remote_root`; remote paths given on the CLI are relative to it.
  * Every job is a *run*: rendered sbatch + params + code snapshot in runs/<run_id>/ on the cluster.
  * Every command supports --json so an agent can parse results; humans get tables.
  * No arbitrary remote exec. The agent gets verbs, not a shell.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path

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
    for key in ("ssh_host", "remote_root"):
        if key not in cfg:
            raise NuhpcError(f"config is missing required key '{key}'")
    cfg.setdefault("local_results", "~/nuhpc-results")
    cfg.setdefault("env_setup", "")
    cfg.setdefault("profiles", {})
    cfg.setdefault("limits", {})
    cfg.setdefault("templates_dir", "")
    cfg.setdefault("globus", {})
    return cfg


def remote_path(cfg: dict, rel: str = "") -> str:
    rel = (rel or "").strip()
    if rel.startswith("/") or ".." in Path(rel).parts:
        raise NuhpcError("remote paths are relative to remote_root (no absolute paths, no '..')")
    root = cfg["remote_root"].rstrip("/")
    return f"{root}/{rel}" if rel else root


# --------------------------------------------------------------------------- transport

class Remote:
    def __init__(self, cfg: dict, dry: bool):
        self.cfg, self.host, self.dry = cfg, cfg["ssh_host"], dry

    def _exec(self, argv: list[str], capture=True, retry=True) -> str:
        if self.dry:
            print("[dry-run] " + " ".join(shlex.quote(a) for a in argv), file=sys.stderr)
            return ""
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                p = subprocess.run(argv, text=True, capture_output=capture)
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

    def run(self, cmd: str, retry=True) -> str:
        # Login shell so `module`, `sbatch`, etc. are on PATH.
        return self._exec(["ssh", "-o", "BatchMode=yes", self.host, "bash -lc " + shlex.quote(cmd)], retry=retry)

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
# Unpacked runs keep the flat layout (outputs/task_K); packed rows write to outputs/task_K/row_R.
# A failing row doesn't stop the others; the task fails at the end if any row did.
hpc_each_row() {{
  if [ ! -f "$HPC_RUN_DIR/packs.json" ]; then
    HPC_ROW="$HPC_TASK_ID" HPC_ROW_PARAMS="$HPC_PARAMS" HPC_ROW_OUT="$HPC_OUT" "$@"; return
  fi
  local r rc=0
  for r in $HPC_ROWS; do
    export HPC_ROW="$r" HPC_ROW_OUT="$HPC_OUT/row_$r" HPC_ROW_PARAMS="$HPC_OUT/row_$r/params.json"
    mkdir -p "$HPC_ROW_OUT"
    sed -n "$((r + 1))p" "$HPC_RUN_DIR/sweep.jsonl" > "$HPC_ROW_PARAMS"
    echo "[nuhpc] row $r: $(cat "$HPC_ROW_PARAMS")"
    "$@" || {{ rc=$?; echo "[nuhpc] row $r failed (exit $rc)"; }}
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
           run_id: str, run_dir: str, job_name: str) -> str:
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
           "DATA": remote_path(cfg, "data"), "REMOTE_ROOT": remote_path(cfg),
           "NODES": str(res["nodes"]), "GPUS_PER_NODE": str(gpn),
           "GPUS": str(gpn * int(res["nodes"])), "CPUS": str(res["cpus"])}
    body = template.read_text()
    body = re.sub(r"\{\{(\w+)\}\}", lambda m: ctx.get(m.group(1), m.group(0)), body)
    left = re.findall(r"\{\{(\w+)\}\}", body)
    if left:
        raise NuhpcError(f"template uses unknown placeholders: {sorted(set(left))}. "
                         f"Per-task values belong in params: use $(hpc_param key).")
    pre = PREAMBLE.format(run_id=run_id, run_dir=run_dir, env_setup=cfg["env_setup"].strip())
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


# --------------------------------------------------------------------------- commands

EXAMPLE_CONFIG = (Path(__file__).parent / "example_config.toml")


def cmd_init(args, cfg, R):
    if CONFIG_PATH.exists() and not args.force:
        raise NuhpcError(f"{CONFIG_PATH} exists (use --force to overwrite)")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(EXAMPLE_CONFIG.read_text())
    emit({"config": str(CONFIG_PATH), "next": "edit it, add the ~/.ssh/config block from the README, "
          "then `nuhpc connect` and `nuhpc check --smoke`"}, args)


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
    root = remote_path(cfg)
    out = R.run(f"echo ok; hostname; mkdir -p {shlex.quote(root)}; df -h {shlex.quote(root)} | tail -1; "
                "sinfo -h -o '%P|%l|%G|%a' | sort -u")
    lines = out.splitlines()
    info = {"ssh": "ok" if lines[:1] == ["ok"] else "dry-run" if args.dry_run else "FAILED",
            "login_node": lines[1] if len(lines) > 1 else "", "remote_root_disk": lines[2] if len(lines) > 2 else "",
            "partitions": [dict(zip(("partition", "timelimit", "gres", "avail"), l.split("|")))
                           for l in lines[3:] if "|" in l]}
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
    keys, packs = [], None
    if getattr(args, "pack", False):
        keys = pack_keys(template)
        if not keys:
            raise NuhpcError(f"template '{template.stem}' does not support --pack (no '# pack-by:' line)")
        packs = pack_rows(tasks, keys)
    n = len(packs) if packs else len(tasks)   # array tasks: what reaches the queue and the limits
    prof, res = resolve_resources(cfg, args)
    gpu_h = gpus_per_node(res) * int(res["nodes"]) * time_hours(res["time"]) * n
    lim = cfg["limits"]
    if lim.get("max_tasks") and n > lim["max_tasks"] and not args.confirm_big:
        raise NuhpcError(f"{n} tasks exceeds limits.max_tasks={lim['max_tasks']}. Needs --confirm-big (human approval).")
    if lim.get("max_gpu_hours_per_submit") and gpu_h > lim["max_gpu_hours_per_submit"] and not args.confirm_big:
        raise NuhpcError(f"worst-case {gpu_h:.1f} GPU-hours exceeds limits.max_gpu_hours_per_submit="
                         f"{lim['max_gpu_hours_per_submit']}. Needs --confirm-big (human approval).")
    max_par = args.max_parallel or lim.get("default_max_parallel")

    run_id = f"{slug(args.name or template.stem)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = remote_path(cfg, f"runs/{run_id}")
    stage = STATE_DIR / "runs" / run_id
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "job.sbatch").write_text(render(cfg, template, res, n, max_par, run_id, run_dir, slug(args.name or template.stem)))
    (stage / "sweep.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    if packs:
        (stage / "packs.json").write_text(json.dumps(packs))
    meta = {"run_id": run_id, "template": template.stem, "project": getattr(args, "project", None),
            "profile": prof, "resources": res, "n_tasks": n,
            "n_rows": len(tasks), "pack_by": keys or None,
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
    out = R.run(f"cd {shlex.quote(run_dir)} && sbatch --parsable job.sbatch", retry=False)
    job_id = out.strip().split(";")[0] if out.strip() else "DRY-RUN"
    result = {**meta, "job_id": job_id, "remote_dir": run_dir, "local_stage": str(stage)}
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
    rows, totals = [], {}
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
               "mem_%": round(100 * max(mem)) if mem else "-"}
        rows.append(row)
        for key in (row["project"], "all"):
            g = totals.setdefault(key, {"project": key, "runs": 0, "gpu_h": 0.0, "req_gpu_h": 0.0, "cpu_h": 0.0})
            g["runs"] += 1
            for f in ("gpu_h", "req_gpu_h", "cpu_h"):
                g[f] = round(g[f] + row[f], 2)
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


def cmd_queue(args, cfg, R):
    out = R.run("squeue --me -h -o '%i|%j|%P|%T|%M|%l|%R'")
    rows = [dict(zip(("job_id", "name", "partition", "state", "elapsed", "limit", "reason"), l.split("|")))
            for l in out.splitlines() if l.strip()]
    emit(rows, args)


def cmd_push(args, cfg, R):
    src = Path(args.local).expanduser().resolve()
    if not src.exists():
        raise NuhpcError(f"not found: {src}")
    dst = remote_path(cfg, f"data/{args.remote or src.name}")
    if args.globus:
        return _globus(cfg, R, args, str(src), dst, to_remote=True)
    R.run(f"mkdir -p {shlex.quote(dst if src.is_dir() else str(Path(dst).parent))}")
    R.rsync(f"{src}/" if src.is_dir() else str(src), f"{cfg['ssh_host']}:{dst}{'/' if src.is_dir() else ''}")
    emit({"pushed": str(src), "remote": dst}, args)


def cmd_pull(args, cfg, R):
    src = remote_path(cfg, args.remote)
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
    out = R.run(f"ls -la {shlex.quote(remote_path(cfg, args.path or ''))}")
    print(out, end="") if not args.json else emit({"listing": out}, args)


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
    sub("connect", cmd_connect, "open a persistent SSH session (interactive: password/Duo happens here)")
    p = sub("check", cmd_check, "verify SSH, remote_root, and list partitions; --smoke also submits an env test job")
    p.add_argument("--smoke", action="store_true")
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
    p = sub("push", cmd_push, "upload data to remote_root/data/")
    p.add_argument("local")
    p.add_argument("remote", nargs="?", help="destination relative to data/ (default: same name)")
    p.add_argument("--globus", action="store_true")
    p = sub("pull", cmd_pull, "download a path (relative to remote_root)")
    p.add_argument("remote")
    p.add_argument("local", nargs="?")
    p.add_argument("--globus", action="store_true")
    p = sub("ls", cmd_ls, "list a directory under remote_root")
    p.add_argument("path", nargs="?")
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
