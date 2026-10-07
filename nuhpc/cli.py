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
import tomllib
from datetime import datetime
from importlib import resources
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("NUHPC_CONFIG", "~/.config/nuhpc/config.toml")).expanduser()
STATE_DIR = Path(os.environ.get("NUHPC_STATE", "~/.local/share/nuhpc")).expanduser()
LEDGER = STATE_DIR / "ledger.jsonl"

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
DEFAULT_EXCLUDES = [".git", "__pycache__", ".venv", "venv", "*.pyc", ".ipynb_checkpoints",
                    "wandb", "outputs", "data", "*.ckpt", "*.safetensors", "*.pt", "*.bin"]
RES_KEYS = ("partition", "time", "cpus", "mem", "gres", "nodes", "account", "constraint")


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

    def _exec(self, argv: list[str], capture=True) -> str:
        if self.dry:
            print("[dry-run] " + " ".join(shlex.quote(a) for a in argv), file=sys.stderr)
            return ""
        try:
            p = subprocess.run(argv, text=True, capture_output=capture)
        except FileNotFoundError:
            raise NuhpcError(f"'{argv[0]}' not found on this machine (install it)")
        if p.returncode != 0:
            err = (p.stderr or "").strip()
            if "Permission denied" in err or "Host key" in err or "BatchMode" in err:
                err += "\n(hint: run `nuhpc connect` to open an authenticated SSH session first)"
            raise NuhpcError(f"command failed ({p.returncode}): {' '.join(argv[:3])} ...\n{err}")
        return p.stdout if capture else ""

    def run(self, cmd: str) -> str:
        # Login shell so `module`, `sbatch`, etc. are on PATH.
        return self._exec(["ssh", "-o", "BatchMode=yes", self.host, "bash -lc " + shlex.quote(cmd)])

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
sed -n "$((HPC_TASK_ID + 1))p" "$HPC_RUN_DIR/sweep.jsonl" > "$HPC_PARAMS"

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


def cmd_submit(args, cfg, R, quiet=False):
    template = find_template(cfg, args.template)
    tasks = build_tasks(args.sweep, args.grid, parse_kv(args.param))
    n = len(tasks)
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
    meta = {"run_id": run_id, "template": template.stem, "profile": prof, "resources": res, "n_tasks": n,
            "max_parallel": max_par, "worst_case_gpu_hours": round(gpu_h, 2), "code": args.code,
            "created": datetime.now().isoformat(timespec="seconds")}
    (stage / "meta.json").write_text(json.dumps(meta, indent=2))

    host = cfg["ssh_host"]
    if args.code:
        code = Path(args.code).expanduser().resolve()
        if not code.is_dir():
            raise NuhpcError(f"--code must be a directory: {code}")
    R.run(f"mkdir -p {shlex.quote(run_dir)}/logs {shlex.quote(run_dir)}/outputs {shlex.quote(run_dir)}/code")
    R.rsync(f"{stage}/", f"{host}:{run_dir}/")
    if args.code:
        ex = [f"--exclude={e}" for e in DEFAULT_EXCLUDES]
        if (code / ".nuhpcignore").exists():
            ex.append(f"--exclude-from={code / '.nuhpcignore'}")
        R.rsync(f"{code}/", f"{host}:{run_dir}/code/", ex)
    out = R.run(f"cd {shlex.quote(run_dir)} && sbatch --parsable job.sbatch")
    job_id = out.strip().split(";")[0] if out.strip() else "DRY-RUN"
    result = {**meta, "job_id": job_id, "remote_dir": run_dir, "local_stage": str(stage)}
    if not args.dry_run:
        ledger_append({k: result[k] for k in ("run_id", "job_id", "template", "profile", "n_tasks",
                                             "remote_dir", "created", "worst_case_gpu_hours")})
    if quiet:
        return {"run_id": run_id, "job_id": job_id}
    emit(result, args)


def cmd_status(args, cfg, R):
    entries = ledger_read()
    if args.run:
        entries = [find_run(r) for r in args.run]
    else:
        entries = entries[-args.last:]
    if not entries:
        return emit([], args)
    ids = ",".join(e["job_id"] for e in entries)
    out = R.run(f"sacct -X -n -P --format=JobID,State,Elapsed,ExitCode -j {ids}")
    counts: dict[str, dict[str, int]] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        base = parts[0].split("_")[0]
        state = parts[1].split()[0]
        counts.setdefault(base, {}).setdefault(state, 0)
        counts[base][state] += array_count(parts[0])
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
    emit(rows, args)


def cmd_logs(args, cfg, R):
    e = find_run(args.run)
    suffix = f"_{args.task}.out" if e["n_tasks"] > 1 else ".out"
    out = R.run(f"tail -n {int(args.lines)} {shlex.quote(e['remote_dir'])}/logs/*{suffix}")
    if args.json:
        emit({"run_id": e["run_id"], "task": args.task, "log": out}, args)
    else:
        print(out, end="")


def cmd_fetch(args, cfg, R):
    e = find_run(args.run)
    dest = Path(cfg["local_results"]).expanduser() / e["run_id"]
    dest.mkdir(parents=True, exist_ok=True)
    extra = []
    max_size = args.max_size or cfg["limits"].get("fetch_max_file_size")
    if max_size:
        extra.append(f"--max-size={max_size}")
    if args.include:
        extra += ["-m", "--include=*/", *[f"--include={p}" for p in args.include], "--exclude=*"]
    for sub in ("outputs", "logs"):
        R.rsync(f"{cfg['ssh_host']}:{e['remote_dir']}/{sub}/", f"{dest}/{sub}/", extra)
    R.rsync(f"{cfg['ssh_host']}:{e['remote_dir']}/meta.json", f"{dest}/meta.json")
    emit({"run_id": e["run_id"], "local": str(dest), "max_file_size": max_size or "none"}, args)


def cmd_cancel(args, cfg, R):
    e = find_run(args.run)
    target = f"{e['job_id']}_{args.task}" if args.task is not None else e["job_id"]
    R.run(f"scancel {shlex.quote(target)}")
    emit({"cancelled": target, "run_id": e["run_id"]}, args)


def cmd_runs(args, cfg, R):
    rows = [{k: e[k] for k in ("run_id", "job_id", "template", "profile", "n_tasks", "created")}
            for e in ledger_read()[-args.last:]]
    emit(rows, args)


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
    out = R._exec(argv)
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
    p.add_argument("--code", help="local project dir to snapshot into the run (respects .nuhpcignore)")
    p.add_argument("-p", "--param", action="append", help="key=value (value parsed as JSON if possible); repeatable")
    p.add_argument("--sweep", help=".jsonl/.json/.csv: one task (array element) per row")
    p.add_argument("--grid", action="append", help="key=v1,v2,... cartesian product; repeatable")
    p.add_argument("--max-parallel", type=int, help="max array tasks running at once")
    p.add_argument("--confirm-big", action="store_true", help="bypass configured limits (human approval only)")
    add_resource_flags(p)

    p = sub("status", cmd_status, "state of runs (default: last 10)")
    p.add_argument("run", nargs="*")
    p.add_argument("--last", type=int, default=10)
    p = sub("logs", cmd_logs, "tail a run's log (per array task)")
    p.add_argument("run")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("-n", "--lines", type=int, default=80)
    p = sub("fetch", cmd_fetch, "rsync a run's outputs/ and logs/ to local_results")
    p.add_argument("run")
    p.add_argument("--include", action="append", help="glob(s) to fetch only, e.g. '*.json'")
    p.add_argument("--max-size", help="skip files larger than this, e.g. 500M")
    p = sub("cancel", cmd_cancel, "scancel a run (or one array task)")
    p.add_argument("run")
    p.add_argument("--task", type=int)
    p = sub("runs", cmd_runs, "list runs submitted from this machine")
    p.add_argument("--last", type=int, default=20)
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
