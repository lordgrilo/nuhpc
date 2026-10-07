"""Offline tests: pure helpers, submit guardrails, and rendered job scripts executed locally
against stub binaries (no cluster, no ssh). Run with: uv run --with pytest pytest"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nuhpc import cli


@pytest.fixture
def env(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'ssh_host = "nowhere"\nremote_root = "{tmp_path}/remote"\n'
                   '[limits]\nmax_gpu_hours_per_submit = 10\nmax_tasks = 5\n')
    return {**os.environ, "NUHPC_CONFIG": str(cfg), "NUHPC_STATE": str(tmp_path / "state")}


def nuhpc(env, *argv):
    p = subprocess.run([sys.executable, "-m", "nuhpc", "--json", "--dry-run", *argv],
                       env=env, capture_output=True, text=True)
    return p.returncode, json.loads(p.stdout)


def stub_bin(tmp_path, **scripts):
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    for name, body in scripts.items():
        (d / name).write_text("#!/bin/bash\n" + body + "\n")
        (d / name).chmod(0o755)
    return d


def run_job(tmp_path, env, result, stubs):
    """Materialise the dry-run stage as the 'remote' run dir and execute job.sbatch locally."""
    run_dir = Path(result["remote_dir"])
    shutil.copytree(result["local_stage"], run_dir)
    (run_dir / "code").mkdir()
    path = f"{stub_bin(tmp_path, **stubs)}:{env['PATH']}"
    return run_dir, subprocess.run(["bash", str(run_dir / "job.sbatch")], capture_output=True,
                                   text=True, env={**env, "PATH": path}, timeout=60)


@pytest.mark.parametrize("t,h", [("30", 0.5), ("01:30:00", 1.5), ("1-12:00:00", 36), ("2-06", 54)])
def test_time_hours(t, h):
    assert cli.time_hours(t) == pytest.approx(h)


def test_grid_times_sweep(tmp_path):
    sweep = tmp_path / "s.jsonl"
    sweep.write_text('{"m": "a"}\n{"m": "b"}\n')
    tasks = cli.build_tasks(str(sweep), ["t=0,0.7"], {"base": 1})
    assert len(tasks) == 4 and tasks[1] == {"base": 1, "m": "a", "t": 0.7}


def test_helpers():
    assert cli.gpus_per_node({"gres": "gpu:a100:4"}) == 4
    assert cli.gpus_per_node({"gres": "gpu:1"}) == 1
    assert cli.array_count("123_[0-7%4]") == 8
    with pytest.raises(cli.NuhpcError):
        cli.remote_path({"remote_root": "/r"}, "../etc")


def test_gpu_hour_limit_needs_confirm_big(env):
    argv = ["submit", "smoke", "--partition", "gpu", "--gres", "gpu:4", "--time", "04:00:00"]
    rc, out = nuhpc(env, *argv)
    assert rc == 1 and "confirm-big" in out["error"]
    rc, out = nuhpc(env, *argv, "--confirm-big")
    assert rc == 0 and out["worst_case_gpu_hours"] == 16


def test_vllm_eval_runs_without_optional_params(tmp_path, env):
    rc, res = nuhpc(env, "submit", "vllm_eval", "--partition", "gpu", "--gres", "gpu:1",
                    "-p", "model=m", "-p", "entry=client.py")
    assert rc == 0
    run_dir, p = run_job(tmp_path, env, res, {"vllm": "sleep 30", "curl": "exit 0",
                                              "python": 'touch "$HPC_OUT/client_ran"'})
    assert p.returncode == 0, p.stdout + p.stderr
    assert (run_dir / "outputs/task_0/client_ran").exists()


def test_missing_required_param_aborts_job(tmp_path, env):
    rc, res = nuhpc(env, "submit", "python", "--partition", "short")
    assert rc == 0
    _, p = run_job(tmp_path, env, res, {"python": "exit 0"})
    assert p.returncode != 0 and "missing required param 'entry'" in p.stderr
