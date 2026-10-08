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
    cfg.write_text(f'ssh_host = "nowhere"\nremote_root = "{tmp_path}/remote"\nlocal_results = "{tmp_path}/results"\n'
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


def test_vllm_eval_health_check_bypasses_cluster_proxy(tmp_path, env):
    # Explorer jobs inherit http_proxy; curl must not send the localhost health check through it.
    rc, res = nuhpc(env, "submit", "vllm_eval", "--partition", "gpu", "--gres", "gpu:1",
                    "-p", "model=m", "-p", "entry=client.py", "-p", "startup_timeout=0")
    assert rc == 0
    curl = '[ -z "${http_proxy:-}" ] || [[ ",${no_proxy:-}," == *",127.0.0.1,"* ]] || exit 7'
    _, p = run_job(tmp_path, {**env, "http_proxy": "http://10.99.0.130:3128"}, res,
                   {"vllm": "sleep 30", "curl": curl, "python": "exit 0"})
    assert p.returncode == 0, p.stdout + p.stderr


def test_pack_rows_groups_by_keys_in_order():
    rows = [{"model": "a", "t": 0}, {"model": "b", "t": 0}, {"model": "a", "t": 1}]
    assert cli.pack_rows(rows, ["model", "vllm_args"]) == [[0, 2], [1]]


def test_pack_requires_template_support(env):
    rc, out = nuhpc(env, "submit", "python", "--partition", "short", "-p", "entry=x.py", "--pack")
    assert rc == 1 and "does not support --pack" in out["error"]


def test_packed_vllm_eval_starts_one_server_per_model(tmp_path, env):
    rc, res = nuhpc(env, "submit", "vllm_eval", "--partition", "gpu", "--gres", "gpu:1", "-p", "entry=client.py",
                    "--grid", "model=a,b", "--grid", "temperature=0,0.7", "--pack")
    assert rc == 0 and res["n_tasks"] == 2 and res["n_rows"] == 4
    run_dir = Path(res["remote_dir"])
    shutil.copytree(res["local_stage"], run_dir)
    (run_dir / "code").mkdir()
    starts = tmp_path / "vllm_starts"
    client = ('while [ $# -gt 0 ]; do case $1 in --params) P=$2; shift;; --out) O=$2; shift;; esac; shift; done; '
              'cp "$P" "$O/seen.json"')
    path = f"{stub_bin(tmp_path, vllm=f'echo x >> {starts}; sleep 30', curl='exit 0', python=client)}:{env['PATH']}"
    for task in (0, 1):
        p = subprocess.run(["bash", str(run_dir / "job.sbatch")], capture_output=True, text=True, timeout=60,
                           env={**env, "PATH": path, "SLURM_ARRAY_TASK_ID": str(task)})
        assert p.returncode == 0, p.stdout + p.stderr
    assert starts.read_text().count("x") == 2
    seen = json.loads((run_dir / "outputs/task_1/row_3/seen.json").read_text())
    assert seen["model"] == "b" and seen["temperature"] == 0.7


@pytest.fixture
def inproc(tmp_path, env, monkeypatch):
    """Run cli.main in-process against a fake transport: replies maps a command prefix to outputs."""
    monkeypatch.setattr(cli, "CONFIG_PATH", Path(env["NUHPC_CONFIG"]))
    monkeypatch.setattr(cli, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(cli, "LEDGER", tmp_path / "state" / "ledger.jsonl")
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    cli.ledger_append({"run_id": "r1-20260101-000000", "job_id": "100", "template": "python", "profile": None,
                       "n_tasks": 3, "remote_dir": "/r/runs/r1-20260101-000000", "created": "x",
                       "worst_case_gpu_hours": 0})
    calls = []

    def install(sacct_replies, other=""):
        it = iter(sacct_replies)

        def fake_run(self, cmd, retry=True):
            calls.append(cmd)
            if "sbatch" in cmd:
                return "555\n"
            if not cmd.startswith("sacct"):
                return other
            r = next(it)
            if isinstance(r, Exception):
                raise r
            return r
        monkeypatch.setattr(cli.Remote, "run", fake_run)
        monkeypatch.setattr(cli.Remote, "rsync", lambda self, *a, **k: None)
        return calls
    return install


def test_wait_reports_final_states_and_failed_task_log(inproc, capsys):
    calls = inproc(["100_[0-2]|PENDING|0:00|0:0\n",
                    "100_0|COMPLETED|1:00|0:0\n100_1|FAILED|0:30|1:0\n100_2|RUNNING|1:00|0:0\n",
                    "100_0|COMPLETED|1:00|0:0\n100_1|FAILED|0:30|1:0\n100_2|COMPLETED|2:00|0:0\n"],
                   other="Traceback: boom\n")
    cli.main(["--json", "wait", "r1", "--interval", "1"])
    out = json.loads(capsys.readouterr().out)
    run = out["runs"][0]
    assert out["timed_out"] is False and run["state"] == "FINISHED_WITH_FAILURES"
    assert run["failed_tasks"] == [1] and "boom" in run["failed_log_tail"]
    assert sum(c.startswith("sacct") for c in calls) == 3


def test_wait_timeout_exits_3_with_states(inproc, capsys):
    inproc(["100_[0-2]|PENDING|0:00|0:0\n"] * 5)
    with pytest.raises(SystemExit) as e:
        cli.main(["--json", "wait", "r1", "--timeout", "0"])
    assert e.value.code == 3
    out = json.loads(capsys.readouterr().out)
    assert out["timed_out"] is True and out["runs"][0]["state"] == "PENDING"


def test_wait_survives_transient_transport_errors(inproc, capsys):
    inproc([cli.NuhpcError("ssh: connect to host: Can't assign requested address")] * 3
           + ["100_0|COMPLETED|1:00|0:0\n100_1|COMPLETED|1:00|0:0\n100_2|COMPLETED|1:00|0:0\n"])
    cli.main(["--json", "wait", "r1"])
    assert json.loads(capsys.readouterr().out)["runs"][0]["state"] == "COMPLETED"


def test_logs_grep_quotes_pattern_for_remote_shell(inproc, capsys):
    calls = inproc([], other="[p1] 24/300 0.54s/trial\n")
    cli.main(["--json", "logs", "r1", "--task", "2", "--grep", "^\\[p|Traceback; rm -rf ~"])
    cmd = calls[-1]
    assert "grep -h -E -e '^\\[p|Traceback; rm -rf ~'" in cmd and cmd.endswith("| tail -n 80")
    assert "s/trial" in json.loads(capsys.readouterr().out)["log"]


def test_cmd_template_runs_any_command(tmp_path, env):
    rc, res = nuhpc(env, "submit", "cmd", "--partition", "short", "-p", 'cmd=echo hello > "$HPC_OUT/out.txt"')
    assert rc == 0
    run_dir, p = run_job(tmp_path, env, res, {})
    assert p.returncode == 0, p.stdout + p.stderr
    assert (run_dir / "outputs/task_0/out.txt").read_text().strip() == "hello"


def test_run_submits_waits_and_returns_output(inproc, capsys, tmp_path):
    inproc(["555|PENDING|0:00|0:0\n", "555|COMPLETED|0:10|0:0\n"], other="hello from the GPU\n")
    cli.main(["--json", "run", "--partition", "gpu-short", "--", "python", "probe.py", "--n", "3"])
    out = json.loads(capsys.readouterr().out)
    assert out["state"] == "COMPLETED" and "hello from the GPU" in out["log"]
    stage = tmp_path / "state" / "runs" / out["run_id"]
    assert json.loads((stage / "sweep.jsonl").read_text())["cmd"] == "python probe.py --n 3"
    assert "#SBATCH --time=00:30:00" in (stage / "job.sbatch").read_text()


def test_run_exits_2_when_the_command_fails(inproc, capsys):
    inproc(["555|FAILED|0:10|1:0\n"], other="Traceback\n")
    with pytest.raises(SystemExit) as e:
        cli.main(["--json", "run", "--partition", "short", "--", "false"])
    assert e.value.code == 2 and json.loads(capsys.readouterr().out)["state"] == "FINISHED_WITH_FAILURES"


def test_transport_retries_repeatable_calls_only(monkeypatch):
    seq, stderr = [], ["ssh: connect to host x port 22: Connection reset"]

    def fake_sp_run(argv, **kw):
        seq.append(argv)
        rc = 0 if len(seq) >= 3 else 255
        return subprocess.CompletedProcess(argv, rc, stdout="ok\n", stderr=stderr[0])
    monkeypatch.setattr(cli.subprocess, "run", fake_sp_run)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    R = cli.Remote({"ssh_host": "h"}, dry=False)
    assert R.run("sacct -j 1") == "ok\n" and len(seq) == 3          # two transient failures, then success
    seq.clear()
    with pytest.raises(cli.NuhpcError):
        R.run("cd x && sbatch --parsable job.sbatch", retry=False)   # never resubmit
    assert len(seq) == 1
    seq.clear(); stderr[0] = "Permission denied (publickey)."
    with pytest.raises(cli.NuhpcError, match="nuhpc connect"):
        R.run("sacct -j 1")                                          # auth failures are not transient
    assert len(seq) == 1


def test_missing_required_param_aborts_job(tmp_path, env):
    rc, res = nuhpc(env, "submit", "python", "--partition", "short")
    assert rc == 0
    _, p = run_job(tmp_path, env, res, {"python": "exit 0"})
    assert p.returncode != 0 and "missing required param 'entry'" in p.stderr
