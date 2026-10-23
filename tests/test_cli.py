"""Offline tests: pure helpers, submit guardrails, and rendered job scripts executed locally
against stub binaries (no cluster, no ssh). Run with: uv run --with pytest pytest"""
import json
import math
import shlex
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
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


# ---- usage accounting: requested vs used, recorded once a run ends ----

SACCT_USAGE = "\n".join([
    "100_0|COMPLETED|0:0|2026-01-01T10:00:00|2026-01-01T10:05:00|2026-01-01T10:35:00|1800|120|d1001|"
    "billing=8,cpu=8,gres/gpu:a100=1,gres/gpu=1,mem=64G,node=1|02:00:00|0|64G",
    "100_0.batch|COMPLETED|0:0|2026-01-01T10:05:00|2026-01-01T10:05:00|2026-01-01T10:35:00|1800||d1001|"
    "cpu=8,gres/gpu:a100=1,gres/gpu=1,mem=64G,node=1|02:00:00|16777216K|",
    # Explorer's form: an untyped gres/gpu count, so the type comes from the node (sinfo)
    "100_1|OUT_OF_MEMORY|0:125|2026-01-01T10:00:00|2026-01-01T10:05:00|2026-01-01T10:15:00|600|120|d1002|"
    "billing=112,cpu=8,gres/gpu=1,mem=64G,node=1|00:05:00|0|64G",
    "100_1.batch|OUT_OF_MEMORY|0:125|2026-01-01T10:05:00|2026-01-01T10:05:00|2026-01-01T10:15:00|600||d1002|"
    "cpu=8,gres/gpu=1,mem=64G,node=1|00:05:00|64G|",
    "100_2|CANCELLED by 42|0:0|2026-01-01T10:00:00|None|2026-01-01T11:00:00|0|120|None assigned||00:00:00||64G",
]) + "\n"
NODE_GRES = "d1001|gpu:a100:3\nd1002|gpu:h200:8(S:0-1)\n"
GPU_SUMMARY = "/r/runs/r1-20260101-000000/usage/task_0_gpu.csv|60|63.5|100|31744\n"
USAGE_REPLY = SACCT_USAGE + f"{cli.SECTION}\n" + NODE_GRES + f"{cli.SECTION}\n" + GPU_SUMMARY


def usage_transport(monkeypatch, reply):
    calls, syncs = [], []

    def fake_run(self, cmd, retry=True):
        calls.append(cmd)
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(cli.Remote, "run", fake_run)
    monkeypatch.setattr(cli.Remote, "rsync", lambda self, src, dst, extra=None: syncs.append((src, dst)))
    return calls, syncs


def test_finished_run_gets_one_usage_record_in_three_places(inproc, monkeypatch, tmp_path, capsys):
    calls, syncs = usage_transport(monkeypatch, USAGE_REPLY)
    (tmp_path / "results/r1-20260101-000000").mkdir(parents=True)          # already fetched
    cli.main(["--json", "usage", "r1"])
    rec = json.loads((tmp_path / "state/usage/r1-20260101-000000.json").read_text())
    t0, t1, t2 = rec["tasks"]
    assert t0["gpu_type"] == "a100" and t0["queue_wait_s"] == 300 and t0["gpu_hours"] == 0.5
    assert t0["cpu_eff"] == 0.5 and t0["max_rss_gb"] == 16 and t0["mem_eff"] == 0.25 and t0["time_eff"] == 0.25
    assert t0["gpu_util_mean"] == 63.5 and t0["gpu_util_max"] == 100 and t0["gpu_mem_max_gb"] == 31.0
    assert t1["state"] == "OUT_OF_MEMORY" and t1["mem_eff"] == 1.0 and t1["gpu_util_mean"] is None
    assert t1["gpu_type"] == "h200"
    assert t2["state"] == "CANCELLED" and t2["queue_wait_s"] is None and t2["gpu_hours"] == 0 and t2["gpu_type"] is None
    assert rec["totals"]["gpu_hours"] == round(0.5 + 600 / 3600, 3)
    assert rec["totals"]["states"] == {"COMPLETED": 1, "OUT_OF_MEMORY": 1, "CANCELLED": 1}
    assert json.loads((tmp_path / "results/r1-20260101-000000/usage.json").read_text()) == rec
    assert syncs == [(f"{tmp_path}/state/usage/", "nowhere:.nuhpc/usage/")]   # the cluster-home copy
    out = json.loads(capsys.readouterr().out)
    assert out["runs"][0]["gpu_h"] == rec["totals"]["gpu_hours"] and out["totals"][-1]["project"] == "all"
    cli.main(["--json", "usage", "r1"])                    # a recorded run is read back, not queried again
    assert len(calls) == 1


def test_unfinished_run_is_not_recorded(inproc, monkeypatch, tmp_path, capsys):
    usage_transport(monkeypatch, USAGE_REPLY.replace("100_2|CANCELLED by 42", "100_2|RUNNING"))
    cli.main(["--json", "usage", "r1"])
    out = json.loads(capsys.readouterr().out)
    assert out["runs"] == [] and out["unfinished"] == ["r1-20260101-000000"]
    assert not (tmp_path / "state/usage/r1-20260101-000000.json").exists()


def test_usage_recording_fails_soft(inproc, monkeypatch, capsys):
    usage_transport(monkeypatch, cli.NuhpcError("ssh: connect to host: Operation timed out"))
    cli.main(["--json", "usage", "r1"])
    cap = capsys.readouterr()
    assert json.loads(cap.out)["unfinished"] == ["r1-20260101-000000"] and "usage not recorded" in cap.err


def test_usage_query_is_one_remote_shell(tmp_path, monkeypatch):
    """The whole remote side, run locally against stub sacct/sinfo, then parsed."""
    monkeypatch.setattr(cli, "STATE_DIR", tmp_path / "state")
    run = tmp_path / "runs/r1-20260101-000000"
    (run / "usage").mkdir(parents=True)
    (run / "usage/task_0_gpu.csv").write_text("2026/01/01 10:00:00.000, 0, 20, 1000\n"
                                              "2026/01/01 10:00:00.000, 1, [N/A], 9\n"
                                              "2026/01/01 10:00:30.000, 0, 80, 3000\n")
    (tmp_path / "acct").write_text(SACCT_USAGE)
    e = {"run_id": run.name, "job_id": "100", "remote_dir": str(run), "n_tasks": 3, "template": "python",
         "profile": None, "created": "x", "worst_case_gpu_hours": 1}
    gone = {**e, "run_id": "gone", "job_id": "101", "remote_dir": str(tmp_path / "deleted-run")}
    sinfo = f'echo "$*" > {tmp_path}/sinfo_args; printf "{NODE_GRES}"'
    env = {**os.environ, "HOME": str(tmp_path / "home"),
           "PATH": f"{stub_bin(tmp_path, sacct=f'cat {tmp_path}/acct', sinfo=sinfo)}:{os.environ['PATH']}"}
    p = subprocess.run(["bash", "-c", cli.usage_query([e, gone])], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    assert (tmp_path / "home/.nuhpc/usage").is_dir()
    assert "-n d1001,d1002 " in (tmp_path / "sinfo_args").read_text()     # only the nodes these runs used
    t = cli.parse_usage(p.stdout, [e, gone])[run.name]["tasks"]
    assert [x["gpu_type"] for x in t] == ["a100", "h200", None]
    assert (t[0]["gpu_samples"], t[0]["gpu_util_mean"], t[0]["gpu_util_max"]) == (2, 50.0, 80)
    stub_bin(tmp_path, sacct="exit 1")                                    # accounting down: fail, don't guess
    assert subprocess.run(["bash", "-c", cli.usage_query([e])], capture_output=True, env=env).returncode == 1


def test_gpu_jobs_sample_utilisation_into_usage_dir(tmp_path, env):
    rc, res = nuhpc(env, "submit", "cmd", "--partition", "gpu", "--gres", "gpu:1", "-p", "cmd=true")
    assert rc == 0
    smi = 'if [ "$1" = -L ]; then echo "GPU 0: A100"; else echo "2026/01/01 10:00:00.000, 0, 42, 1234"; fi'
    run_dir, p = run_job(tmp_path, {**env, "CUDA_VISIBLE_DEVICES": "0"}, res, {"nvidia-smi": smi})
    assert p.returncode == 0, p.stdout + p.stderr
    csv = run_dir / "usage/task_0_gpu.csv"
    for _ in range(50):                                     # the sampler runs in the background
        if csv.exists() and csv.read_text():
            break
        time.sleep(0.1)
    assert csv.read_text().strip().endswith("0, 42, 1234")


def test_project_tag_reaches_meta_and_ledger(inproc, capsys):
    inproc([])
    cli.main(["--json", "submit", "smoke", "--partition", "short", "--project", "miller"])
    assert json.loads(capsys.readouterr().out)["project"] == "miller"
    assert cli.ledger_read()[-1]["project"] == "miller"


def test_fetch_survives_runs_without_usage_dir(inproc, monkeypatch, capsys):
    inproc([])

    def rsync(self, src, dst, extra=None):
        if "/usage/" in src:
            raise cli.NuhpcError("rsync: change_dir failed (23)")
    monkeypatch.setattr(cli.Remote, "rsync", rsync)
    cli.main(["--json", "fetch", "r1"])
    assert json.loads(capsys.readouterr().out)["local"].endswith("r1-20260101-000000")


# ---- env_build: each user's own environment, built in a job ----

def test_jobs_export_remote_root_before_env_setup(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'ssh_host = "nowhere"\nremote_root = "{tmp_path}/remote"\n'
                   'env_setup = "source \\"$HPC_REMOTE_ROOT/envs/llm/bin/activate\\""\n')
    env = {**os.environ, "NUHPC_CONFIG": str(cfg), "NUHPC_STATE": str(tmp_path / "state")}
    rc, res = nuhpc(env, "submit", "cmd", "--partition", "short", "-p", "cmd=true")
    job = (Path(res["local_stage"]) / "job.sbatch").read_text()
    assert job.index(f'export HPC_REMOTE_ROOT="{tmp_path}/remote"') < job.index('source "$HPC_REMOTE_ROOT/envs/llm')


def test_env_build_runs_without_env_setup_and_builds_under_remote_root(tmp_path):
    cfg = tmp_path / "config.toml"   # env_setup that would kill any job that ran it: the env doesn't exist yet
    cfg.write_text(f'ssh_host = "nowhere"\nremote_root = "{tmp_path}/remote"\nenv_setup = "exit 99"\n')
    env = {**os.environ, "NUHPC_CONFIG": str(cfg), "NUHPC_STATE": str(tmp_path / "state")}
    log = tmp_path / "uv.log"
    uv = (f'printf "%s|" "$@" >> {log}; echo >> {log}\n'
          'if [ "$1" = venv ]; then mkdir -p "$4/bin"; printf "#!/bin/bash\\ncat >/dev/null; echo python-ok\\n" > "$4/bin/python"; '
          'chmod +x "$4/bin/python"; fi')
    boot = stub_bin(tmp_path, uv=uv)
    pip = (f'#!/bin/bash\n# stub bootstrap python: "pip install --target DIR uv" puts uv in DIR/bin\n'
           f'[ "$1 $2" = "-m pip" ] && [ "$3" = --version ] && exit 0\n'
           f'mkdir -p "$6/bin" && cp {boot}/uv "$6/bin/uv"\n')
    (tmp_path / "py").write_text(pip)
    (tmp_path / "py").chmod(0o755)
    rc, res = nuhpc(env, "submit", "env_build", "--partition", "short", "-p", f"bootstrap_python={tmp_path}/py")
    assert rc == 0
    assert "exit 99" not in (Path(res["local_stage"]) / "job.sbatch").read_text()
    _, p = run_job(tmp_path, env, res, {})
    assert p.returncode == 0, p.stdout + p.stderr
    p = subprocess.run(["bash", str(Path(res["remote_dir"]) / "job.sbatch")], capture_output=True, text=True,
                       env={**env, "PATH": f"{boot}:{env['PATH']}"})     # a second build reuses the venv
    assert p.returncode == 0, p.stdout + p.stderr
    calls = [l.split("|")[:-1] for l in log.read_text().splitlines()]
    venv = f"{tmp_path}/remote/envs/llm"
    assert calls[0] == ["venv", "--python", "3.12", venv]
    assert calls[1][:3] == ["pip", "install", "--python"] and calls[1][3] == f"{venv}/bin/python"
    assert "--torch-backend=cu129" in calls[1] and any(a.startswith("vllm @ https://") for a in calls[1])
    assert calls[1][-2:] == ["huggingface_hub", "openai"]
    assert [c[0] for c in calls[2:]] == ["pip"]                   # no second venv
    assert "python-ok" in p.stdout and f'source "{venv}/bin/activate"' in p.stdout


# ---- user_dir and project directories: where nuhpc installs and where it keeps work ----

LAB = "/projects/lab/me"


@pytest.fixture
def lab_env(tmp_path):
    """A config in the new style: user_dir plus one declared project; no remote_root."""
    cfg = tmp_path / "lab.toml"
    cfg.write_text(f'ssh_host = "nowhere"\nuser_dir = "{LAB}"\nlocal_results = "{tmp_path}/results"\n'
                   f'[projects.miller]\ndir = "{LAB}/miller"\n')
    return {**os.environ, "NUHPC_CONFIG": str(cfg), "NUHPC_STATE": str(tmp_path / "state")}


@pytest.fixture
def lab_main(lab_env, monkeypatch):
    """cli.main in-process against the lab config, with the transport recording instead of running."""
    monkeypatch.setattr(cli, "CONFIG_PATH", Path(lab_env["NUHPC_CONFIG"]))
    rec = {"run": [], "rsync": [], "reply": ""}
    monkeypatch.setattr(cli.Remote, "run", lambda self, cmd, retry=True: rec["run"].append(cmd) or rec["reply"])
    monkeypatch.setattr(cli.Remote, "rsync", lambda self, src, dst, extra=None: rec["rsync"].append((src, dst)))
    return rec


def job_script(res):
    return (Path(res["local_stage"]) / "job.sbatch").read_text()


def test_default_work_root_is_nested_under_user_dir(lab_env):
    rc, res = nuhpc(lab_env, "submit", "smoke", "--partition", "short")
    assert rc == 0 and res["remote_dir"].startswith(f"{LAB}/nuhpc/runs/")
    job = job_script(res)
    assert f'export HPC_USER_DIR="{LAB}"' in job and f'export HPC_REMOTE_ROOT="{LAB}/nuhpc"' in job
    assert f'df -h "{LAB}/nuhpc"' in job                    # {{REMOTE_ROOT}} is the work root


def test_declared_project_gets_its_own_nested_work_root(lab_env):
    rc, res = nuhpc(lab_env, "submit", "smoke", "--partition", "short", "--project", "miller")
    assert rc == 0 and res["project"] == "miller" and res["remote_dir"].startswith(f"{LAB}/miller/nuhpc/runs/")
    job = job_script(res)
    assert f'export HPC_REMOTE_ROOT="{LAB}/miller/nuhpc"' in job
    assert f'export HPC_USER_DIR="{LAB}"' in job             # installs stay with the user, not the project


def test_undeclared_project_is_only_a_tag(lab_env):
    rc, res = nuhpc(lab_env, "submit", "smoke", "--partition", "short", "--project", "nuhpc-dev")
    assert rc == 0 and res["project"] == "nuhpc-dev" and res["remote_dir"].startswith(f"{LAB}/nuhpc/runs/")


def test_env_build_default_stays_under_user_dir_even_for_a_project(lab_env):
    rc, res = nuhpc(lab_env, "submit", "env_build", "--partition", "short", "--project", "miller")
    assert rc == 0
    assert f'ENV="{LAB}/$ENV"' in job_script(res) and "miller/$ENV" not in job_script(res)


def test_remote_root_alone_behaves_as_before(env):
    """Existing configs: remote_root is the work root, and installs default beside it."""
    rc, res = nuhpc(env, "submit", "env_build", "--partition", "short")
    root = res["remote_dir"].split("/runs/")[0]
    job = job_script(res)
    assert f'export HPC_USER_DIR="{root}"' in job and f'export HPC_REMOTE_ROOT="{root}"' in job
    assert f'ENV="{root}/$ENV"' in job


def test_remote_root_with_user_dir_overrides_only_the_default_work_root(tmp_path):
    cfg = tmp_path / "both.toml"
    cfg.write_text(f'ssh_host = "n"\nuser_dir = "{LAB}"\nremote_root = "{LAB}/nuhpc-old"\n')
    env = {**os.environ, "NUHPC_CONFIG": str(cfg), "NUHPC_STATE": str(tmp_path / "state")}
    rc, res = nuhpc(env, "submit", "env_build", "--partition", "short")
    assert res["remote_dir"].startswith(f"{LAB}/nuhpc-old/runs/") and f'ENV="{LAB}/$ENV"' in job_script(res)


def test_push_pull_and_ls_follow_the_project(lab_main, tmp_path):
    f = tmp_path / "stim.txt"
    f.write_text("x")
    cli.main(["--json", "push", str(f), "--project", "miller"])
    cli.main(["--json", "ls", "sub", "--project", "miller"])
    cli.main(["--json", "pull", "out/a.json", str(tmp_path / "a.json"), "--project", "miller"])
    cli.main(["--json", "push", str(f)])
    assert lab_main["rsync"][0][1] == f"nowhere:{LAB}/miller/nuhpc/data/stim.txt"
    assert f"ls -la {LAB}/miller/nuhpc/sub" in lab_main["run"][1]         # run[0] is push's mkdir
    assert lab_main["rsync"][1][0] == f"nowhere:{LAB}/miller/nuhpc/out/a.json"
    assert lab_main["rsync"][2][1] == f"nowhere:{LAB}/nuhpc/data/stim.txt"       # no project: the default root


@pytest.mark.parametrize("body,msg", [
    ('ssh_host = "n"\n', "user_dir"),
    ('ssh_host = "n"\nuser_dir = "me/folder"\n', "absolute"),
    ('ssh_host = "n"\nuser_dir = "/projects/YOUR_GROUP/YOUR_USER"\n', "placeholder"),
    ('ssh_host = "n"\nremote_root = "/projects/YOUR_GROUP/YOUR_USER/nuhpc"\n', "placeholder"),
    (f'ssh_host = "n"\nuser_dir = "{LAB}"\n[projects.p]\nnote = "no dir"\n', "projects.p"),
    (f'ssh_host = "n"\nuser_dir = "{LAB}"\n[projects.p]\ndir = "relative/dir"\n', "projects.p"),
])
def test_config_problems_are_named(tmp_path, monkeypatch, body, msg):
    cfg = tmp_path / "c.toml"
    cfg.write_text(body)
    monkeypatch.setattr(cli, "CONFIG_PATH", cfg)
    with pytest.raises(cli.NuhpcError, match=msg):
        cli.load_config()


def test_init_writes_user_dir_and_the_default_config_asks_to_be_edited(tmp_path):
    import tomllib
    cfg = tmp_path / "new" / "config.toml"
    env = {**os.environ, "NUHPC_CONFIG": str(cfg), "NUHPC_STATE": str(tmp_path / "state")}
    run = lambda *a: subprocess.run([sys.executable, "-m", "nuhpc", "--json", *a], env=env, capture_output=True, text=True)
    assert run("init", "--user-dir", f"{LAB}/").returncode == 0
    text = cfg.read_text()
    assert tomllib.loads(text)["user_dir"] == LAB
    assert not [l for l in text.splitlines() if "YOUR_" in l and not l.lstrip().startswith("#")]
    cfg.unlink()
    assert run("init").returncode == 0                       # no --user-dir: the placeholder stays...
    p = run("templates")
    assert p.returncode == 1 and "placeholder" in json.loads(p.stdout)["error"]   # ...and is refused


CHECK_OK = "ok\nlogin01\nuser_dir: ok\nvast:/projects 16T 4.0T 12T 26% /projects\nshort|2-00:00:00|(null)|up\n"


def test_check_verifies_user_dir_and_reports_the_work_root(lab_main, capsys):
    lab_main["reply"] = CHECK_OK
    cli.main(["--json", "check", "--project", "miller"])
    out = json.loads(capsys.readouterr().out)
    assert out["user_dir"] == "ok" and out["work_root"] == f"{LAB}/miller/nuhpc"
    assert out["projects"] == {"miller": f"{LAB}/miller"} and out["partitions"][0]["partition"] == "short"
    assert f"mkdir -p {LAB}/miller/nuhpc" in lab_main["run"][0]


def test_check_names_a_missing_user_dir(lab_main, capsys):
    lab_main["reply"] = "ok\nlogin01\nuser_dir: MISSING\n(not checked)\nshort|2-00:00:00|(null)|up\n"
    with pytest.raises(SystemExit):
        cli.main(["--json", "check"])
    err = json.loads(capsys.readouterr().out)["error"]
    assert "user_dir" in err and LAB in err and "MISSING" in err


def test_check_does_not_create_a_missing_user_dir(lab_main, tmp_path):
    """The remote shell itself, run locally: a typo'd user_dir must not be created as a side effect."""
    lab_main["reply"] = CHECK_OK
    cli.main(["--json", "check", "--project", "miller"])
    cmd = lab_main["run"][0].replace(LAB, str(tmp_path / "typo"))
    p = subprocess.run(["bash", "-c", cmd.split("; sinfo")[0]], capture_output=True, text=True)
    assert "user_dir: MISSING" in p.stdout and not (tmp_path / "typo").exists()
    (tmp_path / "typo").mkdir()
    p = subprocess.run(["bash", "-c", cmd.split("; sinfo")[0]], capture_output=True, text=True)
    assert "user_dir: ok" in p.stdout and (tmp_path / "typo/miller/nuhpc").is_dir()


# ---- CPU packing (--rows-per-job) and the small-job checks ----

@pytest.fixture
def pack_env(tmp_path):
    """Config with the Explorer-style big-job thresholds; the small-task limits are left at their defaults."""
    cfg = tmp_path / "pack.toml"
    cfg.write_text(f'ssh_host = "nowhere"\nremote_root = "{tmp_path}/remote"\nlocal_results = "{tmp_path}/results"\n'
                   '[limits]\nmax_tasks = 200\nbig_job_cpus = 20\nbig_job_mem = "60G"\n'
                   '[profiles.cpu]\npartition = "short"\ncpus = 1\nmem = "6G"\ntime = "02:00:00"\n')
    return {**os.environ, "NUHPC_CONFIG": str(cfg), "NUHPC_STATE": str(tmp_path / "state")}


def rows_file(tmp_path, n, **extra):
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps({"entry": "job.py", "i": i, **extra}) + "\n" for i in range(n)))
    return str(p)


def submit_rows(env, tmp_path, n, *flags, template="python"):
    return nuhpc(env, "submit", template, "--profile", "cpu", "--sweep", rows_file(tmp_path, n), *flags)


def test_rows_per_job_groups_rows_and_multiplies_per_row_resources(tmp_path, pack_env):
    rc, res = submit_rows(pack_env, tmp_path, 10, "--rows-per-job", "4")
    assert rc == 0 and res["n_tasks"] == 3 and res["n_rows"] == 10 and res["warnings"] == []
    assert json.loads((Path(res["local_stage"]) / "packs.json").read_text()) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert (res["rows_per_job"], res["parallel"], res["waves"]) == (4, 4, 1)
    assert res["per_row_resources"]["cpus"] == 1 and res["per_row_resources"]["mem"] == "6G"
    assert res["resources"]["cpus"] == 4 and res["resources"]["mem"] == "24G"
    job = job_script(res)
    assert "#SBATCH --cpus-per-task=4" in job and "#SBATCH --mem=24G" in job and "#SBATCH --array=0-2" in job
    assert 'export HPC_PACK_PARALLEL="4"' in job and 'export HPC_ROW_CPUS="1"' in job


def test_parallel_below_rows_per_job_runs_in_waves(tmp_path, pack_env):
    rc, res = submit_rows(pack_env, tmp_path, 8, "--rows-per-job", "8", "--parallel", "4", "--cpus", "2", "--mem", "3G")
    assert rc == 0 and res["n_tasks"] == 1 and res["waves"] == 2
    assert res["resources"]["cpus"] == 8 and res["resources"]["mem"] == "12G"     # 4 rows side by side, 2 CPUs each
    assert 'export HPC_ROW_CPUS="2"' in job_script(res) and 'export HPC_PACK_PARALLEL="4"' in job_script(res)


def test_gpu_rows_run_one_at_a_time_and_resources_are_not_multiplied(tmp_path, pack_env):
    rc, res = submit_rows(pack_env, tmp_path, 6, "--rows-per-job", "3", "--gres", "gpu:1", "--partition", "gpu")
    assert rc == 0 and res["parallel"] == 1 and res["waves"] == 3
    assert res["resources"]["cpus"] == 1 and res["resources"]["mem"] == "6G"


@pytest.mark.parametrize("flags,msg", [
    (["--parallel", "2"], "needs --rows-per-job"),
    (["--rows-per-job", "4", "--pack"], "both group rows"),
    (["--rows-per-job", "0"], "at least 1"),
    (["--rows-per-job", "4", "--parallel", "5"], "exceed"),
    (["--rows-per-job", "4", "--nodes", "2"], "one node"),
    (["--rows-per-job", "4", "--gres", "gpu:1", "--parallel", "2"], "GPU"),
    (["--rows-per-job", "4", "--mem", "6.5G"], "integer"),
])
def test_rows_per_job_flag_errors_are_named(tmp_path, pack_env, flags, msg):
    rc, out = submit_rows(pack_env, tmp_path, 8, *flags)
    assert rc == 1 and msg in out["error"]


def test_templates_that_cannot_run_rows_say_so(tmp_path, pack_env):
    rc, out = submit_rows(pack_env, tmp_path, 4, "--rows-per-job", "2", template="smoke")
    assert rc == 1 and "does not support --rows-per-job" in out["error"]
    rc, out = submit_rows(pack_env, tmp_path, 4, "--rows-per-job", "2", template="vllm_eval")
    assert rc == 1 and "--pack" in out["error"]


def run_packed(tmp_path, env, res, task, stubs, **extra_env):
    """Run one array task of a rendered packed job locally, with stub binaries."""
    run_dir = Path(res["remote_dir"])
    if not run_dir.exists():
        shutil.copytree(res["local_stage"], run_dir)
        (run_dir / "code").mkdir()
    path = f"{stub_bin(tmp_path, **stubs)}:{env['PATH']}"
    p = subprocess.run(["bash", str(run_dir / "job.sbatch")], capture_output=True, text=True, timeout=120,
                       env={**env, "PATH": path, "SLURM_ARRAY_TASK_ID": str(task), **extra_env})
    return run_dir, p


def row_stub(log_dir):
    """A python stub: records how many copies run at once, what it saw, and fails for the row named in FAIL_ROW."""
    return (f'OUT=""; PARAMS=""\n'
            'while [ $# -gt 0 ]; do case $1 in --params) PARAMS=$2; shift;; --out) OUT=$2; shift;; esac; shift; done\n'
            f'D="{log_dir}/running"; mkdir -p "$D/$$"; echo "$(ls "$D" | wc -l | tr -d " ")" >> "{log_dir}/concurrency"\n'
            'echo "$HPC_OUT|${OMP_NUM_THREADS:-}|$(cat "$PARAMS")" > "$OUT/seen"\n'
            'sleep 0.5\n'
            'rmdir "$D/$$"\n'
            '[ "${FAIL_ROW:-}" = "$(basename "$OUT")" ] && exit 7\n'
            'exit 0')


def concurrency(log_dir):
    return [int(x) for x in (log_dir / "concurrency").read_text().split()]


def test_packed_rows_run_side_by_side_each_in_its_own_folder(tmp_path, pack_env):
    rc, res = submit_rows(pack_env, tmp_path, 6, "--rows-per-job", "6", "--parallel", "3", "--cpus", "2")
    assert rc == 0
    log = tmp_path / "log"
    log.mkdir()
    run_dir, p = run_packed(tmp_path, pack_env, res, 0, {"python": row_stub(log)})
    assert p.returncode == 0, p.stdout + p.stderr
    assert max(concurrency(log)) == 3 and len(concurrency(log)) == 6      # never more than 3 at once; all 6 ran
    for r in range(6):
        out, omp, params = (run_dir / f"outputs/task_0/row_{r}/seen").read_text().strip().split("|", 2)
        assert out == str(run_dir / f"outputs/task_0/row_{r}")             # HPC_OUT is the row's own folder
        assert omp == "2" and json.loads(params)["i"] == r                 # the row's CPUs as threads; its own params
        assert (run_dir / f"outputs/task_0/row_{r}/exit_code").read_text().strip() == "0"
    assert "[row 3] [nuhpc] row 3:" in p.stdout                            # output lines carry the row


def test_one_failing_packed_row_fails_the_job_but_not_the_others(tmp_path, pack_env):
    rc, res = submit_rows(pack_env, tmp_path, 4, "--rows-per-job", "4")
    log = tmp_path / "log"
    log.mkdir()
    run_dir, p = run_packed(tmp_path, pack_env, res, 0, {"python": row_stub(log)}, FAIL_ROW="row_1")
    assert p.returncode != 0
    assert (run_dir / "outputs/task_0/row_1/exit_code").read_text().strip() == "7"
    assert all((run_dir / f"outputs/task_0/row_{r}/seen").exists() for r in range(4))   # the rest still ran
    assert "[row 1] [nuhpc] row 1 failed (exit 7)" in p.stdout


def test_rows_per_job_with_parallel_one_runs_rows_in_order_without_forcing_threads(tmp_path, pack_env):
    rc, res = submit_rows(pack_env, tmp_path, 3, "--rows-per-job", "3", "--parallel", "1")
    log = tmp_path / "log"
    log.mkdir()
    run_dir, p = run_packed(tmp_path, pack_env, res, 0, {"python": row_stub(log)})
    assert p.returncode == 0, p.stdout + p.stderr
    assert concurrency(log) == [1, 1, 1] and res["resources"]["cpus"] == 1
    assert (run_dir / "outputs/task_0/row_2/seen").read_text().split("|")[1] == ""     # OMP_NUM_THREADS untouched


def test_packed_cmd_rows_do_not_overwrite_each_others_hpc_out(tmp_path, pack_env):
    sweep = tmp_path / "cmds.jsonl"
    sweep.write_text("".join(json.dumps({"cmd": f'echo row-{i} > "$HPC_OUT/out.txt"'}) + "\n" for i in range(3)))
    rc, res = nuhpc(pack_env, "submit", "cmd", "--profile", "cpu", "--sweep", str(sweep), "--rows-per-job", "3")
    assert rc == 0
    run_dir, p = run_packed(tmp_path, pack_env, res, 0, {})
    assert p.returncode == 0, p.stdout + p.stderr
    assert [(run_dir / f"outputs/task_0/row_{i}/out.txt").read_text().strip() for i in range(3)] == ["row-0", "row-1", "row-2"]


def test_second_array_task_of_a_packed_run_gets_its_own_rows(tmp_path, pack_env):
    rc, res = submit_rows(pack_env, tmp_path, 5, "--rows-per-job", "3")
    log = tmp_path / "log"
    log.mkdir()
    run_dir, p = run_packed(tmp_path, pack_env, res, 1, {"python": row_stub(log)})
    assert p.returncode == 0, p.stdout + p.stderr
    assert sorted(x.name for x in (run_dir / "outputs/task_1").glob("row_*")) == ["row_3", "row_4"]


def test_many_tiny_array_tasks_are_refused_unless_packed(tmp_path, pack_env):
    rc, out = submit_rows(pack_env, tmp_path, 17)
    err = out["error"]
    assert rc == 1 and "17 array tasks" in err and "--confirm-big" in err
    assert "--rows-per-job 10" in err and "2 jobs" in err         # min(20 cpus // 1, 60G // 6G, 17 rows)
    rc, res = submit_rows(pack_env, tmp_path, 17, "--confirm-big")
    assert rc == 0 and any("17 array tasks" in w for w in res["warnings"])
    rc, res = submit_rows(pack_env, tmp_path, 17, "--rows-per-job", "10")
    assert rc == 0 and res["n_tasks"] == 2 and res["warnings"] == []


def test_a_middling_number_of_tiny_tasks_only_warns(tmp_path, pack_env):
    assert submit_rows(pack_env, tmp_path, 8)[1]["warnings"] == []
    rc, res = submit_rows(pack_env, tmp_path, 9)
    assert rc == 0 and len(res["warnings"]) == 1 and "9 array tasks" in res["warnings"][0]
    assert "--rows-per-job" in res["warnings"][0]


def test_gpu_arrays_and_wide_tasks_are_not_small(tmp_path, pack_env):
    rc, res = submit_rows(pack_env, tmp_path, 20, "--gres", "gpu:1", "--partition", "gpu")
    assert rc == 0 and res["warnings"] == []
    rc, res = submit_rows(pack_env, tmp_path, 20, "--cpus", "8")
    assert rc == 0 and res["warnings"] == []


def test_small_task_checks_can_be_switched_off(tmp_path):
    cfg = tmp_path / "off.toml"
    cfg.write_text(f'ssh_host = "n"\nremote_root = "{tmp_path}/r"\n[limits]\nsmall_task_max = 0\nsmall_task_warn = 0\n'
                   '[profiles.cpu]\npartition = "short"\ncpus = 1\n')
    env = {**os.environ, "NUHPC_CONFIG": str(cfg), "NUHPC_STATE": str(tmp_path / "state")}
    rc, res = submit_rows(env, tmp_path, 30)
    assert rc == 0 and res["warnings"] == []


@pytest.mark.parametrize("flags,fragment", [
    (["--rows-per-job", "8", "--cpus", "4"], "32 CPUs"),
    (["--rows-per-job", "4", "--mem", "20G"], "80G"),
])
def test_jobs_bigger_than_most_nodes_warn(tmp_path, pack_env, flags, fragment):
    rc, res = submit_rows(pack_env, tmp_path, 8, *flags)
    assert rc == 0 and any("bigger than most nodes" in w and fragment in w for w in res["warnings"])


def test_gpu_jobs_are_never_judged_against_cpu_node_sizes(tmp_path, pack_env):
    """big_job_cpus/big_job_mem describe the CPU nodes; GPU nodes are far bigger, and a standard A100 or H200 job
    (12 CPUs and 120G, or 16 and 200G) is normal there."""
    for cpus, mem, gres in (("12", "120G", "gpu:a100:1"), ("16", "200G", "gpu:h200:1")):
        rc, res = submit_rows(pack_env, tmp_path, 1, "--cpus", cpus, "--mem", mem, "--gres", gres, "--partition", "gpu-short")
        assert rc == 0 and res["warnings"] == [], (cpus, mem, res["warnings"])


# ---- nuhpc standing: queue pressure and fair share, with a verdict ----

M = "__nuhpc_standing__"
SHARE_ROWS = [      # Account|User|RawShares|NormShares|RawUsage|NormUsage|EffectvUsage|FairShare|LevelFS
    "root||1|1.000000|20306784339||1.000000||",
    " bigacct||1|0.000890|90000000|0.010|0.010||0.089",
    " midacct||1|0.000890|40000000|0.005|0.005||0.178",
    " lab.pi||1|0.000890|68745922|0.003369|0.003369||0.264054",
    "  lab.pi|m.other|1|0.090909|51258530|0.002|0.002500|0.132622|0.121924",
    "  lab.pi|me.pi|1|0.090909|15798936|0.0008|0.000800|0.132704|0.395573",
    "  lab.pi|a.third|1|0.090909|1478310|0.00005|0.000050|0.132786|4.227547",
    "  lab.pi|z.idle|1|0.090909|0|0|0.000000|0.133443|inf",
    " lowacct||1|0.000890|10000000|0.001|0.001||0.89",
    " idleacct||1|0.000890|0|0|0.000000||inf",
]


NOW = datetime(2026, 10, 16, 10, 0, 0)
PART = ["PartitionName=short QoS=short MaxTime=2-00:00:00 ", "PartitionName=gpu QoS=gpu MaxTime=08:00:00 ",
        "PartitionName=gpu-short QoS=gpu-short MaxTime=02:00:00 ", "PartitionName=gpu-interactive QoS=gpu-interactive MaxTime=02:00:00 ",
        "PartitionName=sharing QoS=sharing MaxTime=01:00:00 "]
CORES = ["short|3157/439/700/4296", "gpu|586/550/60/1196", "gpu-short|586/550/60/1196", "gpu-interactive|586/550/60/1196",
         "sharing|4306/3466/6000/13772"]                                   # Partition|allocated/idle/other/total cores
_GPU_NODES = ["a1 allocated gpu:a100:4(S:0-1) gpu:a100:4(IDX:0-3)", "a2 mixed gpu:a100:3 gpu:a100:3(IDX:0-2)",
              "h1 mixed gpu:h200:8(S:0-1) gpu:h200:6(IDX:0-5)", "h2 down* gpu:h200:8(S:0-1) gpu:h200:0(IDX:N/A)",
              "t1 idle gpu:t4:4(S:0-1) gpu:t4:0(IDX:N/A)"]
GPUS = ([f"{p:<16} {n}" for p in ("gpu", "gpu-short", "gpu-interactive") for n in _GPU_NODES]
        + [f"{'sharing':<16} s1 mixed gpu:l40s:3 gpu:l40s:1(IDX:0)", f"{'short':<16} n9 mixed (null) (null)"])
# Partition|State|Reason|Submit|JobId|User, for everyone: the cluster, not just me
CQUEUE = [
    "short|RUNNING|None|2026-10-15T08:00:00|100|u1", "short|RUNNING|None|2026-10-15T08:00:00|101|u2",
    "short|RUNNING|None|2026-10-15T08:00:00|102|u3",
    "short|PENDING|Priority|2026-10-15T22:00:00|103|u1",                      # waits 12 h
    "short|PENDING|Priority|2026-10-16T06:00:00|104_[1-4%2]|u2",              # an array of 4
    "short|PENDING|Resources|2026-10-16T09:00:00|105|u3",
    "short|PENDING|Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions|2026-10-16T09:30:00|106|u4",
    "short|PENDING|Dependency|2026-10-14T09:00:00|107|u4", "short|PENDING|JobHeldUser|2026-10-14T09:00:00|108|u4",
    "short|PENDING|QOSMaxJobsPerUserLimit|2026-10-16T09:00:00|109|u1", "short|PENDING|BeginTime|2026-10-16T09:00:00|110|u1",
    "gpu|RUNNING|None|2026-10-16T01:00:00|200|u1", "gpu|RUNNING|None|2026-10-16T01:00:00|201|u2",
    "gpu|PENDING|Priority|2026-10-16T08:00:00|202|u3",
    "gpu,gpu-short|PENDING|Priority|2026-10-16T09:00:00|203|u4",              # waits in both queues
    "gpu-short|PENDING|Priority|2026-10-16T09:45:00|204|u4"]


def started(part, wait_min, ago_min=60, submit_extra=1000, eligible="yes"):
    """A sacct row for a job that started `ago_min` minutes before NOW after `wait_min` minutes of being eligible."""
    fmt = lambda t: t.strftime("%Y-%m-%dT%H:%M:%S")
    start = NOW - timedelta(minutes=ago_min)
    ready = start - timedelta(minutes=wait_min)
    return f"{part}|{fmt(ready - timedelta(minutes=submit_extra))}|{fmt(ready) if eligible == 'yes' else eligible}|{fmt(start)}"


WAITS = ([started("short", w, ago_min=30 + 40 * i) for i, w in enumerate([5, 10, 60, 120, 300, 900])]      # median 90, p90 300
         + [started("short", 3000, ago_min=72 * 60)]                       # started 3 days ago: outside the 6 h window
         + ["short|2026-10-16T09:00:00|2026-10-16T09:00:00|Unknown"]       # still pending: not a start
         + [started("gpu", w, ago_min=20 * (i + 1)) for i, w in enumerate([1, 2, 3, 4, 5, 6])]            # median 3.5
         + [started("gpu-short", w, ago_min=50) for w in (20, 40, 60)])                                   # too few for a median


def standing_out(queue=None, share=None, qos=None, prio=None, config=None, now="2026-10-16T10:00:00",
                 part=None, cores=None, gpus=None, cqueue=None, waits=None):
    queue = queue if queue is not None else [
        "gpu|PENDING|Priority|2026-10-15T15:18:04|11101394_[0-6%4]|sweep-llama",
        "gpu-short|PENDING|Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions|2026-10-15T14:24:38|11100487|pilot-run",
        "short|RUNNING|None|2026-10-16T08:00:00|11300001|a", "short|RUNNING|None|2026-10-16T08:00:00|11300002|b"]
    blocks = {"me": ["me.pi", "lab.pi"], "now": [now],
              "qos": qos if qos is not None else ["short|50|1000", "gpu|4|8", "gpu-short|2|8", "gpu-interactive|2|4", "sharing|2|4", "normal||"],
              "part": part if part is not None else PART,
              "queue": queue, "share": share if share is not None else SHARE_ROWS,
              "prio": prio if prio is not None else ["11100487 5821 576 2654 2500 0", "11101394 5795 550 2654 2500 0"],
              "config": config if config is not None else [
                  "PriorityDecayHalfLife   = 7-00:00:00", "PriorityWeightAge       = 5000", "PriorityWeightFairShare = 20000",
                  "PriorityWeightJobSize   = 2500", "PriorityWeightPartition = 0", "PriorityWeightQOS       = 2500"],
              "cores": cores if cores is not None else CORES, "gpus": gpus if gpus is not None else GPUS,
              "cqueue": cqueue if cqueue is not None else CQUEUE, "waits": waits if waits is not None else WAITS}
    return "\n".join(f"{M} {k}\n" + "\n".join(v) for k, v in blocks.items()) + "\n"


def standing(out=None, users=False, **thresholds):
    return cli.parse_standing(standing_out() if out is None else out, {"standing": thresholds}, users=users)


def test_standing_reads_the_fair_share_tree():
    f = standing()["fairshare"]
    assert f["label"] == "POOR" and f["factor"] == pytest.approx(0.132704)
    assert f["ratio"] == pytest.approx(0.003369 / 0.000890, rel=1e-3)              # 3.8x its equal share
    assert (f["accounts_above"], f["accounts_total"]) == (2, 5)                    # bigacct and midacct use more; root is not an account
    assert f["fairshare_points"] == round(0.132704 * 20000) and f["max_fairshare_points"] == 20000
    assert f["half_life_days"] == 7 and f["recovery_days"] == pytest.approx(7 * math.log2(1 / 0.264054), rel=1e-3)
    assert f["priority"] == {"priority": 5808, "age": 563, "fairshare": 2654, "jobsize": 2500, "qos": 0}
    assert f["max_priority"] == 30000


def test_standing_splits_the_accounts_usage_without_naming_colleagues():
    f = standing()["fairshare"]
    assert f["n_users"] == 4 and f["your_usage_pct"] == pytest.approx(100 * 15798936 / sum([51258530, 15798936, 1478310, 0]), rel=1e-6)
    assert f["others_pct"] == pytest.approx(100 - f["your_usage_pct"]) and "users" not in f
    names = standing(users=True)["fairshare"]["users"]
    assert [u["user"] for u in names][:2] == ["m.other", "me.pi"] and "a.third" in [u["user"] for u in names]
    assert "m.other" not in json.dumps(standing())


def test_standing_queue_counts_slots_against_the_live_caps_and_the_oldest_wait():
    q = {p["partition"]: p for p in standing()["queue"]["partitions"]}
    assert (q["short"]["running"], q["short"]["cap"], q["short"]["pending"]) == (2, 50, 0)
    assert (q["gpu"]["pending"], q["gpu"]["cap"], q["gpu"]["reasons"]) == (7, 4, {"Priority": 7})   # a pending array of 7 tasks
    assert q["gpu-short"]["oldest_wait_h"] == pytest.approx(19.6, abs=0.1)
    s = standing()
    assert s["queue"]["label"] == "OK" and s["queue"]["waiting_long"] == 2 and s["queue"]["blocked_by_caps"] == 0
    assert s["overall"] == "POOR" and s["verdict"].startswith("POOR") and "3.8x" in s["verdict"]
    assert s["brief"] == "fair share poor 0.13, queue ok"


def test_standing_queue_is_poor_only_when_a_cap_blocks_jobs():
    blocked = ["short|RUNNING|None|2026-10-16T08:00:00|1|a"] * 50 + [
        "short|PENDING|QOSMaxJobsPerUserLimit|2026-10-16T09:00:00|2|b", "short|PENDING|JobArrayTaskLimit|2026-10-16T09:00:00|3_[1-9%3]|c"]
    s = standing(standing_out(queue=blocked))
    assert s["queue"]["label"] == "POOR" and s["queue"]["blocked_by_caps"] == 1       # JobArrayTaskLimit is your own throttle
    near = standing(standing_out(queue=["short|RUNNING|None|2026-10-16T08:00:00|1|a"] * 41))
    assert near["queue"]["label"] == "OK" and "41/50" in near["queue"]["note"]
    calm = standing(standing_out(queue=["short|RUNNING|None|2026-10-16T08:00:00|1|a"] * 5))
    assert calm["queue"]["label"] == "GOOD"
    assert standing(standing_out(queue=[]))["queue"]["label"] == "GOOD"


def test_standing_fair_share_labels_follow_the_thresholds():
    def with_factor(x):
        return [r.replace("|0.132704|", f"|{x}|") for r in SHARE_ROWS]
    assert standing(standing_out(share=with_factor(0.8)))["fairshare"]["label"] == "GOOD"
    assert standing(standing_out(share=with_factor(0.4)))["fairshare"]["label"] == "OK"
    assert standing(standing_out(share=with_factor(0.4)), fairshare_good=0.3)["fairshare"]["label"] == "GOOD"
    s = standing(standing_out(share=with_factor(0.8), queue=["short|RUNNING|None|2026-10-16T08:00:00|1|a"] * 5))
    assert s["overall"] == "GOOD" and s["brief"] == "fair share good 0.80, queue good"


def test_standing_degrades_when_the_cluster_withholds_a_section():
    s = standing(standing_out(share=[]))
    assert s["fairshare"]["label"] == "unknown" and "fair share unavailable" in s["verdict"]
    assert s["overall"] == s["queue"]["label"] == "OK"        # with no fair-share reading, the queue decides (2 jobs wait over 6 h)
    s = standing("")                                  # nothing at all, as in a dry run
    assert s["overall"] == s["fairshare"]["label"] == s["queue"]["label"] == "unknown"   # no data is not good news
    assert s["verdict"].startswith("unknown") and s["brief"] == "fair share unknown, queue unknown"


def test_standing_remote_shell_runs_read_only_queries_and_survives_a_failing_one(tmp_path):
    """The whole remote side, run locally against stub Slurm commands."""
    stubs = {
        "sacctmgr": 'case "$*" in *assoc*) echo lab.pi;; *qos*) printf "short|50|1000\\ngpu-short|2|8\\n";; esac',
        "scontrol": 'case "$*" in *partition*) echo "PartitionName=short AllowGroups=ALL QoS=short MaxTime=2-00:00:00";; '
                    '*config*) printf "PriorityDecayHalfLife   = 7-00:00:00\\nPriorityWeightFairShare = 20000\\nMaxJobCount = 1\\n";; esac',
        "squeue": 'echo "short|RUNNING|None|2026-10-16T08:00:00|11300001|a"',
        "sshare": "printf '%s\\n' " + " ".join(shlex.quote(r) for r in SHARE_ROWS),
        "sprio": 'echo "11100487 5821 576 2654 2500 0"',
        "date": 'echo 2026-10-16T10:00:00'}
    env = {**os.environ, "USER": "me.pi", "PATH": f"{stub_bin(tmp_path, **stubs)}:{os.environ['PATH']}"}
    p = subprocess.run(["bash", "-c", cli.standing_query()], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    s = cli.parse_standing(p.stdout, {})
    assert s["user"] == "me.pi" and s["account"] == "lab.pi" and s["fairshare"]["factor"] == pytest.approx(0.132704)
    assert s["fairshare"]["half_life_days"] == 7 and s["queue"]["partitions"][0]["cap"] == 50
    stub_bin(tmp_path, sshare="exit 1")               # one query failing must not sink the others
    p = subprocess.run(["bash", "-c", cli.standing_query()], capture_output=True, text=True, env=env)
    assert p.returncode == 0 and cli.parse_standing(p.stdout, {})["fairshare"]["label"] == "unknown"
    q = cli.standing_query()                           # read-only: nothing here can change the cluster
    assert not any(w in q for w in ("scancel", "sbatch", "srun", "update", "sacctmgr add", "modify", "delete", " rm ", "kill"))


def test_standing_command_prints_a_report_and_a_json_form(inproc, monkeypatch, capsys):
    inproc([])
    monkeypatch.setattr(cli.Remote, "run", lambda self, cmd, retry=True: standing_out())
    cli.main(["standing"])
    out = capsys.readouterr().out
    assert "POOR" in out and "3.8x" in out and "Queue" in out and "Fair share" in out and "m.other" not in out
    cli.main(["standing", "--brief"])
    assert capsys.readouterr().out.strip().startswith("POOR — fair share POOR")
    cli.main(["--json", "standing"])
    d = json.loads(capsys.readouterr().out)
    assert d["overall"] == "POOR" and d["brief"] == "fair share poor 0.13, queue ok" and d["fairshare"]["accounts_total"] == 5
    cli.main(["--json", "standing", "--users"])
    assert "m.other" in capsys.readouterr().out


# ---- nuhpc standing: how the report looks in a terminal ----

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def shown(text):
    return ANSI.sub("", text)


def render(color=False, width=72, uni=True, **thresholds):
    return cli.render_standing(standing(**thresholds), color=color, width=width, uni=uni)


def test_report_has_a_title_rules_bars_and_aligned_rows():
    text = render()
    lines = text.splitlines()
    assert lines[1] == "Explorer standing · cluster time 2026-10-16 10:00"
    assert lines[3].strip().startswith("[POOR]") and "fair share poor 0.13 · queue ok" in lines[3]
    assert lines[4].strip() == "me.pi · account lab.pi · 4 users"
    assert any(l.startswith("─") and " Queue " in l for l in lines) and any(l.startswith("─") and " Fair share " in l for l in lines)
    assert any(l.strip().startswith("short") and "2 of 50 running" in l and "░" in l for l in lines)
    i = next(i for i, l in enumerate(lines) if l.strip().startswith("gpu ") and "7 pending" in l)
    assert lines[i + 1].strip() == "Priority · oldest 18.7 h"                       # the reason and the wait, never cut off
    j = next(i for i, l in enumerate(lines) if l.strip().startswith("gpu-short"))
    assert lines[j + 1].strip() == "nodes down or reserved · oldest 19.6 h"        # the long Slurm reason, shortened for display
    gpu_short = next(p for p in cli.parse_standing(standing_out(), {})["queue"]["partitions"] if p["partition"] == "gpu-short")
    assert "Nodes required" not in text and "Nodes required" in str(gpu_short["reasons"])
    factor = next(l for l in lines if l.strip().startswith("factor"))
    assert "█" in factor and "0.133" in factor and "ok ≥ 0.25" in factor and "good ≥ 0.5" in factor
    assert any(l.strip().startswith("usage") and "3.8x its equal share" in l and "2 of 5 accounts use more" in l for l in lines)
    assert any(l.strip().startswith("recovery") and "13 days of no use to reach 1x" in l for l in lines)
    assert any(l.strip().startswith("account") and "you" in l and "23%" in l and "77%" in l for l in lines)
    assert any(l.strip().startswith("priority") and "2,654 of 20,000" in l for l in lines)       # factor 0.1327 x weight 20,000
    assert any("8 pending avg 5,808 of 30,000" in l and "fair share 2,654" in l for l in lines)
    assert any(l.strip().startswith("status") and "OK · 2 waiting over 6 h" in l for l in lines)


def test_report_never_runs_past_the_terminal_width():
    for width in (50, 72, 100):
        assert all(len(shown(l)) <= width for l in render(width=width, color=True).splitlines()), width


def test_report_has_an_ascii_form_for_terminals_without_unicode():
    text = render(uni=False)
    assert all(ord(c) < 128 for c in text) and "#" in text and "Queue" in text and "|" in text


def test_colours_change_only_the_escape_codes():
    plain, coloured = render(), render(color=True)
    assert "\x1b[" not in plain and shown(coloured) == plain
    assert "\x1b[31m" in coloured                                  # POOR fair share is red
    good = cli.render_standing(standing(standing_out(share=[r.replace("|0.132704|", "|0.8|") for r in SHARE_ROWS], gpus=[],   # a full GPU type is red too
                                                    queue=["short|RUNNING|None|2026-10-16T08:00:00|1|a"] * 5)),
                               color=True, width=72)
    assert "\x1b[32m" in good and "\x1b[31m" not in good           # GOOD is green, and nothing is red
    ok = cli.render_standing(standing(standing_out(share=[r.replace("|0.132704|", "|0.4|") for r in SHARE_ROWS])), color=True, width=72)
    assert "\x1b[33m" in ok                                        # OK is yellow


@pytest.mark.parametrize("tty,env,expected", [
    (False, {}, False), (True, {}, True), (True, {"NO_COLOR": "1"}, False),
    (False, {"FORCE_COLOR": "1"}, True), (True, {"NO_COLOR": "1", "FORCE_COLOR": "1"}, False), (True, {"NO_COLOR": ""}, True)])
def test_colour_is_for_terminals_unless_told_otherwise(tty, env, expected):
    assert cli.use_color(tty, env) is expected


@pytest.mark.parametrize("frac,filled", [(0, 0), (0.01, 1), (0.5, 10), (1, 20), (1.7, 20), (None, 0), (-1, 0)])
def test_bars_fill_in_proportion_and_stay_inside_their_width(frac, filled):
    b = cli.bar(frac, 20, True)
    assert len(b) == 20 and b.count("█") == filled and b.count("░") == 20 - filled


def test_the_command_colours_only_when_asked_to_and_brief_stays_one_line(inproc, monkeypatch, capsys):
    inproc([])
    monkeypatch.setattr(cli.Remote, "run", lambda self, cmd, retry=True: standing_out())
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    cli.main(["standing"])
    assert "\x1b[" not in capsys.readouterr().out                  # a pipe or a test is not a terminal
    monkeypatch.setenv("FORCE_COLOR", "1")
    cli.main(["standing"])
    assert "\x1b[31m" in capsys.readouterr().out
    cli.main(["standing", "--brief"])
    brief = capsys.readouterr().out
    assert len(brief.strip().splitlines()) == 1 and shown(brief).startswith("POOR — fair share POOR")
    cli.main(["--json", "standing"])
    assert "\x1b[" not in capsys.readouterr().out                  # json is never coloured


@pytest.mark.parametrize("env,encoding,expected", [
    ({}, "utf-8", True), ({"LANG": "en_US.UTF-8"}, "UTF-8", True), ({"LC_ALL": "C"}, "utf-8", False),
    ({"LANG": "en_US.UTF-8", "LC_ALL": "POSIX"}, "utf-8", False), ({"LANG": "C", "LC_CTYPE": "en_GB.UTF-8"}, "utf-8", True),
    ({"LANG": "en_US.ISO-8859-1"}, "ISO-8859-1", False), ({}, "ascii", False), ({}, None, False)])
def test_unicode_is_used_only_when_the_locale_and_the_stream_can_show_it(env, encoding, expected):
    """Python coerces a C locale to UTF-8 for its streams, so the stream's encoding alone proves nothing."""
    assert cli.unicode_ok(env, encoding) is expected


# ---- nuhpc standing: every queue, and how busy the whole cluster is ----

def cluster(out=None, **thresholds):
    return standing(out, **thresholds)["cluster"]


def by_part(c):
    return {p["partition"]: p for p in c["partitions"]}


def without(out, *names):
    """The remote output with whole sections left out, as when a query fails."""
    keep, skip = [], False
    for line in out.splitlines():
        if line.startswith(M + " "):
            skip = line[len(M) + 1:] in names
        if not skip:
            keep.append(line)
    return "\n".join(keep) + "\n"


def test_standing_lists_every_queue_with_its_cap_and_time_limit():
    q = {p["partition"]: p for p in standing()["queue"]["partitions"]}
    assert set(q) == {"short", "gpu", "gpu-short", "gpu-interactive", "sharing"}          # idle queues are listed too
    assert (q["sharing"]["running"], q["sharing"]["pending"], q["sharing"]["cap"], q["sharing"]["limit_h"]) == (0, 0, 2, 1)
    assert q["short"]["limit_h"] == 48 and q["gpu"]["limit_h"] == 8 and q["gpu-interactive"]["cap"] == 2
    assert standing()["queue"]["label"] == "OK"                                              # idle queues change no verdict
    odd = standing(standing_out(part=PART + ["PartitionName=odd QoS=N/A MaxTime=UNLIMITED "]))["queue"]["partitions"]
    assert next(p for p in odd if p["partition"] == "odd")["cap"] is None and next(p for p in odd if p["partition"] == "odd")["limit_h"] is None


def test_cluster_counts_running_waiting_and_held_jobs_per_queue():
    c = by_part(cluster())
    s = c["short"]
    assert (s["running"], s["waiting"], s["held"]) == (3, 7, 4)       # an array of 4 is 4; dependency, hold, cap, start time are held
    assert s["oldest_waiting_h"] == pytest.approx(12.0)
    g, gs = c["gpu"], c["gpu-short"]
    assert (g["running"], g["waiting"]) == (2, 2) and (gs["waiting"], gs["oldest_waiting_h"]) == (2, pytest.approx(1.0))   # waiting in two queues counts in both
    assert (c["sharing"]["running"], c["sharing"]["waiting"], c["sharing"]["oldest_waiting_h"]) == (0, 0, None)


def test_cluster_waits_count_only_starts_inside_the_window_from_when_a_job_became_eligible():
    c = cluster()
    assert c["window_h"] == 6 and by_part(c)["short"]["started"] == 6                 # the 3-day-old start and the pending row are out
    assert by_part(c)["short"]["median_wait_min"] == 90 and by_part(c)["short"]["p90_wait_min"] == 300
    assert by_part(c)["gpu"]["median_wait_min"] == pytest.approx(3.5) and by_part(c)["sharing"]["median_wait_min"] is None
    wide = by_part(cluster(wait_window_h=100))["short"]
    assert wide["started"] == 7 and wide["median_wait_min"] == 120                  # now the 3-day-old start is inside
    held = standing_out(waits=[started("short", 5, ago_min=10, eligible="Unknown")])    # no eligible time: fall back to the submit time
    assert by_part(cluster(held))["short"]["median_wait_min"] is None and by_part(cluster(held))["short"]["started"] == 1
    many = standing_out(waits=[started("short", 5, ago_min=10 + i, eligible="Unknown") for i in range(5)])
    assert by_part(cluster(many))["short"]["median_wait_min"] == 1005


def test_cluster_labels_follow_the_median_wait_and_need_enough_starts():
    c = cluster()
    assert {p["partition"]: p["label"] for p in c["partitions"]} == {
        "short": "OK", "gpu": "GOOD", "gpu-short": "unknown", "gpu-interactive": "GOOD", "sharing": "GOOD"}
    assert c["label"] == "OK" and "short" in c["note"] and "1.5 h" in c["note"]
    strict = cluster(cluster_wait_ok_min=10, cluster_wait_poor_min=80)
    assert by_part(strict)["short"]["label"] == "POOR" and by_part(strict)["gpu"]["label"] == "GOOD" and strict["label"] == "POOR"
    lenient = cluster(cluster_wait_ok_min=100)
    assert by_part(lenient)["short"]["label"] == "GOOD" and lenient["label"] == "GOOD"


def test_cluster_capacity_counts_busy_cores_and_free_gpus_on_usable_nodes():
    c = by_part(cluster())
    assert c["short"]["cores_busy_pct"] == pytest.approx(100 * 3157 / (3157 + 439))   # cores on down or drained nodes are not capacity
    assert c["gpu"]["gpus"] == {"a100": {"total": 7, "used": 7, "free": 0}, "h200": {"total": 8, "used": 6, "free": 2},
                                "t4": {"total": 4, "used": 0, "free": 4}}              # the down h200 node is not counted
    assert c["gpu-short"]["gpus"] == c["gpu"]["gpus"] and c["sharing"]["gpus"] == {"l40s": {"total": 3, "used": 1, "free": 2}}
    assert c["short"]["gpus"] == {}


def test_cluster_load_never_changes_the_overall_verdict():
    calm = ["short|RUNNING|None|2026-10-16T08:00:00|1|a"] * 5
    s = standing(standing_out(share=[r.replace("|0.132704|", "|0.8|") for r in SHARE_ROWS], queue=calm),
                 cluster_wait_ok_min=1, cluster_wait_poor_min=2)
    assert s["cluster"]["label"] == "POOR" and s["overall"] == "GOOD" and s["brief"] == "fair share good 0.80, queue good"


def test_cluster_is_unknown_when_a_query_fails_or_the_scheduler_hides_other_users():
    full = standing_out()
    for gone in ("waits", "cqueue"):
        c = cluster(without(full, gone))
        assert c["label"] == "unknown" and "unavailable" in c["note"], gone
    only_me = standing_out(cqueue=[r.rsplit("|", 1)[0] + "|me.pi" for r in CQUEUE])
    c = cluster(only_me)
    assert c["label"] == "unknown" and "only your own jobs" in c["note"]
    s = standing("")
    assert s["cluster"]["label"] == "unknown" and s["cluster"]["partitions"] == [] and s["advice"] == []
    assert "u1" not in json.dumps(standing()) and "u4" not in json.dumps(standing())          # other users are counted, never named


def advice(out=None, **th):
    return standing(out, **th)["advice"]


def test_advice_is_empty_when_everything_is_good():
    quick = [started("short", 5, ago_min=30 + 10 * i) for i in range(6)]
    s = standing(standing_out(share=[r.replace("|0.132704|", "|0.8|") for r in SHARE_ROWS], gpus=[], waits=quick,
                              queue=["short|RUNNING|None|2026-10-16T08:00:00|1|a"] * 5))
    assert s["overall"] == "GOOD" and s["cluster"]["label"] == "GOOD" and s["advice"] == []


def test_advice_follows_the_caps_the_fair_share_busy_queues_and_full_gpu_types():
    assert [a["code"] for a in advice()] == ["fairshare", "gpu_full"]            # POOR fair share; no A100 is free; queue and cluster are OK
    text = {a["code"]: a["text"] for a in advice()}
    assert "a100" in text["gpu_full"] and "h200 2" in text["gpu_full"] and "t4 4" in text["gpu_full"]
    assert "--time" in text["fairshare"] and "pilot" in text["fairshare"]
    blocked = ["short|RUNNING|None|2026-10-16T08:00:00|1|a"] * 50 + ["short|PENDING|QOSMaxJobsPerUserLimit|2026-10-16T09:00:00|2|b"]
    assert [a["code"] for a in advice(standing_out(queue=blocked))] == ["caps", "fairshare", "gpu_full"]
    assert "--rows-per-job" in {a["code"]: a["text"] for a in advice(standing_out(queue=blocked))}["caps"]
    busy = advice(cluster_wait_ok_min=10, cluster_wait_poor_min=80)
    assert [a["code"] for a in busy] == ["fairshare", "busy_queue", "gpu_full"]
    t = next(a["text"] for a in busy if a["code"] == "busy_queue")
    assert "short" in t and "1.5 h" in t and "--time" in t and "resubmit" in t


@pytest.mark.parametrize("minutes,text", [(45, "45 min"), (90, "1.5 h"), (300, "5 h"), (888, "14.8 h"), (2880, "2 d"), (6000, "4.17 d")])
def test_durations_read_in_the_unit_that_fits(minutes, text):
    assert cli.fmt_dur(minutes) == text


def test_the_wait_window_and_thresholds_come_from_the_standing_table():
    assert cli.standing_thresholds({})["wait_window_h"] == 6 and cli.standing_thresholds({})["cluster_wait_poor_min"] == 120
    assert cli.standing_thresholds({"standing": {"wait_window_h": 12}})["wait_window_h"] == 12
    assert "now-360minutes" in cli.standing_query() and "now-120minutes" in cli.standing_query(2)


def test_standing_remote_shell_also_reads_the_whole_cluster(tmp_path):
    stubs = {
        "sacctmgr": 'case "$*" in *assoc*) echo lab.pi;; esac',
        "scontrol": 'case "$*" in *partition*) echo "PartitionName=short QoS=short MaxTime=2-00:00:00";; esac',
        "squeue": 'case "$*" in *--me*) ;; *-a*) printf "short|RUNNING|None|2026-10-16T08:00:00|6|u8\\nshort|PENDING|Priority|2026-10-16T09:00:00|5|u9\\n";; esac',
        "sinfo": 'case "$*" in *%C*) echo "short|10/90/0/100";; *-N*) echo "gpu g1 mixed gpu:a100:4(S:0-1) gpu:a100:1(IDX:0)";; esac',
        "sacct": 'case "$*" in *-a*) for i in 1 2 3 4 5; do echo "short|2026-10-16T08:00:00|2026-10-16T09:40:00|2026-10-16T09:50:00"; done;; esac',
        "date": "echo 2026-10-16T10:00:00"}
    env = {**os.environ, "USER": "me.pi", "PATH": f"{stub_bin(tmp_path, **stubs)}:{os.environ['PATH']}"}
    p = subprocess.run(["bash", "-c", cli.standing_query()], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    c = cli.parse_standing(p.stdout, {})["cluster"]
    short = by_part(c)["short"]
    assert (short["running"], short["waiting"], short["started"], short["median_wait_min"]) == (1, 1, 5, 10)   # eligible 09:40, started 09:50
    assert short["cores_busy_pct"] == pytest.approx(10.0)
    assert "u9" not in json.dumps(cli.parse_standing(p.stdout, {}))
    assert not any(w in cli.standing_query() for w in ("scancel", "sbatch", "srun", "scontrol update", "kill"))


def test_standing_json_carries_the_cluster_and_the_advice(inproc, monkeypatch, capsys):
    inproc([])
    seen = []
    monkeypatch.setattr(cli.Remote, "run", lambda self, cmd, retry=True: seen.append(cmd) or standing_out())
    cli.main(["--json", "standing"])
    d = json.loads(capsys.readouterr().out)
    assert d["cluster"]["label"] == "OK" and d["advice"][0]["code"] == "fairshare" and "now-360minutes" in seen[0]
    assert d["brief"] == "fair share poor 0.13, queue ok"                              # the morning line is unchanged


# ---- nuhpc standing: how the cluster section looks ----

def test_report_has_every_queue_a_cluster_section_and_advice():
    lines = render(width=100).splitlines()
    assert any(l.strip().startswith("sharing") and "0 of 2 running" in l and "1 h max" in l for l in lines)
    assert any(l.strip().startswith("short") and "2 of 50 running" in l and "2 d max" in l for l in lines)
    assert any(l.startswith("─") and " Cluster " in l for l in lines)
    i = next(i for i, l in enumerate(lines) if l.strip().startswith("short") and "busy" in l)
    assert all(x in lines[i] for x in ("88% busy", "OK", "7 waiting", "4 held"))
    assert lines[i + 1].strip() == "median wait 1.5 h · slowest tenth 5 h · oldest waiting 12 h"
    j = next(i for i, l in enumerate(lines) if l.strip().startswith("gpu-short") and "busy" in l)
    assert "unknown" in lines[j] and lines[j + 1].strip() == "only 3 jobs started in the last 6 h · oldest waiting 1 h"
    k = next(i for i, l in enumerate(lines) if l.strip().startswith("sharing") and "busy" in l)
    assert lines[k + 1].strip() == "no job started in the last 6 h"
    g = next(i for i, l in enumerate(lines) if l.strip().startswith("free GPUs") and "gpu · gpu-interactive · gpu-short" in l)
    assert lines[g + 1].strip() == "a100 0 of 7 · h200 2 of 8 · t4 4 of 4"
    assert any(l.strip().startswith("free GPUs") and l.strip().endswith("sharing") for l in lines) and any(l.strip() == "l40s 2 of 3" for l in lines)
    assert any(l.strip().startswith("status") and "OK · short median wait 1.5 h" in l for l in lines)
    assert any(l.startswith("─") and " Advice " in l for l in lines) and any("pilot" in l for l in lines)


def test_gpu_types_are_coloured_by_how_many_are_free():
    coloured = render(color=True, width=100)
    assert "\x1b[31ma100 0 of 7" in coloured and "\x1b[33mh200 2 of 8" in coloured and "\x1b[32mt4 4 of 4" in coloured


def test_a_good_standing_has_no_advice_section():
    quick = [started("short", 5, ago_min=30 + 10 * i) for i in range(6)]
    s = standing(standing_out(share=[r.replace("|0.132704|", "|0.8|") for r in SHARE_ROWS], gpus=[], waits=quick,
                              queue=["short|RUNNING|None|2026-10-16T08:00:00|1|a"] * 5))
    assert " Advice " not in cli.render_standing(s, width=100)


def test_the_cluster_section_survives_missing_data():
    s = standing(without(standing_out(), "waits", "cqueue", "cores", "gpus"))
    text = cli.render_standing(s, width=72)
    assert "Cluster" in text and "unknown" in text
    assert all(len(shown(l)) <= 72 for l in cli.render_standing(s, color=True, width=72).splitlines())


def test_queue_and_cluster_rows_line_up_whatever_the_queue_names_are():
    for width in (72, 100):
        lines = render(width=width).splitlines()
        rows = [l for l in lines if re.match(r"  (short|gpu|gpu-short|gpu-interactive|sharing) ", l) and any(c in l for c in "█░")]
        cols = {next(i for i, c in enumerate(l) if c in "█░") for l in rows}
        assert len(rows) == 10 and len(cols) == 1, (width, cols)             # five queues, listed twice, gauges in one column


def test_advice_says_whether_it_calls_for_a_plan_or_is_only_a_note():
    blocked = ["short|RUNNING|None|2026-10-16T08:00:00|1|a"] * 50 + ["short|PENDING|QOSMaxJobsPerUserLimit|2026-10-16T09:00:00|2|b"]
    kinds = {a["code"]: a["kind"] for a in advice(standing_out(queue=blocked), cluster_wait_ok_min=10, cluster_wait_poor_min=80)}
    assert kinds == {"caps": "plan", "fairshare": "plan", "busy_queue": "plan", "gpu_full": "note"}   # a full GPU type changes the choice, not the bill


def test_gpu_advice_names_only_the_few_types_that_matter():
    full = [f"{'sharing':<16} nf{i} mixed gpu:f{i}:2 gpu:f{i}:2(IDX:0-1)" for i in range(1, 5)]
    free = [f"{'sharing':<16} ng{g} mixed gpu:g{g}:8 gpu:g{g}:{8 - n}(IDX:0)" for g, n in zip(range(1, 7), (6, 5, 4, 3, 2, 1))]
    text = next(a["text"] for a in advice(standing_out(gpus=full + free)) if a["code"] == "gpu_full")
    assert "No free f1, f2, f3 (+1 more) GPU on sharing" in text
    assert "Free now: g1 6, g2 5, g3 4, g4 3 (+2 more)" in text and "g5" not in text


def test_a_long_detail_line_continues_on_the_next_line_instead_of_being_cut_off():
    text = render(width=72)
    assert "oldest waiting 12 h" in text and "slowest tenth 5 h" in text and "…" not in text
    wide = render(width=100)
    assert "median wait 1.5 h · slowest tenth 5 h · oldest waiting 12 h" in wide           # when it fits it stays on one line


# ---- usage: what was allocated and not used, for the skill's waste check ----

def test_usage_totals_say_how_much_of_what_was_allocated_went_unused(inproc, monkeypatch, tmp_path, capsys):
    """Fixture: one GPU task sampled at 63.5% and one OOM GPU task (billed 18 of 72 hours; a GPU-hour counts as 100 CPU-hours).
    Cores are judged on CPU-only tasks, since a GPU job's cores are incidental, so there is nothing to report here."""
    usage_transport(monkeypatch, USAGE_REPLY)
    (tmp_path / "results/r1-20260101-000000").mkdir(parents=True)
    cli.main(["--json", "usage", "r1"])
    out = json.loads(capsys.readouterr().out)
    t = out["totals"][-1]
    assert t["project"] == "all" and t["gpu_idle_%"] == 36.5 and t["failed_%"] == 25.0 and t["cpu_idle_%"] == "-"
    assert out["runs"][0]["cpu_idle_%"] == "-"


def task(**kw):
    return {"cpu_eff": None, "cpu_hours": 0.0, "gpus": 0, "gpu_util_mean": None, "gpu_hours": 0.0, "state": "COMPLETED", **kw}


def test_waste_weights_each_task_by_its_hours_and_judges_cores_on_cpu_only_tasks():
    tasks = [task(cpu_hours=4.0, cpu_eff=0.5),                                                        # 2 of 4 core-hours used
             task(cpu_hours=1.0, cpu_eff=0.0, state="FAILED"),                                        # none of 1 used, and lost
             task(cpu_hours=2.0, cpu_eff=0.0, gpus=1, gpu_hours=0.5, gpu_util_mean=60.0),             # a GPU job: its idle cores don't count
             task(cpu_hours=1.0, cpu_eff=0.9, gpus=1, gpu_hours=0.25, state="TIMEOUT")]               # no GPU samples: not in the GPU figure
    w = cli.waste(tasks)
    assert w["cpu_idle_%"] == 60.0                       # (4 + 1 - 2) / 5 core-hours
    assert w["gpu_idle_%"] == 40.0                       # one sampled task at 60%
    assert w["failed_%"] == 32.5                         # (1 + 26) of (4 + 1 + 52 + 26) billed hours
    assert cli.waste([task(cpu_hours=2.0, cpu_eff=0.5, gpus=1, gpu_hours=1.0, gpu_util_mean=50.0)])["cpu_idle_%"] == "-"


def test_waste_is_dashes_when_nothing_can_be_measured():
    assert cli.waste([]) == {"cpu_idle_%": "-", "gpu_idle_%": "-", "failed_%": "-"}
    assert cli.waste([task(state="CANCELLED")]) == {"cpu_idle_%": "-", "gpu_idle_%": "-", "failed_%": "-"}      # nothing billed, nothing lost


# ---- sbatch can report an error and still have created the job ----

def flaky_sbatch(monkeypatch, squeue_reply):
    calls = []

    def fake_run(self, cmd, retry=True, input=None):
        calls.append(cmd)
        if "sbatch" in cmd:
            raise cli.NuhpcError("command failed (1): ssh -o BatchMode=yes ...\nsbatch: error: Batch job submission failed: Unexpected message received")
        if cmd.startswith("squeue"):
            return squeue_reply
        return ""
    monkeypatch.setattr(cli.Remote, "run", fake_run)
    monkeypatch.setattr(cli.Remote, "rsync", lambda self, *a, **k: None)
    return calls


def test_a_submission_that_reports_an_error_but_created_the_job_is_adopted_not_repeated(inproc, monkeypatch, capsys):
    inproc([])
    calls = flaky_sbatch(monkeypatch, "4242\n")
    cli.main(["--json", "submit", "cmd", "-p", "cmd=true", "--partition", "short"])
    out = json.loads(capsys.readouterr().out)
    assert out["job_id"] == "4242" and sum("sbatch" in c for c in calls) == 1                  # one sbatch: nothing submitted twice
    assert any(c.startswith("squeue") and out["remote_dir"] in c for c in calls)               # found by the run folder it was submitted from
    assert cli.ledger_read()[-1]["job_id"] == "4242"


def test_a_failed_submission_that_left_no_job_still_fails(inproc, monkeypatch, capsys):
    inproc([])
    flaky_sbatch(monkeypatch, "")
    with pytest.raises(SystemExit):
        cli.main(["submit", "cmd", "-p", "cmd=true", "--partition", "short"])
    assert "Unexpected message received" in capsys.readouterr().err and cli.ledger_read()[-1]["run_id"].startswith("r1-")
