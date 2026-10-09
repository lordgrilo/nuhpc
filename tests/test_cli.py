"""Offline tests: pure helpers, submit guardrails, and rendered job scripts executed locally
against stub binaries (no cluster, no ssh). Run with: uv run --with pytest pytest"""
import json
import os
import shutil
import subprocess
import sys
import time
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
