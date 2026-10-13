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
