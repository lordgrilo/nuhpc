"""nuhpc disk: the scanner that runs on the cluster, the explorer that shows its result, and the command around them.
Offline: the scanner walks a temporary tree, the explorer is driven by keys and read as text."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nuhpc import diskscan


def make(root: Path, spec: dict) -> Path:
    """spec maps a relative path to a size in bytes (a file) or to None (an empty directory)."""
    for rel, size in spec.items():
        p = root / rel
        if size is None:
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * size)
    return root


TREE = {"top.txt": 100, "a/one.bin": 1000, "a/b/two.bin": 2000, "a/b/three.bin": 300, "c/four.bin": 500, "empty/": None}


def by_name(node):
    return {d["name"]: d for d in node.get("dirs", [])}


# ---- the scanner ----

def test_scan_totals_files_and_the_tree(tmp_path):
    s = diskscan.scan(str(make(tmp_path / "r", TREE)), depth=4)
    t = s["tree"]
    assert (t["bytes"], t["files"]) == (3900, 5) and t["own"] == {"files": 1, "bytes": 100}
    d = by_name(t)
    assert (d["a"]["bytes"], d["a"]["files"]) == (3300, 3) and d["a"]["own"] == {"files": 1, "bytes": 1000}
    assert (by_name(d["a"])["b"]["bytes"], by_name(d["a"])["b"]["files"]) == (2300, 2)
    assert (d["c"]["bytes"], d["empty"]["bytes"], d["empty"]["files"]) == (500, 0, 0)
    assert s["root"] == str(tmp_path / "r") and s["version"] == 1 and s["depth"] == 4 and not s["partial"]
    assert t["newest"] is not None and d["empty"]["newest"] is None


def test_scan_keeps_the_largest_direct_files_of_each_folder(tmp_path):
    s = diskscan.scan(str(make(tmp_path / "r", TREE)), depth=4, per_dir=1)
    b = by_name(by_name(s["tree"])["a"])["b"]
    assert [f["name"] for f in b["top"]] == ["two.bin"] and b["top"][0]["bytes"] == 2000       # one of two kept, the biggest
    assert b["own"] == {"files": 2, "bytes": 2300}                                             # the rest is still counted


def test_scan_stops_listing_at_the_depth_but_keeps_the_totals(tmp_path):
    s = diskscan.scan(str(make(tmp_path / "r", TREE)), depth=1)
    a = by_name(s["tree"])["a"]
    assert a["bytes"] == 3300 and a["files"] == 3 and "dirs" not in a and a["deep"] is True    # b is below the depth
    assert by_name(s["tree"])["c"].get("deep") is not True                                     # nothing below c


def test_scan_reports_the_largest_files_with_paths_relative_to_the_root(tmp_path):
    s = diskscan.scan(str(make(tmp_path / "r", TREE)), depth=1, largest=3)
    assert [(f["path"], f["bytes"]) for f in s["largest"]] == [("a/b/two.bin", 2000), ("a/one.bin", 1000), ("c/four.bin", 500)]


def test_scan_without_names_records_no_file_names_but_still_the_totals(tmp_path):
    s = diskscan.scan(str(make(tmp_path / "r", TREE)), depth=1, names=False)
    assert s["names"] is False and s["largest"] == [] and s["tree"]["bytes"] == 3900
    assert "top" not in s["tree"] and all("top" not in d for d in s["tree"]["dirs"])
    assert "top.txt" not in json.dumps(s) and "two.bin" not in json.dumps(s)


def test_scan_does_not_follow_symlinks(tmp_path):
    root = make(tmp_path / "r", {"real/big.bin": 5000})
    (root / "link").symlink_to(root / "real")                 # a link to a folder: not entered
    (root / "real" / "loop").symlink_to(root)                 # a link back up: no cycle
    s = diskscan.scan(str(root), depth=3)
    assert 5000 <= s["tree"]["bytes"] < 6000 and by_name(s["tree"])["real"]["bytes"] < 6000     # 5000 once, plus the links' own sizes (their target paths)
    assert "link" not in by_name(s["tree"])


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read anything")
def test_scan_counts_unreadable_folders_and_carries_on(tmp_path):
    root = make(tmp_path / "r", {"ok/f.bin": 700, "locked/g.bin": 900})
    (root / "locked").chmod(0)
    try:
        s = diskscan.scan(str(root), depth=3)
    finally:
        (root / "locked").chmod(0o755)
    assert s["tree"]["bytes"] == 700 and s["tree"]["unreadable"] == 1 and s["unreadable"] == 1


def test_scan_skips_snapshot_folders(tmp_path):
    s = diskscan.scan(str(make(tmp_path / "r", {"a/f.bin": 10, ".snapshot/old.bin": 99999})), depth=2)
    assert s["tree"]["bytes"] == 10 and ".snapshot" not in by_name(s["tree"])


def test_scan_stops_early_when_its_time_budget_is_spent_and_says_so(tmp_path):
    s = diskscan.scan(str(make(tmp_path / "r", TREE)), depth=4, budget_s=0)
    assert s["partial"] is True and s["unscanned"] > 0 and s["tree"]["own"]["bytes"] == 100     # the root is always read


def test_a_missing_root_is_an_error_not_an_empty_result(tmp_path):
    with pytest.raises(FileNotFoundError):
        diskscan.scan(str(tmp_path / "nope"))


def test_the_scanner_runs_on_the_clusters_python_3_9():
    src = Path(diskscan.__file__).read_text()
    ast.parse(src, feature_version=(3, 9))                    # the compute nodes' system python; no 3.10+ syntax allowed


def test_the_scanner_runs_as_a_script_and_writes_json(tmp_path):
    root = make(tmp_path / "r", TREE)
    out = tmp_path / "disk.json"
    p = subprocess.run([sys.executable, diskscan.__file__, "--root", str(root), "--depth", "2", "--workers", "4",
                        "--out", str(out)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    s = json.loads(out.read_text())
    assert s["tree"]["bytes"] == 3900 and s["workers"] == 4 and s["host"]
    q = subprocess.run([sys.executable, diskscan.__file__, "--root", str(root), "--no-names", "--out", str(out)],
                       capture_output=True, text=True)
    assert q.returncode == 0 and json.loads(out.read_text())["names"] is False


# ---- the explorer: what the scan looks like, and how you move through it ----

import re

from nuhpc import diskview

K, M, G = 1024, 1024 ** 2, 1024 ** 3
NOW, DAY = 1_800_000_000, 86400
SCANNED = NOW - 3 * 3600                                   # ages count from when the scan ran
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def shown(text):
    return ANSI.sub("", text)


def mk(name, dirs=(), own=(0, 0), top=(), age=None, deep=False):
    """A scan node whose totals are the sum of its parts. own is (files, bytes) directly inside it; age is days since the newest change."""
    n = {"name": name, "bytes": own[1] + sum(d["bytes"] for d in dirs), "files": own[0] + sum(d["files"] for d in dirs),
         "newest": None if age is None else SCANNED - int(age * DAY), "own": {"files": own[0], "bytes": own[1]}}
    if top:
        n["top"] = [{"name": a, "bytes": b, "mtime": SCANNED - int(c * DAY)} for a, b, c in top]
    if dirs:
        n["dirs"] = list(dirs)
    if deep:
        n["deep"] = True
    return n


HUB = mk("hub", own=(200, 599 * G), age=12, deep=True)                                    # below the scan's depth: totals only
MODELS = mk("models", dirs=[HUB], own=(14, 1 * G), top=[("tokenizer.bin", 600 * M, 12), ("config.json", 4 * K, 40)], age=12)
ETREE = mk("me", dirs=[MODELS, mk("runs", own=(9000, 80 * G), age=0), mk("envs", own=(2500, 15 * G), age=400),
                      mk("notes", own=(10, 400 * M), age=800)],
          own=(4, 5 * G), top=[("dump.tar", 4 * G, 30), ("readme.md", 4 * K, 5)], age=0)
SCAN = {"version": 1, "root": "/projects/me", "host": "node1", "depth": 2, "names": True, "scanned_at": SCANNED,
        "scanned": "2026-10-23T10:40:00", "partial": False, "unscanned": 0, "unreadable": 0, "n_dirs": 5, "elapsed_s": 41.0,
        "tree": ETREE,
        "largest": [{"path": "models/hub/model-00001.safetensors", "bytes": 300 * G, "mtime": SCANNED - 12 * DAY},
                    {"path": "models/hub/model-00002.safetensors", "bytes": 299 * G, "mtime": SCANNED - 12 * DAY},
                    {"path": "dump.tar", "bytes": 4 * G, "mtime": SCANNED - 30 * DAY}]}


def explorer(**kw):
    return diskview.Explorer(SCAN, now=NOW, **kw)


def lines_of(ex, width=100, height=None, **kw):
    return ex.frame(width=width, height=height, **kw).splitlines()


@pytest.mark.parametrize("n,text", [(0, "0 B"), (1023, "1023 B"), (1024, "1 K"), (1536, "1.5 K"), (600 * M, "600 M"),
                                    (677340334328, "631 G"), (15 * G, "15 G"), (int(26.8 * G), "26.8 G"),
                                    (1000 * K, "0.977 M"), (int(2.5 * 1024 ** 4), "2.5 T")])
def test_sizes_read_like_du_and_df(n, text):
    assert diskview.fmt_size(n) == text


@pytest.mark.parametrize("days,text", [(0, "today"), (0.5, "today"), (1, "1 d"), (13, "13 d"), (14, "2 w"), (45, "6 w"), (89, "12 w"),
                                       (90, "3 mo"), (400, "13 mo"), (729, "24 mo"), (730, "2 y"), (2000, "5 y"), (-3, "today")])
def test_ages_use_the_unit_that_fits(days, text):
    assert diskview.fmt_age(days * DAY) == text


@pytest.mark.parametrize("secs,text", [(5, "just now"), (300, "5 min ago"), (3 * 3600, "3 h ago"), (2 * DAY, "2 d ago")])
def test_the_scan_age_reads_naturally(secs, text):
    assert diskview.fmt_ago(secs) == text


def test_bars_fill_in_eighths_and_keep_their_width():
    assert diskview.hbar(0.5, 20) == "█" * 10 + "░" * 10 and diskview.hbar(0, 20) == "░" * 20 and diskview.hbar(1, 20) == "█" * 20
    b = diskview.hbar(0.01, 20)
    assert len(b) == 20 and b[0] in "▏▎" and b[1:] == "░" * 19                       # a sliver still shows
    assert diskview.hbar(0.5, 10, uni=False) == "#" * 5 + "." * 5 and diskview.hbar(0.001, 10, uni=False) == "#" + "." * 9
    assert len(diskview.hbar(0.337, 13)) == 13


@pytest.mark.parametrize("data,keys", [(b"\x1b[A", ["up"]), (b"\x1b[B", ["down"]), (b"\x1b[C", ["right"]), (b"\x1b[D", ["left"]),
                                       (b"\x1bOA", ["up"]), (b"\x1b[5~", ["pgup"]), (b"\x1b[6~", ["pgdn"]), (b"\x1b[H", ["home"]),
                                       (b"\x1b[F", ["end"]), (b"\r", ["enter"]), (b"\x7f", ["backspace"]), (b"\x1b", ["esc"]),
                                       (b"\x03", ["ctrl-c"]), (b"jjk", ["j", "j", "k"]), (b"\x1b[B\x1b[Bq", ["down", "down", "q"]),
                                       (b"\x1b[99~", [])])
def test_keys_are_read_from_terminal_bytes(data, keys):
    assert diskview.parse_keys(data) == keys


def test_entries_list_folders_files_and_the_rest_biggest_first_with_their_share():
    es = explorer().entries()
    assert [e["name"] for e in es] == ["models/", "runs/", "envs/", "dump.tar", "(2 smaller files)", "notes/", "readme.md"]
    models = es[0]
    assert models["kind"] == "dir" and models["files"] == 214 and models["pct"] == pytest.approx(100 * 600 * G / ETREE["bytes"])
    assert es[3]["kind"] == "file" and es[4]["kind"] == "other" and es[4]["bytes"] == 1 * G - 4 * K
    assert sum(e["pct"] for e in es) == pytest.approx(100)


def test_the_arrow_keys_move_open_and_go_back_remembering_where_you_were():
    ex = explorer()
    assert ex.cursor == 0 and ex.key("down") and ex.cursor == 1 and ex.key("up") and ex.key("up") and ex.cursor == 0
    ex.key("end")
    assert ex.cursor == 6
    ex.key("home")
    ex.key("j"), ex.key("j")                                   # envs
    assert ex.entries()[ex.cursor]["name"] == "envs/"
    ex.key("enter")
    assert ex.path == ["envs"] and ex.cursor == 0
    ex.key("left")
    assert ex.path == [] and ex.cursor == 2                    # back on envs, not at the top
    ex.key("left")
    assert ex.path == [] and ex.key("backspace")               # nothing above the root: stays put
    ex.key("home")
    ex.key("right")
    assert ex.path == ["models"] and [e["name"] for e in ex.entries()][0] == "hub/"


def test_a_folder_below_the_scans_depth_says_so_and_a_file_shows_its_details():
    ex = explorer()
    ex.key("enter"), ex.key("enter")                           # models, then hub
    assert ex.path == ["models", "hub"] and [e["name"] for e in ex.entries()] == ["(200 files)"]
    text = "\n".join(lines_of(ex))
    assert "depth 2" in text and "--depth" in text
    ex = explorer()
    ex.key("end"), ex.key("up"), ex.key("up"), ex.key("up")    # readme.md, then up three: dump.tar
    assert ex.entries()[ex.cursor]["name"] == "dump.tar"
    assert ex.key("enter") and ex.path == [] and "dump.tar" in ex.message and "4 G" in ex.message and "4 w ago" in ex.message


def test_sorting_by_age_puts_the_stalest_first_and_s_toggles_back():
    ex = explorer()
    ex.key("s")
    assert ex.sort == "age" and [e["name"] for e in ex.entries()] == [
        "notes/", "envs/", "dump.tar", "models/", "readme.md", "runs/", "(2 smaller files)"]
    ex.key("s")
    assert ex.sort == "size" and ex.entries()[0]["name"] == "models/"


def test_q_and_ctrl_c_quit_and_anything_else_carries_on():
    ex = explorer()
    assert ex.key("zzz") and ex.key("") and ex.key("pgdn") and ex.key("pgup")
    assert ex.key("q") is False and explorer().key("ctrl-c") is False


def test_the_frame_has_a_title_breadcrumb_rows_and_a_footer():
    ex = explorer()
    lines = lines_of(ex)
    assert lines[0].strip().startswith("nuhpc disk") and "/projects/me" in lines[0]
    assert "3 h ago" in lines[1] and "700 G" in lines[1] and "11,728 files" in lines[1] and lines[2].strip() == "me"
    row = next(l for l in lines if "models/" in l)
    pct = f"{100 * 600 * G / ETREE['bytes']:.1f}%"
    assert all(x in row for x in (pct, "600 G", "214 files", "12 d", "█")) and row.lstrip().startswith("❯")
    assert not next(l for l in lines if "runs/" in l).lstrip().startswith("❯")
    assert "q quit" in lines[-1] and "1/7" in lines[-1] and "sort: size" in lines[-1]
    ex.key("enter")
    assert lines_of(ex)[2].strip() == "me › models"
    dump = next(l for l in lines if "dump.tar" in l)
    assert "4 G" in dump and "4 w" in dump and "files" not in dump
    assert "(2 smaller files)" in next(l for l in lines if "smaller" in l)


def test_the_frame_scrolls_to_keep_the_selected_row_in_view():
    kids = [mk(f"dir{i:02d}", own=(1, (100 - i) * M), age=1) for i in range(30)]
    ex = diskview.Explorer({**SCAN, "tree": mk("many", dirs=kids)}, now=NOW)
    for _ in range(25):
        ex.key("down")
    lines = lines_of(ex, height=12)
    assert len(lines) <= 12 and any("dir25/" in l and l.lstrip().startswith("❯") for l in lines) and not any("dir00/" in l for l in lines)
    assert "26/30" in lines[-1]
    ex.key("home")
    assert any("dir00/" in l for l in lines_of(ex, height=12))


def test_the_largest_files_view_ranks_files_and_reveals_where_one_lives():
    ex = explorer()
    ex.key("l")
    lines = lines_of(ex)
    assert ex.view == "largest" and "largest files" in lines[2]
    rows = [l for l in lines if "model-0000" in l or "dump.tar" in l]
    assert [("model-00001" in r, "model-00002" in r, "dump.tar" in r) for r in rows] == [(True, False, False), (False, True, False), (False, False, True)]
    assert "300 G" in rows[0] and "12 d" in rows[0] and "█" * 20 in rows[0]
    ex.key("enter")
    assert ex.view == "tree" and ex.path == ["models", "hub"]
    ex.key("l"), ex.key("l")
    assert ex.view == "tree"                                   # l toggles
    deep = {**SCAN, "largest": [{"path": "models/hub/snapshots/0123456789abcdef/model-00001.safetensors", "bytes": 300 * G, "mtime": SCANNED - 12 * DAY}]}
    narrow = lines_of(diskview.Explorer(deep, now=NOW, view="largest"), width=50)
    assert any("…" in l and "model-00001.safetensors" in l for l in narrow)           # long paths are cut at the front, keeping the file name


def test_without_file_names_the_largest_view_explains_itself():
    bare = {**SCAN, "names": False, "largest": []}
    ex = diskview.Explorer(bare, now=NOW)
    assert ex.key("l") and ex.view == "tree" and "no file names" in ex.message


def test_a_partial_or_unreadable_scan_is_flagged_in_the_header():
    text = "\n".join(lines_of(diskview.Explorer({**SCAN, "partial": True, "unscanned": 7, "unreadable": 2}, now=NOW)))
    assert "partial" in text and "7 folders not fully read" in text and "2 unreadable" in text


def test_colours_are_pastel_cycle_by_rank_and_fade_with_age():
    plain, coloured = explorer().frame(width=100), explorer().frame(width=100, color=True)
    assert "\x1b[" not in plain and shown(coloured) == plain
    assert "\x1b[38;5;217m" in coloured and "\x1b[38;5;183m" in coloured and "\x1b[38;5;153m" in coloured    # pink, lavender, sky bars
    notes = next(l for l in coloured.splitlines() if "notes/" in l)
    runs = next(l for l in coloured.splitlines() if "runs/" in l)
    assert "\x1b[38;5;211m2 y" in notes or "\x1b[38;5;211m26 mo" in notes                                       # an old folder's age is rose
    assert "\x1b[38;5;158mtoday" in runs                                                                      # a fresh one's is mint


def test_no_row_runs_past_the_terminal_width_and_the_ascii_form_is_ascii():
    for width in (50, 72, 100):
        for view in ("tree", "largest"):
            ex = diskview.Explorer(SCAN, now=NOW, view=view)
            assert all(len(shown(l)) <= width for l in ex.frame(width=width, color=True).splitlines()), (width, view)
    text = explorer().frame(width=80, uni=False)
    assert all(ord(c) < 128 for c in text) and "#" in text and ">" in text
    assert all(ord(c) < 128 for c in diskview.Explorer(SCAN, now=NOW, view="largest").frame(width=80, uni=False))


def test_a_static_frame_shows_every_row_and_no_key_help():
    kids = [mk(f"dir{i:02d}", own=(1, (100 - i) * M), age=1) for i in range(30)]
    lines = lines_of(diskview.Explorer({**SCAN, "tree": mk("many", dirs=kids)}, now=NOW), footer=False)
    assert sum("dir" in l for l in lines) == 30 and not any("q quit" in l for l in lines)


# ---- the command: scopes, the cache, the scan job, and the full-screen loop ----

import fcntl
import pty
import select
import shutil
import struct
import termios
import time

from nuhpc import cli
from test_cli import env, inproc, nuhpc as run_nuhpc  # noqa: F401  (fixtures and the dry-run helper of the main suite)


def cfg_of(tmp_path, **extra):
    return {"user_dir": "/projects/lab/me", "projects": {"alpha": {"dir": "/projects/lab/me/alpha"}}, **extra}


def scan_file(tmp_path, root="/projects/me", name="scan.json"):
    f = tmp_path / name
    f.write_text(json.dumps({**SCAN, "root": root}))
    return f


def args_for(**kw):
    import argparse
    return argparse.Namespace(**{"project": None, "lab": False, "depth": None, **kw})


def test_a_scope_is_your_folder_a_declared_project_or_the_lab_and_nothing_else(tmp_path):
    cfg = cfg_of(tmp_path)
    me = cli.disk_scope(cfg, args_for())
    assert (me["root"], me["depth"], me["names"], me["time"]) == ("/projects/lab/me", 4, True, "00:30:00")
    assert cli.disk_scope(cfg, args_for(depth=6))["depth"] == 6
    alpha = cli.disk_scope(cfg, args_for(project="alpha"))
    assert alpha["root"] == "/projects/lab/me/alpha" and alpha["project"] == "alpha" and alpha["names"] is True
    lab = cli.disk_scope(cfg, args_for(lab=True))
    assert (lab["root"], lab["depth"], lab["names"], lab["time"]) == ("/projects/lab", 1, False, "02:00:00")   # members' totals only, no file names
    assert cli.disk_scope(cfg, args_for(lab=True, depth=5))["depth"] == 1                                      # --depth can't open other people's folders
    with pytest.raises(cli.NuhpcError, match="declared"):
        cli.disk_scope(cfg, args_for(project="nope"))
    with pytest.raises(cli.NuhpcError, match="either"):
        cli.disk_scope(cfg, args_for(project="alpha", lab=True))
    with pytest.raises(cli.NuhpcError, match="group folder"):
        cli.disk_scope({"user_dir": "/projects/me"}, args_for(lab=True))               # the parent would be /projects, shared by everyone


def test_the_cache_key_comes_from_the_folder_that_was_scanned(tmp_path):
    cfg = cfg_of(tmp_path)
    assert cli.scope_key(cfg, "/projects/lab/me") == "user" and cli.scope_key(cfg, "/projects/lab") == "lab"
    assert cli.scope_key(cfg, "/projects/lab/me/alpha") == "project-alpha"
    assert cli.scope_key(cfg, "/somewhere/else") == "other-somewhere-else"


def test_a_saved_scan_prints_as_a_static_report_and_as_json(inproc, tmp_path, capsys):
    inproc([])
    f = scan_file(tmp_path)
    cli.main(["disk", "--from", str(f)])
    out = capsys.readouterr().out
    assert "nuhpc disk" in out and "/projects/me" in out and "models/" in out and "q quit" not in out and "\x1b[" not in out
    cli.main(["disk", "--from", str(f), "models"])
    assert "me › models" in capsys.readouterr().out
    cli.main(["--json", "disk", "--from", str(f)])
    assert json.loads(capsys.readouterr().out)["tree"]["bytes"] == ETREE["bytes"]
    cli.main(["--json", "disk", "--from", str(f), "models"])
    assert json.loads(capsys.readouterr().out)["name"] == "models"
    with pytest.raises(SystemExit):
        cli.main(["disk", "--from", str(f), "models/nope"])
    assert "not in this scan" in capsys.readouterr().err
    (tmp_path / "bad.json").write_text('{"hello": 1}')
    with pytest.raises(SystemExit):
        cli.main(["disk", "--from", str(tmp_path / "bad.json")])
    assert "not a nuhpc disk scan" in capsys.readouterr().err


def test_without_a_scan_the_command_says_how_to_make_one(inproc, tmp_path, capsys):
    inproc([])
    with pytest.raises(SystemExit):
        cli.main(["disk"])
    assert "nuhpc disk --scan" in capsys.readouterr().err
    cache = tmp_path / "state/disk"
    cache.mkdir(parents=True)
    (cache / "user.json").write_text(json.dumps({**SCAN, "root": str(tmp_path / "remote")}))
    cli.main(["disk"])
    out = capsys.readouterr().out
    assert "nuhpc disk" in out and "scanned" in out and "models/" in out


class Cluster:
    """A fake cluster for `disk --scan`: sbatch gives a job id, sacct a state, and rsync of disk.json delivers a scan."""

    def __init__(self, monkeypatch, scan_json, states=("COMPLETED",)):
        self.calls, self.uploads, self.states = [], [], list(states)
        self.scan_json = scan_json

        def run(me, cmd, retry=True):
            self.calls.append(cmd)
            if "sbatch" in cmd:
                return "777\n"
            if cmd.startswith("sacct"):
                st = self.states.pop(0) if len(self.states) > 1 else self.states[0]
                return f"777|{st}|00:02:10|0:0\n"
            return ""

        def rsync(me, src, dst, extra=None):
            if src.rstrip("/").endswith("disk.json") or src.endswith("/disk.json"):
                shutil.copy(self.scan_json, dst)
            elif ":" not in src:                                  # an upload: remember what the folder held
                self.uploads.append((src, sorted(os.listdir(src))))

        monkeypatch.setattr(cli.Remote, "run", run)
        monkeypatch.setattr(cli.Remote, "rsync", rsync)


def staged(tmp_path, run_id_prefix="disk-scan"):
    stage = next((tmp_path / "state/runs").glob(f"{run_id_prefix}-*"))
    return stage, [json.loads(l) for l in (stage / "sweep.jsonl").read_text().splitlines()]


def test_a_job_scan_submits_the_job_waits_caches_the_result_and_shows_it(inproc, monkeypatch, tmp_path, capsys):
    inproc([])
    root = str(tmp_path / "remote")
    cluster = Cluster(monkeypatch, scan_file(tmp_path, root=root))
    cli.main(["disk", "--scan", "--job"])
    out = capsys.readouterr().out
    assert "nuhpc disk" in out and "models/" in out                                            # (a long root is cut to the width)
    stage, rows = staged(tmp_path)
    assert rows == [{"root": root, "depth": 4, "names": True, "budget": 1620}]               # 30 min less 3 for a clean exit
    job = (stage / "job.sbatch").read_text()
    assert "--time=00:30:00" in job and "--cpus-per-task=4" in job and "--partition=short" in job and "diskscan.py" in job
    assert any("diskscan.py" in files for _, files in cluster.uploads if files == ["diskscan.py"])   # the scanner travels with the job
    assert (tmp_path / "state/disk/user.json").exists()
    cached = json.loads((tmp_path / "state/disk/user.json").read_text())
    assert cached["tree"]["bytes"] == ETREE["bytes"] and cached["root"] == root
    assert [e["template"] for e in cli.ledger_read()][-1] == "disk_scan"


def test_a_lab_job_scan_walks_the_group_folder_to_one_level_and_records_no_file_names(inproc, monkeypatch, tmp_path):
    inproc([])
    cluster = Cluster(monkeypatch, scan_file(tmp_path, root=str(tmp_path)))
    cli.main(["disk", "--scan", "--lab", "--job"])
    stage, rows = staged(tmp_path)
    assert rows == [{"root": str(tmp_path), "depth": 1, "names": False, "budget": 7020}]     # 2 h less 3 min
    assert "--time=02:00:00" in (stage / "job.sbatch").read_text()
    assert (tmp_path / "state/disk/lab.json").exists()
    assert cluster.calls


def test_a_scan_still_waiting_exits_3_and_says_how_to_pick_it_up(inproc, monkeypatch, tmp_path, capsys):
    inproc([])
    Cluster(monkeypatch, scan_file(tmp_path), states=("PENDING",))
    with pytest.raises(SystemExit) as e:
        cli.main(["--json", "disk", "--scan", "--job", "--timeout", "0"])
    assert e.value.code == 3
    out = json.loads(capsys.readouterr().out)
    assert out["timed_out"] is True and out["run_id"].startswith("disk-scan-") and f"nuhpc disk --collect {out['run_id']}" in out["next"]
    assert not (tmp_path / "state/disk").exists()


def test_a_scan_job_that_failed_is_an_error_that_points_at_its_log(inproc, monkeypatch, tmp_path, capsys):
    inproc([])
    Cluster(monkeypatch, scan_file(tmp_path), states=("FAILED",))
    with pytest.raises(SystemExit):
        cli.main(["disk", "--scan", "--job"])
    assert "nuhpc logs disk-scan-" in capsys.readouterr().err


def test_collect_fetches_a_finished_scan_into_the_cache_under_its_own_scope(inproc, monkeypatch, tmp_path, capsys):
    inproc([])
    cli.ledger_append({"run_id": "disk-scan-20261023-100000", "job_id": "777", "template": "disk_scan", "profile": None, "n_tasks": 1,
                       "remote_dir": "/r/runs/disk-scan-20261023-100000", "created": "x", "worst_case_gpu_hours": 0})
    Cluster(monkeypatch, scan_file(tmp_path, root=str(tmp_path / "remote")))
    cli.main(["disk", "--collect", "disk-scan-20261023-100000"])
    assert "nuhpc disk" in capsys.readouterr().out and (tmp_path / "state/disk/user.json").exists()


def test_a_dry_run_job_scan_submits_nothing_and_shows_nothing(inproc, monkeypatch, tmp_path, capsys):
    inproc([])
    Cluster(monkeypatch, scan_file(tmp_path))
    cli.main(["--json", "--dry-run", "disk", "--scan", "--job"])
    out = json.loads(capsys.readouterr().out)
    assert out["template"] == "disk_scan" and not (tmp_path / "state/disk").exists()


def test_the_scan_template_runs_the_scanner_with_the_jobs_params(tmp_path, env):
    root = make(tmp_path / "tree", TREE)
    rc, res = run_nuhpc(env, "submit", "disk_scan", "--partition", "short", "-p", f"root={root}", "-p", "depth=2", "-p", "names=false",
                        "-p", "budget=100")
    assert rc == 0
    run_dir = Path(res["remote_dir"])
    shutil.copytree(res["local_stage"], run_dir)
    (run_dir / "code").mkdir()
    shutil.copy(diskscan.__file__, run_dir / "code/diskscan.py")
    p = subprocess.run(["bash", str(run_dir / "job.sbatch")], capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    s = json.loads((run_dir / "outputs/task_0/disk.json").read_text())
    assert s["names"] is False and s["depth"] == 2 and s["tree"]["bytes"] == 3900 and s["largest"] == []
    assert "disk_scan" in [t["template"] for t in json.loads(subprocess.run([sys.executable, "-m", "nuhpc", "--json", "templates"],
                                                                            capture_output=True, text=True, env=env).stdout)]


def test_the_explorer_runs_full_screen_in_a_terminal_and_puts_it_back(tmp_path, env):
    f = scan_file(tmp_path)
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
    term_env = {**env, "TERM": "xterm-256color", "LANG": "en_US.UTF-8"}
    for k in ("NO_COLOR", "FORCE_COLOR", "LC_ALL", "LC_CTYPE"):
        term_env.pop(k, None)
    proc = subprocess.Popen([sys.executable, "-m", "nuhpc", "disk", "--from", str(f)], stdin=slave, stdout=slave, stderr=slave,
                            env=term_env, close_fds=True)
    os.close(slave)
    out = b""

    def drain(wait=1.0):
        nonlocal out
        end = time.time() + wait
        while time.time() < end:
            if select.select([master], [], [], 0.05)[0]:
                try:
                    out += os.read(master, 65536)
                except OSError:
                    return

    drain(1.5)
    for key in (b"\x1b[B", b"\r", b"\x1b[D", b"l", b"l", b"q"):          # down, open, back, largest files, back to folders, quit
        os.write(master, key)
        drain(0.3)
    drain(0.5)
    assert proc.wait(timeout=10) == 0
    os.close(master)
    text = out.decode()
    assert text.index("\x1b[?1049h") < text.rindex("\x1b[?1049l")                                          # entered the alternate screen, then left it
    assert "\x1b[?25h" in text and "\x1b[38;5;217m" in text                                                # the cursor is back; pastel colours were drawn
    assert "models/" in text and "largest files" in text and "model-00001.safetensors" in text


# ---- the quick scan: du, a few at a time, capped, low priority; nothing written on the cluster ----

from nuhpc import diskquick

# What the remote script prints for a root with a/b (with a child x), c/d, c/slow (never finished), c/slow2 (cut off after one child), e.
QUICK = "\n".join([
    "D\t./a", "D\t./a/b", "D\t./c", "D\t./c/d", "D\t./c/slow", "D\t./c/slow2", "D\t./e",
    "F\t100\t1790000000.5\t.", "F\t1000\t1790000100.25\t./a", "F\t500\t1790000050.0\t./e",
    "2000\t1790000200\t./a/b/x", "2300\t1790000250\t./a/b",
    "700\t1790000300\t./c/d",
    "400\t1790000400\t./c/slow2/part",
]) + "\n"


def quick(out=QUICK, depth=4):
    return diskquick.tree_from_quick(out, "/r", depth=depth, elapsed=12.3, host="login")


def sub(node, *names):
    for n in names:
        node = next(d for d in node["dirs"] if d["name"] == n)
    return node


def test_the_quick_tree_adds_unit_sizes_to_the_files_beside_them():
    q = quick()
    assert q["version"] == 1 and q["mode"] == "quick" and q["root"] == "/r" and q["names"] is False and q["largest"] == []
    t = q["tree"]
    assert t["bytes"] == 100 + (1000 + 2300) + (700 + 400) + 500 and t["files"] is None            # files are not counted by du
    a = sub(t, "a")
    assert a["bytes"] == 3300 and a["own"] == {"files": None, "bytes": 1000} and "incomplete" not in a
    b = sub(a, "b")
    assert b["bytes"] == 2300 and sub(b, "x")["bytes"] == 2000 and b["newest"] == 1790000250
    assert t["newest"] == 1790000400 and sub(t, "e")["bytes"] == 500


def test_units_that_did_not_finish_are_flagged_and_their_parents_become_lower_bounds():
    q = quick()
    t, c = q["tree"], sub(q["tree"], "c")
    assert c["bytes"] == 700 + 400 and c["incomplete"] is True and t["incomplete"] is True
    assert sub(c, "slow")["unscanned"] is True and sub(c, "slow")["bytes"] == 0
    s2 = sub(c, "slow2")
    assert s2["bytes"] == 400 and s2["incomplete"] is True and "unscanned" not in s2            # cut off, but one finished child counts
    assert q["partial"] is True and q["unscanned"] == 2 and "incomplete" not in sub(t, "e")          # slow (nothing read) and slow2 (cut off)
    complete = quick(QUICK.replace("D\t./c/slow\n", "").replace("D\t./c/slow2\n", "").replace("400\t1790000400\t./c/slow2/part\n", ""))
    assert complete["partial"] is False and complete["unscanned"] == 0 and "incomplete" not in complete["tree"]


def test_the_quick_tree_lists_only_the_depth_asked_for():
    t = quick(depth=1)["tree"]
    assert sorted(d["name"] for d in t["dirs"]) == ["a", "c", "e"] and t["dirs"][0].get("dirs") is None
    assert sub(t, "a")["deep"] is True and sub(t, "e").get("deep") is not True


def test_garbage_lines_in_the_remote_output_are_ignored():
    q = quick("Welcome\nD\t./a\nF\t10\t1790000000.0\t.\n5\t1790000001\t./a\nnot a record\n\t\t\n[done]\n")
    assert q["tree"]["bytes"] == 15 and sub(q["tree"], "a")["bytes"] == 5


def test_the_remote_script_is_read_only_capped_polite_and_quotes_the_root():
    script = diskquick.quick_script("/projects/lab/me dir", depth=4, budget=60)
    assert "cd '/projects/lab/me dir'" in script and "nice -n 19" in script and "stdbuf -oL du -x --apparent-size" in script
    assert "--time-style=+%s" in script and "-P 6" in script and "$(( $(date +%s) + 60 ))" in script and "-d 2" in script
    assert "-d 0" in diskquick.quick_script("/r", depth=2, budget=60) and "-d 0" in diskquick.quick_script("/r", depth=1, budget=60)
    assert not any(w in script for w in (" rm ", "rm -", "mv ", "chmod", "> /", "sbatch", "scancel", "kill"))
    assert "timeout" in script
    assert script.rstrip().endswith("exit 0")        # xargs exits 123 when a capped du is cut off, which must not discard the records


def test_an_empty_or_unreadable_root_gives_an_empty_tree_not_a_crash():
    q = quick("")
    assert q["tree"]["bytes"] == 0 and q["tree"].get("dirs", []) == [] and q["partial"] is False


# ---- the explorer on a quick scan ----

def test_a_lower_bound_shows_with_a_greater_than_sign_and_missing_counts_leave_blanks():
    ex = diskview.Explorer(quick(), now=1790001000)
    ex.goto("c")
    lines = lines_of(ex, width=100)
    slow2 = next(l for l in lines if "slow2/" in l)
    assert "≥ 400 B" in slow2 or "≥ 400" in slow2
    assert "not read" in next(l for l in lines if "slow/" in l) and "files" not in "\n".join(lines[3:-1])      # no count column in a quick scan
    assert "quick scan" in lines_of(ex)[1] and "partial" in lines_of(ex)[1]
    assert "≥" in "\n".join(lines_of(diskview.Explorer(quick(), now=1790001000)))                               # the root total is a lower bound too
    ascii_text = diskview.Explorer(quick(), now=1790001000).frame(width=80, uni=False)
    assert all(ord(ch) < 128 for ch in ascii_text) and ">=" in ascii_text


def test_files_beside_folders_in_a_quick_scan_are_one_line_without_a_count():
    ex = diskview.Explorer(quick(), now=1790001000)
    ex.goto("a")
    es = ex.entries()
    other = next(e for e in es if e["kind"] == "other")
    assert other["name"] == "(files in this folder)" and other["bytes"] == 1000 and other["files"] is None
    ex.key("end")
    assert ex.key("enter") and "quick scan" in ex.message


def test_the_largest_files_key_explains_that_a_quick_scan_has_no_file_list():
    ex = diskview.Explorer(quick(), now=1790001000)
    assert ex.key("l") and ex.view == "tree" and "--job" in ex.message


# ---- the command: the quick scan is the default ----

class LoginNode:
    """A fake login node: the script arrives on stdin of `bash -s`, and its records come back after some banner noise."""

    def __init__(self, monkeypatch, out=QUICK, noise="Welcome to the cluster\n"):
        self.calls, self.stdin = [], []
        self.out, self.noise = out, noise

        def run(me, cmd, retry=True, input=None):
            self.calls.append((cmd, retry))
            self.stdin.append(input)
            return self.noise + self.out if cmd == "bash -s" else ""
        monkeypatch.setattr(cli.Remote, "run", run)


def test_remote_run_can_feed_a_script_to_the_remote_shell(monkeypatch):
    seen = {}

    def fake(argv, **kw):
        seen.update(argv=argv, **kw)
        return subprocess.CompletedProcess(argv, 0, stdout="out", stderr="")
    monkeypatch.setattr(cli.subprocess, "run", fake)
    assert cli.Remote({"ssh_host": "h"}, False).run("bash -s", retry=False, input="echo hi") == "out"
    assert seen["input"] == "echo hi" and seen["argv"][:3] == ["ssh", "-o", "BatchMode=yes"] and "bash -lc" in seen["argv"][-1]


def test_the_default_scan_runs_du_on_the_login_node_and_submits_no_job(inproc, monkeypatch, tmp_path, capsys):
    inproc([])
    node = LoginNode(monkeypatch)
    cli.main(["disk", "--scan"])
    out = capsys.readouterr()
    assert "nuhpc disk" in out.out and "quick scan" in out.out and "a/" in out.out
    (cmd, retry), = node.calls
    assert cmd == "bash -s" and retry is False                                                  # a capped run is never silently repeated
    script = node.stdin[0]
    root = str(tmp_path / "remote")
    assert f"cd {root}" in script and "$(( $(date +%s) + 60 ))" in script and "-d 2" in script
    cached = json.loads((tmp_path / "state/disk/user.json").read_text())
    assert cached["mode"] == "quick" and cached["root"] == root and cached["tree"]["bytes"] == 5000
    assert [e["template"] for e in cli.ledger_read()] == ["python"]                              # no job: only the fixture's run is in the ledger
    assert "login node" in out.err and "partial" in out.err and "--job" in out.err               # c/slow never finished in the fixture
    assert "2 folders were not fully read" in out.err


def test_the_cap_can_be_raised_and_a_lab_scan_reads_one_level(inproc, monkeypatch, tmp_path):
    inproc([])
    node = LoginNode(monkeypatch)
    cli.main(["disk", "--scan", "--lab", "--budget", "25"])
    script = node.stdin[0]
    assert f"cd {tmp_path} || exit 1" in script and "$(( $(date +%s) + 25 ))" in script and "-d 0" in script
    assert (tmp_path / "state/disk/lab.json").exists()


def test_a_complete_quick_scan_says_nothing_about_being_partial(inproc, monkeypatch, tmp_path, capsys):
    inproc([])
    LoginNode(monkeypatch, out=QUICK.replace("D\t./c/slow\n", "").replace("D\t./c/slow2\n", "").replace("400\t1790000400\t./c/slow2/part\n", ""))
    cli.main(["disk", "--scan"])
    assert "partial" not in capsys.readouterr().err


def test_an_empty_reply_is_an_error_not_an_empty_picture(inproc, monkeypatch, tmp_path, capsys):
    inproc([])
    LoginNode(monkeypatch, out="", noise="")
    with pytest.raises(SystemExit):
        cli.main(["disk", "--scan"])
    assert "nothing came back" in capsys.readouterr().err and not (tmp_path / "state/disk").exists()


def test_a_quick_dry_run_prints_the_command_and_changes_nothing(inproc, monkeypatch, tmp_path):
    inproc([])
    node = LoginNode(monkeypatch)
    cli.main(["--dry-run", "disk", "--scan"])
    assert not (tmp_path / "state/disk").exists()


# ---- the whole disk: how full the filesystem is, and what share of it this folder is ----

FS = {"name": "server:/work", "mount": "/projects", "total": 16 * 1024 * G, "used": 5 * 1024 * G + 920 * M * 1024 // 1024, "avail": 9 * 1024 * G + 700 * M * 1024 // 1024}


def with_fs(**kw):
    fs = {**FS, **kw}
    return {**SCAN, "fs": fs}


def test_the_job_scanner_reports_how_full_the_filesystem_is(tmp_path):
    s = diskscan.scan(str(make(tmp_path / "r", TREE)), depth=1)
    fs = s["fs"]
    assert fs["total"] > 0 and 0 <= fs["used"] <= fs["total"] and 0 <= fs["avail"] <= fs["total"]
    assert os.path.isdir(fs["mount"]) and str(tmp_path / "r").startswith(fs["mount"])           # the mount the root lives on


def test_the_quick_script_asks_df_for_the_mount_and_its_record_is_read_back():
    script = diskquick.quick_script("/r", depth=4, budget=60)
    assert "df -P -B1" in script and script.index("df -P") < script.index("exit 0")
    q = quick("S\tserver:/work\t17592186044416\t6319245000000\t10400000000000\t/projects\n" + QUICK)
    assert q["fs"] == {"name": "server:/work", "total": 17592186044416, "used": 6319245000000, "avail": 10400000000000, "mount": "/projects"}
    assert "fs" not in quick()                                                                  # no record, no claim
    assert "fs" not in quick("S\tserver\tnot-a-number\t1\t2\t/x\n" + QUICK)


def test_the_header_shows_how_full_the_disk_is_and_this_folders_share_of_it():
    ex = diskview.Explorer(with_fs(), now=NOW)
    lines = lines_of(ex)
    assert lines[2].strip().startswith("/projects") and "█" in lines[2] and "% full" in lines[2]
    used_pct = 100 * FS["used"] / FS["total"]
    assert f"{used_pct:.0f}% full" in lines[2] and f"{fmt_size_local(FS['avail'])} free of 16 T" in lines[2]
    share = 100 * ETREE["bytes"] / FS["total"]
    assert f"{share:.1f}%" in lines[3] and "of it" in lines[3] and "of what is used" in lines[3]
    assert lines[4].strip() == "me"                                                               # the breadcrumb moves down two lines
    assert lines_of(diskview.Explorer(SCAN, now=NOW))[2].strip() == "me"                          # no fs, no extra lines


def fmt_size_local(n):
    return diskview.fmt_size(n)


def test_a_lower_bound_folder_share_says_at_least_and_fullness_is_coloured():
    quick_scan = {**quick(), "fs": {**FS}}
    text = "\n".join(lines_of(diskview.Explorer(quick_scan, now=1790001000)))
    assert "≥" in next(l for l in text.splitlines() if "of it" in l)
    def bar_colour(pct):
        total = 1000 * G
        fs = {**FS, "total": total, "used": int(total * pct / 100), "avail": int(total * (100 - pct) / 100)}
        return diskview.Explorer({**SCAN, "fs": fs}, now=NOW).frame(width=100, color=True).splitlines()[2]
    assert "\x1b[38;5;158m" in bar_colour(40) and "\x1b[38;5;229m" in bar_colour(75)               # mint, then lemon
    assert "\x1b[38;5;223m" in bar_colour(90) and "\x1b[38;5;211m" in bar_colour(97)               # peach, then rose


def test_the_disk_lines_fit_every_width_and_the_ascii_form():
    for width in (50, 72, 100):
        ex = diskview.Explorer(with_fs(), now=NOW)
        assert all(len(shown(l)) <= width for l in ex.frame(width=width, color=True).splitlines()), width
    assert all(ord(c) < 128 for c in diskview.Explorer(with_fs(), now=NOW).frame(width=80, uni=False))


def test_the_command_caches_and_shows_the_filesystem_line(inproc, monkeypatch, tmp_path, capsys):
    inproc([])
    LoginNode(monkeypatch, out="S\tserver:/work\t17592186044416\t6319245000000\t10400000000000\t/projects\n" + QUICK)
    cli.main(["disk", "--scan"])
    out = capsys.readouterr().out
    assert "/projects" in out and "% full" in out and "of it" in out
    assert json.loads((tmp_path / "state/disk/user.json").read_text())["fs"]["mount"] == "/projects"


# ---- a quota inside a bigger filesystem: the lab's allocation, and the pool under it ----

PARENT = {"total": 3500 * 1024 * G, "used": 3400 * 1024 * G, "avail": 100 * 1024 * G}


def test_a_folder_quota_is_told_apart_from_the_filesystem_under_it(monkeypatch, tmp_path):
    q = quick("S\tserver:/w\t17042430230528\t13822364483584\t3220065746944\t/projects\nM\t3848290697216000\t3737000000000000\t146000000000000\n" + QUICK)
    assert q["fs"]["parent"] == {"total": 3848290697216000, "used": 3737000000000000, "avail": 146000000000000}
    same = quick("S\tserver:/w\t1000\t400\t600\t/projects\nM\t1000\t400\t600\n" + QUICK)
    assert "parent" not in same["fs"]                                                           # one filesystem, no quota in between
    assert "fs" not in quick("M\t1000\t400\t600\n" + QUICK)                                     # a mount line alone says nothing
    assert "df -P -B1" in diskquick.quick_script("/r") and "M\\t" in diskquick.quick_script("/r")

    import collections
    SV = collections.namedtuple("SV", "f_blocks f_frsize f_bfree f_bavail")
    root = str(make(tmp_path / "r", TREE))
    monkeypatch.setattr(diskscan.os, "statvfs", lambda p: SV(100, 1024, 40, 40) if p == root else SV(1000, 1024, 30, 30))
    monkeypatch.setattr(diskscan.os.path, "ismount", lambda p: p == os.path.dirname(root))
    fs = diskscan.scan(root, depth=1)["fs"]
    assert fs["total"] == 100 * 1024 and fs["parent"] == {"total": 1000 * 1024, "used": 970 * 1024, "avail": 30 * 1024}


def test_the_header_labels_a_quota_and_shows_the_pool_under_it():
    scan = {**SCAN, "fs": {**FS, "parent": PARENT}}
    lines = lines_of(diskview.Explorer(scan, now=NOW))
    assert lines[2].strip().startswith("/projects quota") and "% full" in lines[2]
    assert "of it" in lines[3]
    pool = 100 * PARENT["used"] / PARENT["total"]
    assert f"{pool:.0f}% full" in lines[4] and "filesystem" in lines[4] and "3.32 P of 3.42 P" in lines[4]
    assert lines[5].strip() == "me"
    plain = lines_of(diskview.Explorer({**SCAN, "fs": FS}, now=NOW))
    assert "quota" not in plain[2] and plain[4].strip() == "me"                                  # no parent: as before
    coloured = diskview.Explorer(scan, now=NOW).frame(width=100, color=True).splitlines()[4]
    assert "\x1b[38;5;211m" in coloured                                                          # 97% full is rose
    for width in (50, 72, 100):
        assert all(len(shown(l)) <= width for l in diskview.Explorer(scan, now=NOW).frame(width=width, color=True).splitlines())
    assert all(ord(c) < 128 for c in diskview.Explorer(scan, now=NOW).frame(width=80, uni=False))


def test_a_job_scan_of_a_project_runs_from_your_own_work_root_not_from_inside_the_folder_it_scans(inproc, monkeypatch, tmp_path):
    inproc([])

    cli.CONFIG_PATH.write_text(cli.CONFIG_PATH.read_text() + '\n[projects.alpha]\ndir = "/somewhere/alpha"\n')
    Cluster(monkeypatch, scan_file(tmp_path, root="/somewhere/alpha"))
    cli.main(["disk", "--scan", "--job", "--project", "alpha"])
    entry = cli.ledger_read()[-1]
    assert entry["template"] == "disk_scan" and entry["project"] is None
    assert not entry["remote_dir"].startswith("/somewhere/alpha") and entry["remote_dir"].startswith(str(tmp_path / "remote"))
    _, rows = staged(tmp_path)
    assert rows[0]["root"] == "/somewhere/alpha"                                             # it still scans the project's folder
    assert (tmp_path / "state/disk/project-alpha.json").exists()


# ---- the live view: the cached scan first, a status line while fresh data is collected, then the swap ----

import io
import threading

SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def test_a_status_line_shows_a_spinner_while_collecting_and_a_tick_or_cross_after():
    ex = diskview.Explorer(SCAN, now=NOW)
    assert lines_of(ex)[2].strip() == "me"                                                       # no status, no extra line
    ex.set_status("collecting folder sizes · 3 measured · 12 s", "busy")
    ls = lines_of(ex)
    assert ls[2].strip()[0] in SPIN and "collecting folder sizes · 3 measured · 12 s" in ls[2] and ls[3].strip() == "me"
    ex.tick = 3
    assert lines_of(ex)[2].strip()[0] == SPIN[3]                                                 # the spinner turns with the tick
    ex.set_status("updated just now", "done")
    assert lines_of(ex)[2].strip().startswith("✓") and "updated just now" in lines_of(ex)[2]
    ex.set_status("refresh failed: ssh: connect failed; showing the older data", "error")
    assert lines_of(ex)[2].strip().startswith("✗")
    ex.set_status(None)
    assert lines_of(ex)[2].strip() == "me"
    coloured = {k: diskview.Explorer(SCAN, now=NOW) for k in ("busy", "done", "error")}
    for k, ex2 in coloured.items():
        ex2.set_status("x", k)
    assert "38;5;153" in coloured["busy"].frame(width=100, color=True).splitlines()[2]            # sky
    assert "38;5;158" in coloured["done"].frame(width=100, color=True).splitlines()[2]            # mint
    assert "38;5;211" in coloured["error"].frame(width=100, color=True).splitlines()[2]           # rose
    busy = coloured["busy"]
    assert all(ord(c) < 128 for c in busy.frame(width=80, uni=False)) and busy.frame(width=80, uni=False).splitlines()[2].strip()[0] in "|/-\\"
    for width in (50, 72, 100):
        assert all(len(shown(l)) <= width for l in busy.frame(width=width, color=True).splitlines())


def test_the_footer_offers_r_to_refresh_only_when_a_refresh_is_possible():
    ex = diskview.Explorer(SCAN, now=NOW)
    assert "r refresh" not in lines_of(ex)[-1]
    ex.can_refresh = True
    assert "r refresh" in lines_of(ex)[-1]


def test_a_new_scan_replaces_the_old_one_and_keeps_your_place():
    newer_models = mk("models", dirs=[HUB, mk("extra", own=(1, 2 * G), age=1)], own=(14, 1 * G),
                      top=[("tokenizer.bin", 600 * M, 12), ("config.json", 4 * K, 40)], age=1)
    newer = {**SCAN, "tree": mk("me", dirs=[newer_models, mk("runs", own=(9000, 90 * G), age=0)], own=(4, 5 * G), age=0)}
    ex = diskview.Explorer(SCAN, now=NOW)
    ex.goto("models")
    ex.key("down")
    name = ex.entries()[ex.cursor]["name"]                                                        # a file, whose row moves when the new scan arrives
    ex.set_scan(newer)
    assert ex.path == ["models"] and ex.entries()[ex.cursor]["name"] == name
    assert "extra/" in [e["name"] for e in ex.entries()]
    gone = {**SCAN, "tree": mk("me", dirs=[mk("runs", own=(9000, 80 * G), age=0)], own=(1, 1 * G), age=0)}
    ex.set_scan(gone)                                                                             # the folder you were in has vanished
    assert ex.path == [] and ex.cursor < len(ex.entries())


def test_with_no_scan_yet_the_explorer_waits_and_then_takes_the_first_one():
    ex = diskview.Explorer(diskview.placeholder_scan("/projects/me"), now=NOW)
    ex.set_status("connecting…", "busy")
    text = "\n".join(lines_of(ex))
    assert "/projects/me" in text and "waiting for the first scan" in text and "connecting…" in text
    ex.set_scan(SCAN)
    assert "models/" in "\n".join(lines_of(ex)) and "waiting for the first scan" not in "\n".join(lines_of(ex))


def wait_for_text(read, needle, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if needle in read():
            return True
        time.sleep(0.02)
    return False


def test_the_refresher_runs_work_on_a_thread_and_hands_over_the_result_once():
    gate = threading.Event()

    def work(progress, track):
        progress("collecting 1")
        gate.wait(3)
        progress("collecting 2")
        return SCAN
    r = diskview.Refresher(work)
    assert r.take() is None and not r.running() and r.elapsed() == 0
    r.start()
    assert wait_for_text(lambda: r.status, "collecting 1") and r.running() and r.take() is None and r.elapsed() >= 0
    gate.set()
    r.join(3)
    assert r.take() is SCAN and r.take() is None and not r.running() and r.error is None and r.done
    r.restart()                                                                                    # r: measure again
    r.join(3)
    assert r.take() is SCAN


def test_a_failing_refresh_keeps_its_first_error_line_and_gives_no_scan():
    def work(progress, track):
        raise RuntimeError("ssh: connect to host failed\nmore detail")
    r = diskview.Refresher(work)
    r.start()
    r.join(3)
    assert r.take() is None and r.error == "ssh: connect to host failed" and r.done and not r.running()


def test_stopping_the_refresher_kills_the_processes_the_work_started():
    killed = []

    class Proc:
        def kill(self):
            killed.append(True)
    gate = threading.Event()

    def work(progress, track):
        track(Proc())
        gate.wait(3)
        return SCAN
    r = diskview.Refresher(work)
    r.start()
    assert wait_for_text(lambda: "x" if r._procs else "", "x")
    r.stop()
    assert killed == [True]
    gate.set()
    r.join(3)
    assert r.take() is None                                                                         # a stopped refresh hands over nothing


def test_the_loop_shows_the_old_data_with_a_busy_line_then_swaps_in_the_new_data():
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
    out_f = os.fdopen(os.dup(slave), "w", buffering=1)
    in_f = os.fdopen(slave, "r")
    newer = {**SCAN, "tree": mk("me", dirs=[mk("fresh", own=(1, 2 * 1024 * G), age=0)], own=(1, 1 * G), age=0)}
    gate = threading.Event()

    def work(progress, track):
        progress("collecting folder sizes · 7 measured")
        gate.wait(5)
        return newer
    ex = diskview.Explorer(SCAN, now=NOW)
    ex.can_refresh = True
    ref = diskview.Refresher(work)
    t = threading.Thread(target=diskview.run_explorer, kwargs={"ex": ex, "color": False, "uni": True, "stdin": in_f, "stdout": out_f, "refresher": ref}, daemon=True)
    t.start()
    seen = []

    def read():
        while select.select([master], [], [], 0.02)[0]:
            seen.append(os.read(master, 65536).decode(errors="replace"))
        return "".join(seen)
    assert wait_for_text(read, "models/") and wait_for_text(read, "collecting folder sizes · 7 measured")        # the cached data, at once, with the status
    assert "fresh/" not in read()
    gate.set()
    assert wait_for_text(read, "fresh/") and wait_for_text(read, "updated just now")                           # then the new data, in place
    os.write(master, b"q")
    end = time.time() + 5
    while t.is_alive() and time.time() < end:                 # the terminal is put back only once its output has been read
        read()
        t.join(0.05)
    assert not t.is_alive() and "\x1b[?1049l" in read()
    os.close(master)


# ---- the command: refresh by default in a terminal ----

class StreamNode:
    """A fake login node with a streaming transport: each output line goes to on_line as it arrives."""

    def __init__(self, monkeypatch, out=QUICK, noise=""):
        self.calls = []
        self.out, self.noise = out, noise

        def stream(me, cmd, input=None, on_line=None, on_start=None):
            self.calls.append((cmd, input))
            for line in (self.noise + self.out).splitlines():
                if on_line:
                    on_line(line)
            return self.noise + self.out
        monkeypatch.setattr(cli.Remote, "stream", stream)


def in_a_terminal(monkeypatch):
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    opened = {}

    def fake_explorer(ex, color=True, uni=True, stdin=None, stdout=None, refresher=None):
        opened.update(ex=ex, refresher=refresher, scan_at_open=ex.scan, color=color, uni=uni)
    monkeypatch.setattr(cli.diskview, "run_explorer", fake_explorer)
    return opened


def test_in_a_terminal_the_cached_scan_opens_at_once_and_a_refresh_collects_new_data(inproc, monkeypatch, tmp_path):
    inproc([])
    cache = tmp_path / "state/disk"
    cache.mkdir(parents=True)
    root = str(tmp_path / "remote")
    (cache / "user.json").write_text(json.dumps({**SCAN, "root": root}))
    node = StreamNode(monkeypatch)
    opened = in_a_terminal(monkeypatch)
    cli.main(["disk"])
    assert opened["scan_at_open"]["tree"]["bytes"] == ETREE["bytes"] and opened["ex"].can_refresh is True      # the cached data is what opens
    assert node.calls == []                                                                                      # nothing collected before the screen is up
    ref = opened["refresher"]
    ref.start()
    ref.join(5)
    fresh = ref.take()
    assert fresh["mode"] == "quick" and fresh["tree"]["bytes"] == 5000 and node.calls[0][0] == "bash -s"
    assert json.loads((cache / "user.json").read_text())["mode"] == "quick"                                    # and the cache is updated for next time
    assert "collecting folder sizes" in ref.status or ref.done


def test_with_no_cache_the_explorer_opens_on_a_placeholder_and_the_first_scan_fills_it(inproc, monkeypatch, tmp_path):
    inproc([])
    StreamNode(monkeypatch)
    opened = in_a_terminal(monkeypatch)
    cli.main(["disk"])
    assert opened["scan_at_open"].get("placeholder") is True and opened["scan_at_open"]["root"] == str(tmp_path / "remote")
    ref = opened["refresher"]
    ref.start()
    ref.join(5)
    opened["ex"].set_scan(ref.take())
    assert opened["ex"].scan["tree"]["bytes"] == 5000 and (tmp_path / "state/disk/user.json").exists()


def test_no_refresh_opens_the_cache_without_touching_the_cluster(inproc, monkeypatch, tmp_path):
    inproc([])
    cache = tmp_path / "state/disk"
    cache.mkdir(parents=True)
    (cache / "user.json").write_text(json.dumps({**SCAN, "root": str(tmp_path / "remote")}))
    node = StreamNode(monkeypatch)
    opened = in_a_terminal(monkeypatch)
    cli.main(["disk", "--no-refresh"])
    assert opened["refresher"] is None and opened["ex"].can_refresh is False and node.calls == []


def test_no_refresh_contradicts_asking_for_a_scan_and_a_saved_file_is_never_refreshed(inproc, monkeypatch, tmp_path, capsys):
    inproc([])
    StreamNode(monkeypatch)
    opened = in_a_terminal(monkeypatch)
    with pytest.raises(SystemExit):
        cli.main(["disk", "--no-refresh", "--scan"])
    assert "contradict" in capsys.readouterr().err
    cli.main(["disk", "--from", str(scan_file(tmp_path))])
    assert opened["refresher"] is None


def test_a_job_refresh_reports_queued_running_and_fetching_then_updates_the_cache(inproc, monkeypatch, tmp_path):
    inproc([])
    root = str(tmp_path / "remote")
    cluster = Cluster(monkeypatch, scan_file(tmp_path, root=root), states=("PENDING", "RUNNING", "COMPLETED"))
    opened = in_a_terminal(monkeypatch)
    texts = []
    real = diskview.Refresher._progress
    monkeypatch.setattr(diskview.Refresher, "_progress", lambda self, t: (texts.append(t), real(self, t))[1])
    cli.main(["disk", "--job"])
    ref = opened["refresher"]
    ref.start()
    ref.join(10)
    assert ref.take()["root"] == root and ref.error is None
    joined = " | ".join(texts)
    assert joined.index("submitting") < joined.index("queued") < joined.index("running") < joined.index("fetching")
    assert (tmp_path / "state/disk/user.json").exists() and cli.ledger_read()[-1]["template"] == "disk_scan"


def test_the_streaming_transport_hands_over_lines_as_they_come_and_the_process_for_stopping(monkeypatch):
    class Proc:
        def __init__(self, *a, **k):
            self.stdin, self.stdout, self.stderr = io.StringIO(), iter(["S\tx\n", "5\t1\t./a\n"]), io.StringIO("")
            self.returncode = 0
            Proc.last = self

        def wait(self):
            return self.returncode
    monkeypatch.setattr(cli.subprocess, "Popen", Proc)
    got, started = [], []
    out = cli.Remote({"ssh_host": "h"}, False).stream("bash -s", input="echo hi", on_line=got.append, on_start=started.append)
    assert got == ["S\tx", "5\t1\t./a"] and out == "S\tx\n5\t1\t./a\n" and started == [Proc.last]
    Proc.last.stdin.seek(0) if False else None
    bad = type("Bad", (Proc,), {"__init__": lambda self, *a, **k: (Proc.__init__(self), setattr(self, "stderr", io.StringIO("Permission denied")), setattr(self, "returncode", 255))[0]})
    monkeypatch.setattr(cli.subprocess, "Popen", bad)
    with pytest.raises(cli.NuhpcError, match="Permission denied"):
        cli.Remote({"ssh_host": "h"}, False).stream("bash -s", input="x")
