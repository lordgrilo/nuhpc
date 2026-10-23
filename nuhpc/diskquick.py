"""The quick scan behind `nuhpc disk --scan`: folder sizes from `du`, read over ssh, with no job and nothing written on the cluster.

Why du and not a walker of our own: login nodes cap each user at a fraction of a core and kill processes that use more
(a Python walker with threads was killed about half the time), whereas `du` is C and mostly waits on the file server. So the
remote script runs one capped, lowest-priority `du` per folder two levels down, six at a time, and the result is put together here.

What du cannot give is file counts or file names, so a quick scan has neither; `--job` runs the full scanner on a compute node.
A folder du did not finish before the time cap is `unscanned` (nothing known) or `incomplete` (a lower bound from the parts that
finished), and every folder above it is marked `incomplete`, since its total is then a lower bound too.
"""
from __future__ import annotations

import posixpath
import shlex
import time

PARALLEL = 6


def quick_script(root: str, depth: int = 4, budget: float = 60, parallel: int = PARALLEL) -> str:
    """The remote script (run with `bash -s`). It prints three kinds of record, one per line:
    D<TAB>path (a folder one or two levels down), F<TAB>bytes<TAB>mtime<TAB>folder (a file beside folders, in the root or a
    first-level folder), and du's own `bytes<TAB>mtime<TAB>path` for each second-level folder. All paths are relative to root."""
    d = max(0, depth - 2)
    return f"""cd {shlex.quote(root)} || exit 1
END=$(( $(date +%s) + {int(budget)} )); export END
find . -mindepth 1 -maxdepth 2 -xdev -type d -printf 'D\\t%p\\n' 2>/dev/null
find . -maxdepth 1 -xdev ! -type d -printf 'F\\t%s\\t%T@\\t.\\n' 2>/dev/null
find . -mindepth 2 -maxdepth 2 -xdev ! -type d -printf 'F\\t%s\\t%T@\\t%h\\n' 2>/dev/null
find . -mindepth 2 -maxdepth 2 -xdev -type d -print0 2>/dev/null | xargs -0 -P {int(parallel)} -n 1 sh -c 't=$((END - $(date +%s))); [ "$t" -gt 0 ] && timeout "$t" nice -n 19 stdbuf -oL du -x --apparent-size -B1 --time --time-style=+%s -d {d} -- "$1" 2>/dev/null' _
exit 0
"""


def _rel(path: str) -> str:
    path = path.strip()
    path = path[2:] if path.startswith("./") else path
    return "" if path == "." else path.rstrip("/")


def tree_from_quick(out: str, root: str, depth: int = 4, elapsed: float = 0.0, host: str = "") -> dict:
    """Assemble the scan (the same JSON shape as diskscan's, with `mode: quick`) from the remote script's records.
    Lines that are not records (a login banner, a half line) are ignored."""
    dirs, sizes, own = set(), {}, {}
    for line in out.splitlines():
        f = line.split("\t")
        try:
            if f[0] == "D" and len(f) == 2:
                dirs.add(_rel(f[1]))
            elif f[0] == "F" and len(f) == 4:
                b, m = int(f[1]), int(float(f[2]))
                ob, om = own.get(_rel(f[3]), (0, None))
                own[_rel(f[3])] = (ob + b, m if om is None or m > om else om)
            elif len(f) == 3 and f[0].isdigit() and f[1].lstrip("-").isdigit():
                sizes[_rel(f[2])] = (int(f[0]), int(f[1]))
        except ValueError:
            continue
    known = {p for p in dirs | set(sizes) if p}
    for p in list(known):                                      # every folder above a known one is known
        while "/" in p:
            p = p.rpartition("/")[0]
            known.add(p)
    kids: dict = {}
    for p in known:
        kids.setdefault(p.rpartition("/")[0], []).append(p)
    unscanned = 0

    def build(path: str, level: int) -> dict:
        nonlocal unscanned
        children = [build(c, level + 1) for c in sorted(kids.get(path, []))]
        own_b, own_m = own.get(path, (0, None))
        newest = max([m for m in [own_m] + [c["newest"] for c in children] if m is not None], default=None)
        flags = {}
        if path in sizes:                                      # du finished this folder: its total is exact
            total, mtime = sizes[path]
            newest = mtime if newest is None or mtime > newest else newest
            own_bytes = max(0, total - sum(c["bytes"] for c in children))
        else:
            total = own_b + sum(c["bytes"] for c in children)
            own_bytes = own_b
            if level >= 2:                                     # a second-level folder du did not finish: nothing, or a lower bound
                flags["incomplete"] = True
                unscanned += 1
                if not children:
                    flags["unscanned"] = True
            elif any(c.get("incomplete") for c in children):
                flags["incomplete"] = True
        n = {"name": path.rpartition("/")[2] or posixpath.basename(root.rstrip("/")) or root, "bytes": total, "files": None,
             "newest": newest, "own": {"files": None, "bytes": own_bytes}, **flags}
        if children and level < depth:
            n["dirs"] = sorted(children, key=lambda d: (-d["bytes"], d["name"]))
        elif children:
            n["deep"] = True
        return n

    tree = build("", 0)
    now = time.time()
    return {"version": 1, "mode": "quick", "root": root, "host": host, "depth": depth, "names": False, "workers": PARALLEL,
            "scanned_at": int(now), "scanned": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "elapsed_s": round(elapsed, 1), "n_dirs": len(known), "partial": bool(tree.get("incomplete")),
            "unscanned": unscanned, "unreadable": 0, "tree": tree, "largest": []}
