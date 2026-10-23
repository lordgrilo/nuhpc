"""Measure how much space a directory tree uses: a parallel, read-only walk that writes one JSON tree.

This file runs on the cluster's compute nodes under the system Python (3.9), so it is stdlib only and uses no newer
syntax; a test parses it as 3.9. `nuhpc disk --scan` uploads it with the job; locally, `nuhpc disk` only reads its output.

Sizes are apparent sizes (what `ls -l` reports). Symlinks are counted as the small files they are and never followed,
and `.snapshot` folders are skipped. Each node of the tree has the bytes and files of everything beneath it, the newest
modification time beneath it, and `own`: the files directly inside it. Only `depth` levels of folders are listed; below
that the totals are kept and the node is marked `deep`.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import socket
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

SKIP = {".snapshot"}


def _list_dir(path: str, keep: int, names: bool) -> dict:
    """Read one folder: its subfolders, the size of its files, and (with names) its `keep` largest files."""
    subdirs, files, size, newest, heap, unreadable = [], 0, 0, None, [], False
    try:
        with os.scandir(path) as entries:
            for e in entries:
                try:
                    if not e.is_symlink() and e.is_dir(follow_symlinks=False):
                        if e.name not in SKIP:
                            subdirs.append(e.name)
                        continue
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                files += 1
                size += st.st_size
                mtime = int(st.st_mtime)
                newest = mtime if newest is None or mtime > newest else newest
                if names and keep:
                    item = (st.st_size, e.name, mtime)
                    if len(heap) < keep:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
    except OSError:
        unreadable = True
    return {"subdirs": subdirs, "files": files, "bytes": size, "newest": newest,
            "top": sorted(heap, reverse=True), "unreadable": unreadable}


def _filesystem(root: str) -> dict:
    """How full the filesystem holding `root` is, as `df` would say (sizes in bytes), and where it is mounted. On a cluster with
    directory quotas, `df` inside a group's folder shows the quota; `parent` then holds the size of the whole mount."""
    st = os.statvfs(root)
    mount = root
    while mount != os.sep and not os.path.ismount(mount):
        mount = os.path.dirname(mount)
    fs = {"mount": mount, "total": st.f_blocks * st.f_frsize, "used": (st.f_blocks - st.f_bfree) * st.f_frsize,
          "avail": st.f_bavail * st.f_frsize}
    top = os.statvfs(mount)
    if top.f_blocks * top.f_frsize != fs["total"]:      # the root sits under a quota: the mount is the bigger pool beneath it
        fs["parent"] = {"total": top.f_blocks * top.f_frsize, "used": (top.f_blocks - top.f_bfree) * top.f_frsize,
                        "avail": top.f_bavail * top.f_frsize}
        quota_dir = root                                    # the highest folder that still shows the same size is the one with the quota
        while os.path.dirname(quota_dir) != quota_dir:
            up = os.statvfs(os.path.dirname(quota_dir))
            if up.f_blocks * up.f_frsize != fs["total"]:
                break
            quota_dir = os.path.dirname(quota_dir)
        fs["quota_dir"] = quota_dir
    return fs


def scan(root: str, depth: int = 4, names: bool = True, workers: int = 16, per_dir: int = 8, largest: int = 25,
         budget_s: float | None = None) -> dict:
    """Walk `root` and return the result as a dict (see the module docstring). With `budget_s`, folders not yet
    opened when the time is up are left unread, and the result says so (`partial`, `unscanned`)."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"not a directory: {root}")
    started = time.monotonic()
    deadline = None if budget_s is None else started + budget_s
    keep = max(per_dir, largest)
    info: dict = {}
    unscanned = 0
    biggest: list = []                                        # a min-heap of (bytes, path, mtime), the largest files so far
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending = {pool.submit(_list_dir, root, keep, names): root}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                path = pending.pop(fut)
                r = fut.result()
                for size, name, mtime in r["top"]:            # the global ranking, taken before each folder's list is cut down
                    if largest:
                        item = (size, os.path.join(path, name), mtime)
                        if len(biggest) < largest:
                            heapq.heappush(biggest, item)
                        elif item > biggest[0]:
                            heapq.heapreplace(biggest, item)
                r["top"] = r["top"][:per_dir]
                info[path] = r
                for name in r["subdirs"]:
                    sub = os.path.join(path, name)
                    if deadline is not None and time.monotonic() >= deadline:
                        unscanned += 1
                        info[sub] = {"subdirs": [], "files": 0, "bytes": 0, "newest": None, "top": [],
                                     "unreadable": False, "unscanned": True}
                    else:
                        pending[pool.submit(_list_dir, sub, keep, names)] = sub

    total = {p: {"bytes": r["bytes"], "files": r["files"], "newest": r["newest"], "bad": int(r["unreadable"])}
             for p, r in info.items()}
    for p in sorted(info, key=lambda q: -q.count(os.sep)):    # deepest first: each folder adds itself to its parent
        if p == root:
            continue
        up, t = total[os.path.dirname(p)], total[p]
        up["bytes"] += t["bytes"]
        up["files"] += t["files"]
        up["bad"] += t["bad"]
        if t["newest"] is not None and (up["newest"] is None or t["newest"] > up["newest"]):
            up["newest"] = t["newest"]

    def node(path: str, level: int) -> dict:
        r, t = info[path], total[path]
        n = {"name": os.path.basename(path) or path, "bytes": t["bytes"], "files": t["files"], "newest": t["newest"],
             "own": {"files": r["files"], "bytes": r["bytes"]}}
        if names and r["top"]:
            n["top"] = [{"name": name, "bytes": size, "mtime": mtime} for size, name, mtime in r["top"]]
        if t["bad"]:
            n["unreadable"] = t["bad"]
        if r.get("unscanned"):
            n["unscanned"] = True
        subs = [os.path.join(path, name) for name in r["subdirs"]]
        if subs and level < depth:
            n["dirs"] = sorted((node(s, level + 1) for s in subs), key=lambda d: (-d["bytes"], d["name"]))
        elif subs:
            n["deep"] = True
        return n

    now = time.time()
    return {"version": 1, "root": root, "host": socket.gethostname(), "depth": depth, "names": names, "workers": workers,
            "scanned_at": int(now), "scanned": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "elapsed_s": round(time.monotonic() - started, 1), "n_dirs": len(info),
            "partial": unscanned > 0, "unscanned": unscanned, "unreadable": total[root]["bad"], "fs": _filesystem(root),
            "tree": node(root, 0),
            "largest": ([{"path": os.path.relpath(p, root), "bytes": size, "mtime": mtime}
                         for size, p, mtime in sorted(biggest, reverse=True)] if names else [])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure a directory tree's size, read-only, and write it as JSON.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=4, help="levels of folders to list; deeper ones are only totalled")
    ap.add_argument("--workers", type=int, default=16, help="parallel folder reads")
    ap.add_argument("--no-names", action="store_true", help="record no file names at all (folder names to --depth only)")
    ap.add_argument("--budget", type=float, default=None, help="seconds after which unread folders are left out")
    a = ap.parse_args(argv)
    result = scan(a.root, depth=a.depth, names=not a.no_names, workers=a.workers, budget_s=a.budget)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    os.replace(tmp, a.out)
    t = result["tree"]
    print(f"[diskscan] {result['root']}: {t['bytes']} bytes in {t['files']} files, {result['n_dirs']} folders, "
          f"{result['elapsed_s']} s" + (f", PARTIAL ({result['unscanned']} folders unread)" if result["partial"] else "")
          + (f", {result['unreadable']} unreadable" if result["unreadable"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
