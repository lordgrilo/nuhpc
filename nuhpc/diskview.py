"""The explorer behind `nuhpc disk`: a scan (see diskscan) as bars you move through, in the spirit of `mo analyze`.

Three layers, so almost all of it is testable without a terminal:
  * pure helpers: sizes, ages, bars, and key parsing;
  * `Explorer`, a state machine (`key()` moves, `frame()` draws a frame as text, with or without colour);
  * `run_explorer`, a thin loop that puts the terminal in cbreak mode on the alternate screen and feeds keys to the Explorer.
Colours are 256-colour pastels, cycled by rank; ages fade from mint (fresh) to rose (stale).
"""
from __future__ import annotations

import os
import select
import shutil
import sys
import time

STYLE = {"pink": "38;5;217", "lavender": "38;5;183", "sky": "38;5;153", "mint": "38;5;158", "peach": "38;5;223",
         "lemon": "38;5;229", "rose": "38;5;211", "bold": "1", "dim": "2"}
PALETTE = ("pink", "lavender", "sky", "mint", "peach", "lemon")
DAY = 86400


def fmt_size(n: float) -> str:
    """Sizes the way `du -h` and `df -h` print them (powers of 1024, three significant digits): 631 G, 1.5 K, 600 M."""
    if n < 1024:
        return f"{int(n)} B"
    v = float(n)
    for unit in "KMGTP":
        v /= 1024
        if v < 1000 or unit == "P":
            return f"{v:.3g} {unit}"
    return f"{v:.3g} P"


def fmt_age(seconds: float) -> str:
    """How long ago, in the unit that fits: today, days, weeks, months, years."""
    days = seconds / DAY
    if days < 1:
        return "today"
    d = int(days)
    return f"{d} d" if d < 14 else f"{d // 7} w" if d < 90 else f"{d // 30} mo" if d < 730 else f"{d // 365} y"


def fmt_ago(seconds: float) -> str:
    return ("just now" if seconds < 60 else f"{int(seconds // 60)} min ago" if seconds < 3600
            else f"{int(seconds // 3600)} h ago" if seconds < DAY else f"{int(seconds // DAY)} d ago")


def hbar(frac, width: int, uni: bool = True) -> str:
    """A gauge `width` cells wide, filled to `frac` in eighths of a cell; any nonzero share shows at least a sliver."""
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))
    if not uni:
        n = max(1, round(frac * width)) if frac > 0 else 0
        return "#" * n + "." * (width - n)
    eighths = max(1, round(frac * width * 8)) if frac > 0 else 0
    full, part = divmod(eighths, 8)
    bar = "█" * full + ("▏▎▍▌▋▊▉"[part - 1] if part else "")
    return bar + "░" * (width - len(bar))


_SEQ = {b"[A": "up", b"[B": "down", b"[C": "right", b"[D": "left", b"OA": "up", b"OB": "down", b"OC": "right", b"OD": "left",
        b"[5~": "pgup", b"[6~": "pgdn", b"[H": "home", b"[F": "end", b"[1~": "home", b"[4~": "end", b"OH": "home", b"OF": "end"}


def parse_keys(data: bytes) -> list:
    """Key names from the bytes a terminal sent: arrows and paging keys as words, printable keys as themselves."""
    out, i = [], 0
    while i < len(data):
        b = data[i:i + 1]
        if b == b"\x1b":
            rest = data[i + 1:]
            hit = next((s for s in sorted(_SEQ, key=len, reverse=True) if rest.startswith(s)), None)
            if hit:
                out.append(_SEQ[hit])
                i += 1 + len(hit)
            elif rest[:1] in (b"[", b"O"):                  # some other escape sequence: skip it whole
                j = i + 2
                while j < len(data) and not 0x40 <= data[j] <= 0x7E:
                    j += 1
                i = j + 1
            else:
                out.append("esc")
                i += 1
            continue
        if b in (b"\r", b"\n"):
            out.append("enter")
        elif b in (b"\x7f", b"\x08"):
            out.append("backspace")
        elif b == b"\x03":
            out.append("ctrl-c")
        elif 32 <= data[i] < 127:
            out.append(chr(data[i]))
        i += 1
    return out


class Painter:
    """Rows of (text, *styles) parts, cut to the width by what shows; colour only wraps text in escape codes."""

    def __init__(self, color: bool, width: int, uni: bool = True):
        self.color, self.width, self.uni = color, width, uni

    def paint(self, text: str, *styles) -> str:
        return f"\033[{';'.join(STYLE[s] for s in styles)}m{text}\033[0m" if self.color and styles and text else text

    def row(self, *parts) -> str:
        room, out = self.width, []
        for part in parts:
            text, styles = (part, ()) if isinstance(part, str) else (part[0], part[1:])
            if room <= 0:
                break
            if len(text) > room:
                text = text[:room - 1] + ("…" if self.uni else "~")
            room -= len(text)
            out.append(self.paint(text, *styles))
        return "".join(out)


def _trim(text: str, room: int, uni: bool, front: bool = False) -> str:
    """Fit text into `room` columns, cutting the end (or the front, for paths) and marking the cut."""
    if len(text) <= room:
        return text
    dot = "…" if uni else "~"
    return dot + text[-(room - 1):] if front else text[:room - 1] + dot


def _age_style(seconds) -> str:
    return "mint" if seconds < 30 * DAY else "lemon" if seconds < 180 * DAY else "peach" if seconds < 365 * DAY else "rose"


class Explorer:
    """Where you are in a scan, and how to draw it. `key(name)` returns False when the user quits."""

    def __init__(self, scan: dict, now: float | None = None, sort: str = "size", view: str = "tree"):
        self.scan, self.sort, self.view = scan, sort, view
        self.now = time.time() if now is None else now
        self.ref = scan.get("scanned_at") or self.now         # ages count from when the scan ran
        self.path: list = []                                  # folder names from the root down to the current folder
        self.cursor = self.scroll = 0
        self.message = ""
        self._back: list = []                                 # (cursor, scroll) of each folder above the current one
        self._other = (0, 0)                                  # the cursor and scroll of the view that isn't showing
        self._page = 10

    # ---- what is shown
    def node(self) -> dict:
        n = self.scan["tree"]
        for name in self.path:
            n = next(d for d in n.get("dirs", []) if d["name"] == name)
        return n

    def entries(self) -> list:
        n, total = self.node(), self.node()["bytes"]
        out = [{"kind": "dir", "name": d["name"] + "/", "key": d["name"], "bytes": d["bytes"], "files": d["files"], "mtime": d["newest"],
                "unscanned": bool(d.get("unscanned")), "incomplete": bool(d.get("incomplete"))} for d in n.get("dirs", [])]
        tops = n.get("top", [])
        out += [{"kind": "file", "name": f["name"], "key": f["name"], "bytes": f["bytes"], "files": None, "mtime": f["mtime"]} for f in tops]
        own = n.get("own", {"files": 0, "bytes": 0})
        if own["files"] is None:                              # a quick scan knows the bytes beside the folders, not the count
            if own["bytes"] > 0:
                out.append({"kind": "other", "name": "(files in this folder)", "key": None, "bytes": own["bytes"], "files": None, "mtime": None})
        elif own["files"] - len(tops) > 0:
            rest = own["files"] - len(tops)
            label = f"({rest} smaller file{'s' if rest != 1 else ''})" if tops else f"({rest} file{'s' if rest != 1 else ''})"
            out.append({"kind": "other", "name": label, "key": None, "bytes": own["bytes"] - sum(f["bytes"] for f in tops),
                        "files": rest, "mtime": None})
        for e in out:
            e["pct"] = 100 * e["bytes"] / total if total else 0.0
        if self.sort == "age":                                # the stalest first; rows with no date last
            out.sort(key=lambda e: (e["mtime"] is None, e["mtime"] or 0, e["name"]))
        else:
            out.sort(key=lambda e: (-e["bytes"], e["name"]))
        return out

    def items(self) -> list:
        return self.scan.get("largest", []) if self.view == "largest" else self.entries()

    # ---- keys
    def key(self, k: str) -> bool:
        if k in ("q", "ctrl-c"):
            return False
        self.message = ""
        n = len(self.items())
        last = max(n - 1, 0)
        if k in ("down", "j"):
            self.cursor = min(self.cursor + 1, last)
        elif k in ("up", "k"):
            self.cursor = max(self.cursor - 1, 0)
        elif k == "pgdn":
            self.cursor = min(self.cursor + self._page, last)
        elif k == "pgup":
            self.cursor = max(self.cursor - self._page, 0)
        elif k in ("home", "g"):
            self.cursor = 0
        elif k in ("end", "G"):
            self.cursor = last
        elif k in ("enter", "right"):
            self._open()
        elif k in ("left", "backspace", "esc", "h"):
            self._up()
        elif k == "s" and self.view == "tree":
            self.sort = "age" if self.sort == "size" else "size"
            self.cursor = self.scroll = 0
        elif k == "l":
            self._toggle()
        return True

    def goto(self, sub: str) -> None:
        """Start inside `sub` (folder names separated by /). Raises KeyError if the scan doesn't list it."""
        node, path = self.scan["tree"], []
        for name in [x for x in sub.split("/") if x]:
            child = next((d for d in node.get("dirs", []) if d["name"] == name), None)
            if child is None:
                raise KeyError(sub)
            path.append(name)
            node = child
        self.path, self._back, self.cursor, self.scroll = path, [(0, 0)] * len(path), 0, 0

    def _toggle(self) -> None:
        if self.view == "tree" and not self.scan.get("largest"):
            self.message = ("a quick scan records folder sizes only; run nuhpc disk --scan --job for file counts and the largest files"
                            if self.scan.get("mode") == "quick" else
                            "no file names were recorded in this scan, so there is no largest-files list")
            return
        self.view = "largest" if self.view == "tree" else "tree"
        (self.cursor, self.scroll), self._other = self._other, (self.cursor, self.scroll)

    def _up(self) -> None:
        if self.view == "largest":
            self._toggle()
        elif self.path:
            self.path.pop()
            self.cursor, self.scroll = self._back.pop()

    def _open(self) -> None:
        if self.view == "largest":
            self._reveal()
            return
        es = self.entries()
        if not es:
            return
        e = es[min(self.cursor, len(es) - 1)]
        if e["kind"] == "dir":
            self._back.append((self.cursor, self.scroll))
            self.path.append(e["key"])
            self.cursor = self.scroll = 0
        elif e["kind"] == "file":
            age = fmt_age(self.ref - e["mtime"])
            self.message = f"{e['name']} · {fmt_size(e['bytes'])} · changed {'today' if age == 'today' else age + ' ago'}"
        elif e["files"] is None:
            self.message = f"the files directly in this folder add up to {fmt_size(e['bytes'])} (a quick scan does not count or list them)"
        else:
            self.message = f"{e['files']:,} files, {fmt_size(e['bytes'])} together, too small to list one by one"

    def _reveal(self) -> None:
        """From the largest-files list: go to the folder the file lives in (as deep as the scan lists), with the file selected."""
        big = self.scan.get("largest", [])
        if not big:
            return
        parts = big[min(self.cursor, len(big) - 1)]["path"].split("/")
        node, path = self.scan["tree"], []
        for name in parts[:-1]:
            child = next((d for d in node.get("dirs", []) if d["name"] == name), None)
            if child is None:
                break
            path.append(name)
            node = child
        self.view, self.path, self._back = "tree", path, [(0, 0)] * len(path)
        self.sort, self._other = "size", (0, 0)
        names = [e["key"] for e in self.entries()]
        self.cursor, self.scroll = (names.index(parts[-1]) if len(path) == len(parts) - 1 and parts[-1] in names else 0), 0

    # ---- the frame
    def frame(self, width: int = 80, height: int | None = None, color: bool = False, uni: bool = True, footer: bool = True) -> str:
        P = Painter(color, width, uni)
        sep, crumb = ("·", "›") if uni else ("|", ">")
        s = self.scan
        t = s["tree"]
        size = ("≥ " if uni else ">= ") * bool(t.get("incomplete")) + fmt_size(t["bytes"])
        what = size if t.get("files") is None else f"{size} in {t['files']:,} files"
        facts = f"scanned {fmt_ago(self.now - s['scanned_at'])} {sep} {what}" if s.get("scanned_at") else what
        head = [P.row(" ", ("nuhpc disk", "pink", "bold"), f" {sep} ", (s["root"], "lavender")),
                P.row(" ", (facts, "dim"),
                      *([(f" {sep} quick scan", "dim")] if s.get("mode") == "quick" else []),
                      *([(f" {sep} partial ({s.get('unscanned', 0)} folder{'s' if s.get('unscanned', 0) != 1 else ''} not fully read)", "peach")] if s.get("partial") else []),
                      *([(f" {sep} {s['unreadable']} unreadable", "rose")] if s.get("unreadable") else []))]
        fs = s.get("fs")
        if fs and fs.get("total"):                              # how full the whole disk is, and this folder's share of it
            frac = fs["used"] / fs["total"]
            colour = "mint" if frac < 0.70 else "lemon" if frac < 0.85 else "peach" if frac < 0.95 else "rose"
            head.append(P.row(" ", (fs["mount"] + (" quota" if fs.get("parent") else ""), "lavender"), "  ", (hbar(frac, 20 if width >= 72 else 12, uni), colour), f" {frac * 100:.0f}% full",
                              (f" {sep} {fmt_size(fs['avail'])} free of {fmt_size(fs['total'])}", "dim")))
            lower = ("≥ " if uni else ">= ") if t.get("incomplete") else ""
            used = f", and {lower}{100 * t['bytes'] / fs['used']:.0f}% of what is used" if fs.get("used") else ""
            head.append(P.row("  ", (f"this folder is {lower}{100 * t['bytes'] / fs['total']:.1f}% of it{used}", "dim")))
            pool = fs.get("parent")
            if pool and pool.get("total"):                  # the quota sits inside a bigger filesystem: how full is that?
                pf = pool["used"] / pool["total"]
                head.append(P.row("  ", ("the filesystem under it is ", "dim"), (f"{pf * 100:.0f}% full", "mint" if pf < 0.70 else "lemon" if pf < 0.85
                                 else "peach" if pf < 0.95 else "rose"), (f": {fmt_size(pool['used'])} of {fmt_size(pool['total'])}", "dim")))
        if self.view == "largest":
            head.append(P.row(" ", ("largest files", "bold"), (f" {sep} biggest {len(s.get('largest', []))} in the scan", "dim")))
        else:
            names = [t["name"]] + self.path
            head.append(P.row(" ", *[x for i, nm in enumerate(names) for x in
                                     ([(f" {crumb} ", "dim")] if i else []) + [(nm, "bold" if i == len(names) - 1 else "lavender")]]))
        head.append("")

        items = self.items()
        room = None if height is None else max(1, height - len(head) - (2 if footer else 1))
        self._page = max(1, (room or 10) - 1)
        self.cursor = max(0, min(self.cursor, len(items) - 1))
        if room is not None:                                   # keep the selected row inside the window
            self.scroll = min(self.scroll, self.cursor)
            self.scroll = max(self.scroll, self.cursor - room + 1)
        else:
            self.scroll = 0
        window = items[self.scroll:] if room is None else items[self.scroll:self.scroll + room]
        body = self._largest_rows(P, window, width, uni) if self.view == "largest" else self._tree_rows(P, window, width, uni)
        if not items:
            body = [P.row("  ", ("(empty)", "dim"))]

        note = self.message
        n = self.node()
        if not note and self.view == "tree" and n.get("deep"):
            note = f"this scan listed folders to depth {s['depth']}; below here only totals are kept (rescan deeper with --depth N)"
        elif not note and self.view == "tree" and n.get("unscanned"):
            note = "this folder was not read: the scan ran out of time before reaching it"
        lines = head + body + [P.row(" ", (note, "peach")) if note else ""]
        if footer:
            keys = (f"↑↓ move {sep} ⏎ open {sep} ⌫ back {sep} s sort: {self.sort} {sep} l largest {sep} q quit" if uni else
                    f"up/down move {sep} Enter open {sep} Bksp back {sep} s sort: {self.sort} {sep} l largest {sep} q quit") \
                if self.view == "tree" else \
                (f"↑↓ move {sep} ⏎ show its folder {sep} l or ⌫ back {sep} q quit" if uni else
                 f"up/down move {sep} Enter show its folder {sep} l or Bksp back {sep} q quit")
            pos = f"{min(self.cursor + 1, len(items))}/{len(items)}"
            gap = max(1, width - 1 - len(keys) - len(pos))
            lines.append(P.row(" ", (keys, "dim"), " " * gap, (pos, "dim")))
        return "\n".join(lines)

    def _tree_rows(self, P: Painter, window: list, width: int, uni: bool) -> list:
        bw = 20 if width >= 96 else 14 if width >= 80 else 10 if width >= 60 else 8
        show_files = width >= 80 and any(e["files"] is not None for e in window)
        show_age = width >= 60
        fw = max([len(f"{e['files']:,} files") for e in window if e["files"] is not None] or [0]) if show_files else 0
        mark = "❯" if uni else ">"
        fixed = 2 + 4 + bw + 2 + 6 + 2 + 2 + 8 + (2 + fw if show_files else 0) + (2 + 5 if show_age else 0)
        namew = max(8, width - fixed)
        out = []
        for i, e in enumerate(window, start=self.scroll):
            sel = i == self.cursor
            colour = PALETTE[i % len(PALETTE)]
            age = None if e["mtime"] is None else self.ref - e["mtime"]
            age_txt = "" if age is None else fmt_age(age)
            name = _trim(e["name"], namew, uni)
            unread = e.get("unscanned")
            parts = [(mark, "pink", "bold") if sel else " ", " ", f"{i + 1:>2}  ",
                     (hbar(0 if unread else e["pct"] / 100, bw, uni), "dim" if unread else colour),
                     "  ", ("    ? " if unread else f"{e['pct']:>5.1f}%", "dim" if unread else colour), "  ",
                     (name, "bold") if sel and e["kind"] != "other" else (name, "dim") if e["kind"] == "other" else name,
                     " " * (namew - len(name)), "  ", (f"{'not read':>8}", "dim") if unread else
                     ((f"{('≥ ' if uni else '>= ') + fmt_size(e['bytes']):>8}", "peach") if e.get("incomplete")
                      else (f"{fmt_size(e['bytes']):>8}", "bold"))]
            if show_files:
                parts += ["  ", (f"{e['files']:,} files".rjust(fw) if e["files"] is not None else " " * fw, "dim")]
            if show_age:
                parts += ["  ", " " * (5 - len(age_txt)), (age_txt, _age_style(age)) if age is not None else ""]
            out.append(P.row(*parts))
        return out

    def _largest_rows(self, P: Painter, window: list, width: int, uni: bool) -> list:
        big = self.scan.get("largest", [])
        top = big[0]["bytes"] if big else 1
        bw = 20 if width >= 96 else 14 if width >= 80 else 10 if width >= 60 else 0
        show_age = width >= 60
        mark = "❯" if uni else ">"
        fixed = 2 + 4 + (bw + 2 if bw else 0) + 7 + 2 + (5 + 2 if show_age else 0)
        pathw = max(10, width - fixed)
        out = []
        for i, f in enumerate(window, start=self.scroll):
            sel = i == self.cursor
            colour = PALETTE[i % len(PALETTE)]
            age = self.ref - f["mtime"]
            age_txt = fmt_age(age)
            parts = [(mark, "pink", "bold") if sel else " ", " ", f"{i + 1:>2}  "]
            if bw:
                parts += [(hbar(f["bytes"] / top if top else 0, bw, uni), colour), "  "]
            parts += [(f"{fmt_size(f['bytes']):>7}", "bold"), "  "]
            if show_age:
                parts += [" " * (5 - len(age_txt)), (age_txt, _age_style(age)), "  "]
            path = _trim(f["path"], pathw, uni, front=True)
            parts.append((path, "bold") if sel else path)
            out.append(P.row(*parts))
        return out


def run_explorer(ex: Explorer, color: bool = True, uni: bool = True, stdin=None, stdout=None) -> None:
    """Run the explorer full screen until the user quits. The terminal is always put back as it was."""
    import termios
    import tty
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    fd = stdin.fileno()
    saved = termios.tcgetattr(fd)
    last, dirty = None, True
    try:
        tty.setcbreak(fd)                                    # no echo, no line buffering; Ctrl-C still interrupts
        stdout.write("\033[?1049h\033[?25l")                 # the alternate screen, and no cursor
        stdout.flush()
        while True:
            size = shutil.get_terminal_size((80, 24))
            if dirty or size != last:
                frame = ex.frame(width=min(size.columns, 120), height=size.lines, color=color, uni=uni)
                stdout.write("\033[H" + "\033[K\n".join(frame.splitlines()) + "\033[K\033[J")
                stdout.flush()
                last, dirty = size, False
            if not select.select([fd], [], [], 0.4)[0]:     # a quiet moment: loop to notice a resized window
                continue
            data = os.read(fd, 64)
            if not data:
                return
            for k in parse_keys(data):
                dirty = True
                if not ex.key(k):
                    return
    except KeyboardInterrupt:
        pass
    finally:
        stdout.write("\033[?25h\033[?1049l")
        stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
