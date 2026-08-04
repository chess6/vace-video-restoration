#!/usr/bin/env python
"""Extract material an agent has been explicitly permitted to look at.

CLAUDE.md rule 2b forbids an agent forming conclusions about what is depicted in
the user's media. The user may grant narrow exceptions. This is the only way to
act on one, so that the permission is written down, bounded, and checkable
instead of being something an agent remembers having been told.

The allowlist lives at `intermediate/inspection_allowlist.txt`, untracked,
because writing the user's filenames into a tracked file is itself the leak the
privacy guard exists to prevent. One entry per line:

    path/relative/to/project/root [start_sec-end_sec]

Anything not listed is refused. A listed video may only be read inside its
listed interval - asking for a second outside it is refused too, so a permission
granted for one moment cannot quietly become a permission for the whole file.

Rule 1 is NOT relaxed by any of this. Frames are written to disk and their paths
printed. Nothing is opened in a viewer, ever, and this script has no way to.

    scripts/inspect_allowed.py --list
    scripts/inspect_allowed.py --path inputs/... --start 47 --end 52 --frames 6
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, setup_logging  # noqa: E402

ALLOWLIST = "inspection_allowlist.txt"


def load_allowlist() -> list[dict]:
    p = P.intermediate / ALLOWLIST
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        entry = {"path": parts[0], "start": None, "end": None}
        if len(parts) > 1 and "-" in parts[1]:
            a, b = parts[1].split("-", 1)
            entry["start"], entry["end"] = float(a), float(b)
        out.append(entry)
    return out


def permitted(path: Path, start: float | None, end: float | None):
    """(ok, reason). Refuses anything not listed, and any second outside the
    listed window."""
    try:
        rel = str(Path(path).resolve().relative_to(P.root.resolve()))
    except ValueError:
        return False, "outside the project root"
    for e in load_allowlist():
        if e["path"] != rel:
            continue
        if e["start"] is None:
            return True, "whole file permitted"
        if start is None or end is None:
            return False, (f"only {e['start']:g}-{e['end']:g}s is permitted; "
                           f"ask for an interval")
        if start < e["start"] - 1e-6 or end > e["end"] + 1e-6:
            return False, (f"{start:g}-{end:g}s is outside the permitted "
                           f"{e['start']:g}-{e['end']:g}s")
        return True, f"within the permitted {e['start']:g}-{e['end']:g}s"
    return False, "not on the allowlist"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", type=Path, default=None)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--frames", type=int, default=6,
                    help="Evenly spaced stills to pull from a video")
    ap.add_argument("--out", type=Path, default=None,
                    help="Directory for the extracted stills")
    ap.add_argument("--max-width", type=int, default=960)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("inspect_allowed", args.verbose)
    entries = load_allowlist()
    if args.list or args.path is None:
        if not entries:
            log.info("Nothing is permitted for inspection. rule 2b applies in "
                     "full.")
            return 0
        log.info("Permitted for inspection (%d entry/entries):", len(entries))
        for e in entries:
            log.info("  %s%s", e["path"],
                     "" if e["start"] is None
                     else f"   {e['start']:g}-{e['end']:g}s only")
        return 0 if args.list else 1

    ok, why = permitted(args.path, args.start, args.end)
    if not ok:
        log.error("REFUSED: %s (%s)", args.path, why)
        log.error("Rule 2b stands for everything not explicitly listed in %s.",
                  ALLOWLIST)
        return 1
    if not args.path.exists():
        log.error("%s does not exist", args.path)
        return 1

    out = args.out or (P.intermediate / "_inspect")
    out.mkdir(parents=True, exist_ok=True)
    log.info("Permitted: %s (%s)", args.path.name, why)

    if args.path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        dst = out / f"still_{args.path.stem}.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(args.path),
                        "-vf", f"scale='min({args.max_width},iw)':-2",
                        str(dst)], check=True)
        log.info("-> %s", dst)
        return 0

    n = max(1, args.frames)
    span = (args.end - args.start) if args.end else 0.0
    for i in range(n):
        t = args.start + (span * i / max(1, n - 1) if n > 1 else 0.0)
        dst = out / f"frame_{i:02d}_t{t:.2f}.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}",
                        "-i", str(args.path), "-frames:v", "1",
                        "-vf", f"scale='min({args.max_width},iw)':-2",
                        str(dst)], check=True)
        log.info("-> %s  (t=%.2fs)", dst.name, t)
    log.info("Stills in %s. Nothing was displayed; rule 1 is unaffected.", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
