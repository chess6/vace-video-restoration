#!/usr/bin/env python3
"""Locate a file anywhere in this repository, by name, path or content.

Why this exists. The tree carries several large directories that are not this
project's own work -- the installed ComfyUI, the virtualenv, downloaded model
weights, caches, generated intermediates. A plain `find` from the repo root
spends most of its time inside them and buries the handful of interesting hits
under thousands of vendored ones. This walks the tree with those pruned by
default, and takes `--all` to look everywhere when that is genuinely what is
wanted.

It is read-only: it opens files only to match `--grep`, and it writes nothing,
opens nothing, and prints nothing but paths and structural facts (size, mtime).
Nothing here decodes or previews media -- CLAUDE.md rules 1 and 2b.

    scripts/find_file.py roi                     # name contains "roi"
    scripts/find_file.py '*_mask.png'            # glob on the filename
    scripts/find_file.py 'intermediate/**/*.mp4' --path
    scripts/find_file.py --ext py --grep 'def resolve_targets'
    scripts/find_file.py chunk --long --sort mtime
    scripts/find_file.py cudnn --all             # include venv, ComfyUI, models

Stdlib only, so it runs with the system python as well as the project venv.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Directory names pruned unless --all. These are third-party installs, caches,
# and bulk artefacts -- none of them is where a file of this project's own lives.
PRUNE_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "ComfyUI",
    "__pycache__",
    "models",
    "node_modules",
    "venv",
    ".venv",
    "wheels",
}

# Extensions never opened for --grep. Reading these as text is pointless, and
# for the media ones it is also forbidden (rule 2b): the search skips them
# rather than sniffing their bytes.
BINARY_EXT = {
    ".7z", ".avi", ".bin", ".ckpt", ".dylib", ".gif", ".gz", ".ico", ".jpeg",
    ".jpg", ".mkv", ".mov", ".mp3", ".mp4", ".npy", ".npz", ".o", ".onnx",
    ".pdf", ".png", ".pt", ".pth", ".pyc", ".safetensors", ".so", ".tar",
    ".tif", ".tiff", ".wav", ".webm", ".webp", ".whl", ".zip",
}

GREP_MAX_BYTES = 8 * 1024 * 1024  # a match beyond this is not what anyone meant


def human(n: int) -> str:
    """Byte count as a short fixed-width string."""
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:6.0f}{unit}" if unit == "B" else f"{n:6.1f}{unit}"
        n /= 1024.0
    return f"{n:6.1f}T"


def name_matches(candidate: str, pattern: str) -> bool:
    """Glob if the pattern looks like one, case-insensitive substring otherwise.

    Typing a bare word is by far the common case, and requiring `*word*` for it
    is friction with no upside; a pattern carrying a wildcard clearly means the
    glob, so the two intents separate cleanly without a flag.
    """
    if any(c in pattern for c in "*?["):
        return fnmatch.fnmatch(candidate.lower(), pattern.lower())
    return pattern.lower() in candidate.lower()


def content_matches(path: Path, rx: re.Pattern) -> bool:
    """True if the file is readable text and some line matches."""
    if path.suffix.lower() in BINARY_EXT:
        return False
    try:
        if path.stat().st_size > GREP_MAX_BYTES:
            return False
        with path.open("r", encoding="utf-8", errors="strict") as fh:
            return any(rx.search(line) for line in fh)
    except (UnicodeDecodeError, OSError):
        return False


def walk(root: Path, args) -> list[Path]:
    """Depth-first walk with the prune list applied, newest-first collection."""
    hits: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        if not args.all:
            dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        if not args.hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        entries = []
        if args.type in ("f", "any"):
            entries += filenames
        if args.type in ("d", "any"):
            entries += dirnames

        for entry in entries:
            if not args.hidden and entry.startswith("."):
                continue
            path = here / entry
            rel = path.relative_to(REPO).as_posix()

            if args.pattern:
                subject = rel if args.path else entry
                if not name_matches(subject, args.pattern):
                    continue
            if args.ext and path.suffix.lower().lstrip(".") not in args.ext:
                continue
            if args.grep and (path.is_dir() or not content_matches(path, args.grep)):
                continue
            hits.append(path)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Find a file in this repository by name, path or content.",
        epilog="Read-only: nothing is written, opened or displayed.",
    )
    ap.add_argument("pattern", nargs="?", help="filename substring, or a glob "
                    "if it contains * ? [ ]")
    ap.add_argument("--path", action="store_true",
                    help="match the pattern against the repo-relative path "
                         "rather than the bare filename")
    ap.add_argument("--ext", help="comma-separated extensions to keep, e.g. py,sh")
    ap.add_argument("--grep", metavar="REGEX",
                    help="keep only text files containing this regex")
    ap.add_argument("-i", "--ignore-case", action="store_true",
                    help="case-insensitive --grep")
    ap.add_argument("--type", choices=("f", "d", "any"), default="f",
                    help="files, directories, or both (default: f)")
    ap.add_argument("--root", default=".",
                    help="subtree to search, relative to the repo root")
    ap.add_argument("--all", action="store_true",
                    help=f"do not prune {', '.join(sorted(PRUNE_DIRS))}")
    ap.add_argument("--hidden", action="store_true",
                    help="include dot-files and dot-directories")
    ap.add_argument("--sort", choices=("path", "mtime", "size"), default="path",
                    help="ordering; mtime and size are newest/largest first")
    ap.add_argument("-l", "--long", action="store_true",
                    help="print size and modification time alongside each path")
    ap.add_argument("-n", "--limit", type=int, default=200,
                    help="stop printing after this many hits (default: 200)")
    args = ap.parse_args()

    if not args.pattern and not args.grep and not args.ext:
        ap.error("give a pattern, or --grep, or --ext -- "
                 "otherwise this just lists the whole tree")

    root = (REPO / args.root).resolve()
    if not root.is_dir():
        print(f"error: no such directory under the repo: {args.root}",
              file=sys.stderr)
        return 2
    if REPO not in root.parents and root != REPO:
        print("error: --root must stay inside the repository", file=sys.stderr)
        return 2

    if args.ext:
        args.ext = {e.strip().lower().lstrip(".") for e in args.ext.split(",")}
    if args.grep:
        args.grep = re.compile(args.grep, re.IGNORECASE if args.ignore_case else 0)

    hits = walk(root, args)

    if args.sort == "mtime":
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    elif args.sort == "size":
        hits.sort(key=lambda p: p.stat().st_size if p.is_file() else 0,
                  reverse=True)
    else:
        hits.sort(key=lambda p: p.relative_to(REPO).as_posix())

    shown = hits[: args.limit] if args.limit > 0 else hits
    for path in shown:
        rel = path.relative_to(REPO).as_posix()
        if args.long:
            st = path.stat()
            when = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            size = human(st.st_size) if path.is_file() else "     -"
            kind = "d" if path.is_dir() else "f"
            print(f"{kind} {size}  {when}  {rel}")
        else:
            print(rel)

    if not hits:
        hint = "" if args.all else "  (try --all to include pruned directories)"
        print(f"no match{hint}", file=sys.stderr)
        return 1
    if len(hits) > len(shown):
        print(f"... {len(hits) - len(shown)} more; raise --limit",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
