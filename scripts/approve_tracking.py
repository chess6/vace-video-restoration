#!/usr/bin/env python
"""Record that a human has looked at a tracked figure and confirmed who it is.

Generation refuses to start without this. That is deliberate: an automatic
track once ran at 0.72 confidence, flagged nothing, and was the wrong person -
and four variants and about an hour of GPU time were spent on it before anyone
looked. No confidence score can settle "is this the right person"; only someone
who knows them can.

The approval is bound to the mask's CONTENT, not to the shot name. Re-tracking
writes new pixels, which invalidates the approval automatically, so an approval
can never be inherited by a mask nobody has seen.

    scripts/approve_tracking.py --shot shot0000
    scripts/approve_tracking.py --shot shot0000 --revoke
    scripts/approve_tracking.py --list

Look at intermediate/masks/review/<shot>_review.png first. The script will not
show it to you (CLAUDE.md rule 1) and cannot check it for you.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, file_digest, load_manifest, rel, save_manifest, setup_logging,
)


def approval_valid(shot: dict) -> bool:
    """True only if the approved mask is still the mask on disk."""
    ap = shot.get("tracking_approved")
    if not ap:
        return False
    mv = P.masks / f"{shot['shot_id']}_mask.mkv"
    return bool(mv.exists() and ap.get("mask_digest") == file_digest(mv))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", nargs="*", default=None)
    ap.add_argument("--all", action="store_true",
                    help="Approve every shot with a mask (say so only if you "
                         "have actually looked at every review sheet)")
    ap.add_argument("--revoke", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--note", default="", help="Why, in your words")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("approve_tracking", args.verbose)
    man = load_manifest()

    if args.list:
        for s in man["shots"]:
            mv = P.masks / f"{s['shot_id']}_mask.mkv"
            if not mv.exists():
                continue
            ok = approval_valid(s)
            stale = bool(s.get("tracking_approved")) and not ok
            log.info("%-12s status=%-14s approved=%s%s", s["shot_id"],
                     s.get("subject_status", "pending"), "yes" if ok else "no",
                     "  (approval is STALE: the mask changed since)" if stale
                     else "")
        return 0

    if not args.shot and not args.all:
        log.error("Pass --shot <id> [...] or --all, or --list to see the state.")
        return 1

    want = set(args.shot or [])
    touched = 0
    for s in man["shots"]:
        if not args.all and s["shot_id"] not in want:
            continue
        mv = P.masks / f"{s['shot_id']}_mask.mkv"
        if not mv.exists():
            log.warning("%s: no mask; nothing to approve", s["shot_id"])
            continue
        if args.revoke:
            s.pop("tracking_approved", None)
            log.info("%s: approval revoked", s["shot_id"])
        else:
            if s.get("subject_status") in ("needs_user", "failed"):
                log.warning("%s: status is %r. Approving overrides an automatic "
                            "flag - only do this if the review sheet really "
                            "shows the right person.", s["shot_id"],
                            s.get("subject_status"))
            s["tracking_approved"] = {
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "mask": rel(mv),
                # Content-bound: re-tracking changes this and the approval dies.
                "mask_digest": file_digest(mv),
                "review_sheet": rel(P.masks / "review" / f"{s['shot_id']}_review.png"),
                "note": args.note,
            }
            log.info("%s: approved (mask %s)", s["shot_id"],
                     s["tracking_approved"]["mask_digest"])
        touched += 1

    if not touched:
        log.error("No matching shot with a mask.")
        return 1
    save_manifest(man)
    log.info("%d shot(s) updated.", touched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
