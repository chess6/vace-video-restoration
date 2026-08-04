#!/usr/bin/env python
"""Ask the user which figure is the target, in the least painful way possible.

When no face is resolvable in a shot there is no automatic answer, and guessing
is what put a stranger through a full generation. This does not guess. It writes
a numbered contact sheet of every detected person and prints the box for each
number, so the answer can be given as "candidate 4" instead of by reading pixel
coordinates off a still.

Nothing is displayed (CLAUDE.md rule 1): the sheet is written to disk and its
path is printed.

    scripts/pick_subject.py --shot shot0000
    # then, with the number chosen:
    scripts/track_subject.py --shot shot0000 --force \\
        --init-box <x0,y0,x1,y1> --seed-frame <frame>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, load_manifest, rel, setup_logging  # noqa: E402

COLOURS = [(255, 64, 64), (64, 220, 64), (80, 160, 255), (255, 200, 40),
           (255, 90, 255), (60, 230, 230), (255, 140, 60), (180, 120, 255)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--frames", type=int, default=4,
                    help="How many frames across the shot to show")
    ap.add_argument("--max-candidates", type=int, default=6)
    ap.add_argument("--scale", type=float, default=1.5,
                    help="Enlarge the sheet; the source is low resolution")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import cv2
    from PIL import Image
    import track_subject as T

    log = setup_logging("pick_subject", args.verbose)
    man = load_manifest()
    shot = next((s for s in man["shots"] if s["shot_id"] == args.shot), None)
    if shot is None:
        log.error("Unknown shot %s", args.shot)
        return 1
    work = P.root / man["normalized"]["work_path"]
    a, b = int(shot["start_frame"]), int(shot["end_frame"])
    picks = sorted(set(int(x) for x in np.linspace(a, b - 1, args.frames)))

    models = T.Models(log)
    cap = cv2.VideoCapture(str(work))
    panels, table = [], []
    idx = 0
    for want in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, want)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = models.detect_people(Image.fromarray(rgb))[:args.max_candidates]
        canvas = frame.copy()
        for bx in boxes:
            x0, y0, x1, y1 = (int(v) for v in bx[:4])
            col = COLOURS[idx % len(COLOURS)]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), col, 2)
            cv2.rectangle(canvas, (x0, max(0, y0 - 22)), (x0 + 34, y0), col, -1)
            cv2.putText(canvas, str(idx), (x0 + 6, max(14, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            table.append({"id": idx, "shot_frame": want - a, "abs_frame": want,
                          "box": [round(float(v), 1) for v in bx[:4]],
                          "det_score": round(float(bx[4]), 3),
                          "height_frac": round((y1 - y0) / canvas.shape[0], 3)})
            idx += 1
        cv2.putText(canvas, f"frame {want - a}", (8, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(canvas)
    cap.release()
    if not panels:
        log.error("Could not read any frame of %s", args.shot)
        return 1

    cols = 2
    rows = [np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)]
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
    sheet = np.vstack(rows)
    if args.scale != 1.0:
        sheet = cv2.resize(sheet, None, fx=args.scale, fy=args.scale,
                           interpolation=cv2.INTER_NEAREST)

    out_dir = P.masks / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{args.shot}_candidates.png"
    cv2.imwrite(str(png), sheet)
    (out_dir / f"{args.shot}_candidates.json").write_text(
        json.dumps({"shot_id": args.shot, "candidates": table}, indent=2) + "\n")

    log.info("=" * 66)
    log.info("Numbered candidates -> %s", rel(png))
    log.info("Open it yourself; nothing was displayed.")
    log.info("")
    for t in table:
        log.info("  [%d] frame %-3d  box %-32s  height %4.1f%% of frame  det %.2f",
                 t["id"], t["shot_frame"], ",".join(str(int(v)) for v in t["box"]),
                 100 * t["height_frac"], t["det_score"])
    log.info("")
    log.info("Tell me the number, or seed it directly:")
    if table:
        t = table[0]
        log.info("  scripts/track_subject.py --shot %s --force --seed-frame %d \\",
                 args.shot, t["shot_frame"])
        log.info("      --init-box %s",
                 ",".join(str(int(v)) for v in t["box"]))
    log.info("A positive point works too: --init-points x,y,+")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
