#!/usr/bin/env python
"""Phase 9e - map an ROI generation back into the full frame.

Exact inverse of scripts/make_roi.py: each generated frame is scaled from the
generation canvas back down to its crop size and written at that crop's original
position, then the subject is composited onto the restored full-frame plate with
the same narrow centred band used everywhere else.

Only the region inside the tracked mask is taken from the generation. Everything
else is the plate, untouched: the ROI pass saw a different framing, so its idea
of the background is not comparable and must not leak in.

    scripts/map_roi_back.py --shot shot0000 --roi-output <mkv> \
        --background <plate.mkv> --out <mkv>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, load_config, load_manifest, probe_dims_fps, probe_frames, rel, setup_logging,
)
from composite_subject import alpha_from_mask  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--roi-output", type=Path, required=True)
    ap.add_argument("--background", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--band-px", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import cv2
    log = setup_logging("map_roi_back", args.verbose)
    cfg = load_config(args.config)
    man = load_manifest()
    comp = cfg.get("composite", {})
    band = args.band_px if args.band_px is not None else int(comp.get("band_px", 3))

    roi_file = P.intermediate / "roi" / f"{args.shot}_roi.json"
    if not roi_file.exists():
        log.error("No ROI transform at %s", roi_file)
        return 1
    roi = json.loads(roi_file.read_text())
    cw, ch = roi["crop_w"], roi["crop_h"]
    W = int(man["normalized"]["width"])
    H = int(man["normalized"]["height"])
    fps = int(man["normalized"]["fps"])
    mask_video = P.masks / f"{args.shot}_mask.mkv"

    for name, p in (("roi output", args.roi_output), ("background", args.background),
                    ("mask", mask_video)):
        if not p.exists():
            log.error("%s missing: %s", name, p)
            return 1

    caps = {k: cv2.VideoCapture(str(v)) for k, v in
            (("roi", args.roi_output), ("bg", args.background), ("mask", mask_video))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "ffv1", "-level", "3",
         "-pix_fmt", "yuv420p", "-an", str(args.out)], stdin=subprocess.PIPE)

    i = 0
    while True:
        okr, fr = caps["roi"].read()
        okb, fb = caps["bg"].read()
        okm, fm = caps["mask"].read()
        if not (okr and okb and okm):
            break
        x = roi["x"][min(i, len(roi["x"]) - 1)]
        y = roi["y"][min(i, len(roi["y"]) - 1)]
        # generation canvas -> crop size -> full-frame position
        patch = cv2.resize(fr, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
        placed = fb.copy()
        placed[y:y + ch, x:x + cw] = patch
        a = alpha_from_mask(cv2.cvtColor(fm, cv2.COLOR_BGR2GRAY), band,
                            bool(comp.get("center_band", True)))[:, :, None]
        out = (placed.astype(np.float32) * a + fb.astype(np.float32) * (1.0 - a))
        ff.stdin.write(cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2RGB).tobytes())
        i += 1
    for c in caps.values():
        c.release()
    ff.stdin.close()
    ff.wait()
    if ff.returncode != 0 or i == 0:
        log.error("ffmpeg failed or no frames written")
        return 1
    got = probe_frames(args.out)
    if got != i:
        log.error("wrote %d frames but %s decodes %d", i, args.out.name, got)
        return 1
    log.info("Mapped %d ROI frame(s) back at %.2fx -> %s", i, roi["scale"],
             rel(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
