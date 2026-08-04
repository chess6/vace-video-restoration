#!/usr/bin/env python
"""Phase 8c - separate the things that must stay IN FRONT of the subject.

The subject mask says what to regenerate. It says nothing about what is between
the camera and the subject: another person crossing in front, a held object, a
foreground limb. Regenerating the figure and compositing it straight onto the
plate paints over all of those, which is what makes interactions read wrongly.

This builds a second mask - the occluders - so compositing can use an explicit
layer order:

    1. restored environment (the SeedVR2 plate)
    2. the generated target figure
    3. preserved foreground: occluders, held objects, other people

Occluders are taken from the ORIGINAL footage, never regenerated: they are not
the subject, so there is no reason to synthesise them and every reason not to.

It also answers the question the brief asks directly - whether mask dilation
lets the regenerated figure spill over someone else. `grow` pixels of dilation
are applied to the subject mask, and the overlap with the occluder mask is
measured and reported. A non-zero overlap is a real defect, not a rounding
detail: it is the generated figure painting onto another person.

    scripts/make_occluders.py [--pilot] [--shot shot0000]
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
    P, load_config, load_manifest, pilot_chunks, probe_frames, rel,
    save_manifest, setup_logging,
)

# SegFormer clothes classes that indicate a PERSON (any person, not only ours).
PERSON_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--shot", nargs="*", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import cv2
    from PIL import Image
    from make_reference_pack import Parser

    log = setup_logging("make_occluders", args.verbose)
    cfg = load_config(args.config)
    man = load_manifest()
    grow = int(cfg["mask"].get("grow", 0))
    work = P.root / man["normalized"]["work_path"]
    fps = int(man["normalized"]["fps"])
    parser = Parser(log)

    chunks = pilot_chunks(man) if args.pilot else man["chunks"]
    shot_ids = sorted({c["shot_id"] for c in chunks})
    if args.shot:
        shot_ids = [s for s in shot_ids if s in set(args.shot)]

    out_dir = P.masks / "occluders"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in shot_ids:
        mask_video = P.masks / f"{sid}_mask.mkv"
        if not mask_video.exists():
            log.warning("%s: no subject mask; skipping", sid)
            continue
        shot = next(s for s in man["shots"] if s["shot_id"] == sid)
        cap = cv2.VideoCapture(str(work))
        mcap = cv2.VideoCapture(str(mask_video))
        # seek to the shot start on the working stream
        for _ in range(int(shot["start_frame"])):
            cap.read()

        dst = out_dir / f"{sid}_occluders.mkv"
        ff = None
        n = 0
        overlaps, occ_fracs = [], []
        subj_cov, occluded_frames = [], []
        k_grow = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (2 * grow + 1, 2 * grow + 1))
                  if grow > 0 else None)
        while True:
            ok, f = cap.read()
            okm, m = mcap.read()
            if not (ok and okm):
                break
            subj = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) > 127
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            labels = parser.parse(Image.fromarray(rgb))
            people = np.isin(labels, list(PERSON_IDS))
            # Anyone who is a person but not our tracked subject. Eroding the
            # subject side slightly avoids labelling the subject's own outline
            # as an occluder purely from parser/tracker disagreement.
            # Occluders are defined against the subject at its TRUE extent, with
            # no dilation. Subtracting the DILATED mask here would make the
            # dilation test below tautological: it would be measuring the
            # overlap of a set with its own complement, which is empty whatever
            # the dilation radius is. That is the bug this replaces.
            occ = people & ~subj
            occ = cv2.morphologyEx(occ.astype(np.uint8), cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8)).astype(bool)

            # Now the dilation test is a real question: growth is applied AFTER
            # the occluder set is fixed, so it genuinely can reach into it.
            if k_grow is not None:
                grown = cv2.dilate(subj.astype(np.uint8), k_grow).astype(bool)
                inter = int((grown & occ).sum())
                overlaps.append(inter / max(1, int(grown.sum())))
            occ_fracs.append(float(occ.mean()))

            # How much of the SUBJECT is actually occluded on this frame? This is
            # the number that says whether foreground layering matters here at
            # all - occluders elsewhere in the frame are irrelevant to it.
            near = cv2.dilate(subj.astype(np.uint8),
                              np.ones((9, 9), np.uint8)).astype(bool)
            touching = occ & near
            cov = int(touching.sum()) / max(1, int(subj.sum()))
            subj_cov.append(cov)
            if cov > 0.002:
                occluded_frames.append(n)

            if ff is None:
                h, w = occ.shape
                ff = subprocess.Popen(
                    ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo",
                     "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(fps),
                     "-i", "-", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray",
                     "-an", str(dst)], stdin=subprocess.PIPE)
            ff.stdin.write((occ.astype(np.uint8) * 255).tobytes())
            n += 1
        cap.release(); mcap.release()
        if ff is None:
            log.warning("%s: no frames", sid)
            continue
        ff.stdin.close(); ff.wait()
        if ff.returncode != 0 or probe_frames(dst) != n:
            log.error("%s: occluder mask write failed", sid)
            return 1

        worst = max(overlaps) if overlaps else 0.0
        mean_occ = float(np.mean(occ_fracs)) if occ_fracs else 0.0
        mean_cov = float(np.mean(subj_cov)) if subj_cov else 0.0
        max_cov = float(np.max(subj_cov)) if subj_cov else 0.0
        # Temporal stability of the occluder boundary: how much the occluder mask
        # changes frame to frame, relative to its size. High values mean a
        # shimmering edge, which is visible even when the layering is correct.
        shimmer = None
        if len(occ_fracs) > 1:
            d = np.abs(np.diff(occ_fracs))
            shimmer = float(np.mean(d) / max(1e-6, float(np.mean(occ_fracs))))
        log.info("%s: %d frame(s); occluders cover %.2f%% of frame on average",
                 sid, n, 100 * mean_occ)
        log.info("%s: %d/%d frame(s) have foreground ACTUALLY occluding the "
                 "subject; subject alpha covered mean %.2f%% max %.2f%%",
                 sid, len(occluded_frames), n, 100 * mean_cov, 100 * max_cov)
        if shimmer is not None:
            log.info("%s: occluder boundary shimmer %.3f (frame-to-frame area "
                     "change / mean area; lower is steadier)", sid, shimmer)
        if overlaps:
            if worst > 0.001:
                log.warning("%s: DILATION OVERLAP - growing the subject mask by "
                            "%d px reaches into another person/object on up to "
                            "%.2f%% of its area. The generated figure would paint "
                            "over them; reduce mask.grow or rely on the occluder "
                            "layer.", sid, grow, 100 * worst)
            else:
                log.info("%s: dilation check OK - %d px of growth never reaches "
                         "an occluder (worst %.4f%%)", sid, grow, 100 * worst)
        shot["occluders"] = {"path": rel(dst), "mean_frame_fraction": round(mean_occ, 5),
                             "worst_dilation_overlap": round(worst, 6),
                             "dilation_px": grow,
                             "frames_with_occlusion": len(occluded_frames),
                             "frames_total": n,
                             "subject_alpha_covered_mean": round(mean_cov, 5),
                             "subject_alpha_covered_max": round(max_cov, 5),
                             "boundary_shimmer": (round(shimmer, 4)
                                                  if shimmer is not None else None),
                             "independent_masks": True}
    save_manifest(man)
    log.info("Occluder masks -> %s", rel(out_dir))
    log.info("Composite order: plate, then generated figure, then these on top.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
