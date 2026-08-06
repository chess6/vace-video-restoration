#!/usr/bin/env python
"""Phase 8c - separate the things that must stay IN FRONT of the subject.

The subject mask says what to regenerate. It says nothing about what is between
the camera and the subject: another candidate crossing in front, a held object, a
foreground occluder. Regenerating the subject and compositing it onto the
plate paints over all of those, which is what makes interactions read wrongly.

This builds a second mask - the occluders - so compositing can use an explicit
layer order:

    1. restored environment (the SeedVR2 plate)
    2. the generated target subject
    3. preserved foreground: occluders, held objects, other candidates

Occluders are taken from the ORIGINAL footage, never regenerated: they are not
the subject, so there is no reason to synthesise them and every reason not to.

It also answers the question the brief asks directly - whether mask dilation
lets the regenerated subject spill over someone else. `grow` pixels of dilation
are applied to the subject mask, and the overlap with the occluder mask is
measured and reported. A non-zero overlap is a real defect, not a rounding
detail: it is the generated subject painting onto another candidate.

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

# SegFormer attributes classes that indicate a CANDIDATE (any candidate, not only ours).
CANDIDATE_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}

# Depth Anything returns relative INVERSE depth, normalised per shot by
# make_depth.py, so a brighter pixel is a NEARER one. Everything below depends on
# that direction; it is stated here rather than buried in a comparison.
NEARER_IS_BRIGHTER = True
# How much nearer, in grey levels, an occluder must be before it is accepted as
# being in front. Depth is a smooth estimate and the boundary between two candidates
# is exactly where it is least certain, so a bare > would call noise an occlusion.
DEPTH_MARGIN = 6.0


def verified_foreground(occ: np.ndarray, subj: np.ndarray,
                        depth: np.ndarray | None, reach: int = 9):
    """The occluder pixels that are actually IN FRONT of the subject.

    Being near the subject in the image plane is not being in front of it. A
    candidate a metre behind the subject, or a static scene edge beside them,
    touches the silhouette in 2D and occludes nothing. Counting those was
    overstating occlusion, and the layering decision it feeds is about depth
    order.

    Each connected occluder component is tested on its own - one distant subject
    must not disqualify a near one - by comparing its median depth against the
    depth of the subject pixels it touches. Components that are not nearer are
    dropped.

    Returns (verified_foreground, touching) so the caller can report genuine
    occlusion and mere proximity as the separate things they are.
    """
    import cv2
    k = np.ones((reach, reach), np.uint8)
    touching = occ & cv2.dilate(subj.astype(np.uint8), k).astype(bool)
    if depth is None or not touching.any():
        return np.zeros_like(occ), touching

    verified = np.zeros_like(occ)
    n, labels = cv2.connectedComponents(touching.astype(np.uint8))
    for i in range(1, n):
        comp = labels == i
        if comp.sum() < 16:
            continue
        # The subject pixels this component actually abuts - not the whole
        # subject, whose far side may be at a completely different depth.
        edge = subj & cv2.dilate(comp.astype(np.uint8), k).astype(bool)
        if edge.sum() < 16:
            continue
        d_occ = float(np.median(depth[comp]))
        d_sub = float(np.median(depth[edge]))
        nearer = (d_occ > d_sub + DEPTH_MARGIN if NEARER_IS_BRIGHTER
                  else d_occ < d_sub - DEPTH_MARGIN)
        if nearer:
            verified |= comp
    return verified, touching


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
    from composite_subject import alpha_from_mask
    from make_reference_pack import Parser

    log = setup_logging("make_occluders", args.verbose)
    cfg = load_config(args.config)
    man = load_manifest()
    grow = int(cfg["mask"].get("grow", 0))
    comp = cfg.get("composite", {})
    band = int(comp.get("band_px", 3))
    center_band = bool(comp.get("center_band", True))
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

        # Depth for this shot, when one chunk covers it exactly. Anything else
        # would need per-chunk seeking and overlap handling; rather than guess an
        # alignment, depth ordering is reported as unavailable and only proximity
        # is stated - see verified_foreground().
        dcap, depth_src = None, None
        shot_chunks = [c for c in man["chunks"] if c["shot_id"] == sid]
        exact = [c for c in shot_chunks
                 if int(c["start_frame"]) == int(shot["start_frame"])
                 and int(c["end_frame"]) == int(shot["end_frame"])]
        if exact and (P.root / exact[0]["depth_path"]).exists():
            depth_src = P.root / exact[0]["depth_path"]
            dcap = cv2.VideoCapture(str(depth_src))
        else:
            log.warning("%s: no single chunk covers this shot, so depth is not "
                        "frame-aligned here. Occlusion cannot be depth-verified; "
                        "only proximity is reported.", sid)

        dst = out_dir / f"{sid}_occluders.mkv"
        ff = None
        n = 0
        overlaps, occ_fracs = [], []
        subj_cov, occluded_frames = [], []
        proximity, proximity_frames = [], []
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
            candidates = np.isin(labels, list(CANDIDATE_IDS))
            # Anyone who is a candidate but not our tracked subject. Eroding the
            # subject side slightly avoids labelling the subject's own outline
            # as an occluder purely from parser/tracker disagreement.
            # Occluders are defined against the subject at its TRUE extent, with
            # no dilation. Subtracting the DILATED mask here would make the
            # dilation test below tautological: it would be measuring the
            # overlap of a set with its own complement, which is empty whatever
            # the dilation radius is. That is the bug this replaces.
            occ = candidates & ~subj
            occ = cv2.morphologyEx(occ.astype(np.uint8), cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8)).astype(bool)

            # Now the dilation test is a real question: growth is applied AFTER
            # the occluder set is fixed, so it genuinely can reach into it.
            if k_grow is not None:
                grown = cv2.dilate(subj.astype(np.uint8), k_grow).astype(bool)
                inter = int((grown & occ).sum())
                overlaps.append(inter / max(1, int(grown.sum())))
            occ_fracs.append(float(occ.mean()))

            # How much of the SUBJECT is actually occluded on this frame?
            #
            # Measured as the weighted intersection between the alpha the
            # generated subject would be composited with BEFORE any occluder is
            # applied, and the depth-verified foreground. That is the quantity
            # the layer exists to control: how much of the subject a real
            # foreground object takes away.
            #
            # The previous number counted occluder pixels inside a 9x9
            # neighbourhood of the mask and called the result subject-alpha
            # coverage. It was neither - it was proximity, unweighted and
            # unverified, and it counted anything beside the subject as being in
            # front of it. Both are kept, named for what they are.
            depth_frame = None
            if dcap is not None:
                okd, fd = dcap.read()
                if okd:
                    depth_frame = cv2.cvtColor(fd, cv2.COLOR_BGR2GRAY).astype(np.float32)
            fg, touching = verified_foreground(occ, subj, depth_frame)
            a_sub = alpha_from_mask(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), band,
                                    center_band)
            a_total = float(a_sub.sum())
            cov = float((a_sub * fg).sum()) / max(1e-6, a_total)
            prox = int(touching.sum()) / max(1, int(subj.sum()))
            subj_cov.append(cov)
            proximity.append(prox)
            if cov > 0.002:
                occluded_frames.append(n)
            if prox > 0.002:
                proximity_frames.append(n)

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
        if dcap is not None:
            dcap.release()
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
        mean_prox = float(np.mean(proximity)) if proximity else 0.0
        log.info("%s: depth-verified occlusion on %d/%d frame(s); subject alpha "
                 "actually taken by foreground mean %.2f%% max %.2f%%",
                 sid, len(occluded_frames), n, 100 * mean_cov, 100 * max_cov)
        log.info("%s: proximity (occluder pixels merely NEXT TO the subject, not "
                 "verified in front) on %d/%d frame(s), mean %.2f%% - reported "
                 "separately because it is not occlusion%s", sid,
                 len(proximity_frames), n, 100 * mean_prox,
                 "" if dcap is not None or depth_src else
                 "; no aligned depth here, so nothing could be verified")
        if shimmer is not None:
            log.info("%s: occluder boundary shimmer %.3f (frame-to-frame area "
                     "change / mean area; lower is steadier)", sid, shimmer)
        if overlaps:
            if worst > 0.001:
                log.warning("%s: DILATION OVERLAP - growing the subject mask by "
                            "%d px reaches into another candidate/object on up to "
                            "%.2f%% of its area. The generated subject would paint "
                            "over them; reduce mask.grow or rely on the occluder "
                            "layer.", sid, grow, 100 * worst)
            else:
                log.info("%s: dilation check OK - %d px of growth never reaches "
                         "an occluder (worst %.4f%%)", sid, grow, 100 * worst)
        shot["occluders"] = {"path": rel(dst), "mean_frame_fraction": round(mean_occ, 5),
                             "worst_dilation_overlap": round(worst, 6),
                             "dilation_px": grow,
                             "frames_with_verified_occlusion": len(occluded_frames),
                             "frames_total": n,
                             # Weighted intersection of the pre-occlusion subject
                             # alpha with the depth-verified foreground.
                             "subject_alpha_occluded_mean": round(mean_cov, 5),
                             "subject_alpha_occluded_max": round(max_cov, 5),
                             "depth_verified": bool(depth_src is not None),
                             "depth_source": rel(depth_src) if depth_src else None,
                             # NOT occlusion: occluder pixels within 9 px of the
                             # subject, unweighted and unverified.
                             "proximity_mean": round(mean_prox, 5),
                             "frames_with_proximity": len(proximity_frames),
                             "boundary_shimmer": (round(shimmer, 4)
                                                  if shimmer is not None else None),
                             "independent_masks": True}
    save_manifest(man)
    log.info("Occluder masks -> %s", rel(out_dir))
    log.info("Composite order: plate, then generated subject, then these on top.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
