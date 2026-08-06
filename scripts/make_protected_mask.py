#!/usr/bin/env python
"""Phase 8d - the protected-attribute regeneration submask.

The full subject mask says "this whole subject is the subject". Handing that to
VACE says something much stronger: regenerate all of it - every attribute, every
boundary, every accessory, every covering. With external references in play that
is an invitation to redesign the attributes the source actually carries, and it is
what produced a repainted attribute at dE 25 and an invented accessory.

This derives a much smaller mask: only the regions a reference can legitimately
improve - the anchor region - and only where the parser is CONFIDENT the region is
exposed. Everything else stays black and is supplied by the SeedVR2 plate, so
the existing attribute is restored rather than regenerated.

Fail closed, in three ways that all matter at low resolution:

  * low parser confidence is treated as attribute, not as exposed;
  * a margin is eroded away from every boundary with an attribute or covering
    class, because that boundary is exactly where the parser is least reliable
    and where an attribute edge would be lost;
  * a region must persist across neighbouring frames to be regenerated, so a
    single frame's misparse cannot open a hole in the attributes.

It also answers a question the pack needs: is the source anchor COVERED? If it is,
an external reference showing it uncovered must not be allowed to condition the
anchor at all - that would remove a covering the source actually carries.

    scripts/make_protected_mask.py [--pilot] [--shot shot0000]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, load_config, load_manifest, pilot_chunks, probe_frames, rel,
    save_manifest, setup_logging,
)

# Confidence below which a pixel is not trusted to be anything. Deliberately
# high: the cost of a false "exposed" is deleting an attribute, the cost of a
# false "covered" is a slightly smaller improvement.
MIN_CONF = 0.70
# Erosion, in pixels, away from any attribute/covering boundary.
SAFETY_PX = 3
# A pixel must be exposed on at least this fraction of a small temporal window.
PERSIST = 0.75
WINDOW = 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--shot", nargs="*", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--min-conf", type=float, default=MIN_CONF)
    ap.add_argument("--safety-px", type=int, default=SAFETY_PX)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import cv2
    from PIL import Image
    from make_reference_pack import ANCHOR_REGION, ATTRIBUTE, COVERING, LBL, Parser

    log = setup_logging("make_protected_mask", args.verbose)
    cfg = load_config(args.config)
    man = load_manifest()
    work = P.root / man["normalized"]["work_path"]
    fps = int(man["normalized"]["fps"])
    parser = Parser(log)

    # Classes by role. Anything not explicitly regenerable is protected.
    REGENERABLE = {"face", "hair"}
    ids = {n: i for i, n in LBL.items()}
    regen_ids = [ids[n] for n in REGENERABLE]
    protect_ids = [ids[n] for n in (ATTRIBUTE | COVERING)]

    chunks = pilot_chunks(man) if args.pilot else man["chunks"]
    shot_ids = sorted({c["shot_id"] for c in chunks})
    if args.shot:
        shot_ids = [s for s in shot_ids if s in set(args.shot)]

    out_dir = P.masks / "protected"
    out_dir.mkdir(parents=True, exist_ok=True)

    for sid in shot_ids:
        mask_video = P.masks / f"{sid}_mask.mkv"
        if not mask_video.exists():
            log.warning("%s: no subject mask; skipping", sid)
            continue
        shot = next(s for s in man["shots"] if s["shot_id"] == sid)
        cap = cv2.VideoCapture(str(work))
        mcap = cv2.VideoCapture(str(mask_video))
        for _ in range(int(shot["start_frame"])):
            cap.read()

        exposed_stack, subj_stack = [], []
        anchor_px, exposed_px, cover_px = 0, 0, 0
        while True:
            ok, f = cap.read()
            okm, m = mcap.read()
            if not (ok and okm):
                break
            subj = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) > 127
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            lab, conf = parser.parse_prob(Image.fromarray(rgb))

            confident = conf >= args.min_conf
            regen = np.isin(lab, regen_ids) & confident & subj
            protect = (np.isin(lab, protect_ids) | ~confident) & subj

            # Pull back from every protected boundary. The parser is least
            # reliable exactly at an attribute edge, and that edge is what must
            # survive.
            if args.safety_px > 0 and protect.any():
                k = np.ones((2 * args.safety_px + 1,) * 2, np.uint8)
                near_protected = cv2.dilate(protect.astype(np.uint8), k).astype(bool)
                regen &= ~near_protected
            exposed_stack.append(regen)
            subj_stack.append(subj)

            # Anchor-region composition, for the covering question.
            anchor = np.isin(lab, [ids[n] for n in ANCHOR_REGION]) & subj
            anchor_px += int(anchor.sum())
            exposed_px += int((np.isin(lab, [ids["face"]]) & confident & subj).sum())
            cover_px += int((np.isin(lab, [ids[n] for n in COVERING])
                             & confident & subj).sum())
        cap.release(); mcap.release()
        if not exposed_stack:
            log.warning("%s: no frames", sid)
            continue

        # Temporal persistence: a region opened by one frame's misparse is not a
        # region. Averaging over a short window and thresholding removes those.
        n = len(exposed_stack)
        arr = np.stack(exposed_stack).astype(np.float32)
        half = WINDOW // 2
        persist = np.empty_like(arr)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            persist[i] = arr[lo:hi].mean(axis=0)
        # AND with the current frame. The window is a plain temporal mean with no
        # motion compensation, so a neighbouring frame vouches for a pixel at its
        # own coordinates, not at this frame's. Without this intersection a
        # region exposed in the neighbours but covered *now* is opened for
        # regeneration over an attribute present in this very frame - the
        # exact failure the authority split exists to prevent. Persistence may
        # only ever withdraw permission, never grant it.
        final = (persist >= PERSIST) & (arr > 0.5)

        h, w = final[0].shape
        dst = out_dir / f"{sid}_protected.mkv"
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "ffv1",
             "-level", "3", "-pix_fmt", "gray", "-an", str(dst)],
            stdin=subprocess.PIPE)
        for fr in final:
            ff.stdin.write((fr.astype(np.uint8) * 255).tobytes())
        ff.stdin.close(); ff.wait()
        if ff.returncode != 0 or probe_frames(dst) != n:
            log.error("%s: protected mask write failed", sid)
            return 1

        subj_total = float(np.stack(subj_stack).sum())
        regen_total = float(final.sum())
        frac = regen_total / max(1.0, subj_total)
        anchor_exposed = exposed_px / max(1, anchor_px)
        covered = bool(cover_px > exposed_px) or anchor_exposed < 0.15

        log.info("%s: %d frame(s). Regeneration submask covers %.2f%% of the "
                 "tracked subject; the other %.2f%% is protected and comes from "
                 "the plate.", sid, n, 100 * frac, 100 * (1 - frac))
        log.info("%s: anchor-region composition - %.1f%% confidently exposed, "
                 "%.1f%% covering classes -> source anchor is %s", sid,
                 100 * anchor_exposed, 100 * cover_px / max(1, anchor_px),
                 "COVERED" if covered else "exposed")
        if covered:
            log.warning("%s: the source anchor reads as COVERED. External "
                        "references showing it uncovered must not condition the "
                        "anchor here - that would remove an attribute the source "
                        "carries. Anchor conditioning is disabled for this shot.",
                        sid)
        if frac < 0.02:
            log.warning("%s: almost nothing is confidently exposed (%.2f%%). At "
                        "this resolution there is no region a reference can "
                        "legitimately improve, so VACE would be redrawing "
                        "protected pixels. The SeedVR2 plate alone is the "
                        "correct output for this shot.", sid, 100 * frac)

        shot["protected_mask"] = {
            "path": rel(dst),
            "regenerable_fraction": round(frac, 5),
            "protected_fraction": round(1 - frac, 5),
            "min_confidence": args.min_conf,
            "safety_px": args.safety_px,
            "persistence": PERSIST,
            "regenerable_classes": sorted(REGENERABLE),
            "source_anchor_exposed_fraction": round(anchor_exposed, 4),
            "source_anchor_covered": covered,
            "anchor_conditioning_allowed": not covered,
            "frames": n,
        }
    save_manifest(man)
    log.info("Protected submasks -> %s", rel(out_dir))
    log.info("White = VACE may regenerate. Black = the plate supplies it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
