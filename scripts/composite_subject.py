#!/usr/bin/env python
"""Phase 9d - lay a VACE-generated subject over a restored background plate.

The second of the two integration paths:

  in_vace     VACE preserves the plate itself, because control_video outside the
              mask IS the background. One pass, no seam - but the background is
              only as stable as the model's willingness to leave the preserved
              region alone.
  composite   (this script) The subject is cut out of the VACE output with the
              lossless tracked mask and laid over the plate afterwards. The
              background is then bit-exact by construction; the cost is a real
              edge to manage.

The edge is deliberately narrow. A wide alpha ramp is what produces halos and
the pasted-on look around hair and clothing, so the ramp is a few pixels wide and
is centred on the silhouette (eroded by half the band before ramping) rather than
grown outwards from it. Feathering happens exactly once on each path: the
workflow's FeatherMask for in_vace, this band for composite - never both.

    scripts/composite_subject.py --chunk shot0000_c000 \
        --subject outputs/restored_480p/shot0000_c000.mp4 \
        --background intermediate/background/<profile>/<hash>/bg_...mkv \
        --out outputs/comparisons/composited.mkv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, composite_key, load_config, load_manifest, probe_dims_fps, probe_frames,
    rel, setup_logging,
)


def alpha_from_mask(mask_u8: np.ndarray, band_px: int, center: bool) -> np.ndarray:
    """Narrow, centred alpha ramp from a hard mask. Returns float32 in [0, 1].

    The tracked mask is binary (white = subject). Blurring it directly would push
    the ramp outward and leave a bright rim of regenerated pixels laid over the
    restored background - the halo. Eroding by half the band first puts the ramp
    across the silhouette instead of outside it.
    """
    import cv2
    m = (mask_u8 > 127).astype(np.uint8) * 255
    if band_px <= 0:
        return (m > 127).astype(np.float32)
    if center:
        r = max(1, band_px // 2)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m = cv2.erode(m, k)
    r = band_px * 2 + 1
    a = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (r, r), 0)
    return np.clip(a, 0.0, 1.0)


def occluder_alpha(occ_u8: np.ndarray, band_px: int) -> np.ndarray:
    """How much of the generated figure survives where something is in front.

    Returns 1 away from the occluder, 0 in its opaque core, and a ramp between -
    laid ONE-SIDED, entirely inside the occluder. That direction matters. A
    symmetric ramp would spread half its width outwards, blending generated
    figure over pixels that belong to whoever is actually in front, which is the
    same over-painting the occluder layer exists to prevent. Growing it inwards
    can only ever lose a sliver of occluder to the figure, never the reverse.

    A fully hard edge was the previous behaviour. It is correct about ownership
    but reads as a cut-out, because the tracked boundary lands on a different
    pixel each frame; a couple of pixels of ramp absorbs that jitter without
    giving the figure any ground.
    """
    import cv2
    occ = occ_u8 > 127
    a = np.ones(occ.shape, np.float32)
    if not occ.any():
        return a
    if band_px <= 0:
        a[occ] = 0.0
        return a
    # Distance INTO the occluder: 0 at its silhouette, growing inwards.
    d = cv2.distanceTransform(occ.astype(np.uint8), cv2.DIST_L2, 3)
    a[occ] = np.clip(1.0 - d[occ] / float(band_px), 0.0, 1.0)
    return a


def stabilize_occluder_alpha(cur: np.ndarray, prev: np.ndarray | None,
                             smooth: float) -> np.ndarray:
    """Carry part of the previous frame's foreground alpha into this one, WITHOUT
    letting it contradict the current frame.

    A plain temporal blend is wrong in two ways that both show on screen. Where
    an occluder has moved on, the old alpha trails behind it as a smear of
    transparency over background it no longer covers. Where it has just arrived,
    the old alpha eats into its core and the generated figure shows through a
    solid object.

    So the blend is confined to the current frame's ramp. Outside the current
    occluder the alpha is pinned to 1, and throughout its opaque core to 0; only
    the narrow boundary band - the part that was jittering in the first place -
    is allowed to remember anything. Those two regions are exactly the pixels
    where `cur` is 1.0 or 0.0 by construction, so no extra mask is needed.
    """
    if prev is None or smooth <= 0 or prev.shape != cur.shape:
        return cur
    sm = (1.0 - smooth) * cur + smooth * prev
    pinned = (cur == 1.0) | (cur == 0.0)
    sm[pinned] = cur[pinned]
    return np.clip(sm, 0.0, 1.0)


def composite(subject: Path, background: Path, mask: Path, out: Path,
              band_px: int, center: bool, fps: int, log,
              occluders: Path | None = None, occ_band_px: int = 2,
              occ_smooth: float = 0.0) -> dict:
    """Per-frame alpha composite, in explicit layer order. Streams to disk.

        1. restored environment (background plate)
        2. generated target figure
        3. preserved foreground: occluders, held objects, other people

    Layer 3 is taken from the plate, never regenerated: the occluder's own pixels
    are the RESTORED original, so they get SeedVR2's detail without ever passing
    through VACE. Without this layer the figure is pasted over anyone crossing in
    front of it, which reads as broken interaction however good the figure looks.
    """
    import cv2
    import subprocess

    srcs = {"subject": subject, "background": background, "mask": mask}
    if occluders is not None:
        srcs["occ"] = occluders
    caps = {n: cv2.VideoCapture(str(p)) for n, p in srcs.items()}
    for n, c in caps.items():
        if not c.isOpened():
            raise RuntimeError(f"cannot open {n}: {locals()[n] if n in locals() else n}")

    w, h, _ = probe_dims_fps(subject)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p", "-an", str(out)],
        stdin=subprocess.PIPE)

    n = 0
    alpha_sum = 0.0
    occ_prev: np.ndarray | None = None
    occ_frames = 0
    while True:
        oks, fs = caps["subject"].read()
        okb, fb = caps["background"].read()
        okm, fm = caps["mask"].read()
        if not (oks and okb and okm):
            break
        if fb.shape[:2] != fs.shape[:2] or fm.shape[:2] != fs.shape[:2]:
            raise RuntimeError(
                f"frame {n}: subject {fs.shape[:2]}, background {fb.shape[:2]}, "
                f"mask {fm.shape[:2]} disagree")
        a = alpha_from_mask(cv2.cvtColor(fm, cv2.COLOR_BGR2GRAY), band_px, center)
        if "occ" in caps:
            oko, fo = caps["occ"].read()
            if oko:
                # Layer 3: anything in front of the subject wins, with an opaque
                # core and a narrow one-sided boundary (see occluder_alpha).
                # The occluder silhouette is re-segmented every frame and its
                # boundary lands a pixel or two differently each time. The blend
                # steadies that, confined to the current frame's ramp so it can
                # neither trail behind a moving occluder nor hollow out its core.
                oa = stabilize_occluder_alpha(
                    occluder_alpha(cv2.cvtColor(fo, cv2.COLOR_BGR2GRAY),
                                   occ_band_px),
                    occ_prev, occ_smooth)
                occ_prev = oa
                if (oa < 1.0).any():
                    occ_frames += 1
                a = a * oa
        alpha_sum += float(a.mean())
        a3 = a[:, :, None]
        blended = (fs.astype(np.float32) * a3 + fb.astype(np.float32) * (1.0 - a3))
        ff.stdin.write(cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_BGR2RGB).tobytes())
        n += 1
    for c in caps.values():
        c.release()
    ff.stdin.close()
    ff.wait()
    if ff.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {out}")
    if n == 0:
        raise RuntimeError("no frames composited")

    got = probe_frames(out)
    if got != n:
        raise RuntimeError(f"wrote {n} frames but {out.name} decodes {got}")
    log.info("Composited %d frame(s) -> %s (mean subject coverage %.2f%%%s)",
             n, rel(out), 100 * alpha_sum / n,
             f"; foreground layer active on {occ_frames}/{n} frame(s), "
             f"{occ_band_px} px one-sided boundary, temporal smoothing "
             f"{occ_smooth:.2f}" if "occ" in caps else "")
    return {"frames": n, "mean_alpha": alpha_sum / n,
            "occluded_frames": occ_frames, "occluder_band_px": occ_band_px,
            "occluder_temporal_smooth": occ_smooth}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--chunk", required=True, help="Chunk id, for the mask and geometry")
    ap.add_argument("--subject", type=Path, required=True, help="VACE output to cut from")
    ap.add_argument("--background", type=Path, required=True, help="Restored plate")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--band-px", type=int, default=None)
    ap.add_argument("--occluders", type=Path, default=None,
                    help="Foreground mask kept ABOVE the generated figure. "
                         "Defaults to the shot's occluder mask when it exists.")
    ap.add_argument("--redo", action="store_true",
                    help="Re-composite even when the composite key still matches")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("composite_subject", args.verbose)
    cfg = load_config(args.config)
    man = load_manifest()
    comp = cfg.get("composite", {})
    band = args.band_px if args.band_px is not None else int(comp.get("band_px", 3))

    c = next((x for x in man["chunks"] if x["chunk_id"] == args.chunk), None)
    if c is None:
        log.error("Unknown chunk %s", args.chunk)
        return 1
    mask = P.root / c["mask_path"]
    for name, p in (("subject", args.subject), ("background", args.background),
                    ("mask", mask)):
        if not p.exists():
            log.error("%s missing: %s", name, p)
            return 1

    # All three must describe the same frames at the same size, or the composite
    # would blend different moments together.
    ns = {n: probe_frames(p) for n, p in
          (("subject", args.subject), ("background", args.background), ("mask", mask))}
    if len(set(ns.values())) != 1:
        log.error("Frame counts disagree: %s", ns)
        return 1
    log.info("Compositing %s: %d frames, band %d px, centred=%s",
             args.chunk, ns["subject"], band, comp.get("center_band", True))

    occ = args.occluders
    if occ is None:
        shot = next((s for s in man["shots"] if s["shot_id"] == c["shot_id"]), {})
        rec = (shot or {}).get("occluders") or {}
        cand = P.root / rec["path"] if rec.get("path") else None
        occ = cand if cand and cand.exists() else None
    if occ is not None:
        log.info("Foreground layer: %s kept above the generated figure", rel(occ))
    settings = {"band_px": band, "center_band": bool(comp.get("center_band", True)),
                "occluder_band_px": int(comp.get("occluder_band_px", 2)),
                "occluder_temporal_smooth": float(
                    comp.get("occluder_temporal_smooth", 0.35))}
    key = composite_key(args.subject, args.background, mask, occ, settings)
    # Its own key, decided on its own inputs. A compositing change re-composites;
    # it never marks the generation behind it stale.
    side = args.out.with_suffix(args.out.suffix + ".key.json")
    if (args.out.exists() and side.exists() and not args.redo
            and json.loads(side.read_text()).get("composite_key") == key):
        log.info("Composite is already current for these inputs and settings "
                 "(key %s). Pass --redo to force.", key)
        return 0

    stats = composite(args.subject, args.background, mask, args.out, band,
                      settings["center_band"], int(c["fps"]), log,
                      occluders=occ, occ_band_px=settings["occluder_band_px"],
                      occ_smooth=settings["occluder_temporal_smooth"])
    side.write_text(json.dumps(
        {"composite_key": key, "settings": settings, "inputs": {
            "vace_output": rel(args.subject), "plate": rel(args.background),
            "mask": rel(mask), "occluders": rel(occ) if occ else None},
         **stats}, indent=2) + "\n")
    log.info("Composite key %s -> %s", key, rel(side))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
