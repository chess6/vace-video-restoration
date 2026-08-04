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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, load_config, load_manifest, probe_dims_fps, probe_frames, rel,
    setup_logging,
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


def composite(subject: Path, background: Path, mask: Path, out: Path,
              band_px: int, center: bool, fps: int, log) -> dict:
    """Per-frame alpha composite. Streams; never holds the clip in RAM."""
    import cv2
    import subprocess

    caps = {n: cv2.VideoCapture(str(p))
            for n, p in (("subject", subject), ("background", background), ("mask", mask))}
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
    log.info("Composited %d frame(s) -> %s (mean subject coverage %.2f%%)",
             n, rel(out), 100 * alpha_sum / n)
    return {"frames": n, "mean_alpha": alpha_sum / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--chunk", required=True, help="Chunk id, for the mask and geometry")
    ap.add_argument("--subject", type=Path, required=True, help="VACE output to cut from")
    ap.add_argument("--background", type=Path, required=True, help="Restored plate")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--band-px", type=int, default=None)
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

    composite(args.subject, args.background, mask, args.out, band,
              bool(comp.get("center_band", True)), int(c["fps"]), log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
