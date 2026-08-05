#!/usr/bin/env python
"""Is any of this better than simply upscaling the source to 720p?

The pipeline's own working stream already answers half the question by existing:
`preprocess_source.py` scales the source to the working geometry with **Lanczos**
(`scale=...:flags=lanczos`), so `intermediate/normalized/work_*.mp4` IS the
default-upscaler baseline, frame-aligned with every restored variant by
construction. Nothing needs re-upscaling; it needs putting side by side.

Two artefacts, because they answer different questions:

  * **2-up videos at native resolution.** Baseline left, candidate right, no
    scaling anywhere. Motion artefacts - flicker, crawling texture, temporal
    instability - are invisible in stills and are the usual way a neural
    upscaler loses to Lanczos.
  * **100% crops.** A 960x720 panel shrunk to fit a grid is a picture of a
    resampler, not of the restoration. These are cut, never resized, from the
    tracked subject's box, which is where the detail argument is actually had.

Sharpness accompanies both, and is a PROXY: docs/STATE.md records that ringing
and noise raise high-frequency energy too, so it is reported against the
baseline rather than as a score.

Rule 1: everything is written to disk and the paths are printed. Nothing is
displayed, and no frame is described.

    scripts/compare_720p.py baseline=path candidate=path [...] [--mask M]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, setup_logging  # noqa: E402


def read_frames(path: Path, limit: int = 400) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    while len(out) < limit:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return np.asarray(out)


def sharpness(frames: np.ndarray, idx, box=None) -> float:
    import cv2
    vals = []
    for i in idx:
        f = frames[i]
        if box:
            x0, y0, x1, y1 = box
            f = f[y0:y1, x0:x1]
        vals.append(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),
                                  cv2.CV_64F).var())
    return float(np.median(vals))


def label_bar(width: int, text: str, height: int = 28) -> np.ndarray:
    import cv2
    bar = np.zeros((height, width, 3), np.uint8)
    cv2.putText(bar, text, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def read_masks(mask_path: Path) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(mask_path))
    out = []
    while True:
        ok, m = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) > 127)
    cap.release()
    return np.asarray(out)


def mask_box(masks: np.ndarray, pad: float = 0.15):
    """Union of the mask's box, so one crop works for every frame."""
    xs0, ys0, xs1, ys1 = [], [], [], []
    h = w = 0
    for g in masks:
        if g.sum() < 64:
            continue
        ys, xs = np.where(g)
        xs0.append(xs.min()); xs1.append(xs.max())
        ys0.append(ys.min()); ys1.append(ys.max())
        h, w = g.shape
    if not xs0:
        return None
    x0, x1 = min(xs0), max(xs1)
    y0, y1 = min(ys0), max(ys1)
    dx, dy = int((x1 - x0) * pad), int((y1 - y0) * pad)
    return (max(0, x0 - dx), max(0, y0 - dy), min(w, x1 + dx), min(h, y1 + dy))


def sharpness_in_mask(frames: np.ndarray, masks: np.ndarray, idx) -> float | None:
    """Sharpness over the mask's PIXELS, not its bounding box.

    A bounding box around a head is mostly not the head: the protected submask
    is 4.42% of the figure and its box is twenty times that, so a box-based
    number is dominated by pixels the plate supplied and reports the plate's
    sharpness back as if VACE had produced it. This is the only measurement that
    sees what was actually regenerated.
    """
    import cv2
    vals = []
    for i in idx:
        if i >= len(masks):
            break
        m = masks[i]
        if m.sum() < 64:
            continue
        lap = cv2.Laplacian(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY),
                            cv2.CV_64F)
        vals.append(float(lap[m].var()))
    return float(np.median(vals)) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("streams", nargs="+", metavar="LABEL=PATH",
                    help="First one is the baseline everything is compared to")
    ap.add_argument("--mask", type=Path, default=None,
                    help="Subject mask video; crops are taken from its box")
    ap.add_argument("--crops", type=int, default=4, help="Crop sheets to write")
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("compare_720p", args.verbose)
    import cv2

    pairs = []
    for s in args.streams:
        if "=" not in s:
            log.error("expected LABEL=PATH, got %r", s)
            return 1
        lab, p = s.split("=", 1)
        path = Path(p)
        if not path.exists():
            log.error("%s: no such file: %s", lab, path)
            return 1
        pairs.append((lab, path))

    out = args.out or (P.outputs / "compare_720p")
    out.mkdir(parents=True, exist_ok=True)

    loaded = []
    for lab, path in pairs:
        fr = read_frames(path)
        if fr.size == 0:
            log.error("%s: decoded 0 frames", lab)
            return 1
        loaded.append((lab, path, fr))
        log.info("%-22s %d frames  %dx%d", lab, len(fr), fr.shape[2], fr.shape[1])

    # Refuse to compare streams that are not the same thing frame for frame.
    # A 2-up of two different geometries or lengths is a picture of an
    # alignment bug, and it would be read as a difference in restoration.
    base_lab, base_path, base = loaded[0]
    for lab, _, fr in loaded[1:]:
        if fr.shape[1:3] != base.shape[1:3]:
            log.error("%s is %dx%d but %s is %dx%d; refusing to resize either - "
                      "that would compare resamplers, not restorations.",
                      lab, fr.shape[2], fr.shape[1], base_lab,
                      base.shape[2], base.shape[1])
            return 1
    n = min(len(fr) for _, _, fr in loaded)
    if any(len(fr) != n for _, _, fr in loaded):
        log.warning("frame counts differ; comparing the first %d of each", n)

    idx = np.linspace(0, n - 1, min(16, n)).astype(int)
    masks = read_masks(args.mask) if args.mask else None
    box = mask_box(masks) if masks is not None and len(masks) else None
    if box:
        log.info("mask box for 100%% crops: x %d-%d, y %d-%d (%dx%d px); the "
                 "mask itself covers %.2f%% of it",
                 box[0], box[2], box[1], box[3], box[2] - box[0], box[3] - box[1],
                 100 * masks.sum() / max(1, len(masks) * (box[2] - box[0]) *
                                         (box[3] - box[1])))

    # ---- measured, so the pictures are not judged alone ---------------------
    stats = {}
    b_full = sharpness(base, idx)
    b_crop = sharpness(base, idx, box) if box else None
    b_in = sharpness_in_mask(base, masks, idx) if masks is not None else None
    log.info("=" * 78)
    log.info("%-22s %8s %9s %8s %9s %8s %9s", "stream", "frame", "vs base",
             "box", "vs base", "in-mask", "vs base")
    for lab, path, fr in loaded:
        f_full = sharpness(fr[:n], idx)
        f_crop = sharpness(fr[:n], idx, box) if box else None
        f_in = sharpness_in_mask(fr[:n], masks, idx) if masks is not None else None
        stats[lab] = {"frame_sharpness": round(f_full, 2),
                      "frame_vs_baseline_pct": round(100 * (f_full / b_full - 1), 1),
                      "box_sharpness": round(f_crop, 2) if f_crop else None,
                      "box_vs_baseline_pct":
                          round(100 * (f_crop / b_crop - 1), 1) if f_crop else None,
                      "in_mask_sharpness": round(f_in, 2) if f_in else None,
                      "in_mask_vs_baseline_pct":
                          round(100 * (f_in / b_in - 1), 1) if f_in and b_in else None,
                      "source": str(path)}
        log.info("%-22s %8.1f %+8.1f%% %8s %9s %8s %9s", lab, f_full,
                 100 * (f_full / b_full - 1),
                 f"{f_crop:.1f}" if f_crop else "-",
                 f"{100 * (f_crop / b_crop - 1):+.1f}%" if f_crop else "-",
                 f"{f_in:.1f}" if f_in else "-",
                 f"{100 * (f_in / b_in - 1):+.1f}%" if f_in and b_in else "-")
    log.info("=" * 78)
    log.info("`in-mask` is the only column that sees what was regenerated. The "
             "box around a head is mostly not the head, so `box` still reports "
             "pixels the plate supplied.")
    log.info("Sharpness is a proxy. Ringing and noise raise it too; the clips "
             "are the evidence, these are the check.")

    # ---- 2-up videos, native resolution, no scaling -------------------------
    h, w = base.shape[1], base.shape[2]
    for lab, path, fr in loaded[1:]:
        dst = out / f"{lab}_vs_{base_lab}.mp4"
        vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 16,
                             (w * 2, h + 28))
        for i in range(n):
            top = np.hstack([label_bar(w, f"{base_lab}  (baseline)"),
                             label_bar(w, lab)])
            vw.write(np.vstack([top, np.hstack([base[i], fr[i]])]))
        vw.release()
        # mp4v is not friendly to every player; re-encode to H.264 and verify
        # the frame count survived (rule 4).
        final = out / f"{lab}_vs_{base_lab}_h264.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(dst),
                        "-c:v", "libx264", "-crf", str(args.crf), "-preset",
                        "slow", "-pix_fmt", "yuv420p", str(final)], check=True)
        got = len(read_frames(final))
        if got != n:
            log.error("%s: %d frames in, %d out", final.name, n, got)
            return 1
        dst.unlink(missing_ok=True)
        log.info("wrote %s", final)

    # ---- 100%% crops, cut and never resized ---------------------------------
    if box:
        x0, y0, x1, y1 = box
        for k, i in enumerate(np.linspace(0, n - 1, args.crops).astype(int)):
            panels = []
            for lab, _, fr in loaded:
                crop = fr[i][y0:y1, x0:x1]
                panels.append(np.vstack([label_bar(crop.shape[1], lab), crop]))
            sheet = np.hstack(panels)
            dst = out / f"crop_frame{i:03d}.png"
            cv2.imwrite(str(dst), sheet)
            log.info("wrote %s  (%dx%d, 100%% scale)", dst, sheet.shape[1],
                     sheet.shape[0])

    (out / "compare_720p.json").write_text(json.dumps(
        {"baseline": base_lab, "frames": int(n),
         "subject_box": list(box) if box else None, "streams": stats}, indent=2))
    log.info("wrote %s", out / "compare_720p.json")
    log.info("Open them yourself; nothing was displayed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
