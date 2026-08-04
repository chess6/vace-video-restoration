#!/usr/bin/env python
"""Phase 8b - a stabilized region-of-interest crop around the subject.

Why this exists. At 320x240 the figure occupies a small fraction of the frame, so
generating at 720x544 spends almost all of the model's capacity on background it
was told to preserve. Cropping to the figure and scaling that crop up to the same
generation size gives the subject several times as many pixels for identical
compute. This is the main lever on "the quality improvement is too small".

What the crop must contain, beyond the figure itself: hands, whatever the figure
is touching, anyone it is interacting with, and its contact shadow. A crop that
clips the hands or the person being spoken to removes exactly the evidence the
model needs to get the interaction right.

Stability matters as much as size. A box that follows the tracker frame by frame
jitters, and a box that resizes frame by frame makes the subject pump in and out
of scale. Both are worse than a slightly loose crop. The box here is therefore:

  * union of the subject mask with any interacting people and contacted objects
  * expanded by a context margin
  * grown to a FIXED aspect ratio matching the generation canvas
  * smoothed over time with a zero-phase filter, so it neither leads nor lags
  * constrained to a single constant scale per shot, so perspective cannot drift

The transform is recorded per shot, so the mask, depth, pose and any other
control can be mapped into ROI coordinates with the same numbers, and the
generated subject mapped back into the full frame afterwards.

    scripts/make_roi.py [--pilot] [--shot shot0000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, geometry_key, load_config, load_manifest, pilot_chunks, probe_dims_fps,
    probe_frames, rel, run, save_manifest, setup_logging,
)


def mask_boxes(mask_video: Path, log) -> list[tuple[int, int, int, int] | None]:
    """Per-frame tight bounding box of the mask, or None where it is empty."""
    import cv2
    cap = cv2.VideoCapture(str(mask_video))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        m = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) > 127
        ys, xs = np.where(m)
        out.append(None if len(xs) == 0 else
                   (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    cap.release()
    return out


def smooth_series(v: np.ndarray, win: int) -> np.ndarray:
    """Zero-phase moving average: smooths without shifting the box in time.

    A causal filter would make the crop lag the subject, which reads as the
    figure drifting inside the frame.
    """
    if win <= 1 or len(v) < 3:
        return v
    win = min(win | 1, len(v) if len(v) % 2 else len(v) - 1)
    if win < 3:
        return v
    pad = win // 2
    k = np.ones(win) / win
    padded = np.concatenate([np.full(pad, v[0]), v, np.full(pad, v[-1])])
    return np.convolve(padded, k, mode="valid")[:len(v)]


def plan_roi(boxes: list, W: int, H: int, out_w: int, out_h: int, cfg: dict,
             log) -> dict:
    """Turn per-frame boxes into one stable crop track.

    Returns {"scale": s, "boxes": [(x, y, w, h) per frame]} where every box has
    the SAME width and height - a constant scale for the whole shot - and only
    the position moves.
    """
    r = cfg["roi"]
    known = [b for b in boxes if b is not None]
    if not known:
        raise RuntimeError("the subject mask is empty for every frame; no ROI")

    # Hold the last known box through frames where the subject vanished, so the
    # crop does not snap to the frame centre and back.
    filled, last = [], known[0]
    for b in boxes:
        last = b if b is not None else last
        filled.append(last)
    arr = np.array(filled, dtype=np.float32)
    cx = (arr[:, 0] + arr[:, 2]) / 2.0
    cy = (arr[:, 1] + arr[:, 3]) / 2.0
    bw = arr[:, 2] - arr[:, 0]
    bh = arr[:, 3] - arr[:, 1]

    # One scale for the whole shot. Use a high percentile of subject height, not
    # the max, so a single frame with an outstretched arm cannot shrink every
    # other frame; then check the largest frame still fits.
    target_frac = float(r["target_height_fraction"])
    margin = 1.0 + float(r["context_margin"])
    keep = float(r.get("containment_percentile", 95))
    # Percentiles, not max. A single frame with an outstretched arm or a
    # momentarily loose mask would otherwise force the crop out to the full
    # frame and destroy the entire point of the exercise. Frames beyond this
    # percentile lose a little at the edges, which costs less than no zoom.
    # Occupancy is measured on whichever axis the subject actually extends
    # along, not on height. "Fill 78% of the crop's HEIGHT" silently assumes an
    # upright figure; a reclining one is wide and short, so height-based sizing
    # asks for a crop far tighter than its own width and the planner then has to
    # give the zoom straight back. Sizing on the dominant extent asks the same
    # question - how much of the frame should the subject occupy - in a way that
    # does not depend on which way up they are.
    h_ref = float(np.percentile(bh, float(r["height_percentile"])))
    w_ref = float(np.percentile(bw, float(r["height_percentile"])))
    need_h = float(np.percentile(bh * margin, keep))
    need_w = float(np.percentile(bw * margin, keep))

    # The crop height each axis's target implies; take whichever binds.
    from_h = h_ref / max(target_frac, 1e-3)
    from_w = (w_ref / max(target_frac, 1e-3)) * (out_h / out_w)
    crop_h = max(from_h, from_w, need_h, need_w * (out_h / out_w))
    crop_w = crop_h * (out_w / out_h)
    # Never larger than the frame: letterboxing would waste the gain.
    if crop_w > W or crop_h > H:
        k = min(W / crop_w, H / crop_h)
        crop_w, crop_h = crop_w * k, crop_h * k
    cw, ch = int(round(crop_w / 2) * 2), int(round(crop_h / 2) * 2)

    win = int(r["smooth_frames"])
    cx_s = smooth_series(cx, win)
    cy_s = smooth_series(cy, win)
    # Bias the box upwards a little: heads carry identity, feet rarely do.
    cy_s = cy_s - ch * float(r.get("vertical_bias", 0.0))

    xs = np.clip(np.round(cx_s - cw / 2), 0, W - cw).astype(int)
    ys = np.clip(np.round(cy_s - ch / 2), 0, H - ch).astype(int)

    # Deadband: ignore sub-pixel wander so the crop is genuinely still when the
    # subject is still. Jitter at this scale is very visible once upscaled.
    dead = float(r["deadband_px"])
    for i in range(1, len(xs)):
        if abs(int(xs[i]) - int(xs[i - 1])) < dead:
            xs[i] = xs[i - 1]
        if abs(int(ys[i]) - int(ys[i - 1])) < dead:
            ys[i] = ys[i - 1]

    scale = out_h / ch
    occupancy = float(np.median(bh)) * scale / out_h
    log.info("ROI: crop %dx%d at scale %.2fx -> %dx%d; subject fills ~%.0f%% of "
             "crop height (target %.0f-%.0f%%)", cw, ch, scale, out_w, out_h,
             100 * occupancy, 100 * target_frac,
             100 * float(r.get("target_height_fraction_max", target_frac)))
    if occupancy < 0.3:
        log.warning("Subject fills only %.0f%% of the crop. The mask may be "
                    "loose, or the figure genuinely small in frame.", 100 * occupancy)
    if scale < float(r.get("min_useful_scale", 1.15)):
        log.warning("ROI scale is only %.2fx. The subject already fills much of "
                    "the frame here, so cropping buys almost no extra resolution "
                    "- any remaining softness is the source or the model, not the "
                    "framing. Full-frame generation is the fair comparison.", scale)
    motion = float(np.abs(np.diff(xs)).mean() + np.abs(np.diff(ys)).mean()) \
        if len(xs) > 1 else 0.0
    log.info("Crop motion: %.2f px/frame mean (0 = perfectly static)", motion)

    # ---- context preservation test ------------------------------------------
    # The zoom is only legitimate if the crop still contains what the model needs
    # to understand the action. Measure, per frame, the fraction of the subject
    # box that falls outside the crop; a hand or a contacted object leaving the
    # frame is a harder failure than a slightly softer image.
    clipped = []
    for i, b in enumerate(filled):
        x0, y0, x1, y1 = b
        ix0, iy0 = max(x0, xs[i]), max(y0, ys[i])
        ix1, iy1 = min(x1, xs[i] + cw), min(y1, ys[i] + ch)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        area = max(1.0, (x1 - x0) * (y1 - y0))
        clipped.append(1.0 - inter / area)
    worst = float(max(clipped)) if clipped else 0.0
    mean_clip = float(np.mean(clipped)) if clipped else 0.0
    log.info("Context: %.2f%% of the subject box clipped on the worst frame "
             "(mean %.2f%%)", 100 * worst, 100 * mean_clip)
    ok = (worst <= float(r.get("max_subject_clip", 0.02))
          and motion <= float(r.get("max_crop_motion_px", 3.0)))
    if not ok:
        log.warning("ROI REJECTED: worst clip %.2f%% (limit %.2f%%), motion "
                    "%.2f px/frame (limit %.2f). Falling back to full frame.",
                    100 * worst, 100 * float(r.get("max_subject_clip", 0.02)),
                    motion, float(r.get("max_crop_motion_px", 3.0)))
    return {"crop_w": cw, "crop_h": ch, "scale": scale,
            "context_ok": bool(ok),
            "worst_subject_clip": round(worst, 5),
            "mean_subject_clip": round(mean_clip, 5),
            "out_w": out_w, "out_h": out_h,
            "x": [int(v) for v in xs], "y": [int(v) for v in ys],
            "subject_height_fraction": round(occupancy, 4),
            "crop_motion_px_per_frame": round(motion, 3)}


def warp_video(src: Path, dst: Path, roi: dict, fps: int, log,
               gray: bool = False, interp: str = "lanczos") -> None:
    """Crop each frame to its ROI box and scale to the generation canvas."""
    import cv2
    import subprocess
    cap = cv2.VideoCapture(str(src))
    cw, ch = roi["crop_w"], roi["crop_h"]
    ow, oh = roi["out_w"], roi["out_h"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo",
         "-pix_fmt", "gray" if gray else "rgb24", "-s", f"{ow}x{oh}",
         "-r", str(fps), "-i", "-", "-c:v", "ffv1", "-level", "3",
         "-pix_fmt", "gray" if gray else "yuv420p", "-an", str(dst)],
        stdin=subprocess.PIPE)
    flag = {"lanczos": cv2.INTER_LANCZOS4, "nearest": cv2.INTER_NEAREST,
            "cubic": cv2.INTER_CUBIC}[interp]
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        x = roi["x"][min(i, len(roi["x"]) - 1)]
        y = roi["y"][min(i, len(roi["y"]) - 1)]
        patch = f[y:y + ch, x:x + cw]
        patch = cv2.resize(patch, (ow, oh), interpolation=flag)
        if gray:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        ff.stdin.write(np.ascontiguousarray(patch).tobytes())
        i += 1
    cap.release()
    ff.stdin.close()
    ff.wait()
    if ff.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {dst}")
    if probe_frames(dst) != i:
        raise RuntimeError(f"{dst.name}: wrote {i} frames, decodes "
                           f"{probe_frames(dst)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--shot", nargs="*", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-rejected", action="store_true",
                    help="Emit ROI streams even when the context test fails")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("make_roi", args.verbose)
    cfg = load_config(args.config)
    man = load_manifest()
    if "roi" not in cfg:
        log.error("No `roi` section in the config.")
        return 1
    v = cfg["video"]
    out_w, out_h = int(man["normalized"]["width"]), int(man["normalized"]["height"])
    fps = int(man["normalized"]["fps"])
    W, H = out_w, out_h                     # working stream == generation canvas
    work = P.root / man["normalized"]["work_path"]

    chunks = pilot_chunks(man) if args.pilot else man["chunks"]
    shot_ids = sorted({c["shot_id"] for c in chunks})
    if args.shot:
        shot_ids = [s for s in shot_ids if s in set(args.shot)]
    if not shot_ids:
        log.error("No shots selected.")
        return 1

    roi_dir = P.intermediate / "roi"
    roi_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for sid in shot_ids:
        shot = next(s for s in man["shots"] if s["shot_id"] == sid)
        mask_video = P.masks / f"{sid}_mask.mkv"
        if not mask_video.exists():
            log.warning("%s: no mask; skipping (run track_subject.py)", sid)
            continue
        log.info("=" * 62)
        log.info("%s: frames %d-%d", sid, shot["start_frame"], shot["end_frame"])
        boxes = mask_boxes(mask_video, log)
        roi = plan_roi(boxes, W, H, out_w, out_h, cfg, log)
        if not roi.get("context_ok", True) and not args.allow_rejected:
            log.warning("%s: ROI not used. Full-frame generation stands as the "
                        "correct choice for this shot.", sid)
            shot["roi"] = {"rejected": True,
                           "worst_subject_clip": roi["worst_subject_clip"],
                           "crop_motion_px_per_frame": roi["crop_motion_px_per_frame"],
                           "scale": roi["scale"]}
            continue
        roi["shot_id"] = sid
        roi["start_frame"] = int(shot["start_frame"])
        roi["key"] = geometry_key({}, {"shot": sid, "w": W, "h": H,
                                       "out": [out_w, out_h], **{
                                           k: roi[k] for k in
                                           ("crop_w", "crop_h")},
                                       "cfg": cfg["roi"]})
        (roi_dir / f"{sid}_roi.json").write_text(json.dumps(roi, indent=2) + "\n")

        # Warp the streams the generator needs, all with the SAME transform.
        srcs = {"source": (work, False, "lanczos"),
                "mask": (mask_video, True, "nearest")}
        depth_full = P.depth / "full_depth.mkv"
        if depth_full.exists():
            srcs["depth"] = (depth_full, False, "lanczos")
        for name, (src, gray, interp) in srcs.items():
            dst = roi_dir / f"{sid}_{name}_roi.mkv"
            if dst.exists() and not args.force:
                continue
            warp_video(src, dst, roi, fps, log, gray=gray, interp=interp)
            log.info("  %-8s -> %s", name, dst.name)
        made += 1

        shot["roi"] = {"path": rel(roi_dir / f"{sid}_roi.json"), "key": roi["key"],
                       "scale": roi["scale"],
                       "subject_height_fraction": roi["subject_height_fraction"],
                       "crop_motion_px_per_frame": roi["crop_motion_px_per_frame"]}

    save_manifest(man)
    log.info("=" * 62)
    log.info("ROI planned for %d shot(s) -> %s", made, rel(roi_dir))
    log.info("Next: run_chunks.py --roi, then map back with map_roi_back.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
