#!/usr/bin/env python
"""Phase 7 - depth control videos from the normalized working stream.

Depth (not pose, not edges) is the baseline structural control: the goal is to
preserve the whole figure's shape, volume and motion, not just facial landmarks.

Runs Depth Anything V2 (Large) as a SEPARATE stage from VACE so the two never
compete for VRAM. ComfyUI can stay running; this uses its own process.

Two outputs:
  intermediate/depth/full_depth.mkv     - one pass over the whole working stream
  intermediate/depth/<chunk_id>_depth.mkv - frame-exact slices per chunk

Slicing uses the `trim` filter with frame indices, never time-based seeking, so
depth frames stay index-aligned with the source and mask streams.

    scripts/make_depth.py [--config ...] [--only shot0000_c000] [--force]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, human_time, load_config, load_manifest, probe_dims_fps, probe_frames,
    require_cuda, require_tools, run, save_manifest, setup_logging, slice_frames,
    vram_snapshot,
)

MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"


def build_full_canny(work: Path, out: Path, width: int, height: int, fps: int,
                     total_frames: int, lo: int, hi: int, log) -> None:
    """Alternative edge control profile. Cheap, no model, no VRAM.

    Kept available per the brief but NOT the baseline: edges alone describe the
    silhouette and creases but carry no volume, so the model has less to go on
    for a whole figure than depth gives it. Useful as a comparison, or on
    material where Depth Anything is unstable.
    """
    import cv2
    import subprocess
    cap = cv2.VideoCapture(str(work))
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p", "-an", str(out)],
        stdin=subprocess.PIPE)
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(g, lo, hi)
        ff.stdin.write(np.repeat(e[:, :, None], 3, axis=2).tobytes())
        written += 1
    cap.release()
    ff.stdin.close()
    ff.wait()
    if written != total_frames:
        raise RuntimeError(f"Canny pass produced {written} frames, expected "
                           f"{total_frames}")
    log.info("Canny edge pass: %d frames", written)


def build_full_depth(work: Path, out: Path, width: int, height: int, fps: int,
                     total_frames: int, batch: int, log) -> None:
    """Single sequential pass. Writes a grayscale-in-RGB depth video."""
    import cv2
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    dev = require_cuda(log)
    log.info("Loading %s", MODEL_ID)
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_ID, dtype=torch.float16).to(dev).eval()

    cap = cv2.VideoCapture(str(work))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open working stream {work}")

    out.parent.mkdir(parents=True, exist_ok=True)
    # Pipe raw frames into ffmpeg: avoids writing 28k PNGs to disk.
    import subprocess
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-",
         "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p", "-an", str(out)],
        stdin=subprocess.PIPE)

    written = 0
    t0 = time.time()
    buf: list[np.ndarray] = []

    def flush(frames: list[np.ndarray]) -> None:
        nonlocal written
        if not frames:
            return
        with torch.inference_mode():
            inputs = proc(images=frames, return_tensors="pt").to(dev)
            pred = model(**inputs).predicted_depth      # (B, h, w)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1).float(), size=(height, width),
                mode="bicubic", align_corners=False).squeeze(1)
        d = pred.cpu().numpy()
        for m in d:
            # Per-frame min/max normalisation. This is the convention the VACE /
            # ControlNet depth conditioning expects: near = bright, far = dark.
            lo, hi = float(m.min()), float(m.max())
            g = np.zeros_like(m, dtype=np.uint8) if hi - lo < 1e-6 else \
                ((m - lo) / (hi - lo) * 255.0).astype(np.uint8)
            ff.stdin.write(np.repeat(g[:, :, None], 3, axis=2).tobytes())
            written += 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(buf) >= batch:
            flush(buf)
            buf = []
            if written % (batch * 10) == 0:
                el = time.time() - t0
                rate = written / max(el, 1e-6)
                log.info("depth %d/%d frames (%.1f fps, eta %s)", written,
                         total_frames, rate,
                         human_time((total_frames - written) / max(rate, 1e-6)))
    flush(buf)
    cap.release()
    ff.stdin.close()
    ff.wait()

    del model
    torch.cuda.empty_cache()
    log.info("Full depth pass: %d frames in %s (peak VRAM %s)", written,
             human_time(time.time() - t0), vram_snapshot())
    if written != total_frames:
        raise RuntimeError(
            f"Depth pass produced {written} frames but the working stream has "
            f"{total_frames}. Depth and source would be misaligned; aborting.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="Limit to these chunk ids")
    ap.add_argument("--batch", type=int, default=4, help="Frames per forward pass")
    ap.add_argument("--mode", choices=["depth", "canny"], default="depth",
                    help="Structural control profile. 'depth' is the baseline; "
                         "'canny' is a cheap edge alternative kept for comparison. "
                         "Pose control would need a DWPose/OpenPose model, which "
                         "is deliberately not installed.")
    ap.add_argument("--canny-low", type=int, default=80)
    ap.add_argument("--canny-high", type=int, default=160)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("make_depth", args.verbose)
    require_tools("ffmpeg", "ffprobe")
    cfg = load_config(args.config)
    man = load_manifest()

    work = P.root / man["normalized"]["work_path"]
    width = man["normalized"]["width"]
    height = man["normalized"]["height"]
    fps = man["normalized"]["fps"]
    total = man["normalized"]["total_frames"]

    if not work.exists():
        log.error("Working stream missing: %s. Run preprocess_source.py first.", work)
        return 1

    P.depth.mkdir(parents=True, exist_ok=True)
    full = P.depth / ("full_depth.mkv" if args.mode == "depth" else "full_canny.mkv")

    if args.force or not full.exists() or probe_frames(full) != total:
        log.info("Building full-length %s control (%d frames @ %dx%d)",
                 args.mode, total, width, height)
        if args.mode == "depth":
            build_full_depth(work, full, width, height, fps, total, args.batch, log)
        else:
            build_full_canny(work, full, width, height, fps, total,
                             args.canny_low, args.canny_high, log)
    else:
        log.info("%s exists with %d frames, skipping (--force to rebuild)",
                 full.name, total)

    dw, dh, dfps = probe_dims_fps(full)
    dn = probe_frames(full)
    if (dw, dh) != (width, height) or dn != total or abs(dfps - fps) > 0.02:
        log.error("Full depth mismatch: %dx%d %d frames %.3f fps vs expected "
                  "%dx%d %d frames %d fps", dw, dh, dn, dfps, width, height, total, fps)
        return 1
    log.info("Full depth verified: %dx%d, %d frames, %.3f fps", dw, dh, dn, dfps)

    # ---- per-chunk slices ----------------------------------------------------
    chunks = man["chunks"]
    if args.only:
        chunks = [c for c in chunks if c["chunk_id"] in set(args.only)]
    made = skipped = 0
    for c in chunks:
        dst = P.root / c["depth_path"]
        dst = dst.with_suffix(".mkv")
        if not args.force and dst.exists() and probe_frames(dst) == c["n_frames"]:
            skipped += 1
            c["depth_path"] = str(dst.relative_to(P.root))
            continue
        slice_frames(full, dst, c["start_frame"], c["end_frame"], fps, log,
                     lossless=True, gray=True)
        n = probe_frames(dst)
        if n != c["n_frames"]:
            log.error("%s: sliced %d frames, expected %d", c["chunk_id"], n, c["n_frames"])
            return 1
        c["depth_path"] = str(dst.relative_to(P.root))
        made += 1
        if made % 25 == 0:
            log.info("sliced %d chunk depth clips", made)

    save_manifest(man)
    log.info("Depth ready: %d built, %d already present, %d chunk(s) total",
             made, skipped, len(chunks))
    log.info("NOTE: depth is the control signal INSIDE the subject mask only. The "
             "workflow composites original RGB outside the mask via "
             "ImageCompositeMasked, because WanVaceToVideo derives the preserved "
             "region from control_video*(1-mask).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
