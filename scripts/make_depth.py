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
    P, check_geometry, geometry_key, human_time, load_config, load_manifest,
    probe_dims_fps, probe_frames, require_cuda, require_tools, run, save_manifest,
    setup_logging, slice_frames, vram_snapshot,
)

MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
# Exact revision, so the control signal cannot change under the same code.
MODEL_REV = "7581137eff8d4e94f6e796d3baea0e9fa79b22d2"


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


def calibrate_ranges(work: Path, shots: list[tuple[int, int]], infer, total_frames: int,
                     samples: int, log) -> tuple[np.ndarray, np.ndarray]:
    """Fixed normalisation range per shot, as a (lo, hi) value for every frame.

    Depth Anything returns relative inverse depth on an arbitrary scale, so the
    raw value for one physical distance drifts from frame to frame. Normalising
    each frame by its own min/max therefore re-maps the same physical depth to a
    different grey level whenever anything enters or leaves the frame, and that
    flicker goes straight into the control signal the sampler follows.

    A range fixed per shot removes that: within a shot the mapping is constant,
    so equal depths stay equally bright. It is calibrated per shot rather than
    over the whole video because a cut can change the depth range completely,
    and robust percentiles rather than min/max so one speck cannot set the scale.
    """
    import cv2
    lo_pf = np.zeros(total_frames, dtype=np.float32)
    hi_pf = np.ones(total_frames, dtype=np.float32)

    wanted: dict[int, int] = {}          # frame index -> shot index
    for si, (a, b) in enumerate(shots):
        n = min(samples, b - a)
        for f in np.linspace(a, b - 1, n):
            wanted[int(round(f))] = si

    cap = cv2.VideoCapture(str(work))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open working stream {work}")
    per_shot: dict[int, list[np.ndarray]] = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            d = infer([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)])[0]
            per_shot.setdefault(wanted[idx], []).append(d)
        idx += 1
    cap.release()

    for si, (a, b) in enumerate(shots):
        ds = per_shot.get(si)
        if not ds:
            continue
        allv = np.concatenate([d.ravel() for d in ds])
        lo, hi = np.percentile(allv, 1.0), np.percentile(allv, 99.0)
        if hi - lo < 1e-6:
            lo, hi = float(allv.min()), float(allv.max()) or (lo + 1.0)
        lo_pf[a:b], hi_pf[a:b] = lo, hi
        log.info("depth range for frames %d-%d: [%.3f, %.3f] from %d sample(s)",
                 a, b, lo, hi, len(ds))
    return lo_pf, hi_pf


def build_full_depth(work: Path, out: Path, width: int, height: int, fps: int,
                     total_frames: int, batch: int, log,
                     shots: list[tuple[int, int]] | None = None,
                     calib_samples: int = 24) -> None:
    """Single sequential pass. Writes a grayscale-in-RGB depth video."""
    import cv2
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    dev = require_cuda(log)
    log.info("Loading %s", MODEL_ID)
    proc = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REV)
    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_ID, revision=MODEL_REV, dtype=torch.float16).to(dev).eval()

    def infer(frames: list[np.ndarray]) -> np.ndarray:
        with torch.inference_mode():
            inputs = proc(images=frames, return_tensors="pt").to(dev)
            pred = model(**inputs).predicted_depth
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1).float(), size=(height, width),
                mode="bicubic", align_corners=False).squeeze(1)
        return pred.cpu().numpy()

    shots = shots or [(0, total_frames)]
    log.info("Calibrating a fixed depth range per shot (%d shot(s), <=%d samples "
             "each) so equal depths keep equal brightness", len(shots), calib_samples)
    lo_pf, hi_pf = calibrate_ranges(work, shots, infer, total_frames,
                                    calib_samples, log)

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
        for m in infer(frames):
            # Near = bright, far = dark, as the VACE / ControlNet depth
            # conditioning expects. The range is the SHOT's, fixed by
            # calibrate_ranges, not this frame's own min/max: a per-frame range
            # would make the same physical depth flicker between frames.
            lo, hi = float(lo_pf[written]), float(hi_pf[written])
            g = (np.zeros_like(m, dtype=np.uint8) if hi - lo < 1e-6 else
                 (np.clip((m - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8))
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
    ap.add_argument("--calib-samples", type=int, default=24,
                    help="Frames sampled per shot to fix that shot's depth range")
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

    # Refuse to reuse depth built for a different geometry or control profile.
    # Two separate keys: the plain geometry one every stage shares, and a
    # depth-specific one that also covers the control profile, so switching
    # depth -> canny invalidates the depth videos without invalidating anything
    # else in the run.
    if not args.force:
        check_geometry(man, log, stage="make_depth")
        check_geometry(man, log, extra={"control": args.mode}, stage="make_depth",
                       key_name="depth_key")
    else:
        man["geometry_key"] = geometry_key(man)
        man["depth_key"] = geometry_key(man, {"control": args.mode})

    if args.force or not full.exists() or probe_frames(full) != total:
        log.info("Building full-length %s control (%d frames @ %dx%d)",
                 args.mode, total, width, height)
        if args.mode == "depth":
            shot_ranges = [(int(s["start_frame"]), int(s["end_frame"]))
                           for s in man.get("shots", [])] or [(0, total)]
            build_full_depth(work, full, width, height, fps, total, args.batch, log,
                             shots=shot_ranges, calib_samples=args.calib_samples)
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
