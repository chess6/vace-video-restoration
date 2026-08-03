#!/usr/bin/env python
"""Phase 10 - assemble restored chunks into a master, with audio.

Handles:
  * removal of duplicate overlap frames between adjacent chunks
  * a seam chosen inside the overlap, with a short cross-dissolve
  * NO cross-dissolve across a scene cut - shots are concatenated hard
  * shots with no restored output are passed through from the normalized source,
    so the master always has the full duration
  * audio remuxed from the UNTOUCHED original
  * an explicit A/V duration and sync check on the result

    scripts/assemble.py                       # 480p master
    scripts/assemble.py --pilot               # just the pilot chunks
    scripts/assemble.py --deliver 720p        # plus a resize stage
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
    P, ffprobe_json, human_size, human_time, load_config, load_manifest,
    parse_fraction, probe_dims_fps, probe_frames, require_tools, run,
    setup_logging, slice_frames,
)


def read_frames(path: Path) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out) if out else np.zeros((0, 0, 0, 3), np.uint8)


class FrameWriter:
    """Streaming ffv1 writer, so a long shot never has to fit in RAM."""

    def __init__(self, path: Path, w: int, h: int, fps: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.count = path, 0
        self.ff = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
             "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p", "-an", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frames: np.ndarray) -> None:
        for f in frames:
            self.ff.stdin.write(np.ascontiguousarray(f).tobytes())
        self.count += len(frames)

    def close(self) -> int:
        self.ff.stdin.close()
        self.ff.wait()
        if self.ff.returncode != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")
        if self.count == 0:
            raise ValueError(f"refusing to write an empty video to {self.path}")
        return self.count


def stitch_shot(chunks: list[dict], fps: int, out_path: Path, log) -> int:
    """Concatenate one shot's chunks, dropping duplicated overlap frames.

    Streams to disk holding at most two chunks in memory. A shot can be the whole
    video when there are no scene cuts (28 800 frames at 832x480 would be ~34 GB
    as raw RGB), so buffering the shot is not an option on a 15 GiB machine.

    The seam sits inside the overlap with a short linear cross-dissolve. Both
    chunks genuinely contain those frames, so this blends two renderings of the
    same content rather than inventing a transition.
    """
    ordered = sorted(chunks, key=lambda c: c["start_frame"])

    def load(c: dict) -> np.ndarray:
        p = P.root / c["output_path"]
        if not p.exists():
            raise FileNotFoundError(f"{c['chunk_id']}: missing output {p}")
        f = read_frames(p)
        if len(f) != c["n_frames"]:
            raise RuntimeError(f"{c['chunk_id']}: {len(f)} frames, expected "
                               f"{c['n_frames']}")
        return f

    pending = load(ordered[0])
    H, W = pending.shape[1], pending.shape[2]
    writer = FrameWriter(out_path, W, H, fps)

    for c in ordered[1:]:
        cur = load(c)
        ov = int(c["overlap_prev"])
        if ov <= 0:
            writer.write(pending)
            pending = cur
            continue
        if ov > len(pending):
            raise RuntimeError(f"{c['chunk_id']}: overlap {ov} exceeds the "
                               f"{len(pending)} frames available from the previous chunk")
        blend = min(ov, 8)
        keep = len(pending) - ov
        writer.write(pending[:keep])
        a = pending[keep:keep + blend].astype(np.float32)
        b = cur[:blend].astype(np.float32)
        w = np.linspace(0.0, 1.0, blend, dtype=np.float32)[:, None, None, None]
        writer.write((a * (1 - w) + b * w).astype(np.uint8))
        pending = cur[blend:]
        log.debug("seam: overlap %d, dissolve %d, written %d", ov, blend, writer.count)

    writer.write(pending)
    return writer.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--deliver", choices=["720p", "1080p"], default=None,
                    help="Additional Lanczos resize stage from the 480p master")
    ap.add_argument("--interpolate-fps", type=float, default=None,
                    help="Optional frame interpolation target (e.g. 30)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("assemble", args.verbose)
    require_tools("ffmpeg", "ffprobe")
    cfg = load_config(args.config)
    man = load_manifest()
    fps = man["normalized"]["fps"]
    out_fps = int(cfg["output"]["fps"])

    chunks = [c for c in man["chunks"] if c["status"] == "done"]
    if args.pilot:
        chunks = [c for c in chunks if c.get("is_pilot")]
    if not chunks:
        log.error("No completed chunks to assemble. Run scripts/run_chunks.py first.")
        return 1
    log.info("Assembling %d restored chunk(s)", len(chunks))

    by_shot: dict[str, list[dict]] = {}
    for c in chunks:
        by_shot.setdefault(c["shot_id"], []).append(c)

    tmp = P.intermediate / "_assembly"
    tmp.mkdir(parents=True, exist_ok=True)
    segments: list[tuple[int, Path]] = []      # (start_frame, path)

    for shot_id, cs in sorted(by_shot.items(), key=lambda kv: min(c["start_frame"] for c in kv[1])):
        seg = tmp / f"{shot_id}_stitched.mkv"
        n_written = stitch_shot(cs, fps, seg, log)
        start = min(c["start_frame"] for c in cs)
        segments.append((start, seg))
        log.info("shot %s: %d chunk(s) -> %d frames (%s)", shot_id, len(cs),
                 n_written, human_time(n_written / fps))

    # ---- fill unrestored gaps from the normalized source ---------------------
    if not args.pilot:
        work = P.root / man["normalized"]["work_path"]
        total = man["normalized"]["total_frames"]
        covered: list[tuple[int, int]] = []
        for start, seg in segments:
            covered.append((start, start + probe_frames(seg)))
        covered.sort()
        gaps: list[tuple[int, int]] = []
        cursor = 0
        for a, b in covered:
            if a > cursor:
                gaps.append((cursor, a))
            cursor = max(cursor, b)
        if cursor < total:
            gaps.append((cursor, total))
        for a, b in gaps:
            if b - a <= 0:
                continue
            g = tmp / f"gap_{a:07d}.mkv"
            slice_frames(work, g, a, b, fps, log, lossless=True)
            segments.append((a, g))
            log.warning("Gap %d-%d (%s) had no restored output; passing the "
                        "original through so duration is preserved.",
                        a, b, human_time((b - a) / fps))

    segments.sort(key=lambda s: s[0])

    # ---- concat (hard cuts between shots: no dissolve across a scene cut) ----
    listf = tmp / "concat.txt"
    listf.write_text("".join(f"file '{p.resolve()}'\n" for _, p in segments))
    silent = tmp / "master_silent.mkv"
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(listf), "-c", "copy", str(silent)], log)
    n_master = probe_frames(silent)
    log.info("Concatenated %d segment(s) -> %d frames (%s)", len(segments),
             n_master, human_time(n_master / fps))

    # ---- audio from the UNTOUCHED original -----------------------------------
    P.final.mkdir(parents=True, exist_ok=True)
    P.restored_480p.mkdir(parents=True, exist_ok=True)
    name = "pilot_master" if args.pilot else "restored_master_480p"
    master = args.out or (P.final / f"{name}.mp4")

    src = Path(man["source"]["path"])
    has_audio = man["source"]["has_audio"] and not args.no_audio and src.exists()

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(silent)]
    if has_audio:
        if args.pilot and man.get("pilot"):
            # take the matching slice of the original audio
            cmd += ["-ss", str(man["pilot"]["start_sec"]),
                    "-t", str(man["pilot"]["duration_sec"]), "-i", str(src)]
        else:
            cmd += ["-i", str(src)]
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k",
                "-shortest"]
    else:
        cmd += ["-map", "0:v:0", "-an"]
    cmd += ["-c:v", "libx264", "-crf", str(cfg["output"]["master_crf"]),
            "-preset", cfg["output"]["master_preset"], "-pix_fmt", "yuv420p",
            "-r", str(out_fps), str(master)]
    run(cmd, log)
    log.info("Master: %s (%s)", master, human_size(master.stat().st_size))

    # ---- verify A/V sync ------------------------------------------------------
    pj = ffprobe_json(master)
    vs = next(s for s in pj["streams"] if s["codec_type"] == "video")
    aud = [s for s in pj["streams"] if s["codec_type"] == "audio"]
    vdur = float(vs.get("duration") or pj["format"]["duration"])
    checks = {"video_duration_sec": round(vdur, 3),
              "video_frames": probe_frames(master),
              "video_fps": round(parse_fraction(vs.get("avg_frame_rate")), 4)}
    ok = True
    if aud:
        adur = float(aud[0].get("duration") or pj["format"]["duration"])
        drift = abs(vdur - adur)
        checks.update(audio_duration_sec=round(adur, 3), av_drift_sec=round(drift, 3))
        if drift > 0.15:
            log.error("A/V drift %.3fs exceeds 0.15s tolerance", drift)
            ok = False
        else:
            log.info("A/V sync OK: video %.3fs, audio %.3fs, drift %.3fs",
                     vdur, adur, drift)
    else:
        log.info("No audio track in the master (source had none, or --no-audio).")

    expected = (man["pilot"]["duration_sec"] if args.pilot and man.get("pilot")
                else man["normalized"]["duration_sec"])
    checks["expected_duration_sec"] = round(expected, 3)
    if abs(vdur - expected) > 0.25:
        log.error("Duration %.3fs differs from the expected %.3fs by more than "
                  "0.25s - frames were lost or duplicated.", vdur, expected)
        ok = False
    else:
        log.info("Duration preserved: %.3fs vs expected %.3fs", vdur, expected)

    # ---- optional delivery resizes -------------------------------------------
    deliverables = {"master_480p": str(master.relative_to(P.root))}
    if args.deliver:
        h = 720 if args.deliver == "720p" else 1080
        w, mh, _ = probe_dims_fps(master)
        tw = int(round(w * h / mh / 2) * 2)
        dst = P.final / f"{name}_{args.deliver}_lanczos.mp4"
        acopy = ["-c:a", "copy"] if aud else ["-an"]
        run(["ffmpeg", "-y", "-v", "error", "-i", str(master),
             "-vf", f"scale={tw}:{h}:flags=lanczos", *acopy,
             "-c:v", "libx264", "-crf", "16", "-preset", "slow",
             "-pix_fmt", "yuv420p", str(dst)], log)
        deliverables[f"lanczos_{args.deliver}"] = str(dst.relative_to(P.root))
        log.info("Lanczos %s: %s", args.deliver, dst.name)
        log.info("Compare this against any neural upscaler before choosing one "
                 "(scripts/compare_upscalers.py).")

    if args.interpolate_fps:
        dst = P.final / f"{name}_{int(args.interpolate_fps)}fps.mp4"
        acopy = ["-c:a", "copy"] if aud else ["-an"]
        run(["ffmpeg", "-y", "-v", "error", "-i", str(master),
             "-vf", f"minterpolate=fps={args.interpolate_fps}:mi_mode=mci:"
                    "mc_mode=aobmc:vsbmc=1", *acopy,
             "-c:v", "libx264", "-crf", "16", "-preset", "slow",
             "-pix_fmt", "yuv420p", str(dst)], log)
        deliverables[f"interpolated_{int(args.interpolate_fps)}fps"] = \
            str(dst.relative_to(P.root))
        log.info("Interpolated to %.3g fps: %s", args.interpolate_fps, dst.name)

    P.reports.mkdir(parents=True, exist_ok=True)
    (P.reports / ("assembly_pilot.json" if args.pilot else "assembly.json")).write_text(
        json.dumps({"checks": checks, "passed": ok, "segments": len(segments),
                    "chunks": len(chunks), "deliverables": deliverables}, indent=2))

    log.info("=" * 62)
    log.info("Assembly %s", "OK" if ok else "FAILED VERIFICATION")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
