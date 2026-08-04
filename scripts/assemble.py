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


def plan_stitch(spans: list[tuple[int, int]], blend_max: int = 8) -> list[tuple]:
    """Decide, from absolute frame spans alone, what to emit and in what order.

    `spans` is [(start_frame, n_frames), ...] for the chunks of one shot, in any
    order. Returns a list of operations, each covering a contiguous run of the
    output timeline and, together, covering [min(start), max(end)) exactly once:

        ("copy",   i, off, n)                 n frames from chunk i at offset off
        ("blend",  i, off_i, j, off_j, n)     cross-dissolve chunk i into chunk j
        ("source", a, n)                      n frames from the normalized source
        ("skip",   i)                         chunk i adds nothing; not emitted

    Kept free of I/O so the arithmetic can be tested exhaustively over shot
    lengths and missing-chunk patterns without generating any video.
    """
    order = sorted(range(len(spans)), key=lambda i: (spans[i][0], spans[i][1]))
    ops: list[tuple] = []
    first = order[0]
    pend_i, pend_start, pend_len = first, spans[first][0], spans[first][1]
    pend_off = 0            # how much of chunk pend_i has already been consumed

    for i in order[1:]:
        cur_start, cur_len = spans[i]
        pend_end = pend_start + pend_len
        cur_end = cur_start + cur_len

        if cur_end <= pend_end:
            ops.append(("skip", i))
            continue

        if cur_start >= pend_end:
            ops.append(("copy", pend_i, pend_off, pend_len))
            if cur_start > pend_end:
                ops.append(("source", pend_end, cur_start - pend_end))
            pend_i, pend_start, pend_len, pend_off = i, cur_start, cur_len, 0
            continue

        ov_start = max(cur_start, pend_start)
        ov = pend_end - ov_start
        blend = min(ov, blend_max)
        keep = ov_start - pend_start
        off = ov_start - cur_start
        if keep:
            ops.append(("copy", pend_i, pend_off, keep))
        ops.append(("blend", pend_i, pend_off + keep, i, off, blend))
        pend_i = i
        pend_off = off + blend
        pend_start = ov_start + blend
        pend_len = cur_len - pend_off

    ops.append(("copy", pend_i, pend_off, pend_len))
    return ops


def op_frames(op: tuple) -> int:
    """How many output frames an operation contributes."""
    return {"copy": lambda o: o[3], "blend": lambda o: o[5],
            "source": lambda o: o[2], "skip": lambda o: 0}[op[0]](op)


def stitch_shot(chunks: list[dict], fps: int, out_path: Path, log,
                work: Path | None = None) -> tuple[int, int]:
    """Concatenate one shot's chunks onto the shot's own absolute frame timeline.

    Returns (span_start, n_written), where the result covers exactly
    [span_start, span_start + n_written) in working-stream frame indices.

    Everything is computed from absolute frame positions rather than from the
    stored `overlap_prev`. Two things make that necessary:

      * The last window of a shot is snapped BACKWARDS to end on the shot end, so
        its overlap with the previous window can be far larger than the nominal
        one - larger, in fact, than what is left of the previous chunk once its
        own seam has been consumed. Comparing a full-chunk overlap against a
        partly-consumed buffer both loses frames and, past a point, aborts.
      * Chunks may be missing (a failed generation, or --only on a subset). Those
        are real holes on the timeline, not a reason to slide later footage
        earlier. Each hole is filled from the normalized source so that every
        frame keeps its original position.

    Streams to disk holding at most two chunks in memory: a shot can be the whole
    video when there are no scene cuts (28 800 frames at 832x480 would be ~34 GB
    as raw RGB), so buffering the shot is not an option on a 15 GiB machine.

    The seam sits inside the overlap with a short linear cross-dissolve. Both
    chunks genuinely contain those frames, so this blends two renderings of the
    same content rather than inventing a transition.
    """
    ordered = sorted(chunks, key=lambda c: (c["start_frame"], c["end_frame"]))

    def load(c: dict) -> np.ndarray:
        p = P.root / c["output_path"]
        if not p.exists():
            raise FileNotFoundError(f"{c['chunk_id']}: missing output {p}")
        f = read_frames(p)
        if len(f) != c["n_frames"]:
            raise RuntimeError(f"{c['chunk_id']}: {len(f)} frames, expected "
                               f"{c['n_frames']}")
        return f

    def source_range(a: int, b: int) -> np.ndarray:
        """Unrestored frames [a, b) from the normalized working stream."""
        if work is None:
            raise RuntimeError(
                f"frames {a}-{b} have no restored chunk and no source stream was "
                f"given to fill them from")
        tmpf = out_path.parent / f"_hole_{a:07d}_{b:07d}.mkv"
        slice_frames(work, tmpf, a, b, fps, log, lossless=True)
        fr = read_frames(tmpf)
        tmpf.unlink(missing_ok=True)
        if len(fr) != b - a:
            raise RuntimeError(f"source fill {a}-{b}: got {len(fr)} frames")
        return fr

    spans = [(c["start_frame"], c["n_frames"]) for c in ordered]
    ops = plan_stitch(spans)
    span_start = min(s for s, _ in spans)
    span_end = max(s + n for s, n in spans)

    # At most two decoded chunks are held at once: the plan only ever refers to
    # the chunk being consumed and the one being blended into it.
    cache: dict[int, np.ndarray] = {}

    def frames_of(i: int) -> np.ndarray:
        if i not in cache:
            cache.clear()
            cache[i] = load(ordered[i])
        return cache[i]

    def pair(i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        a = cache.get(i)
        if a is None:
            a = load(ordered[i])
        b = load(ordered[j])
        cache.clear()
        cache[j] = b
        return a, b

    writer = None
    for op in ops:
        if op[0] == "skip":
            log.warning("%s is already covered by earlier chunks; skipped",
                        ordered[op[1]]["chunk_id"])
            continue
        if op[0] == "copy":
            _, i, off, n = op
            block = frames_of(i)[off:off + n]
        elif op[0] == "blend":
            _, i, off_i, j, off_j, n = op
            fa, fb = pair(i, j)
            a = fa[off_i:off_i + n].astype(np.float32)
            b = fb[off_j:off_j + n].astype(np.float32)
            w = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None, None, None]
            block = (a * (1 - w) + b * w).astype(np.uint8)
            log.debug("seam: dissolve %d frame(s) between %s and %s", n,
                      ordered[i]["chunk_id"], ordered[j]["chunk_id"])
        else:                                   # "source"
            _, a0, n = op
            log.warning("Frames %d-%d have no restored chunk; filling them from "
                        "the original so later frames keep their position.",
                        a0, a0 + n)
            block = source_range(a0, a0 + n)
        if writer is None:
            writer = FrameWriter(out_path, block.shape[2], block.shape[1], fps)
        writer.write(block)

    n = writer.close()
    expect = span_end - span_start
    if n != expect:
        raise RuntimeError(f"stitched {n} frames but the span {span_start}-"
                           f"{span_end} needs {expect}")
    return span_start, n


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
    work = P.root / man["normalized"]["work_path"]
    segments: list[tuple[int, int, Path]] = []      # (start_frame, n_frames, path)

    for shot_id, cs in sorted(by_shot.items(), key=lambda kv: min(c["start_frame"] for c in kv[1])):
        seg = tmp / f"{shot_id}_stitched.mkv"
        start, n_written = stitch_shot(cs, fps, seg, log, work=work)
        segments.append((start, n_written, seg))
        log.info("shot %s: %d chunk(s) -> frames %d-%d (%s)", shot_id, len(cs),
                 start, start + n_written, human_time(n_written / fps))

    # ---- fill unrestored gaps from the normalized source ---------------------
    # Holes INSIDE a shot were already filled in place by stitch_shot. What is
    # left is everything outside the restored spans: shots that were never run,
    # sub-legal shots that produced no window at all, and the head and tail.
    # For the pilot the timeline is only the pilot window; for a full assembly it
    # is the whole working stream.
    if args.pilot and man.get("pilot"):
        t_start, t_end = int(man["pilot"]["start_frame"]), int(man["pilot"]["end_frame"])
    else:
        t_start, t_end = 0, int(man["normalized"]["total_frames"])

    covered = sorted((s, s + n) for s, n, _ in segments)
    gaps: list[tuple[int, int]] = []
    cursor = t_start
    for a, b in covered:
        if a > cursor:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < t_end:
        gaps.append((cursor, t_end))
    for a, b in gaps:
        a, b = max(a, t_start), min(b, t_end)
        if b - a <= 0:
            continue
        g = tmp / f"gap_{a:07d}.mkv"
        slice_frames(work, g, a, b, fps, log, lossless=True)
        segments.append((a, b - a, g))
        log.warning("Gap %d-%d (%s) had no restored output; passing the "
                    "original through so duration is preserved.",
                    a, b, human_time((b - a) / fps))

    segments.sort(key=lambda s: s[0])

    # ---- concat (hard cuts between shots: no dissolve across a scene cut) ----
    listf = tmp / "concat.txt"
    listf.write_text("".join(f"file '{p.resolve()}'\n" for _, _, p in segments))
    silent = tmp / "master_silent.mkv"
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(listf), "-c", "copy", str(silent)], log)
    n_master = probe_frames(silent)
    span_start = segments[0][0]
    log.info("Concatenated %d segment(s) -> %d frames covering %d-%d (%s)",
             len(segments), n_master, span_start, span_start + n_master,
             human_time(n_master / fps))

    # ---- crop to the exact requested window ----------------------------------
    # A pilot is an exact frame interval, but the chunks covering it are whole
    # 4n+1 windows that spill past both ends. The audio below is cut at the exact
    # pilot timestamp, so without this the picture and the sound would describe
    # different moments.
    if (span_start, n_master) != (t_start, t_end - t_start):
        lo, hi = t_start - span_start, t_end - span_start
        if lo < 0 or hi > n_master:
            raise RuntimeError(f"requested window {t_start}-{t_end} is not inside "
                               f"the assembled span {span_start}-{span_start + n_master}")
        cropped = tmp / "master_silent_cropped.mkv"
        run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-vf",
             f"trim=start_frame={lo}:end_frame={hi},setpts=PTS-STARTPTS",
             "-c:v", "ffv1", "-level", "3", "-an", str(cropped)], log)
        n_cropped = probe_frames(cropped)
        if n_cropped != t_end - t_start:
            raise RuntimeError(f"crop produced {n_cropped} frames, expected "
                               f"{t_end - t_start}")
        log.info("Cropped to the requested window: frames %d-%d (%d frames)",
                 t_start, t_end, n_cropped)
        silent, n_master, span_start = cropped, n_cropped, t_start

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
