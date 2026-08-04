#!/usr/bin/env python
"""Phase 4.0 - cut one short random clip out of each source video.

Pre-pipeline sampler. Runs BEFORE preprocess_source.py, so unlike
extract_pilot.py it needs no chunk manifest: it only reads the originals with
ffprobe/ffmpeg and writes copies. The originals are opened read-only and are
never modified, moved or re-encoded in place.

Each clip keeps its source's resolution, frame rate and audio, so a clip is a
drop-in `--source` for inspect_source.py / preprocess_source.py and every stage
downstream of them.

The start offset is drawn uniformly at random from the middle of each file
(the first and last `--margin` fraction are skipped, so clips do not land on
titles or credits), then snapped to a frame boundary. The draw is seeded, so
re-running reproduces the same clips; use a different --seed for a new sample.

    scripts/sample_clips.py [--seconds 5] [--seed 20260803] [--force]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, VIDEO_EXTS, ffprobe_json, human_size, human_time, parse_fraction,
    probe_frames, require_tools, run, setup_logging,
)


def probe_basics(path: Path) -> dict:
    """Duration, fps, dimensions and audio presence - structural facts only."""
    d = ffprobe_json(path)
    v = next((s for s in d["streams"] if s.get("codec_type") == "video"), None)
    if v is None:
        raise RuntimeError(f"No video stream in {path.name}")
    fps = parse_fraction(v.get("avg_frame_rate") or v.get("r_frame_rate"))
    if fps <= 0:
        raise RuntimeError(f"Could not determine frame rate of {path.name}")
    dur = float(d["format"].get("duration") or 0.0)
    if dur <= 0:
        raise RuntimeError(f"Could not determine duration of {path.name}")
    return {
        "duration": dur,
        "fps": fps,
        "width": int(v["width"]),
        "height": int(v["height"]),
        "has_audio": any(s.get("codec_type") == "audio" for s in d["streams"]),
    }


def cut(src: Path, dst: Path, start_sec: float, dur_sec: float, has_audio: bool,
        log) -> None:
    """Re-encode a [start, start+dur) window. `-ss` before `-i` seeks accurately
    when the output is re-encoded, and re-encoding (rather than a stream copy)
    is what makes the cut land on the requested frame instead of the nearest
    preceding keyframe. Quality is kept high because this clip is the input to
    the restoration, not a deliverable.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    audio = (["-c:a", "aac", "-b:a", "192k"] if has_audio else ["-an"])
    run(["ffmpeg", "-y", "-v", "error",
         "-ss", f"{start_sec:.6f}", "-i", str(src), "-t", f"{dur_sec:.6f}",
         "-map", "0:v:0", *(["-map", "0:a:0"] if has_audio else []),
         "-c:v", "libx264", "-crf", "12", "-preset", "slow",
         "-pix_fmt", "yuv420p", *audio, str(dst)], log)


def verify(dst: Path, want_frames: int, info: dict, dur_sec: float, log) -> list[str]:
    """Exit code 0 proves ffmpeg ran, not that the clip is right (rule 4)."""
    problems = []
    if not dst.exists() or dst.stat().st_size == 0:
        return [f"{dst.name}: missing or empty"]
    got = probe_frames(dst)
    if abs(got - want_frames) > 2:
        problems.append(f"{dst.name}: {got} frames, expected ~{want_frames}")
    out = probe_basics(dst)
    if (out["width"], out["height"]) != (info["width"], info["height"]):
        problems.append(
            f"{dst.name}: {out['width']}x{out['height']}, "
            f"expected {info['width']}x{info['height']}")
    if abs(out["fps"] - info["fps"]) > 0.02:
        problems.append(f"{dst.name}: {out['fps']:.3f} fps, expected {info['fps']:.3f}")
    if abs(out["duration"] - dur_sec) > 0.2:
        problems.append(
            f"{dst.name}: {out['duration']:.3f} s, expected {dur_sec:.3f} s")
    if info["has_audio"] and not out["has_audio"]:
        problems.append(f"{dst.name}: audio stream lost")
    for p in problems:
        log.error("VERIFY FAIL %s", p)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=5.0, help="Clip length")
    ap.add_argument("--seed", type=int, default=20260803,
                    help="Seed for the start-offset draw; same seed = same clips")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="Fraction of each file skipped at the head and tail")
    ap.add_argument("--sources", type=Path, nargs="*", default=None,
                    help="Explicit source files. Default: every video in inputs/source/")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output directory (default intermediate/clips)")
    ap.add_argument("--force", action="store_true", help="Re-cut clips that already exist")
    args = ap.parse_args()

    log = setup_logging("sample_clips")
    require_tools("ffmpeg", "ffprobe")

    out_dir = args.out or (P.intermediate / "clips")
    srcs = args.sources or sorted(
        p for p in P.source.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not srcs:
        log.error("No source videos found in %s", P.source)
        return 1
    log.info("Sampling %.1f s from %d source video(s), seed=%d",
             args.seconds, len(srcs), args.seed)

    rng = random.Random(args.seed)
    entries, problems = [], []

    for src in srcs:
        info = probe_basics(src)
        fps = info["fps"]
        n_frames = max(1, round(args.seconds * fps))
        dur = n_frames / fps
        # Draw inside [margin, 1-margin] of the file, leaving room for the clip.
        lo = info["duration"] * args.margin
        hi = info["duration"] * (1.0 - args.margin) - dur
        if hi <= lo:                       # short file: fall back to the whole span
            lo, hi = 0.0, max(0.0, info["duration"] - dur)
        # Draw first regardless of --force, so one source being skipped does not
        # shift the offsets drawn for the others.
        start = round(rng.uniform(lo, hi) * fps) / fps
        dst = out_dir / f"{src.stem}_clip{int(round(args.seconds))}s.mp4"

        if dst.exists() and not args.force:
            log.info("%-28s exists, skipping (use --force to re-cut)", dst.name)
        else:
            log.info("%-14s %dx%d @ %.3f fps, %s -> cut %d frames at %s",
                     src.name, info["width"], info["height"], fps,
                     human_time(info["duration"]), n_frames, human_time(start))
            cut(src, dst, start, dur, info["has_audio"], log)
            problems += verify(dst, n_frames, info, dur, log)

        if dst.exists():
            entries.append({
                "source": src.name,
                "clip": str(dst.relative_to(P.root)),
                "start_sec": round(start, 6),
                "start_frame": round(start * fps),
                "frames": n_frames,
                "duration_sec": round(dur, 6),
                "fps": fps,
                "width": info["width"],
                "height": info["height"],
                "has_audio": info["has_audio"],
                "size": dst.stat().st_size,
            })
            log.info("%-28s %s  start=%s  %d frames",
                     dst.name, human_size(dst.stat().st_size),
                     human_time(entries[-1]["start_sec"]), n_frames)

    # Lands under intermediate/, which .gitignore denies wholesale, because it
    # carries the user's filenames and durations (rule 2a).
    manifest = out_dir / "clips.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(
        {"seed": args.seed, "seconds": args.seconds, "margin": args.margin,
         "clips": entries}, indent=2) + "\n")
    log.info("Wrote %s", manifest)

    if problems:
        log.error("%d verification problem(s); clips are NOT trustworthy", len(problems))
        return 1
    log.info("OK - %d clip(s) verified in %s", len(entries), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
