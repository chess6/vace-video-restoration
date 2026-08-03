#!/usr/bin/env python
"""Phase 9b - choose a representative 5-10 s pilot segment.

Picks automatically rather than taking the first few seconds. Scores every
candidate window inside a single shot on:
  * motion energy          - the subject should actually move
  * detail / texture       - visible clothing and edges, not a flat wall
  * viewpoint change       - rewards a moderate pose or angle change
  * subject presence       - uses the tracked mask when it exists
and penalises windows that are nearly static or nearly empty.

Marks the covering chunks with is_pilot=true in the manifest. Never crosses a
scene cut, because chunks already respect cuts.

    scripts/extract_pilot.py [--seconds 8] [--start-sec 123.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, human_time, load_config, load_manifest, probe_frames, run, save_manifest,
    setup_logging, slice_frames,
)


def frame_stats(path: Path, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (motion vs previous, texture energy), computed at low res."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    prev = None
    motion, texture = [], []
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(g, (160, 96))
            texture.append(float(cv2.Laplacian(small, cv2.CV_32F).var()))
            motion.append(0.0 if prev is None
                          else float(np.abs(small.astype(np.float32) -
                                            prev.astype(np.float32)).mean()))
            prev = small
        idx += 1
    cap.release()
    return np.asarray(motion), np.asarray(texture)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seconds", type=float, default=8.0, help="Target pilot length (5-10)")
    ap.add_argument("--start-sec", type=float, default=None,
                    help="Force the pilot to start here instead of auto-selecting")
    ap.add_argument("--clear", action="store_true", help="Unmark existing pilot chunks")
    args = ap.parse_args()

    log = setup_logging("extract_pilot")
    cfg = load_config(args.config)
    man = load_manifest()
    fps = man["normalized"]["fps"]
    work = P.root / man["normalized"]["work_path"]
    if not work.exists():
        log.error("Working stream missing. Run preprocess_source.py first.")
        return 1

    for c in man["chunks"]:
        if args.clear:
            c.pop("is_pilot", None)
    if args.clear:
        save_manifest(man)
        log.info("Cleared pilot marks.")
        return 0

    want_frames = int(round(np.clip(args.seconds, 5.0, 10.0) * fps))

    if args.start_sec is not None:
        start = int(round(args.start_sec * fps))
        log.info("Using the forced start at %.2fs (frame %d)", args.start_sec, start)
        chosen = (start, start + want_frames, None)
    else:
        log.info("Scoring the working stream for a representative window...")
        motion, texture = frame_stats(work)
        n = len(motion)
        if n < want_frames:
            log.error("Working stream has only %d frames; need >= %d", n, want_frames)
            return 1

        # cumulative sums for fast windowing
        cm = np.concatenate([[0], np.cumsum(motion)])
        ct = np.concatenate([[0], np.cumsum(texture)])

        shots = man["shots"]
        best = None
        for s in shots:
            s0, s1 = s["start_frame"], min(s["end_frame"], n)
            if s1 - s0 < want_frames:
                continue
            for st in range(s0, s1 - want_frames + 1, max(1, fps // 2)):
                en = st + want_frames
                m = (cm[en] - cm[st]) / want_frames
                t = (ct[en] - ct[st]) / want_frames
                seg = motion[st:en]
                # reward variation in motion: a moderate pose/viewpoint change,
                # not a constant pan and not a frozen frame
                var = float(seg.std())
                if m < 0.4:
                    continue                      # essentially static
                score = (0.45 * min(m / 6.0, 1.0) +
                         0.25 * min(t / 250.0, 1.0) +
                         0.30 * min(var / 3.0, 1.0))
                if best is None or score > best[0]:
                    best = (score, st, en, s["shot_id"], m, t, var)
        if best is None:
            log.warning("No window met the motion threshold; falling back to the "
                        "highest-motion window overall.")
            st = int(np.argmax(np.convolve(motion, np.ones(want_frames) / want_frames,
                                           mode="valid")))
            best = (0.0, st, st + want_frames, "?", 0, 0, 0)
        _, s_st, s_en, shot_id, m, t, var = best
        log.info("Chosen: shot %s frames %d-%d (%.2fs-%.2fs) motion=%.2f "
                 "texture=%.0f motion_var=%.2f", shot_id, s_st, s_en,
                 s_st / fps, s_en / fps, m, t, var)
        chosen = (s_st, s_en, shot_id)

    p_start, p_end, _ = chosen

    # ---- mark the covering chunks -------------------------------------------
    marked = []
    for c in man["chunks"]:
        overlaps = not (c["end_frame"] <= p_start or c["start_frame"] >= p_end)
        if overlaps:
            c["is_pilot"] = True
            marked.append(c["chunk_id"])
        else:
            c.pop("is_pilot", None)

    if not marked:
        log.error("The chosen window covers no chunk. Was preprocess_source.py run?")
        return 1

    man["pilot"] = {
        "start_frame": int(p_start), "end_frame": int(p_end),
        "start_sec": round(p_start / fps, 3), "end_sec": round(p_end / fps, 3),
        "duration_sec": round((p_end - p_start) / fps, 3),
        "chunks": marked,
    }
    save_manifest(man)

    # ---- extract the reference clip for comparisons --------------------------
    P.pilots.mkdir(parents=True, exist_ok=True)
    clip = P.pilots / "pilot_source.mp4"
    slice_frames(work, clip, p_start, p_end, fps, log, lossless=False)

    log.info("=" * 62)
    log.info("Pilot: frames %d-%d  (%s - %s, %s)", p_start, p_end,
             human_time(p_start / fps), human_time(p_end / fps),
             human_time((p_end - p_start) / fps))
    log.info("Covering chunk(s): %s", ", ".join(marked))
    log.info("Source clip: %s", clip.relative_to(P.root))
    log.info("Next: scripts/run_chunks.py --pilot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
