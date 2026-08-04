#!/usr/bin/env python
"""Phase 4 - normalize the source, detect scene cuts, build the chunk manifest.

Pipeline:
  1. Normalize a WORKING COPY to constant frame rate at the original resolution
     and fps (square pixels, deinterlaced if needed). The original is untouched.
  2. Build the VACE working stream: resample to model fps (16), scale-and-pad to
     the configured WxH without distorting, multiples of 16.
  3. Detect scene cuts on the VACE stream.
  4. Split each shot into 4n+1 frame chunks with overlap; chunks never cross a cut.
  5. Write intermediate/chunk_manifest.json.

Timing is tracked in BOTH timebases: frame indices in the 16 fps working stream,
and seconds in the original source timebase. Audio is remuxed later from the
untouched original using the source-seconds values, so audio never drifts.

    scripts/preprocess_source.py [--config configs/local_1p3b.yaml] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, VIDEO_EXTS, Chunk, Shot, ffprobe_json, find_single, human_size, human_time,
    load_config, nearest_valid_length, parse_fraction, probe_frames, rel,
    require_tools, round_to_16, run, save_manifest, setup_logging,
)


# ---------------------------------------------------------------------------

def compute_fit(src_w: int, src_h: int, sar: float, tgt_w: int, tgt_h: int) -> dict:
    """Scale-and-pad geometry. Never distorts: one uniform scale factor."""
    disp_w = src_w * sar          # square-pixel width
    scale = min(tgt_w / disp_w, tgt_h / src_h)
    new_w = int(round(disp_w * scale))
    new_h = int(round(src_h * scale))
    # ffmpeg scale must produce even dims for yuv420p
    new_w -= new_w % 2
    new_h -= new_h % 2
    pad_x = (tgt_w - new_w) // 2
    pad_y = (tgt_h - new_h) // 2
    return {"scale_w": new_w, "scale_h": new_h, "pad_x": pad_x, "pad_y": pad_y,
            "scale_factor": scale,
            "pad_fraction": 1.0 - (new_w * new_h) / (tgt_w * tgt_h)}


def suggest_target(src_w: int, src_h: int, sar: float, tgt_w: int, tgt_h: int) -> tuple[int, int]:
    """Target dims matching the source AR, both multiples of 16, ~same pixel budget."""
    ar = (src_w * sar) / src_h
    budget = tgt_w * tgt_h
    h = round_to_16(int(round((budget / ar) ** 0.5)))
    w = round_to_16(int(round(h * ar)))
    return w, h


def build_shots(cut_frames: list[int], total: int, min_len: int) -> list[Shot]:
    """One shot per detected cut interval. Shots are NEVER merged across a cut.

    A short interval used to be merged into the preceding shot, which produced a
    shot spanning the cut and, from there, chunks spanning it too - the one thing
    the chunking is supposed to guarantee never happens. A short interval is now
    kept as its own shot: `chunk_shot` gives it the longest legal window that
    fits, and anything below the 5-frame minimum simply yields no window and is
    passed through unrestored at assembly. `min_len` therefore only labels a shot
    as short; it never changes a boundary.
    """
    bounds = sorted(set([0] + [c for c in cut_frames if 0 < c < total] + [total]))
    shots: list[Shot] = []
    for s, e in zip(bounds, bounds[1:]):
        shots.append(Shot(shot_id=f"shot{len(shots):04d}", start_frame=s, end_frame=e,
                          src_start_sec=0.0, src_end_sec=0.0, n_frames=e - s))
    return shots


def chunk_shot(shot: Shot, chunk_frames: int, overlap: int) -> list[tuple[int, int, int]]:
    """Split one shot into (start, end, overlap_prev) windows, all 4n+1 long.

    Windows never cross the shot boundary. The final window is snapped BACK from
    the shot end so it stays a legal length and stays inside the shot, which means
    its overlap with the previous window may be larger than the nominal overlap.
    """
    out: list[tuple[int, int, int]] = []
    n = shot.n_frames
    if n < 5:
        return out          # shorter than the minimum legal VACE length (4*1+1)

    if n <= chunk_frames:
        length = nearest_valid_length(n)
        out.append((shot.start_frame, shot.start_frame + length, 0))
        if length < n:
            # The shot length is not itself 4n+1, so the first window leaves a
            # tail of up to 3 frames uncovered. Add a second window anchored to
            # the shot END. It overlaps heavily, which costs one extra
            # generation, but it means every frame of the shot is restored
            # rather than silently falling back to the unrestored source.
            s2 = shot.end_frame - length
            ov = (shot.start_frame + length) - s2
            out.append((s2, s2 + length, max(0, ov)))
        return out

    stride = chunk_frames - overlap
    pos = shot.start_frame
    prev_end = None
    while True:
        end = pos + chunk_frames
        if end >= shot.end_frame:
            # last window: pull it back so it ends exactly at the shot end
            start = max(shot.start_frame, shot.end_frame - chunk_frames)
            length = nearest_valid_length(min(chunk_frames, shot.end_frame - start))
            start = shot.end_frame - length
            ov = 0 if prev_end is None else max(0, prev_end - start)
            if prev_end is None or start + length > prev_end:
                out.append((start, start + length, ov))
            break
        ov = 0 if prev_end is None else max(0, prev_end - pos)
        out.append((pos, end, ov))
        prev_end = end
        pos += stride
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="Re-encode even if outputs exist")
    ap.add_argument("--deinterlace", action="store_true", help="Apply yadif")
    ap.add_argument("--auto-aspect", action="store_true",
                    help="Override config W/H with dims matching the true source AR")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("preprocess_source", args.verbose)
    require_tools("ffmpeg", "ffprobe")
    cfg = load_config(args.config)
    v = cfg["video"]

    src = args.source or find_single(P.source, VIDEO_EXTS, "source video")
    log.info("Source (read-only): %s", src)

    probe = ffprobe_json(src)
    vs = next(s for s in probe["streams"] if s.get("codec_type") == "video")
    has_audio = any(s.get("codec_type") == "audio" for s in probe["streams"])
    src_w, src_h = int(vs["width"]), int(vs["height"])
    src_fps = parse_fraction(vs.get("avg_frame_rate")) or parse_fraction(vs.get("r_frame_rate"))
    duration = float(probe["format"].get("duration") or 0)
    sar_s = vs.get("sample_aspect_ratio", "1:1")
    try:
        sar = float(Fraction(*(int(x) for x in sar_s.split(":")))) if sar_s and sar_s != "0:1" else 1.0
    except Exception:
        sar = 1.0
    interlaced = vs.get("field_order", "progressive") not in ("progressive", None, "")

    log.info("Source: %dx%d SAR %s (display AR %.3f), %.3f fps, %s, audio=%s",
             src_w, src_h, sar_s, (src_w * sar) / src_h, src_fps,
             human_time(duration), has_audio)

    # ---- resolution decision -------------------------------------------------
    tgt_w, tgt_h = v["width"], v["height"]
    fit = compute_fit(src_w, src_h, sar, tgt_w, tgt_h)
    if args.auto_aspect:
        tgt_w, tgt_h = suggest_target(src_w, src_h, sar, v["width"], v["height"])
        fit = compute_fit(src_w, src_h, sar, tgt_w, tgt_h)
        log.info("--auto-aspect: using %dx%d instead of %dx%d",
                 tgt_w, tgt_h, v["width"], v["height"])
    elif fit["pad_fraction"] > 0.12:
        alt_w, alt_h = suggest_target(src_w, src_h, sar, tgt_w, tgt_h)
        log.warning(
            "Target %dx%d wastes %.0f%% of the frame on padding for this source AR. "
            "%dx%d matches the source aspect (both multiples of 16). "
            "Re-run with --auto-aspect, or set video.width/height in %s.",
            tgt_w, tgt_h, 100 * fit["pad_fraction"], alt_w, alt_h, cfg["_config_path"])

    log.info("Fit: scale to %dx%d, pad to %dx%d (offset %d,%d), %.1f%% padding",
             fit["scale_w"], fit["scale_h"], tgt_w, tgt_h,
             fit["pad_x"], fit["pad_y"], 100 * fit["pad_fraction"])

    P.normalized.mkdir(parents=True, exist_ok=True)

    # ---- 1. CFR working copy at native size ---------------------------------
    cfr = P.normalized / "source_cfr.mp4"
    vf_cfr = []
    if interlaced or args.deinterlace:
        vf_cfr.append("yadif=mode=0")
        log.info("Deinterlacing with yadif")
    if abs(sar - 1.0) > 1e-3:
        vf_cfr.append(f"scale={int(round(src_w*sar))}:{src_h}:flags=lanczos,setsar=1")
    vf_cfr.append("format=yuv420p")
    if args.force or not cfr.exists():
        log.info("Normalizing to CFR %.4f fps -> %s", src_fps, cfr.name)
        run(["ffmpeg", "-y", "-i", str(src), "-map", "0:v:0",
             "-vf", ",".join(vf_cfr), "-r", f"{src_fps:.10f}", "-vsync", "cfr",
             "-c:v", "libx264", "-crf", "16", "-preset", "medium",
             "-pix_fmt", "yuv420p", "-an", str(cfr)], log)
    else:
        log.info("CFR copy exists, skipping (use --force to rebuild)")

    # ---- 2. VACE working stream at model fps + target geometry --------------
    model_fps = v["model_fps"]
    work = P.normalized / f"work_{tgt_w}x{tgt_h}_{model_fps}fps.mp4"
    pad_color = v.get("pad_color", "black")
    vf_work = (f"scale={fit['scale_w']}:{fit['scale_h']}:flags=lanczos,"
               f"pad={tgt_w}:{tgt_h}:{fit['pad_x']}:{fit['pad_y']}:color={pad_color},"
               f"setsar=1,format=yuv420p")
    if args.force or not work.exists():
        log.info("Building VACE working stream %dx%d @ %d fps -> %s",
                 tgt_w, tgt_h, model_fps, work.name)
        run(["ffmpeg", "-y", "-i", str(cfr), "-map", "0:v:0",
             "-vf", vf_work, "-r", str(model_fps), "-vsync", "cfr",
             "-c:v", "libx264", "-crf", "14", "-preset", "medium",
             "-pix_fmt", "yuv420p", "-an", str(work)], log)
    else:
        log.info("Working stream exists, skipping (use --force to rebuild)")

    wprobe = ffprobe_json(work)
    wvs = next(s for s in wprobe["streams"] if s.get("codec_type") == "video")
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(work)])
    total_frames = int(p.stdout.strip())
    real_frames = total_frames
    log.info("Working stream: %sx%s, %d frames, %.4f s",
             wvs["width"], wvs["height"], total_frames, total_frames / model_fps)

    # ---- 2b. tail-pad a short stream up to one legal inference length --------
    # A stream just under the chunk size used to be split into two nearly
    # identical windows - 80 frames became 0-77 and 3-80, a 74-frame overlap and
    # two full generations for one 5 s clip. VACE lengths are 4n+1, so instead
    # hold the last frame for a few frames and generate ONCE. The padding is
    # dropped again when the interval is cropped, so no padded frame reaches the
    # output. Only done when the whole stream fits a single chunk and there is no
    # cut to respect.
    chunk_frames_cfg = int(v["chunk_frames"])
    if real_frames < chunk_frames_cfg and (real_frames - 1) % 4 != 0:
        padded = nearest_valid_length(real_frames)
        if padded < real_frames:
            padded += 4
        if padded <= chunk_frames_cfg:
            pad_n = padded - real_frames
            work_padded = P.normalized / (work.stem + f"_pad{padded}.mp4")
            if args.force or not work_padded.exists():
                log.info("Tail-padding %d -> %d frames (hold the last frame %d "
                         "time(s)) so this clip is ONE %d-frame inference instead "
                         "of two overlapping ones.", real_frames, padded, pad_n,
                         padded)
                run(["ffmpeg", "-y", "-v", "error", "-i", str(work), "-vf",
                     f"tpad=stop_mode=clone:stop_duration={pad_n / model_fps:.6f}",
                     "-r", str(model_fps), "-vsync", "cfr",
                     "-c:v", "libx264", "-crf", "14", "-preset", "medium",
                     "-pix_fmt", "yuv420p", "-an", str(work_padded)], log)
            got = probe_frames(work_padded)
            if got != padded:
                raise RuntimeError(f"tail-pad produced {got} frames, wanted {padded}")
            work, total_frames = work_padded, padded
            log.info("Working stream is now %d frames (%d real + %d padding)",
                     total_frames, real_frames, pad_n)

    # ---- 3. scene detection --------------------------------------------------
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    sd = cfg["scene_detect"]
    log.info("Detecting scene cuts (ContentDetector threshold=%s)...", sd["threshold"])
    video = open_video(str(work))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=float(sd["threshold"]),
                                    min_scene_len=int(sd["min_scene_len_frames"])))
    sm.detect_scenes(video, show_progress=False)
    scenes = sm.get_scene_list()
    cut_frames = [s[0].get_frames() for s in scenes if s[0].get_frames() > 0]
    log.info("Detected %d scene cut(s) -> %d shot(s)", len(cut_frames), len(cut_frames) + 1)

    shots = build_shots(cut_frames, total_frames, int(v["min_shot_frames"]))
    for s in shots:
        s.src_start_sec = s.start_frame / model_fps
        s.src_end_sec = s.end_frame / model_fps

    # ---- 4. chunking ---------------------------------------------------------
    chunk_frames, overlap = int(v["chunk_frames"]), int(v["chunk_overlap"])
    chunks: list[Chunk] = []
    skipped_short = 0
    for s in shots:
        windows = chunk_shot(s, chunk_frames, overlap)
        if not windows:
            skipped_short += 1
            log.warning("Shot %s is only %d frames (< 5); no legal VACE window fits. "
                        "It will be passed through unrestored at assembly.",
                        s.shot_id, s.n_frames)
            continue
        for i, (st, en, ov) in enumerate(windows):
            cid = f"{s.shot_id}_c{i:03d}"
            chunks.append(Chunk(
                chunk_id=cid, shot_id=s.shot_id,
                start_frame=st, end_frame=en, n_frames=en - st, overlap_prev=ov,
                src_start_sec=st / model_fps, src_end_sec=en / model_fps,
                width=tgt_w, height=tgt_h, fps=model_fps,
                seed=int(cfg["sampling"]["seed"]),
                prompt=cfg["prompt"]["positive"].strip(),
                negative_prompt=cfg["prompt"]["negative"].strip(),
                # Derived from P, not hard-coded: under VACE_RUN these live in
                # runs/<name>/. Consumers resolve them as P.root / <path>, so a
                # literal "intermediate/..." would send every later stage back to
                # the unnamespaced tree.
                reference_sheet=rel(P.reference_sheets / "reference_sheet.png"),
                depth_path=rel(P.depth / f"{cid}_depth.mp4"),
                mask_path=rel(P.masks / f"{cid}_mask.mp4"),
                control_path=rel(P.chunks / f"{cid}_control.mp4"),
                output_path=rel(P.restored_480p / f"{cid}.mkv"),
            ))

    bad = [c for c in chunks if (c.n_frames - 1) % 4 != 0]
    if bad:
        raise RuntimeError(f"Internal error: {len(bad)} chunk(s) not 4n+1: "
                           f"{[(c.chunk_id, c.n_frames) for c in bad[:5]]}")

    manifest = {
        "schema_version": 2,
        "config_path": cfg["_config_path"],
        "profile": cfg["profile"],
        "source": {
            "path": str(src), "filename": src.name,
            "width": src_w, "height": src_h, "sample_aspect_ratio": sar_s,
            "fps": src_fps, "duration_sec": duration, "has_audio": has_audio,
            "interlaced": interlaced,
        },
        "normalized": {
            "real_frames": int(real_frames),
            "pad_frames": int(total_frames - real_frames),
            "cfr_path": str(cfr.relative_to(P.root)),
            "work_path": str(work.relative_to(P.root)),
            "width": tgt_w, "height": tgt_h, "fps": model_fps,
            "total_frames": total_frames,
            "duration_sec": total_frames / model_fps,
            "fit": fit,
        },
        # Everything needed to map a working-stream frame back to source time.
        "timing": {
            "model_fps": model_fps,
            "source_fps": src_fps,
            "frame_to_source_sec": f"src_sec = frame / {model_fps}",
            "note": ("The working stream is resampled to model fps but has the same "
                     "wall-clock duration as the CFR copy, so frame/model_fps is a "
                     "valid source timestamp. Audio is remuxed from the ORIGINAL file "
                     "at assembly time using these seconds."),
        },
        "scene_cuts": cut_frames,
        "shots": [vars(s) for s in shots],
        "chunks": [c.to_dict() for c in chunks],
        "constraints": {
            "valid_lengths": "4n+1 (WanVaceToVideo length step=4 from min=1)",
            "dim_multiple": 16,
            "mask_polarity": "white=regenerate, black=preserve",
            "source": "ComfyUI/comfy_extras/nodes_wan.py::WanVaceToVideo",
        },
    }
    save_manifest(manifest)

    gen_frames = sum(c.n_frames for c in chunks)
    log.info("-" * 62)
    log.info("Shots        : %d (%d too short to restore)", len(shots), skipped_short)
    log.info("Chunks       : %d", len(chunks))
    log.info("Frames to gen: %d (%.1fx the %d source frames, due to overlap)",
             gen_frames, gen_frames / max(total_frames, 1), total_frames)
    log.info("Manifest     : %s", P.manifest)
    log.info("-" * 62)
    log.info("Next: scripts/prepare_references.py, then scripts/make_depth.py "
             "and scripts/track_subject.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
