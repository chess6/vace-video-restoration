#!/usr/bin/env python
"""Phase 4.1 - inspect the source video with ffprobe.

Read-only. Writes reports/source_info.json and reports/source_info.md.
Never modifies, moves or re-encodes the original.

    scripts/inspect_source.py [--source inputs/source/foo.mp4]
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, VIDEO_EXTS, ffprobe_json, find_single, human_size, human_time,
    parse_fraction, require_tools, setup_logging,
)


def count_frames_exact(path: Path, log) -> int | None:
    """nb_frames is often absent or wrong on VBR/streamed files.

    Fall back to a full packet count, which is exact but reads the file.
    """
    from common import run
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", str(path)], check=False)
    txt = (p.stdout or "").strip()
    return int(txt) if txt.isdigit() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=None,
                    help="Source video. Default: the single file in inputs/source/")
    ap.add_argument("--exact-frames", action="store_true",
                    help="Count packets for an exact frame count (slower, reads whole file)")
    args = ap.parse_args()

    log = setup_logging("inspect_source")
    require_tools("ffprobe")

    src = args.source or find_single(P.source, VIDEO_EXTS, "source video")
    if not src.exists():
        log.error("Source not found: %s", src)
        return 1
    log.info("Inspecting (read-only): %s", src)

    probe = ffprobe_json(src)
    vstreams = [s for s in probe["streams"] if s.get("codec_type") == "video"]
    astreams = [s for s in probe["streams"] if s.get("codec_type") == "audio"]
    sstreams = [s for s in probe["streams"] if s.get("codec_type") == "subtitle"]
    if not vstreams:
        log.error("No video stream in %s", src)
        return 1
    v = vstreams[0]

    fmt = probe["format"]
    r_fps = parse_fraction(v.get("r_frame_rate"))
    avg_fps = parse_fraction(v.get("avg_frame_rate"))
    duration = float(fmt.get("duration") or v.get("duration") or 0.0)

    nb_frames = int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None
    frame_source = "stream.nb_frames"
    if args.exact_frames or nb_frames is None:
        exact = count_frames_exact(src, log)
        if exact:
            nb_frames, frame_source = exact, "packet count (exact)"
    if nb_frames is None and avg_fps and duration:
        nb_frames, frame_source = int(round(duration * avg_fps)), "estimated duration*fps"

    w, h = int(v["width"]), int(v["height"])
    sar = v.get("sample_aspect_ratio", "1:1")
    dar = v.get("display_aspect_ratio")
    try:
        sar_f = Fraction(*(int(x) for x in sar.split(":"))) if sar and sar != "0:1" else Fraction(1)
    except Exception:
        sar_f = Fraction(1)
    display_w = int(round(w * float(sar_f)))
    storage_ar = w / h
    display_ar = display_w / h

    # CFR/VFR: r_frame_rate is the "base" rate; avg differs on VFR content.
    is_vfr = bool(r_fps and avg_fps and abs(r_fps - avg_fps) / max(r_fps, 1e-9) > 0.01)

    info = {
        "path": str(src),
        "filename": src.name,
        "size_bytes": src.stat().st_size,
        "size_human": human_size(src.stat().st_size),
        "container": fmt.get("format_name"),
        "duration_sec": duration,
        "duration_human": human_time(duration),
        "bit_rate": fmt.get("bit_rate"),
        "video": {
            "codec": v.get("codec_name"),
            "profile": v.get("profile"),
            "pix_fmt": v.get("pix_fmt"),
            "width": w, "height": h,
            "coded_width": v.get("coded_width"), "coded_height": v.get("coded_height"),
            "sample_aspect_ratio": sar,
            "display_aspect_ratio": dar,
            "storage_aspect_ratio": round(storage_ar, 6),
            "display_aspect_ratio_value": round(display_ar, 6),
            "display_width_square_px": display_w,
            "r_frame_rate": v.get("r_frame_rate"),
            "avg_frame_rate": v.get("avg_frame_rate"),
            "r_fps": round(r_fps, 6),
            "avg_fps": round(avg_fps, 6),
            "likely_vfr": is_vfr,
            "nb_frames": nb_frames,
            "frame_count_source": frame_source,
            "time_base": v.get("time_base"),
            "field_order": v.get("field_order", "progressive"),
            "color_range": v.get("color_range"),
            "color_space": v.get("color_space"),
            "color_transfer": v.get("color_transfer"),
            "color_primaries": v.get("color_primaries"),
            "rotation": next((int(sd.get("rotation", 0))
                              for sd in v.get("side_data_list", [])
                              if "rotation" in sd), 0),
        },
        "audio_streams": [{
            "index": a.get("index"), "codec": a.get("codec_name"),
            "sample_rate": a.get("sample_rate"), "channels": a.get("channels"),
            "channel_layout": a.get("channel_layout"), "language":
                (a.get("tags") or {}).get("language"),
            "duration_sec": float(a.get("duration") or 0) or None,
        } for a in astreams],
        "subtitle_streams": [{"index": s.get("index"), "codec": s.get("codec_name")}
                             for s in sstreams],
        "n_chapters": len(probe.get("chapters", [])),
    }

    P.reports.mkdir(parents=True, exist_ok=True)
    (P.reports / "source_info.json").write_text(json.dumps(info, indent=2))

    # ---- human-readable summary + warnings ---------------------------------
    warn: list[str] = []
    if is_vfr:
        warn.append(
            f"Variable frame rate detected (r={r_fps:.3f} vs avg={avg_fps:.3f}). "
            "Normalization to CFR is mandatory or frame indices will drift from "
            "audio. preprocess_source.py handles this.")
    if info["video"]["field_order"] not in ("progressive", None, ""):
        warn.append(
            f"Interlaced content ({info['video']['field_order']}). preprocess_source.py "
            "will need --deinterlace, otherwise depth and masks will comb.")
    if abs(display_ar - storage_ar) > 0.01:
        warn.append(
            f"Non-square pixels (SAR {sar}). Storage AR {storage_ar:.3f} != display AR "
            f"{display_ar:.3f}. Normalization scales to square pixels at "
            f"{display_w}x{h} before padding, so the subject is not stretched.")
    if not astreams:
        warn.append("No audio stream. The assembly step will skip audio remuxing.")
    if info["video"]["rotation"]:
        warn.append(f"Rotation metadata {info['video']['rotation']}deg will be baked in "
                    "during normalization.")

    lines = [
        "# Source video inspection", "",
        f"Generated from `{src}` (read-only; the original is never modified).", "",
        "| Field | Value |", "|---|---|",
        f"| File | `{src.name}` |",
        f"| Size | {info['size_human']} |",
        f"| Container | {info['container']} |",
        f"| Duration | {info['duration_human']} ({duration:.3f} s) |",
        f"| Video codec | {v.get('codec_name')} / {v.get('profile')} / {v.get('pix_fmt')} |",
        f"| Stored size | {w}x{h} |",
        f"| Display size | {display_w}x{h} (SAR {sar}, DAR {dar}) |",
        f"| Frame rate | r={r_fps:.4f}, avg={avg_fps:.4f}{' (VFR)' if is_vfr else ' (CFR)'} |",
        f"| Frame count | {nb_frames} ({frame_source}) |",
        f"| Field order | {info['video']['field_order']} |",
        f"| Colour | range={v.get('color_range')} space={v.get('color_space')} "
        f"trc={v.get('color_transfer')} prim={v.get('color_primaries')} |",
        f"| Audio streams | {len(astreams)} |",
        f"| Subtitle streams | {len(sstreams)} |",
        "",
    ]
    for a in info["audio_streams"]:
        lines.append(f"- audio #{a['index']}: {a['codec']} {a['sample_rate']} Hz "
                     f"{a['channels']}ch ({a['channel_layout']}) lang={a['language']}")
    lines += ["", "## Warnings", ""]
    lines += [f"- {w_}" for w_ in warn] or ["- None."]
    (P.reports / "source_info.md").write_text("\n".join(lines) + "\n")

    log.info("%s  %dx%d (display %dx%d)  %.3f fps  %s frames  %s",
             v.get("codec_name"), w, h, display_w, h, avg_fps or r_fps,
             nb_frames, info["duration_human"])
    for w_ in warn:
        log.warning(w_)
    log.info("Wrote reports/source_info.json and reports/source_info.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
