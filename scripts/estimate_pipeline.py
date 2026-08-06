#!/usr/bin/env python
"""Phase 10c - separate SeedVR2 and VACE cost, and what the full video would cost.

Reads measurements that were actually taken on this machine - never a guess:
  * SeedVR2   reports/background_runtime.json, written by restore_background.py
  * VACE      per-chunk duration_sec / peak_vram_mb recorded in the manifest

and projects them onto the whole source. Every projection states which
measurement it came from and how many samples were behind it, because a
one-chunk sample is a weak basis for a multi-day estimate and should look like
one.

    scripts/estimate_pipeline.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, human_size, human_time, load_config, load_manifest, rel, setup_logging,
)


def vace_samples(man: dict) -> list[dict]:
    """Every completed VACE generation in the manifest, baseline or variant."""
    out = []
    for c in man.get("chunks", []):
        for variant, r in (c.get("runs") or {}).items():
            if r.get("status") == "done" and r.get("duration_sec"):
                out.append({"chunk": c["chunk_id"], "variant": variant,
                            "frames": c["n_frames"],
                            "seconds": float(r["duration_sec"]),
                            "peak_vram_mb": r.get("peak_vram_mb") or 0,
                            "background": r.get("background")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--source-seconds", type=float, default=None,
                    help="Duration of the WHOLE source video, when the manifest "
                         "describes only a sampled clip. Projections use this "
                         "instead of the clip length.")
    args = ap.parse_args()

    log = setup_logging("estimate_pipeline")
    cfg = load_config(args.config)
    man = load_manifest()
    n = man["normalized"]
    fps = int(n["fps"])
    clip_frames = int(n["total_frames"])
    # A pilot is measured on a sampled clip, but the question being asked is what
    # the whole video costs. Project onto the real duration when it is given.
    total_frames = (int(round(args.source_seconds * fps)) if args.source_seconds
                    else clip_frames)
    scale = total_frames / max(clip_frames, 1)
    chunks = man.get("chunks", [])
    gen_frames = int(sum(c["n_frames"] for c in chunks
                         if c.get("status") != "skipped") * scale)

    lines: list[str] = []
    lines.append("# Pipeline cost: SeedVR2 and VACE measured separately\n")
    lines.append(f"Source: {total_frames} frames at {n['width']}x{n['height']} "
                 f"@ {fps} fps ({human_time(total_frames / fps)}).\n")
    if args.source_seconds:
        lines.append(f"> Measured on a {clip_frames}-frame sampled clip and scaled "
                     f"x{scale:.0f} to the full source. Per-frame costs are "
                     f"measured; the multiplication is arithmetic.\n")
    lines.append(f"Chunking gives {len(chunks)} chunk(s), {gen_frames} generated "
                 f"frames ({gen_frames / max(total_frames,1):.2f}x the source, "
                 f"the excess being chunk overlap).\n")

    totals = {}

    # ---- SeedVR2 -------------------------------------------------------------
    bp = P.reports / "background_runtime.json"
    lines.append("\n## SeedVR2 background restoration\n")
    if bp.exists():
        b = json.loads(bp.read_text())
        runs = b.get("runs", [])
        by_profile: dict[str, list[dict]] = {}
        for r in runs:
            by_profile.setdefault(r.get("profile", "?"), []).append(r)
        lines.append("| Profile | Samples | Frames | s/frame | Peak VRAM | Bytes/frame | Full video |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for prof, rs in sorted(by_profile.items()):
            fr = sum(r["frames"] for r in rs)
            se = sum(r["seconds"] for r in rs)
            by = sum(r["bytes"] for r in rs)
            spf = se / max(fr, 1)
            peak = max(r["peak_vram_mb"] for r in rs)
            # SeedVR2 runs once per distinct interval, i.e. over the SOURCE
            # timeline, not the overlapped chunk timeline.
            full_s = spf * total_frames
            totals[f"seedvr2_{prof}"] = {"seconds": full_s,
                                         "bytes": (by / max(fr, 1)) * total_frames}
            lines.append(f"| `{prof}` | {len(rs)} | {fr} | {spf:.2f} | "
                         f"{peak} MiB | {human_size(by / max(fr,1))} | "
                         f"**{human_time(full_s)}** |")
        lines.append("\nSeedVR2 is billed per *source* frame: the cache is keyed by "
                     "interval, so overlapping chunks reuse one restoration and the "
                     "stage does not pay the overlap multiplier that VACE does.\n")
    else:
        lines.append("_No SeedVR2 measurement yet. Run "
                     "`scripts/restore_background.py`._\n")

    # ---- VACE ----------------------------------------------------------------
    lines.append("\n## VACE subject generation\n")
    vs = vace_samples(man)
    if vs:
        by_variant: dict[str, list[dict]] = {}
        for s in vs:
            by_variant.setdefault(s["variant"], []).append(s)
        lines.append("| Variant | Samples | Frames | s/frame | Peak VRAM | Full video |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for var, ss in sorted(by_variant.items()):
            fr = sum(s["frames"] for s in ss)
            se = sum(s["seconds"] for s in ss)
            spf = se / max(fr, 1)
            peak = max(s["peak_vram_mb"] for s in ss)
            full_s = spf * gen_frames
            totals[f"vace_{var}"] = {"seconds": full_s, "bytes": 0}
            lines.append(f"| `{var}` | {len(ss)} | {fr} | {spf:.2f} | {peak} MiB | "
                         f"**{human_time(full_s)}** |")
        lines.append("\nVACE is billed per *generated* frame, which includes the "
                     "chunk overlap, hence the multiplier above.\n")
    else:
        lines.append("_No completed VACE chunk yet._\n")

    # ---- combined ------------------------------------------------------------
    lines.append("\n## One full pass over the whole video\n")
    if totals:
        bg_keys = [k for k in totals if k.startswith("seedvr2_")]
        vace_keys = [k for k in totals if k.startswith("vace_")]
        if bg_keys and vace_keys:
            cheap_bg = min(bg_keys, key=lambda k: totals[k]["seconds"])
            cheap_v = min(vace_keys, key=lambda k: totals[k]["seconds"])
            combined = totals[cheap_bg]["seconds"] + totals[cheap_v]["seconds"]
            lines.append(f"Choosing one background profile (`{cheap_bg}`) and one "
                         f"VACE variant (`{cheap_v}`):\n")
            lines.append(f"- SeedVR2: **{human_time(totals[cheap_bg]['seconds'])}**")
            lines.append(f"- VACE:    **{human_time(totals[cheap_v]['seconds'])}**")
            lines.append(f"- Total:   **{human_time(combined)}** "
                         f"({combined / 3600:.1f} h, {combined / 86400:.1f} days)\n")
            lines.append(f"- Background plates on disk: "
                         f"**{human_size(totals[cheap_bg]['bytes'])}** per profile\n")
            lines.append("Adding the background stage therefore costs roughly "
                         f"{100 * totals[cheap_bg]['seconds'] / max(totals[cheap_v]['seconds'], 1):.0f}% "
                         "on top of VACE alone, for a restoration that covers the "
                         "entire frame rather than just the subject.\n")
    lines.append("\n> These come from a very small number of samples on one clip. "
                 "They are the right order of magnitude, not a schedule. Re-run "
                 "this after more chunks have completed before planning around "
                 "the numbers.\n")

    out = args.out or (P.reports / "pipeline_estimates.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", rel(out))
    for line in lines:
        if line.startswith(("- ", "| `")) or "**" in line:
            log.info("%s", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
