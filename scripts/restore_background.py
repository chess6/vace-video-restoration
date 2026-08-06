#!/usr/bin/env python
"""Phase 7b - full-frame background restoration with SeedVR2 3B.

Runs BEFORE VACE and produces the plate the subject is later composited onto.
It restores the whole frame - environment, props, distant subjects, signage - and
VACE then replaces only the masked main subject on top of it.

What this stage deliberately does NOT do: it never feeds anything structural.
Scene cuts, chunk timing, depth, masks and subject tracking are all derived from
the ORIGINAL normalized stream and are never recomputed from SeedVR2 output. A
background profile can therefore be changed, or this stage skipped entirely,
without moving a single chunk boundary or shifting the tracked subject.

Caching: results are keyed by (source interval, configuration hash), so the same
interval restored with the same settings is computed once and then reused across
every VACE seed, reference variant and integration path. Changing any setting
that affects the pixels changes the hash and forces a rebuild.

VRAM: 3B in fp8 only, VAE-tiled decode, and SeedVR2's own temporal chunking so
peak VRAM is set by frames_per_chunk rather than by clip length. The 7B model is
never used locally.

    scripts/restore_background.py --profile background_conservative
    scripts/restore_background.py --all-profiles [--only shot0000_c000]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_client import ComfyClient, load_api_workflow, set_input  # noqa: E402
from common import (  # noqa: E402
    P, geometry_key, human_size, human_time, load_config, load_manifest,
    probe_dims_fps, probe_frames, rel, run, save_manifest, setup_logging,
    slice_frames,
)


def profile_hash(cfg: dict, profile: str, width: int, height: int) -> str:
    """Configuration hash for the cache key.

    Covers every setting that changes the restored pixels, so a cached span is
    reused only when it would be bit-identical to recomputing it. Deliberately
    excludes VACE's seed, prompts and reference sheet: those do not affect the
    background, which is exactly why the cache survives across them.
    """
    b = cfg["background"]
    p = b["profiles"][profile]
    payload = {
        "profile": profile, "model": b["model"], "vae": b["vae"],
        "weight_dtype": b["weight_dtype"],
        "target_short_edge": b["target_short_edge"],
        "chunking_mode": b.get("chunking_mode", "manual"),
        "frames_per_chunk": b["frames_per_chunk"],
        "temporal_overlap": b["temporal_overlap"],
        "vae_tile_size": b["vae_tile_size"], "vae_tile_overlap": b["vae_tile_overlap"],
        "steps": p["steps"], "cfg": p["cfg"], "denoise": p["denoise"],
        "color_correction": p["color_correction"],
        "sampler": cfg["sampling"]["sampler"], "scheduler": cfg["sampling"]["scheduler"],
        "seed": cfg["sampling"]["seed"],
        "width": width, "height": height,
    }
    return geometry_key({}, payload)


def cache_dir(profile: str, phash: str) -> Path:
    return P.intermediate / "background" / profile / phash


def restore_interval(client: ComfyClient, wf_path: Path, work: Path, start: int,
                     end: int, fps: int, width: int, height: int, dst: Path,
                     short_edge: int, timeout: float, log,
                     model: str | None = None, weight_dtype: str | None = None) -> dict:
    """Restore [start, end) of the working stream into `dst`. Returns run stats."""
    src = P.comfy_input / "bg_source.mp4"
    slice_frames(work, src, start, end, fps, log, lossless=False)
    n_in = probe_frames(src)
    if n_in != end - start:
        raise RuntimeError(f"source slice {start}-{end} has {n_in} frames")

    wf = load_api_workflow(wf_path)
    set_input(wf, "LoadVideo", "file", src.name)
    # Re-assert the checkpoint at RUN time. build_workflows.py bakes the config's
    # model into UNETLoader at BUILD time, so the graph on disk goes stale the
    # moment the config changes or --model is passed - and nothing noticed,
    # because the model name reaches the cache hash and the recorded metadata by
    # a different route. That combination is the dangerous one: a run would be
    # filed as 7B, force a cache rebuild, and execute 3B.
    if model:
        set_input(wf, "UNETLoader", "unet_name", model)
    if weight_dtype:
        set_input(wf, "UNETLoader", "weight_dtype", weight_dtype)
    # The workflow is built from the config, but --auto-aspect may have chosen a
    # different working geometry. Patch both resize nodes so the plate comes back
    # pixel-aligned with the depth, the mask and the original.
    from build_workflows import seedvr2_target_size
    tw, th = seedvr2_target_size(width, height, short_edge)
    set_input(wf, "ImageScale", "width", tw, title_contains="Resize")
    set_input(wf, "ImageScale", "height", th, title_contains="Resize")
    set_input(wf, "ImageScale", "width", width, title_contains="Back to")
    set_input(wf, "ImageScale", "height", height, title_contains="Back to")

    t0 = time.time()
    hist = client.run(wf, timeout=timeout)
    dt = time.time() - t0
    outs = ComfyClient.output_files(hist, P.comfy_output)
    if not outs:
        raise RuntimeError("SeedVR2 workflow produced no output")
    produced = outs[0]

    # Exit code proves it ran; decode and measure before trusting it (rule 4).
    n_out = probe_frames(produced)
    w_out, h_out, fps_out = probe_dims_fps(produced)
    if n_out != end - start:
        raise RuntimeError(f"restored {n_out} frames, expected {end - start}")
    if (w_out, h_out) != (width, height):
        raise RuntimeError(f"restored {w_out}x{h_out}, expected {width}x{height}")
    if abs(fps_out - fps) > 0.02:
        raise RuntimeError(f"restored at {fps_out} fps, expected {fps}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    # Keep the plate lossless: it is re-read by VACE and by the compositor, and a
    # lossy intermediate here would show up as background drift between paths.
    run(["ffmpeg", "-y", "-v", "error", "-i", str(produced),
         "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p", "-an", str(dst)], log)
    produced.unlink(missing_ok=True)
    src.unlink(missing_ok=True)
    if probe_frames(dst) != end - start:
        raise RuntimeError(f"cached plate {dst.name} lost frames during archiving")

    return {"seconds": round(dt, 2), "peak_vram_mb": hist.get("_peak_vram_mb", 0),
            "frames": n_out, "bytes": dst.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None, help="Background profile to run")
    ap.add_argument("--all-profiles", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="Limit to these chunk ids")
    ap.add_argument("--pilot", action="store_true", help="Only chunks tagged is_pilot")
    ap.add_argument("--force", action="store_true", help="Ignore the cache")
    ap.add_argument("--model", default=None,
                    help="Override background.model, e.g. a 7B checkpoint. The "
                         "model name is part of the cache hash, so a different "
                         "model rebuilds rather than silently reusing a plate "
                         "built by another one.")
    ap.add_argument("--timeout", type=float, default=5400)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("restore_background", args.verbose)
    cfg = load_config(args.config)
    if args.model:
        cfg["background"]["model"] = args.model
    man = load_manifest()
    b = cfg.get("background", {})
    if not b.get("enabled"):
        log.error("background.enabled is false in the config; nothing to do.")
        return 1

    profiles = (list(b["profiles"]) if args.all_profiles
                else [args.profile or cfg["composite"].get("profile")
                      or list(b["profiles"])[0]])
    for p in profiles:
        if p not in b["profiles"]:
            log.error("Unknown profile %r. Available: %s", p, ", ".join(b["profiles"]))
            return 1

    client = ComfyClient(cfg["runtime"]["comfy_host"], int(cfg["runtime"]["comfy_port"]), log)
    if not client.is_up():
        log.error("ComfyUI is not running. Start it: scripts/start_comfyui.sh --daemon")
        return 1

    work = P.root / man["normalized"]["work_path"]
    if not work.exists():
        log.error("Working stream missing: %s. Run preprocess_source.py first.", work)
        return 1
    width = int(man["normalized"]["width"])
    height = int(man["normalized"]["height"])
    fps = int(man["normalized"]["fps"])

    chunks = man["chunks"]
    if args.only:
        chunks = [c for c in chunks if c["chunk_id"] in set(args.only)]
    elif args.pilot:
        chunks = [c for c in chunks if c.get("is_pilot")]
        if not chunks:
            log.error("No pilot chunks. Run scripts/extract_pilot.py first.")
            return 1
    chunks = [c for c in chunks if c.get("status") != "skipped"]
    if not chunks:
        log.error("No chunks selected.")
        return 1

    stats: list[dict] = []
    for profile in profiles:
        phash = profile_hash(cfg, profile, width, height)
        out_dir = cache_dir(profile, phash)
        wf_path = P.workflows / f"seedvr2_{profile}_api.json"
        if not wf_path.exists():
            log.error("Missing %s. Run scripts/build_workflows.py", wf_path)
            return 1
        log.info("=" * 62)
        log.info("Profile %s  hash=%s  %dx%d @ %d fps", profile, phash, width, height, fps)
        log.info("Cache: %s", out_dir)

        # Distinct intervals only: overlapping chunks share frames, and the cache
        # is keyed by interval, so an interval restored once is reused by every
        # chunk that needs it.
        intervals = sorted({(int(c["start_frame"]), int(c["end_frame"])) for c in chunks})
        log.info("%d chunk(s) -> %d distinct interval(s)", len(chunks), len(intervals))

        built = reused = 0
        for i, (a, z) in enumerate(intervals):
            dst = out_dir / f"bg_{a:07d}_{z:07d}.mkv"
            if dst.exists() and not args.force and probe_frames(dst) == z - a:
                log.info("[%d/%d] %d-%d cached", i + 1, len(intervals), a, z)
                reused += 1
                continue
            log.info("[%d/%d] %d-%d restoring (%d frames)...",
                     i + 1, len(intervals), a, z, z - a)
            s = restore_interval(client, wf_path, work, a, z, fps, width, height,
                                 dst, int(b["target_short_edge"]), args.timeout, log,
                                 model=b["model"], weight_dtype=b["weight_dtype"])
            s.update(profile=profile, start=a, end=z)
            stats.append(s)
            built += 1
            log.info("        %s, peak VRAM %s MiB, %s -> %s",
                     human_time(s["seconds"]), s["peak_vram_mb"],
                     human_size(s["bytes"]), dst.name)

        # Record where each chunk's plate lives, per profile, so run_chunks.py and
        # the compositor can find it without recomputing the hash.
        for c in chunks:
            src_path = out_dir / f"bg_{int(c['start_frame']):07d}_{int(c['end_frame']):07d}.mkv"
            # Not setdefault: an older manifest can carry an explicit null here
            # (a run_chunks variant field once collided with this key), and
            # setdefault would hand back that None.
            if not isinstance(c.get("background"), dict):
                c["background"] = {}
            c["background"][profile] = {
                "path": rel(src_path), "config_hash": phash}
        man.setdefault("background_profiles", {})[profile] = {
            "config_hash": phash, "cache_dir": rel(out_dir),
            "model": b["model"], "weight_dtype": b["weight_dtype"],
        }
        save_manifest(man)
        if stats:
            write_runtime_report(stats, width, height, fps, log)
        log.info("Profile %s: %d built, %d reused from cache", profile, built, reused)

    if stats:
        write_runtime_report(stats, width, height, fps, log)
    return 0


def write_runtime_report(stats: list[dict], width: int, height: int, fps: int,
                         log) -> None:
    """Append these measurements to reports/background_runtime.json.

    Called after EVERY profile, not once at the end: writing only at the end
    meant a failure in a later profile silently discarded the timings already
    measured for the earlier ones.
    """
    if stats:
        tot = sum(s["seconds"] for s in stats)
        fr = sum(s["frames"] for s in stats)
        by = sum(s["bytes"] for s in stats)
        peak = max(s["peak_vram_mb"] for s in stats)
        log.info("=" * 62)
        log.info("SeedVR2 totals: %s for %d frame(s) (%.2f s/frame), peak VRAM %s MiB, %s on disk",
                 human_time(tot), fr, tot / max(fr, 1), peak, human_size(by))
        P.reports.mkdir(parents=True, exist_ok=True)
        rp = P.reports / "background_runtime.json"
        prev = json.loads(rp.read_text()).get("runs", []) if rp.exists() else []
        rp.write_text(json.dumps({
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "geometry": {"width": width, "height": height, "fps": fps},
            "seconds_per_frame": round(tot / max(fr, 1), 4),
            "peak_vram_mb": peak,
            "bytes_per_frame": round(by / max(fr, 1)),
            "runs": prev + stats,
        }, indent=2) + "\n")
        log.info("Runtime report -> %s", rel(rp))


if __name__ == "__main__":
    raise SystemExit(main())
