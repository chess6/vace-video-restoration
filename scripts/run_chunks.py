#!/usr/bin/env python
"""Phases 9/11 - run VACE over chunks from the manifest. Resumable.

Every chunk records its own status in the manifest, so an interrupted run is
resumed by simply invoking the script again. `--resume-failed` retries only the
chunks that errored, leaving completed work alone.

Safety: this script will NOT process the whole video unless it is given an
explicit chunk selection or --all. scripts/run_full.sh is the only entry point
that runs everything, and it demands --confirm-full-run.

    scripts/run_chunks.py --pilot                 # the pilot chunks only
    scripts/run_chunks.py --only shot0000_c000
    scripts/run_chunks.py --limit 3
    scripts/run_chunks.py --resume-failed
    scripts/run_chunks.py --all                   # used by run_full.sh
    scripts/run_chunks.py --pilot --no-reference  # ablation
    scripts/run_chunks.py --pilot --seed 12345 --tag seedB
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_workflows import graph_name  # noqa: E402
from comfy_client import ComfyClient, load_api_workflow, set_input  # noqa: E402
from common import (  # noqa: E402
    P, assert_aligned, check_geometry, human_time, load_config, load_manifest,
    probe_frames, rel, run, save_manifest, setup_logging, slice_frames,
)

# Key under which the unablated run is recorded. Its fields are also mirrored to
# the top level of the chunk, which is what assemble.py and the run_full.sh
# preflight read.
BASELINE = "baseline"


def default_tag(args) -> str:
    """Name the variant from the flags that make it one. Empty for the baseline."""
    parts = []
    if getattr(args, "background", None):
        # Short, stable name: "bg_conservative" from "background_conservative".
        parts.append("bg_" + args.background.replace("background_", ""))
    if args.no_reference:
        parts.append("noref")
    if args.no_mask:
        parts.append("nomask")
    if args.seed is not None:
        parts.append(f"seed{args.seed}")
    if args.steps:
        parts.append(f"steps{args.steps}")
    return "_".join(parts)


def run_attempts(c: dict, variant: str) -> int:
    runs = c.get("runs", {})
    if variant in runs:
        return int(runs[variant].get("attempts", 0))
    return int(c.get("attempts", 0)) if variant == BASELINE else 0


def run_status(c: dict, variant: str) -> str:
    """Status of one variant of one chunk, tolerating manifests written before
    per-variant records existed (their baseline lives only at the top level)."""
    runs = c.get("runs", {})
    if variant in runs:
        return runs[variant].get("status", "pending")
    return c.get("status", "pending") if variant == BASELINE else "pending"


def record_run(c: dict, variant: str, **fields) -> None:
    """Write one variant's result. Only the baseline is mirrored to the top level,
    so an ablation can never overwrite the result the master is assembled from."""
    c.setdefault("runs", {}).setdefault(variant, {}).update(fields)
    if variant == BASELINE:
        c.update(fields)


def ensure_source_chunk(man: dict, c: dict, log, force=False) -> Path:
    """Frame-exact slice of the normalized working stream for this chunk."""
    dst = (P.root / c["control_path"]).with_suffix(".mp4")
    if dst.exists() and not force and probe_frames(dst) == c["n_frames"]:
        return dst
    work = P.root / man["normalized"]["work_path"]
    # h264 here (not ffv1): ComfyUI's LoadVideo reads it directly and the source
    # is already lossy 240p, so a visually lossless crf is plenty.
    slice_frames(work, dst, c["start_frame"], c["end_frame"], c["fps"], log,
                 lossless=False)
    n = probe_frames(dst)
    if n != c["n_frames"]:
        raise RuntimeError(f"{c['chunk_id']}: source slice has {n} frames, "
                           f"expected {c['n_frames']}")
    return dst


def stage(src: Path, name: str) -> str:
    """Copy an asset into ComfyUI/input under a stable name; return that name."""
    P.comfy_input.mkdir(parents=True, exist_ok=True)
    dst = P.comfy_input / name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sel = ap.add_argument_group("selection")
    sel.add_argument("--only", nargs="*", default=None)
    sel.add_argument("--pilot", action="store_true", help="Only chunks tagged is_pilot")
    sel.add_argument("--limit", type=int, default=None)
    sel.add_argument("--all", action="store_true", help="Every pending chunk")
    sel.add_argument("--resume-failed", action="store_true",
                     help="Retry only chunks with status=failed")
    sel.add_argument("--redo", action="store_true", help="Re-run even if done")
    var = ap.add_argument_group("variants")
    var.add_argument("--no-reference", action="store_true",
                     help="Ablation: drop the reference sheet ONLY, keeping the mask")
    var.add_argument("--no-mask", action="store_true",
                     help="Ablation: drop the subject mask ONLY, keeping the reference")
    var.add_argument("--background", default=None, metavar="PROFILE",
                     help="Preserve a SeedVR2-restored plate instead of the "
                          "original outside the mask (background profile name)")
    var.add_argument("--seed", type=int, default=None, help="Override the seed")
    var.add_argument("--steps", type=int, default=None)
    var.add_argument("--tag", default="", help="Suffix for output filenames")
    ap.add_argument("--timeout", type=float, default=5400)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("run_chunks", args.verbose)
    cfg = load_config(args.config)
    man = load_manifest()
    client = ComfyClient(cfg["runtime"]["comfy_host"],
                         int(cfg["runtime"]["comfy_port"]), log)

    if not client.is_up():
        log.error("ComfyUI is not running. Start it: scripts/start_comfyui.sh --daemon")
        return 1

    use_mask = not args.no_mask
    use_ref = not args.no_reference

    # A variant must never be written into the baseline's record: the baseline and
    # its ablations are different results for the same chunk. `tag` names the
    # variant, and is derived from the flags when not given explicitly so that a
    # forgotten --tag cannot silently overwrite the baseline. Resolved before
    # selection, because "which chunks are done/failed" is a per-variant question.
    tag = args.tag or default_tag(args)
    variant = tag or BASELINE
    if tag and not args.tag:
        log.info("No --tag given for a non-baseline run; using --tag %s", tag)

    # ---- select chunks -------------------------------------------------------
    chunks = man["chunks"]
    if args.only:
        want = set(args.only)
        chunks = [c for c in chunks if c["chunk_id"] in want]
    elif args.pilot:
        chunks = [c for c in chunks if c.get("is_pilot")]
        if not chunks:
            log.error("No pilot chunks. Run scripts/extract_pilot.py first.")
            return 1
    elif args.resume_failed:
        chunks = [c for c in chunks if run_status(c, variant) == "failed"]
    elif args.all:
        pass
    else:
        log.error("Refusing to guess the scope. Pass one of: --pilot, --only, "
                  "--limit, --resume-failed, or --all.")
        return 1

    if not args.redo:
        # "skipped" means a shot with no subject in it: there is nothing to
        # regenerate, and assembly passes the original frames through.
        chunks = [c for c in chunks if run_status(c, variant) not in ("done", "skipped")]
    if args.limit:
        chunks = chunks[:args.limit]
    if not chunks:
        log.info("Nothing to do: no chunks matched (variant %r already done?). "
                 "Pass --redo to regenerate.", variant)
        return 0

    use_bg = bool(args.background)
    if use_bg and not use_mask:
        log.error("--background needs a mask: without one there is no preserved "
                  "region, so the restored plate would be regenerated over.")
        return 1
    wf_name = graph_name(use_mask, use_ref, use_bg)
    wf_path = P.workflows / f"{wf_name}_api.json"
    if not wf_path.exists():
        log.error("Missing %s. Run scripts/build_workflows.py", wf_path)
        return 1
    log.info("Workflow: %s (mask=%s reference=%s)", wf_name, use_mask, use_ref)
    log.info("Variant: %s", variant)
    log.info("Chunks to process: %d", len(chunks))

    ref_sheet = P.reference_sheets / "reference_sheet.png"
    if use_ref and not ref_sheet.exists():
        log.error("Reference sheet missing: %s. Run scripts/prepare_references.py",
                  ref_sheet)
        return 1

    # ---- pre-flight ----------------------------------------------------------
    # Check every selected chunk has its control streams BEFORE generating
    # anything. A chunk takes ~16 minutes on this GPU, so discovering a missing
    # mask on chunk 40 of 395 is an expensive way to find out.
    missing: list[str] = []
    for c in chunks:
        if not (P.root / c["depth_path"]).exists():
            missing.append(f"{c['chunk_id']}: depth {c['depth_path']}")
        if use_mask and not (P.root / c["mask_path"]).exists():
            missing.append(f"{c['chunk_id']}: mask {c['mask_path']}")
        if use_bg:
            bg = (c.get("background") or {}).get(args.background)
            if not bg or not (P.root / bg["path"]).exists():
                missing.append(f"{c['chunk_id']}: background plate for "
                               f"{args.background} (run restore_background.py)")
    if missing:
        log.error("Pre-flight failed: %d missing control stream(s). "
                  "Nothing was generated.", len(missing))
        for m in missing[:12]:
            log.error("  %s", m)
        if len(missing) > 12:
            log.error("  ... and %d more", len(missing) - 12)
        shots_missing = sorted({m.split(":")[0].rsplit("_c", 1)[0]
                                for m in missing if "mask" in m})
        if shots_missing:
            log.error("Untracked shot(s): %s", ", ".join(shots_missing))
            log.error("Fix with: scripts/track_subject.py"
                      + (f" --shot {' '.join(shots_missing)}" if len(shots_missing) < 10 else ""))
        if any("depth" in m for m in missing):
            log.error("Missing depth: run scripts/make_depth.py")
        return 1
    check_geometry(man, log, stage="run_chunks")
    log.info("Pre-flight OK: depth%s present for all %d chunk(s)",
             " and masks" if use_mask else "", len(chunks))

    P.restored_480p.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    t_all = time.time()
    durations: list[float] = []

    for i, c in enumerate(chunks):
        cid = c["chunk_id"]
        suffix = f"_{tag}" if tag else ""
        log.info("-" * 62)
        log.info("[%d/%d] %s  frames %d-%d (%d)  %dx%d", i + 1, len(chunks), cid,
                 c["start_frame"], c["end_frame"], c["n_frames"], c["width"], c["height"])

        try:
            src_chunk = ensure_source_chunk(man, c, log)
            depth = (P.root / c["depth_path"])
            if not depth.exists():
                raise FileNotFoundError(f"depth missing: {depth}. Run make_depth.py")
            mask = (P.root / c["mask_path"]) if use_mask else None
            if use_mask and not mask.exists():
                raise FileNotFoundError(f"mask missing: {mask}. Run track_subject.py")

            # Last point before ~16 minutes of GPU time: prove the three streams
            # really do describe the same frames at the same size. A one-frame or
            # one-pixel disagreement here is invisible in the output but wrong.
            bg_path = None
            if use_bg:
                bg_path = P.root / (c.get("background") or {})[args.background]["path"]
                if not bg_path.exists():
                    raise FileNotFoundError(
                        f"background plate missing: {bg_path}. "
                        f"Run restore_background.py --profile {args.background}")

            streams = {"source": src_chunk, "depth": depth}
            if use_mask:
                streams["mask"] = mask
            if bg_path is not None:
                streams["background"] = bg_path
            assert_aligned(streams, c["n_frames"], c["width"], c["height"],
                           float(c["fps"]), log)

            # ComfyUI LoadVideo reads only from its input directory
            # ComfyUI's LoadVideo reads from its input directory, so the control
            # streams are re-encoded into it. `-qp 0` is mathematically lossless
            # in luma: the mask edge and the depth gradient reach the sampler
            # exactly as computed. They used to be re-encoded lossily, which
            # rounded mask edges outward and so regenerated pixels just outside
            # the tracked boundary.
            LOSSLESS = ["-c:v", "libx264", "-qp", "0", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p"]
            src_name = stage(src_chunk, f"chunk_source_{cid}.mp4")
            dep_mp4 = P.comfy_input / f"chunk_depth_{cid}.mp4"
            run(["ffmpeg", "-y", "-v", "error", "-i", str(depth), *LOSSLESS,
                 str(dep_mp4)], log)
            dep_name = dep_mp4.name
            if use_mask:
                msk_mp4 = P.comfy_input / f"chunk_mask_{cid}.mp4"
                run(["ffmpeg", "-y", "-v", "error", "-i", str(mask), *LOSSLESS,
                     str(msk_mp4)], log)
            if use_bg:
                bg_mp4 = P.comfy_input / f"chunk_background_{cid}.mp4"
                run(["ffmpeg", "-y", "-v", "error", "-i", str(bg_path), *LOSSLESS,
                     str(bg_mp4)], log)
            if use_ref:
                stage(ref_sheet, "reference_sheet.png")

            # ---- patch the workflow -------------------------------------------
            wf = load_api_workflow(wf_path)
            set_input(wf, "LoadVideo", "file", src_name, title_contains="source")
            set_input(wf, "LoadVideo", "file", dep_name, title_contains="depth")
            if use_mask:
                set_input(wf, "LoadVideo", "file", msk_mp4.name, title_contains="mask")
            if use_bg:
                set_input(wf, "LoadVideo", "file", bg_mp4.name,
                          title_contains="background")
            if use_ref:
                set_input(wf, "LoadImage", "image", "reference_sheet.png")
            set_input(wf, "WanVaceToVideo", "width", c["width"])
            set_input(wf, "WanVaceToVideo", "height", c["height"])
            set_input(wf, "WanVaceToVideo", "length", c["n_frames"])
            set_input(wf, "KSampler", "seed", args.seed if args.seed is not None else c["seed"])
            if args.steps:
                set_input(wf, "KSampler", "steps", args.steps)
            set_input(wf, "CreateVideo", "fps", float(c["fps"]))
            set_input(wf, "SaveVideo", "filename_prefix", f"vace/{cid}{suffix}")

            record_run(c, variant, status="running",
                       attempts=run_attempts(c, variant) + 1)
            save_manifest(man)

            t0 = time.time()
            hist = client.run(wf, timeout=args.timeout)
            dt = time.time() - t0
            durations.append(dt)

            outs = ComfyClient.output_files(hist, P.comfy_output)
            if not outs:
                raise RuntimeError("workflow produced no output file")
            produced = outs[0]
            n = probe_frames(produced)
            if n != c["n_frames"]:
                raise RuntimeError(f"output has {n} frames, expected {c['n_frames']}")

            dest = P.restored_480p / f"{cid}{suffix}.mp4"
            shutil.move(str(produced), dest)

            record_run(c, variant, status="done", error="",
                       duration_sec=round(dt, 2),
                       peak_vram_mb=hist.get("_peak_vram_mb", 0),
                       workflow=wf_name, use_mask=use_mask, use_reference=use_ref,
                       background_profile=args.background,
                       seed=args.seed if args.seed is not None else c["seed"],
                       output_path=rel(dest))
            save_manifest(man)
            done += 1
            log.info("done in %s (%.2f s/frame, peak VRAM %s MiB) -> %s",
                     human_time(dt), dt / c["n_frames"],
                     hist.get("_peak_vram_mb", 0), dest.name)

            eta = (len(chunks) - i - 1) * (sum(durations) / len(durations))
            log.info("progress %d/%d, eta %s", i + 1, len(chunks), human_time(eta))

        except Exception as e:
            failed += 1
            record_run(c, variant, status="failed", error=str(e)[:2000])
            save_manifest(man)
            log.error("%s FAILED: %s", cid, str(e)[:1500])
        finally:
            for pat in (f"chunk_source_{cid}.mp4", f"chunk_depth_{cid}.mp4",
                        f"chunk_mask_{cid}.mp4", f"chunk_background_{cid}.mp4"):
                fp = P.comfy_input / pat
                if fp.exists():
                    fp.unlink()

    log.info("=" * 62)
    log.info("Finished: %d done, %d failed, in %s", done, failed,
             human_time(time.time() - t_all))
    if durations:
        avg = sum(durations) / len(durations)
        fpc = chunks[0]["n_frames"]
        log.info("Average %s per chunk (%.2f s per generated frame)",
                 human_time(avg), avg / fpc)
    if failed:
        log.warning("Retry just the failures with: scripts/run_chunks.py --resume-failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
