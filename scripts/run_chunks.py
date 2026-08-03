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
from comfy_client import ComfyClient, load_api_workflow, set_input  # noqa: E402
from common import (  # noqa: E402
    P, human_time, load_config, load_manifest, probe_frames, run, save_manifest,
    setup_logging, slice_frames,
)


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
                     help="Ablation: drop the reference sheet (uses the unmasked graph)")
    var.add_argument("--no-mask", action="store_true",
                     help="Ablation: drop the subject mask")
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
        chunks = [c for c in chunks if c["status"] == "failed"]
    elif args.all:
        pass
    else:
        log.error("Refusing to guess the scope. Pass one of: --pilot, --only, "
                  "--limit, --resume-failed, or --all.")
        return 1

    if not args.redo:
        chunks = [c for c in chunks if c["status"] != "done"]
    if args.limit:
        chunks = chunks[:args.limit]
    if not chunks:
        log.info("Nothing to do: no chunks matched (all already done?).")
        return 0

    use_mask = not args.no_mask
    use_ref = not args.no_reference
    wf_name = ("vace_masked_depth_v2v_1p3b" if (use_mask and use_ref)
               else "vace_unmasked_compare")
    if wf_name == "vace_unmasked_compare":
        # That graph has neither a mask LoadVideo nor a reference LoadImage, so
        # dropping either one drops both. Say so rather than failing later when
        # a node that does not exist is patched.
        if use_mask or use_ref:
            log.warning("The ablation graph carries no mask and no reference; "
                        "--no-reference and --no-mask both select it. Running "
                        "with neither.")
        use_mask = use_ref = False
    wf_path = P.workflows / f"{wf_name}_api.json"
    if not wf_path.exists():
        log.error("Missing %s. Run scripts/build_workflows.py", wf_path)
        return 1
    log.info("Workflow: %s (mask=%s reference=%s)", wf_name, use_mask, use_ref)
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
    log.info("Pre-flight OK: depth%s present for all %d chunk(s)",
             " and masks" if use_mask else "", len(chunks))

    P.restored_480p.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    t_all = time.time()
    durations: list[float] = []

    for i, c in enumerate(chunks):
        cid = c["chunk_id"]
        tag = f"_{args.tag}" if args.tag else ""
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

            # ComfyUI LoadVideo reads only from its input directory
            src_name = stage(src_chunk, f"chunk_source_{cid}.mp4")
            dep_mp4 = P.comfy_input / f"chunk_depth_{cid}.mp4"
            run(["ffmpeg", "-y", "-v", "error", "-i", str(depth), "-c:v", "libx264",
                 "-crf", "12", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                 str(dep_mp4)], log)
            dep_name = dep_mp4.name
            if use_mask:
                msk_mp4 = P.comfy_input / f"chunk_mask_{cid}.mp4"
                run(["ffmpeg", "-y", "-v", "error", "-i", str(mask), "-c:v", "libx264",
                     "-crf", "8", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                     str(msk_mp4)], log)
            if use_ref:
                stage(ref_sheet, "reference_sheet.png")

            # ---- patch the workflow -------------------------------------------
            wf = load_api_workflow(wf_path)
            set_input(wf, "LoadVideo", "file", src_name, title_contains="source")
            set_input(wf, "LoadVideo", "file", dep_name, title_contains="depth")
            if use_mask:
                set_input(wf, "LoadVideo", "file", msk_mp4.name, title_contains="mask")
            if use_ref:
                set_input(wf, "LoadImage", "image", "reference_sheet.png")
            set_input(wf, "WanVaceToVideo", "width", c["width"])
            set_input(wf, "WanVaceToVideo", "height", c["height"])
            set_input(wf, "WanVaceToVideo", "length", c["n_frames"])
            set_input(wf, "KSampler", "seed", args.seed if args.seed is not None else c["seed"])
            if args.steps:
                set_input(wf, "KSampler", "steps", args.steps)
            set_input(wf, "CreateVideo", "fps", float(c["fps"]))
            set_input(wf, "SaveVideo", "filename_prefix", f"vace/{cid}{tag}")

            c["status"] = "running"
            c["attempts"] = c.get("attempts", 0) + 1
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

            dest = P.restored_480p / f"{cid}{tag}.mp4"
            shutil.move(str(produced), dest)

            c.update(status="done", error="", duration_sec=round(dt, 2),
                     peak_vram_mb=hist.get("_peak_vram_mb", 0),
                     output_path=str(dest.relative_to(P.root)))
            save_manifest(man)
            done += 1
            log.info("done in %s (%.2f s/frame, peak VRAM %s MiB) -> %s",
                     human_time(dt), dt / c["n_frames"], c["peak_vram_mb"], dest.name)

            eta = (len(chunks) - i - 1) * (sum(durations) / len(durations))
            log.info("progress %d/%d, eta %s", i + 1, len(chunks), human_time(eta))

        except Exception as e:
            failed += 1
            c.update(status="failed", error=str(e)[:2000])
            save_manifest(man)
            log.error("%s FAILED: %s", cid, str(e)[:1500])
        finally:
            for pat in (f"chunk_source_{cid}.mp4", f"chunk_depth_{cid}.mp4",
                        f"chunk_mask_{cid}.mp4"):
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
