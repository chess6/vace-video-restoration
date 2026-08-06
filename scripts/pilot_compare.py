#!/usr/bin/env python
"""Phase 10 - build the controlled background/subject pilot comparison.

Every variant covers the SAME interval and uses the SAME mask, depth, reference
sheet, prompt and VACE seed. The only thing that changes between them is what
the environment is restored with, and how the subject is integrated onto it.

Variants:
  1 lanczos_original          the original, Lanczos-enlarged to working geometry
                              (this is exactly the stream VACE takes as source)
  2 seedvr2_conservative      full-frame SeedVR2, structural/temporal fidelity
  3 seedvr2_aggressive        full-frame SeedVR2, stronger invented detail
  4 vace_over_original        VACE subject, original preserved outside the mask
  5 vace_pathA_conservative   VACE preserving the conservative plate directly
    vace_pathB_conservative   VACE subject composited onto that plate afterwards
  6 vace_pathA_aggressive     VACE preserving the aggressive plate directly
    vace_pathB_aggressive     VACE subject composited onto that plate afterwards

Path A vs Path B is the integration question: A trusts VACE to leave the
preserved region alone, B guarantees the background bit-exactly and manages an
edge instead. Both are built so scripts/evaluate_pilot.py can measure which
actually holds its background and which has the cleaner edge.

This script does NOT generate: it expects run_chunks.py to have produced the
VACE outputs and restore_background.py the plates. It assembles, composites and
writes reports/pilot_variants.json.

    scripts/pilot_compare.py [--skip-missing]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, human_size, load_config, load_manifest, pilot_chunks, pilot_interval,
    probe_dims_fps, probe_frames, rel, run, setup_logging, slice_frames,
)

# variant -> (vace tag or None, background profile or None, path, description)
# tag None means "not a VACE variant". Paths:
#   "A" preserved in-VACE      "B" composited on the subject mask
#   "R" ROI, mapped back       "P" composited on the PROTECTED submask
# "P" exists because the retained pipeline runs run_chunks --protected, which
# regenerates only the anchor region. Compositing that on the SUBJECT mask would take the
# whole subject from VACE's VAE round-trip of the plate and lose ~8% of the
# attribute's detail for nothing; the protected submask is the correct alpha.
# Without a P entry the protected comparison was not reproducible through this
# script at all - it only existed as commands typed by hand.
PLAN = [
    ("lanczos_original", None, None, None,
     "Original, Lanczos-enlarged to working geometry. The baseline everything "
     "else has to beat."),
    ("seedvr2_conservative", None, "background_conservative", None,
     "SeedVR2 full-frame restoration only, no subject replacement."),
    ("seedvr2_aggressive", None, "background_aggressive", None,
     "SeedVR2 full-frame restoration only, no subject replacement."),
    ("vace_over_original", "baseline", None, "A",
     "VACE subject with the ORIGINAL preserved outside the mask."),
    ("vace_pathA_conservative", "bg_conservative", "background_conservative", "A",
     "VACE preserving the conservative plate directly (control_video outside "
     "the mask IS the restored background)."),
    ("vace_pathB_conservative", "baseline", "background_conservative", "B",
     "Baseline VACE subject composited onto the conservative plate with the "
     "lossless mask and a narrow band."),
    ("vace_pathA_aggressive", "bg_aggressive", "background_aggressive", "A",
     "VACE preserving the aggressive plate directly."),
    ("vace_pathB_aggressive", "baseline", "background_aggressive", "B",
     "Baseline VACE subject composited onto the aggressive plate."),
    ("vace_protected_conservative", "prot_bg_conservative",
     "background_conservative", "P",
     "Protected run: VACE regenerates only the confidently exposed anchor region, "
     "composited onto the conservative plate using that same submask, so the "
     "attribute stays plate-exact."),
    ("vace_protected_aggressive", "prot_bg_aggressive",
     "background_aggressive", "P",
     "Protected run over the aggressive plate."),
    ("vace_roi_conservative", "roi", "background_conservative", "R",
     "Subject generated on the stabilized ROI crop, then mapped back to full "
     "frame over the conservative plate. Only the masked subject comes from the "
     "ROI pass: it saw a different framing, so its idea of the background is "
     "not comparable and must not leak in."),
]


def vace_output(c: dict, tag: str) -> Path | None:
    """Where run_chunks.py put one variant's generation."""
    runs = c.get("runs", {})
    if tag in runs and runs[tag].get("output_path"):
        return P.root / runs[tag]["output_path"]
    if tag == "baseline" and c.get("output_path"):
        return P.root / c["output_path"]
    return None


def plate_path(c: dict, profile: str) -> Path | None:
    bg = (c.get("background") or {}).get(profile)
    return (P.root / bg["path"]) if bg else None


def assemble_interval(chunks: list[dict], sources: list[Path], a: int, b: int,
                      fps: int, work: Path, dst: Path, log) -> Path:
    """Stitch per-chunk videos onto the timeline and cut exactly [a, b).

    A pilot interval can span several chunks, and the last window of a shot is
    snapped backwards so two of them may overlap by nearly their whole length.
    Reuses the assembler's planner rather than reimplementing that arithmetic.
    """
    from assemble import stitch_shot
    if len(chunks) == 1 and int(chunks[0]["start_frame"]) == a and \
            int(chunks[0]["end_frame"]) == b:
        run(["ffmpeg", "-y", "-v", "error", "-i", str(sources[0]),
             "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p", "-an",
             str(dst)], log)
        return dst

    # stitch_shot reads output_path off each chunk, so point it at this variant's
    # files without disturbing the manifest.
    shim = [{**c, "output_path": str(p)} for c, p in zip(chunks, sources)]
    stitched = dst.with_name(dst.stem + "_stitched.mkv")
    span_start, n = stitch_shot(shim, fps, stitched, log, work=work)
    lo, hi = a - span_start, b - span_start
    if lo < 0 or hi > n:
        raise RuntimeError(f"interval {a}-{b} outside assembled span "
                           f"{span_start}-{span_start + n}")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(stitched), "-vf",
         f"trim=start_frame={lo}:end_frame={hi},setpts=PTS-STARTPTS",
         "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p", "-an", str(dst)], log)
    stitched.unlink(missing_ok=True)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-missing", action="store_true",
                    help="Build what exists instead of failing on the first gap")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("pilot_compare", args.verbose)
    cfg = load_config(args.config)
    man = load_manifest()
    pilot = man.get("pilot")
    if not pilot:
        log.error("No pilot recorded. Run scripts/extract_pilot.py first.")
        return 1
    pchunks = pilot_chunks(man)
    if not pchunks:
        log.error("No chunk intersects the pilot interval.")
        return 1
    a, b = pilot_interval(man)
    n_expect = b - a
    out_dir = P.comparisons
    out_dir.mkdir(parents=True, exist_ok=True)
    work = P.root / man["normalized"]["work_path"]
    fps = int(man["normalized"]["fps"])
    w, h = int(man["normalized"]["width"]), int(man["normalized"]["height"])
    log.info("Pilot interval %d-%d (%d frames, %.4fs) at %dx%d @ %d fps across "
             "%d chunk(s): %s", a, b, n_expect, n_expect / fps, w, h, fps,
             len(pchunks), ", ".join(c["chunk_id"] for c in pchunks))
    origin = {k: v for k, v in (man.get("pilot") or {}).items()
              if k.startswith("origin")}
    if origin:
        log.info("Origin: %s %.3f-%.3fs%s", origin.get("origin_source"),
                 origin.get("origin_start_sec", 0), origin.get("origin_end_sec", 0),
                 " (exact)" if origin.get("origin_exact") else "")

    variants: dict[str, dict] = {}
    missing: list[str] = []

    for name, tag, profile, path, describes in PLAN:
        dst = out_dir / f"{name}.mkv"
        plate = plate_path(pchunks[0], profile) if profile else None

        # Resolve this variant's per-chunk sources across EVERY pilot chunk.
        parts: list[Path] = []
        gap = None
        for c in pchunks:
            if tag is None and profile is None:
                p = (P.root / c["control_path"]).with_suffix(".mp4")
            elif tag is None:
                p = plate_path(c, profile)
            elif path == "A":
                p = vace_output(c, tag)
            else:                                  # composited/mapped below
                p = None
            if path not in ("B", "R"):
                if p is None or not Path(p).exists():
                    gap = (f"{name}: {c['chunk_id']} missing "
                           + (f"VACE output for tag {tag!r}" if tag
                              else f"plate for {profile!r}"))
                    break
                parts.append(Path(p))
        if gap:
            missing.append(gap)
            continue

        if path == "R":
            # The ROI pass generated on a crop. map_roi_back.py is the exact
            # inverse of the warp and lays only the masked subject back over the
            # plate, so this variant differs from path A in subject resolution
            # and in nothing else.
            for c in pchunks:
                roi_out = vace_output(c, tag)
                pl = plate_path(c, profile)
                if roi_out is None or not roi_out.exists() or pl is None \
                        or not pl.exists():
                    gap = (f"{name}: {c['chunk_id']} needs tag {tag!r} and plate "
                           f"{profile!r}")
                    break
                part = out_dir / f"_{name}_{c['chunk_id']}.mkv"
                r = run([str(P.venv_python), str(P.scripts / "map_roi_back.py"),
                         "--shot", c["shot_id"], "--roi-output", str(roi_out),
                         "--background", str(pl), "--out", str(part)],
                        log, check=False)
                if r.returncode != 0:
                    gap = (f"{name}: ROI map-back failed "
                           f"({(r.stderr or '').strip()[-200:]})")
                    break
                parts.append(part)
            if gap:
                missing.append(gap)
                continue

        if path in ("B", "P"):
            for c in pchunks:
                subject = vace_output(c, tag)
                pl = plate_path(c, profile)
                if subject is None or not subject.exists() or pl is None or not pl.exists():
                    gap = f"{name}: {c['chunk_id']} needs tag {tag!r} and plate {profile!r}"
                    break
                extra = []
                if path == "P":
                    shot = next((x for x in man["shots"]
                                 if x["shot_id"] == c["shot_id"]), None)
                    pm = ((shot or {}).get("protected_mask") or {}).get("path")
                    if not pm or not (P.root / pm).exists():
                        gap = (f"{name}: {c['shot_id']} has no protected submask; "
                               f"run scripts/make_protected_mask.py")
                        break
                    extra = ["--mask", str(P.root / pm)]
                part = out_dir / f"_{name}_{c['chunk_id']}.mkv"
                r = run([str(P.venv_python), str(P.scripts / "composite_subject.py"),
                         "--chunk", c["chunk_id"], "--subject", str(subject),
                         "--background", str(pl), "--out", str(part)] + extra,
                        log, check=False)
                if r.returncode != 0:
                    gap = f"{name}: compositing failed ({(r.stderr or '').strip()[-200:]})"
                    break
                parts.append(part)
            if gap:
                missing.append(gap)
                continue
            log.info("%-28s composited %d chunk(s) over %s", name, len(parts), profile)

        try:
            assemble_interval(pchunks, parts, a, b, fps, work, dst, log)
        except Exception as e:
            missing.append(f"{name}: assembly failed ({e})")
            continue
        finally:
            for part in parts:
                if part.name.startswith("_"):
                    part.unlink(missing_ok=True)

        got = probe_frames(dst)
        gw, gh, _ = probe_dims_fps(dst)
        if got != n_expect or (gw, gh) != (w, h):
            missing.append(f"{name}: {got} frames at {gw}x{gh}, "
                           f"expected {n_expect} at {w}x{h}")
            continue
        variants[name] = {
            "path": rel(dst),
            "plate": rel(plate) if plate else None,
            "vace_tag": tag, "profile": profile, "integration": path,
            "describes": describes,
            "bytes": dst.stat().st_size,
        }
        log.info("%-28s %s  %d frames  %s", name, dst.name, got,
                 human_size(dst.stat().st_size))

    if missing:
        log.warning("%d variant(s) could not be built:", len(missing))
        for m in missing:
            log.warning("  %s", m)
        if not args.skip_missing:
            log.error("Refusing to write a partial comparison. Generate the "
                      "missing pieces, or pass --skip-missing.")
            return 1

    # One mask covering the whole interval, so the evaluator measures inside and
    # outside the subject over the same frames as the variants. Built from the
    # per-SHOT masks, which are continuous, rather than the per-chunk slices,
    # which overlap.
    interval_mask = out_dir / "_interval_mask.mkv"
    shot_ids = sorted({c["shot_id"] for c in pchunks})
    if len(shot_ids) == 1:
        sm = P.masks / f"{shot_ids[0]}_mask.mkv"
        shot = next(s for s in man["shots"] if s["shot_id"] == shot_ids[0])
        if sm.exists():
            lo = a - int(shot["start_frame"])
            slice_frames(sm, interval_mask, lo, lo + n_expect, fps, log,
                         lossless=True, gray=True)
        else:
            log.warning("No mask for %s; metrics will have no subject region.",
                        shot_ids[0])
    else:
        log.warning("Pilot spans %d shots (%s); per-shot mask stitching is not "
                    "implemented, so metrics will use the first shot's mask.",
                    len(shot_ids), ", ".join(shot_ids))
        sm = P.masks / f"{shot_ids[0]}_mask.mkv"
        shot = next(s for s in man["shots"] if s["shot_id"] == shot_ids[0])
        if sm.exists():
            lo = max(0, a - int(shot["start_frame"]))
            slice_frames(sm, interval_mask, lo, lo + n_expect, fps, log,
                         lossless=True, gray=True)

    spec = {"chunks": [c["chunk_id"] for c in pchunks], "pilot": pilot,
            "interval": {"start_frame": a, "end_frame": b, "frames": n_expect},
            "source": rel(out_dir / "lanczos_original.mkv"),
            "mask": rel(interval_mask),
            "geometry": {"width": w, "height": h, "fps": fps, "frames": n_expect},
            "variants": variants, "missing": missing}
    rp = P.reports / "pilot_variants.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(spec, indent=2) + "\n")
    log.info("Built %d variant(s) -> %s", len(variants), rel(rp))

    # Side-by-side sheets, written to disk only. Nothing is ever displayed.
    if len(variants) >= 2:
        names = list(variants)
        # Prefer a layout that needs no padding: a synthetic `color` source has
        # no natural duration and xstack then writes nothing at all. Fall back to
        # dropping the remainder rather than padding.
        cols = next((c for c in range(min(4, len(names)), 1, -1)
                     if len(names) % c == 0), len(names))
        rows = len(names) // cols
        names = names[:rows * cols]
        # hstack each row then vstack the rows, rather than one xstack with a
        # computed layout string: same result, far less that can go quietly wrong.
        tw, th = (w // 2) // 2 * 2, (h // 2) // 2 * 2      # half size, even
        inputs, filt = [], []
        for i, n in enumerate(names):
            inputs += ["-i", str(P.root / variants[n]["path"])]
            filt.append(f"[{i}:v]scale={tw}:{th},drawtext=text='{n}':"
                        f"x=6:y=6:fontsize=14:fontcolor=white:box=1:"
                        f"boxcolor=black@0.5[v{i}]")
        for r_i in range(rows):
            ins = "".join(f"[v{r_i * cols + ci}]" for ci in range(cols))
            filt.append(f"{ins}hstack=inputs={cols}[row{r_i}]")
        rows_in = "".join(f"[row{r_i}]" for r_i in range(rows))
        filt.append(f"{rows_in}vstack=inputs={rows}[out]"
                    if rows > 1 else f"[row0]copy[out]")
        grid = out_dir / "pilot_grid.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex",
               ";".join(filt), "-map", "[out]", "-c:v", "libx264", "-crf", "16",
               "-preset", "medium", "-pix_fmt", "yuv420p", str(grid)]
        r = run(cmd, log, check=False)
        if r.returncode == 0 and grid.exists() and grid.stat().st_size > 0:
            log.info("Comparison grid -> %s (%s)", rel(grid),
                     human_size(grid.stat().st_size))
        else:
            # Remove the stub ffmpeg leaves behind, so a zero-byte file is never
            # mistaken for a deliverable.
            grid.unlink(missing_ok=True)
            log.warning("Grid build failed (%s); the individual variants are "
                        "still in %s",
                        (r.stderr or "").strip().splitlines()[-1:] or "no stderr",
                        rel(out_dir))

    log.info("Next: scripts/evaluate_pilot.py, then judge them yourself and "
             "fill in reports/pilot_results.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
