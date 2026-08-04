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
    P, human_size, load_config, load_manifest, probe_dims_fps, probe_frames, rel,
    run, setup_logging,
)

# variant -> (vace tag or None, background profile or None, path, description)
# tag None means "not a VACE variant"; path "A" = preserved in-VACE, "B" = composited.
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
    bg = c.get("background", {}).get(profile)
    return (P.root / bg["path"]) if bg else None


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
    cid = pilot["chunks"][0]
    c = next(x for x in man["chunks"] if x["chunk_id"] == cid)
    out_dir = P.comparisons
    out_dir.mkdir(parents=True, exist_ok=True)

    src_chunk = (P.root / c["control_path"]).with_suffix(".mp4")
    if not src_chunk.exists():
        log.error("Source chunk missing: %s. Run run_chunks.py (it slices it).", src_chunk)
        return 1
    n_expect = int(c["n_frames"])
    w, h, fps = probe_dims_fps(src_chunk)
    log.info("Pilot %s: %d frames at %dx%d @ %.0f fps", cid, n_expect, w, h, fps)

    variants: dict[str, dict] = {}
    missing: list[str] = []

    for name, tag, profile, path, describes in PLAN:
        dst = out_dir / f"{name}.mkv"
        plate = plate_path(c, profile) if profile else None

        if tag is None and profile is None:
            src = src_chunk                       # variant 1
        elif tag is None:
            src = plate                           # variants 2 and 3
        elif path == "A":
            src = vace_output(c, tag)             # VACE generated it directly
        else:
            src = None                            # variant needs compositing

        if path == "B":
            subject = vace_output(c, tag)
            if subject is None or not subject.exists() or plate is None or not plate.exists():
                missing.append(f"{name}: needs VACE tag {tag!r} and plate {profile!r}")
                continue
            log.info("%-28s compositing subject over %s", name, profile)
            r = run([str(P.venv_python), str(P.scripts / "composite_subject.py"),
                     "--chunk", cid, "--subject", str(subject),
                     "--background", str(plate), "--out", str(dst)],
                    log, check=False)
            if r.returncode != 0:
                missing.append(f"{name}: compositing failed ({r.stderr.strip()[-200:]})")
                continue
        else:
            if src is None or not Path(src).exists():
                missing.append(f"{name}: missing "
                               + (f"VACE output for tag {tag!r}" if tag
                                  else f"plate for {profile!r}"))
                continue
            # Normalise every variant to the same lossless container so the
            # metrics compare pixels, not codecs.
            run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                 "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p", "-an",
                 str(dst)], log)

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

    spec = {"chunk": cid, "pilot": pilot,
            "source": rel(src_chunk), "mask": c["mask_path"],
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
