#!/usr/bin/env python
"""Collect everything that needs a human eye into one zip.

The pipeline never opens a viewer (CLAUDE.md rule 1), so every judgement call
depends on you finding the right files. This gathers them into a single archive
with a README explaining what each one is and what to look for, and prints the
path. It does not open anything.

Videos are transcoded to H.264 for the bundle: the working copies are lossless
FFV1 in Matroska, which many players will not touch, and eight of them is ~90 MB.
The transcode is visually transparent at the default CRF, and the lossless
originals stay where they were. If you are chasing a subtle edge artefact and
want to be certain the codec is not responsible, pass --lossless.

    scripts/make_review_bundle.py
    scripts/make_review_bundle.py --lossless --out /tmp/review.zip

The archive contains material derived from your footage, so it is written under
outputs/ (git-ignored, like every zip in this project) and must never be
committed or uploaded anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, human_size, load_manifest, probe_frames, rel, run, setup_logging,
)

# What to look for in each artefact. Keyed by the name inside the archive.
NOTES = {
    "pilot_grid.mp4":
        "START HERE. All variants side by side, same frames. Scan for: which "
        "environment looks like the same place; whether the figure looks pasted "
        "on; any rim or glow along the figure's edge.",
    "variants/lanczos_original.mp4":
        "The baseline. Everything else has to beat this to be worth its runtime.",
    "variants/seedvr2_conservative.mp4":
        "Background restoration alone, fidelity-first. Check signage, text and "
        "any distant faces against the baseline - those are where invented "
        "detail does real damage.",
    "variants/seedvr2_aggressive.mp4":
        "Background restoration alone, detail-first. Measured as the least "
        "temporally stable and the most inventive variant. Compare its signage "
        "and distant faces to the conservative one before trusting it.",
    "variants/vace_over_original.mp4":
        "Subject replacement with NO background stage. Measured as the only "
        "variant steadier than its own source. If the environment does not "
        "actually need rescuing, this is the cheap answer.",
    "variants/vace_pathA_conservative.mp4":
        "VACE preserving the conservative plate itself. Measured to hold only "
        "~65% of background pixels - look for the environment drifting.",
    "variants/vace_pathB_conservative.mp4":
        "Same subject composited onto that plate. Background is held verbatim. "
        "This is the configured default; check its edge against path A.",
    "variants/vace_pathA_aggressive.mp4":
        "VACE preserving the aggressive plate itself. Highest measured halo.",
    "variants/vace_pathB_aggressive.mp4":
        "Same subject composited onto the aggressive plate. Best measured "
        "sharpness balance of any variant, but the most invented detail.",
    "masks/":
        "Tracking review sheets. Does the red region cover the WHOLE figure - "
        "hair, hands, feet, bag - and nothing else? If it caught the wrong "
        "person, every subject judgement below is meaningless.",
    "references/shot0000_reference_pack.png":
        "THE image VACE is actually conditioned on for this shot. Three panels: "
        "an identity view, the garment taken from this interval's own footage, "
        "and an alternate head angle. The two external panels must show face, "
        "hair and skin on flat grey - if you can see clothing in either of "
        "them, the garment authority has leaked and everything below is void.",
    "references/shot0000_pack.json":
        "Provenance for that image: which photographs, why they were chosen, "
        "their identity agreement, and the measured colour distance between "
        "their clothes and this interval's. That distance is a diagnostic only.",
    "references/reference_sheet.png":
        "The older GLOBAL sheet, kept only as a fallback for shots without a "
        "pack. If a pack exists above, judge that instead - this one may "
        "predate it.",
    "references/contact_sheet.png":
        "Every candidate reference, kept and rejected. Did anything good get "
        "dropped, or anything bad kept?",
    "reports/pilot_findings.md":
        "START HERE. What was measured, what it means, what is NOT concluded, "
        "and one earlier claim corrected.",
    "reports/":
        "Measurements, not verdicts. pilot_metrics.json has the numbers behind "
        "the notes above; pipeline_estimates.md has runtime, VRAM and disk.",
}


def add_video(zf: zipfile.ZipFile, src: Path, arc: str, tmp: Path, crf: int,
              lossless: bool, log) -> int:
    """Put one video in the archive, transcoded unless --lossless."""
    if lossless:
        zf.write(src, arc)
        return src.stat().st_size
    out = tmp / (Path(arc).stem + ".mp4")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c:v", "libx264",
         "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out)], log)
    n_in, n_out = probe_frames(src), probe_frames(out)
    if n_in != n_out:
        raise RuntimeError(f"{src.name}: transcode changed {n_in} -> {n_out} frames")
    zf.write(out, arc)
    size = out.stat().st_size
    out.unlink(missing_ok=True)
    return size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--crf", type=int, default=12,
                    help="H.264 quality for the bundled copies (lower = better)")
    ap.add_argument("--lossless", action="store_true",
                    help="Bundle the original FFV1 files instead of transcoding")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("make_review_bundle", args.verbose)
    out = args.out or (P.outputs / "review_bundle.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = P.intermediate / "_review_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    manifest_note = ""
    try:
        man = load_manifest()
        n = man["normalized"]
        pilot = man.get("pilot") or {}
        manifest_note = (
            f"Pilot interval: frames {pilot.get('start_frame')}-{pilot.get('end_frame')} "
            f"({pilot.get('duration_sec')}s) at {n['width']}x{n['height']} @ {n['fps']} fps.\n"
            f"Every variant covers this same interval with the same mask, depth, "
            f"reference sheet, prompt and VACE seed.\n")
    except Exception as e:                       # a bundle is still useful without it
        log.warning("Could not read the manifest (%s); bundling anyway.", e)

    included: list[tuple[str, int]] = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        # ---- comparison videos ------------------------------------------------
        comp = P.comparisons
        grid = comp / "pilot_grid.mp4"
        if grid.exists() and grid.stat().st_size > 0:
            sz = add_video(zf, grid, "pilot_grid.mp4", tmp, args.crf, args.lossless, log)
            included.append(("pilot_grid.mp4", sz))
        for v in sorted(comp.glob("*.mkv")):
            arc = f"variants/{v.stem}." + ("mkv" if args.lossless else "mp4")
            sz = add_video(zf, v, arc, tmp, args.crf, args.lossless, log)
            included.append((arc, sz))

        # ---- stills that need judging ----------------------------------------
        # The PER-SHOT packs first. These are what actually conditions the
        # generation, and the only artefact that shows whether the external
        # clothing was really stripped. The global sheet below is a fallback and
        # is often older than the pack, so shipping it alone invites the reviewer
        # to check the wrong image.
        pack_dir = P.intermediate / "reference_packs"
        if pack_dir.exists():
            for p in sorted(pack_dir.glob("*_reference_pack.png")):
                zf.write(p, f"references/{p.name}")
                included.append((f"references/{p.name}", p.stat().st_size))
            for p in sorted(pack_dir.glob("*_pack.json")):
                zf.write(p, f"references/{p.name}")
                included.append((f"references/{p.name}", p.stat().st_size))
        for sheet in ("reference_sheet.png", "contact_sheet.png"):
            p = P.reference_sheets / sheet
            if p.exists():
                zf.write(p, f"references/{sheet}")
                included.append((f"references/{sheet}", p.stat().st_size))
        review = P.masks / "review"
        if review.exists():
            for p in sorted(review.glob("*.png")):
                zf.write(p, f"masks/{p.name}")
                included.append((f"masks/{p.name}", p.stat().st_size))

        # ---- reports ----------------------------------------------------------
        for rp in ("pilot_findings.md", "pilot_metrics.json",
                   "pipeline_estimates.md", "tracking_report.json",
                   "background_runtime.json", "pilot_variants.json"):
            p = P.reports / rp
            if p.exists():
                zf.write(p, f"reports/{rp}")
                included.append((f"reports/{rp}", p.stat().st_size))
        tmpl = P.root / "reports" / "pilot_results.md"
        if tmpl.exists():
            zf.write(tmpl, "reports/pilot_results_TEMPLATE.md")
            included.append(("reports/pilot_results_TEMPLATE.md", tmpl.stat().st_size))

        # ---- the guide ---------------------------------------------------------
        lines = [
            "WHAT TO LOOK AT, AND WHAT FOR",
            "=" * 60, "",
            f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}.",
            manifest_note,
            "Nothing here was opened or judged automatically. The numbers in "
            "reports/ are measurements; whether the result is GOOD is the one "
            "thing a metric cannot tell you.",
            "",
            "Suggested order: pilot_grid.mp4 first, then masks/ (if the mask is "
            "wrong nothing else matters), then the individual variants.",
            "",
        ]
        for arc, _ in included:
            note = NOTES.get(arc) or NOTES.get(arc.split("/")[0] + "/")
            if note:
                lines.append(arc)
                lines.append(textwrap.fill(note, 76, initial_indent="    ",
                                           subsequent_indent="    "))
                lines.append("")
        lines += ["", "-" * 60,
                  "When you have decided, record it in "
                  "reports/pilot_results.md (template included).",
                  "",
                  "This archive contains material derived from your footage. It "
                  "is written to a git-ignored location; do not commit or upload it."]
        zf.writestr("README.txt", "\n".join(lines) + "\n")

    total = sum(s for _, s in included)
    log.info("Bundled %d file(s), %s of content -> %s (%s)",
             len(included), human_size(total), rel(out), human_size(out.stat().st_size))
    for arc, sz in included:
        log.info("   %-46s %s", arc, human_size(sz))
    log.info("Open it yourself; nothing was displayed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
