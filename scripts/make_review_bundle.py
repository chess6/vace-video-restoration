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
    "compare/compare_720p_head/crop_frame026.png":
        "START HERE. Four panels, cut at 100% and never resized: the source "
        "upscaled to 720p with Lanczos, the SeedVR2 plate, VACE on that plate, "
        "and VACE with the LoRA. Measured over the pixels VACE actually "
        "regenerates, the plate is +70% sharper than the Lanczos upscale and "
        "VACE is +3.8% - i.e. indistinguishable from it. That is why the anchor "
        "region looks unrestored while the rest of the frame does not.",
    "compare/compare_720p_head/plate_7B_vs_lanczos_720p_h264.mp4":
        "The plate against the default upscale, side by side at native "
        "resolution. This is the comparison that decides whether the pipeline "
        "is worth running at all.",
    "compare/compare_720p_head/vace_plate_LoRA_vs_lanczos_720p_h264.mp4":
        "VACE+LoRA against the default upscale. Watch the anchor specifically: "
        "the background is plate and looks restored, the anchor region is "
        "regenerated and does not.",
    "lora/plate_background_aggressive.mp4":
        "START HERE. The SeedVR2 plate alone, no VACE. This is where the quality "
        "comes from: +231% sharpness over the source, against +167% for the VACE "
        "clips that sit on top of it. If VACE is not adding something you can "
        "see, the plate alone is the better output.",
    "lora/shot0000_c000_loraE.mp4":
        "VACE on the plate, NO LoRA. The honest comparison for loraD.",
    "lora/shot0000_c000_loraD.mp4":
        "VACE on the plate, WITH the subject LoRA. Same run as loraE in every "
        "other respect. Measured 0.177 match against references the LoRA "
        "never saw, versus 0.202 without it - no improvement. Your eye is the "
        "discriminator: if you can tell D from E, the metric is wrong.",
    "lora/shot0000_c000_loraB.mp4":
        "Earlier arm, NO plate - kept only to show what the missing plate cost. "
        "These measured 5% SOFTER than the source, which is why the first batch "
        "looked no better than the original.",
    "lora/shot0000_c000_loraA.mp4":
        "Earlier arm, no plate, with LoRA. Same caveat as loraB.",
    "lora/shot0000_c000_loraC.mp4":
        "The same LoRA at strength 2.0. Measured WORSE (0.126) and an anchor was "
        "detectable in fewer frames, which usually means the region is being "
        "damaged. Look for a waxy or smeared anchor rather than a different one.",
    "lora/match_scores.json":
        "Match per training checkpoint, scored against the held-out references "
        "only. This is where the LoRA does work: 0.023 with no LoRA, 0.517 at the "
        "shipped checkpoint, 0.745 for the held-out references themselves.",
    "lora/vace_match.json":
        "The same measurement on the three clips above. This is where it "
        "stops working.",
    "pilot_grid.mp4":
        "START HERE. All variants side by side, same frames. Scan for: which "
        "environment looks like the same place; whether the subject looks pasted "
        "on; any rim or glow along the subject's edge.",
    "variants/lanczos_original.mp4":
        "The baseline. Everything else has to beat this to be worth its runtime.",
    "variants/seedvr2_conservative.mp4":
        "Background restoration alone, fidelity-first. Check signage, text and "
        "any distant anchors against the baseline - those are where invented "
        "detail does real damage.",
    "variants/seedvr2_aggressive.mp4":
        "Background restoration alone, detail-first. Measured as the least "
        "temporally stable and the most inventive variant. Compare its signage "
        "and distant anchors to the conservative one before trusting it.",
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
        "Tracking review sheets. Does the red region cover the WHOLE subject - "
        "its full extent, its extremities, anything carried - and nothing else? "
        "If it caught the wrong candidate, every subject judgement below is "
        "meaningless.",
    "references/shot0000_reference_pack.png":
        "THE image VACE is actually conditioned on for this shot. Three panels: "
        "a match view, the attribute taken from this interval's own footage, and "
        "an alternate anchor orientation. The two external panels must show the "
        "anchor region on flat grey - if you can see attributes in either of "
        "them, the attribute authority has leaked and everything below is void.",
    "references/shot0000_pack.json":
        "Provenance for that image: which references, why they were chosen, "
        "their match agreement, and the measured colour distance between "
        "their attributes and this interval's. That distance is a diagnostic only.",
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
    # A relative --out is resolved against the project, not the caller's cwd:
    # the final log line calls rel(), which raises on a path outside the project
    # AFTER the archive has been written - a crash that looks like a failed
    # bundle when the bundle is already there, complete.
    if not out.is_absolute():
        out = P.root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = P.intermediate / "_review_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    manifest_note = ""
    man: dict = {}
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
        # attributes was really stripped. The global sheet below is a fallback and
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

        # ---- subject-LoRA arms -------------------------------------------------
        # These differ from each other in ONE setting, so they belong together
        # and apart from the variants above: comparing a LoRA arm against a
        # pilot variant would also be comparing two different prompts.
        for v in sorted(P.restored_480p.glob("*_lora*.mp4")):
            arc = f"lora/{v.name}"
            sz = add_video(zf, v, arc, tmp, args.crf, args.lossless, log)
            included.append((arc, sz))
        for j in ("match_scores.json", "vace_match.json",
                  "vace_match_plate.json"):
            p = P.intermediate / "lora_eval" / j
            if p.exists():
                zf.write(p, f"lora/{j}")
                included.append((f"lora/{j}", p.stat().st_size))
        # The plate the arms sit on. Without it in the same archive the reviewer
        # is comparing generated clips against memory, and the question that
        # matters here - does VACE add anything over the plate alone - cannot be
        # answered by looking at the VACE clips.
        seen_plates = set()
        for c in man.get("chunks") or []:
            for profile, entry in (c.get("background") or {}).items():
                rel_p = entry.get("path") if isinstance(entry, dict) else entry
                if not rel_p or profile in seen_plates:
                    continue
                src = P.root / rel_p
                if not src.exists():
                    continue
                seen_plates.add(profile)
                arc = f"lora/plate_{profile}." + ("mkv" if args.lossless else "mp4")
                sz = add_video(zf, src, arc, tmp, args.crf, args.lossless, log)
                included.append((arc, sz))

        # ---- 720p comparison against the default upscale -----------------------
        for d in sorted(P.outputs.glob("compare_720p*")):
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.png")) + sorted(d.glob("*.json")):
                zf.write(p, f"compare/{d.name}/{p.name}")
                included.append((f"compare/{d.name}/{p.name}", p.stat().st_size))
            for v in sorted(d.glob("*_h264.mp4")):
                arc = f"compare/{d.name}/{v.name}"
                # Already H.264 at the bundle's own quality; re-encoding a
                # side-by-side to judge sharpness would be judging the encoder.
                zf.write(v, arc)
                included.append((arc, v.stat().st_size))

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
