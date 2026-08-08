#!/usr/bin/env python
"""Ask the user, per training crop, whether the region is visible AFTER the trainer sees it.

THE QUESTION THIS EXISTS TO ANSWER. Adaptation is the one lever shown to move the
problematic region, and it moved it using a dataset that was never built to
contain the region and has never been checked for whether it does. Whether the
training material holds usable evidence of the region **after the trainer's
resize and crop** is the difference between "the ceiling is the model" and "the
ceiling is the dataset". Nothing generated can settle it.

WHY IT IS A MANIFEST AND NOT AN AUTOMATED AUDIT. CLAUDE.md rule 2b: an agent may
not inspect the content of the user's media. No detector here can report whether
the region is present either - that is the same saturation problem that has made
every metric in this project blind to the defect. So the pixels are transformed
programmatically, written to disk, and judged by the user, who may look.

WHAT MAKES IT HONEST. The crops are shown AS THE TRAINER SEES THEM: resized into
the configured bucket and centre-cropped exactly as the training pipeline does.
Judging the originals would answer a different question - a region can be plainly
visible at full size and be resampled into four pixels by the time it reaches the
optimizer.

Writes a directory of transformed crops plus a CSV with one row per crop, for the
user to mark present / partial / absent. Displays nothing (rule 1); no filename,
count or content is echoed anywhere tracked.

    scripts/build_evidence_manifest.py --dataset intermediate/lora_dataset \\
        --bucket 1024 --out intermediate/evidence_audit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("VACE_ROOT", Path(__file__).resolve().parent.parent))
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def die(m: str) -> None:
    print(f"FATAL: {m}", file=sys.stderr)
    raise SystemExit(1)


def trainer_view(im, bucket: int):
    """Resize and centre-crop exactly as the trainer does before the optimizer.

    Short side to `bucket`, then a centre crop to `bucket` square. That is the
    standard bucketed pipeline, and it is the transform that decides how many
    pixels the region actually contributes - which is the whole point of judging
    the transformed crop rather than the original.
    """
    from PIL import Image
    w, h = im.size
    s = bucket / min(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    w, h = im.size
    left, top = (w - bucket) // 2, (h - bucket) // 2
    return im.crop((left, top, left + bucket, top + bucket))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True,
                    help="the training dataset directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bucket", type=int, default=1024,
                    help="the trainer's bucket resolution. Read it from the "
                         "training config; guessing changes the answer.")
    ap.add_argument("--grid", type=int, default=128,
                    help="coordinate grid spacing, so a box can be read off")
    args = ap.parse_args()

    if not args.dataset.exists():
        die(f"{args.dataset} does not exist")
    from PIL import Image, ImageDraw

    crops = sorted(p for p in args.dataset.rglob("*")
                   if p.suffix.lower() in IMAGE_EXTS)
    if not crops:
        die(f"no images under {args.dataset}")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(crops):
        im = Image.open(p).convert("RGB")
        w0, h0 = im.size
        view = trainer_view(im, args.bucket)
        d = ImageDraw.Draw(view, "RGBA")
        for x in range(0, args.bucket + 1, args.grid):
            d.line([(x, 0), (x, args.bucket)], fill=(255, 0, 0, 120), width=1)
            d.line([(0, x), (args.bucket, x)], fill=(255, 0, 0, 120), width=1)
            d.text((x + 3, 3), str(x), fill=(255, 60, 60, 255))
            d.text((3, x + 3), str(x), fill=(255, 60, 60, 255))
        # Numbered, never named. The crop filenames derive from the user's
        # material (rule 2a), so the manifest keys on an index and a digest and
        # the reviewer matches by looking, not by reading a name.
        cid = f"crop_{i:03d}"
        view.save(args.out / f"{cid}.png")
        rows.append({
            "crop_id": cid,
            "sha256_12": hashlib.sha256(p.read_bytes()).hexdigest()[:12],
            "original_px": f"{w0}x{h0}",
            "trainer_px": f"{args.bucket}x{args.bucket}",
            "downscale": f"{args.bucket / min(w0, h0):.3f}",
            "split": "holdout" if "holdout" in p.parts else "train",
            "region_visible": "",          # present | partial | absent
            "region_box_xyxy": "",         # optional, read off the grid
            "notes": "",
        })

    man = args.out / "evidence_manifest.csv"
    with man.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    (args.out / "README.txt").write_text(
        "EVIDENCE AUDIT - one question per crop.\n\n"
        "Each PNG is a training crop AS THE TRAINER SEES IT: resized into the\n"
        f"configured bucket ({args.bucket}px short side) and centre-cropped, with a\n"
        "coordinate grid burned in. Judging the originals would answer a different\n"
        "question - a region can be obvious at full size and be resampled into a\n"
        "handful of pixels by the time it reaches the optimizer.\n\n"
        "In evidence_manifest.csv, fill in region_visible for each row:\n\n"
        "  present  the region is clearly visible and its structure is legible\n"
        "  partial  it is in frame but too small, blurred, cropped or occluded\n"
        "           to show how it is actually shaped\n"
        "  absent   not in frame at all\n\n"
        "Optionally give region_box_xyxy, read off the grid, for any row marked\n"
        "present - that turns the audit into training signal rather than only a\n"
        "yes/no, because a focus-masked adaptation needs to know WHERE.\n\n"
        "WHY YOU AND NOT A SCRIPT. Rule 2b forbids an agent inspecting your media,\n"
        "and no detector in this project reports whether the region is present -\n"
        "that is the same blindness that has made every metric here useless on\n"
        "this defect.\n\n"
        "WHAT THE ANSWER DECIDES\n"
        "  mostly present  -> the dataset contains the region and a region-aware or\n"
        "                     focus-masked adaptation is the indicated next step\n"
        "  mostly partial  -> the region is there but not at usable resolution;\n"
        "                     the bucket or the crop strategy is the constraint\n"
        "  mostly absent   -> subject-specific accuracy CANNOT be learned from this\n"
        "                     dataset, and that has to be stated rather than worked\n"
        "                     around. New source material, or a fill model for the\n"
        "                     region alone.\n")

    print(f"{len(rows)} crop(s) -> {args.out}")
    print(f"manifest: {man}")
    print("Nothing was inspected to produce this; every judgement is yours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
