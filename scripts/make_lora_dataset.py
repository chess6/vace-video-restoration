#!/usr/bin/env python
"""Phase 5c - export a subject-LoRA training set from the verified references.

A LoRA learns EVERYTHING in its training images, permanently. That makes the
dataset the whole safety question, and two choices follow from it.

**Head crops, not greyed panels.** make_reference_pack.py greys the wardrobe out
of a full-frame panel, which is right for conditioning: the model sees the panel
once and the grey is obviously not content. Training is different. A large flat
mid-grey field appears in every image, so the LoRA would learn "this person
comes with a grey background" and reproduce it. Cropping tight to the head
excludes the wardrobe by FRAMING instead of by masking, so there is nothing
artificial to learn. Same authority split, no grey.

**The verified face, never the largest.** Crops are taken around the instance
identity.resolve_targets agreed on, honouring the run's exclusion list. Cropping
around the biggest face in a group shot would train the LoRA on a stranger.

**A held-out split, because the alternative is unfalsifiable.** The identity bank
is built from these same photographs, so scoring a LoRA's output against a bank
that includes its own training images is the "self-comparison as evidence" trap
recorded in docs/STATE.md. A few images are therefore reserved, never trained
on, and exist solely to answer "did this actually move identity". The split is
deterministic and spread across the identity ranking, so the held-out set is
neither the best nor the worst photographs.

Originals are opened read-only (rule 2). Output lands under intermediate/, which
.gitignore denies wholesale, because filenames and crops of the user's material
must never reach a remote (rule 2a).

    scripts/make_lora_dataset.py [--holdout 3] [--margin 1.9] [--trigger TOKEN]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, setup_logging  # noqa: E402


def crop_head(im, box, margin: float, min_px: int):
    """A square crop centred on the verified face, widened to include hair.

    Square because trainers bucket by aspect ratio and a mixed-aspect set of 13
    images fragments into buckets too small to batch. Clamped to the image, so a
    face near an edge yields a smaller crop rather than a padded one - padding
    would be another constant artefact for the LoRA to learn.
    """
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) * margin / 2.0
    # bias upward: hair sits above the face box, chin needs less room
    cy -= half * 0.10
    L, T = int(round(cx - half)), int(round(cy - half))
    Rt, B = int(round(cx + half)), int(round(cy + half))
    L, T = max(0, L), max(0, T)
    Rt, B = min(im.width, Rt), min(im.height, B)
    if Rt - L < min_px or B - T < min_px:
        return None
    return im.crop((L, T, Rt, B))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", type=int, default=3,
                    help="Images reserved for evaluation, never trained on")
    ap.add_argument("--margin", type=float, default=1.9,
                    help="Crop size as a multiple of the face box")
    ap.add_argument("--min-px", type=int, default=256,
                    help="Reject crops smaller than this")
    ap.add_argument("--max-px", type=int, default=1024,
                    help="Downscale crops larger than this; never upscale")
    ap.add_argument("--trigger", default=None,
                    help="Caption token written beside each image. Omit for no "
                         "captions; some trainers prefer filename-derived ones.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("make_lora_dataset", args.verbose)
    from PIL import Image, ImageOps
    import track_subject as T
    from identity import load_exclusions, reference_files, resolve_targets

    models = T.Models(log)
    res = resolve_targets(reference_files(), models, log, load_exclusions(log))
    per = res.get("per_image") or {}
    if not per:
        log.error("No verified target face in any reference. Refusing to build a "
                  "training set: a LoRA trained on an unverified face bakes the "
                  "wrong person into the weights permanently.")
        return 1

    # Rank by consensus agreement so the split can be spread across it rather
    # than reserving the best or the worst images.
    ranked = sorted(per.values(), key=lambda v: -float(v.get("agreement") or 0.0))

    out_dir = args.out or (P.intermediate / "lora_dataset")
    # Clear previous crops. Names carry the rank index, so a re-run with a
    # different --min-px or holdout leaves the old ones behind under different
    # names and the trainer, which globs the directory, would train on both -
    # the same face twice at different crops, silently double-weighted.
    for split in ("train", "holdout"):
        d = out_dir / split
        d.mkdir(parents=True, exist_ok=True)
        for old in list(d.glob("*.png")) + list(d.glob("*.txt")):
            old.unlink()

    manifest = {"train": [], "holdout": [], "rejected": [],
                "margin": args.margin, "trigger": args.trigger,
                "source": "identity.resolve_targets + exclusions",
                "note": "head crops, not greyed panels: a flat grey field would "
                        "be learned as background"}

    # Crop first, split second. Splitting over the ranking BEFORE cropping spends
    # holdout slots on images the crop filter then rejects: on the first real run
    # two of three reserved images fell below --min-px and the evaluation set
    # arrived with one image in it. Only images that survive the crop can be
    # reserved, so the ranking that matters is the surviving one.
    kept = []
    for v in ranked:
        f = Path(v["file"])
        try:
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        except Exception as e:
            manifest["rejected"].append({"file": f.name, "why": str(e)})
            log.info("%-34s rejected: %s", f.name, e)
            continue
        crop = crop_head(im, v["instance"]["box"], args.margin, args.min_px)
        if crop is None:
            # Report the size it would have had: the floor is a judgement call and
            # "rejected" alone gives no way to tell a near miss from a thumbnail.
            full = crop_head(im, v["instance"]["box"], args.margin, 0)
            got = f"{min(full.size)}px" if full is not None else "empty"
            manifest["rejected"].append(
                {"file": f.name, "why": f"crop {got}, below --min-px {args.min_px}",
                 "short_edge_px": min(full.size) if full is not None else 0})
            log.info("%-34s rejected: crop %s, below --min-px %d",
                     f.name, got, args.min_px)
            continue
        if max(crop.size) > args.max_px:
            s = args.max_px / max(crop.size)
            crop = crop.resize((round(crop.width * s), round(crop.height * s)),
                               Image.LANCZOS)
        kept.append((v, f, crop))

    n_hold = max(0, min(args.holdout, len(kept) - 1))
    # every k-th, offset to avoid taking the single strongest image
    hold_idx = set()
    if n_hold:
        step = max(1, len(kept) // (n_hold + 1))
        hold_idx = {min(len(kept) - 1, (i + 1) * step) for i in range(n_hold)}
    if n_hold and len(hold_idx) < args.holdout:
        log.warning("Asked for %d held-out image(s), reserving %d: only %d "
                    "image(s) survived cropping.", args.holdout, len(hold_idx),
                    len(kept))

    for i, (v, f, crop) in enumerate(kept):
        split = "holdout" if i in hold_idx else "train"
        dst = out_dir / split / f"{i:02d}_{f.stem}.png"
        crop.save(dst)
        if args.trigger:
            dst.with_suffix(".txt").write_text(args.trigger + "\n")
        manifest[split].append({
            "file": f.name, "crop": dst.name,
            "crop_px": list(crop.size),
            "agreement": round(float(v.get("agreement") or 0.0), 4),
            "face_pixels": int(v.get("face_pixels") or 0),
        })
        log.info("%-34s %-8s %dx%d  agreement %.3f", f.name, split,
                 crop.width, crop.height, float(v.get("agreement") or 0.0))

    (out_dir / "dataset.json").write_text(json.dumps(manifest, indent=2))
    log.info("=" * 62)
    log.info("train %d, holdout %d, rejected %d -> %s",
             len(manifest["train"]), len(manifest["holdout"]),
             len(manifest["rejected"]), out_dir)
    if len(manifest["train"]) < 8:
        log.warning("Only %d training image(s). Thin for a subject LoRA; expect "
                    "to need a lower rank and fewer steps to avoid overfitting.",
                    len(manifest["train"]))
    log.info("Held-out images are the ONLY valid way to ask whether the LoRA "
             "improved identity. Scoring against a bank built from the training "
             "images returns a high number by construction.")
    log.info("Originals in %s were not modified.", P.references)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
