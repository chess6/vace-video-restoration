#!/usr/bin/env python
"""Phase 11c - strip the invented garment out of generated candidates.

A full-figure candidate is safe as identity evidence and dangerous as a texture
donor. The LoRA behind it was trained on head crops, so it knows a face and hair
and nothing else: everything below the neck is the base model inventing
clothing, and it is not the clothing in the source interval. docs/STATE.md is
explicit that the garment in the source is the sole ground truth, and this
project has already measured what an invented one looks like.

A reference-based super-resolution model cannot be told to take the face and
leave the shirt - it transfers whatever it matched. So the shirt is removed
before the image can ever reach one.

This deliberately reuses `make_reference_pack.mask_to_identity` rather than
implementing a second version. That function is where the authority split lives,
including the detail that matters most here: the feather ramps INWARD, because a
plain outward blur was measured leaving up to 83/255 of the original pixel in a
ring around the head - and the pixels around a head are neck and shoulders,
which is exactly the hint that would instruct a model about sleeves.

SCOPE. Two rules are available and they disagree on limbs.

  `head`          {hair, face}. The rule make_reference_pack.py enforces, on the
                  reasoning recorded there: how much arm or leg is visible is a
                  fact about what the person is WEARING, so a reference with bare
                  arms instructs a generative model to remove the source's
                  sleeves.
  `not_clothing`  {hair, face, arms, legs} - everything the parser does not call
                  a garment, an accessory or background. The default here, at the
                  user's instruction.

The sleeve hazard is real but belongs to a path that is now closed: it was about
CONDITIONING a model that regenerates the garment, and VACE is out - the plate
supplies the garment and nothing regenerates it. What remains is the transfer
hazard: a RefSR step matching across the whole figure could paste the
reference's bare skin where the source has a sleeve. The mitigation is to
restrict the transfer region rather than to blind the reference, which is why
this default is safe HERE and would not be safe in the reference pack.

    scripts/grey_candidate_garments.py --in DIR [--scope not_clothing|head]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, setup_logging  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, required=True,
                    help="Directory of generated candidates (searched recursively)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--scope", choices=["not_clothing", "head"],
                    default="not_clothing",
                    help="not_clothing keeps skin and hair; head keeps only "
                         "{hair, face}, the reference-pack rule")
    ap.add_argument("--feather", type=int, default=3)
    ap.add_argument("--min-kept", type=float, default=0.01,
                    help="Reject a candidate keeping less than this fraction: "
                         "if the parser found almost no head, the greyed image "
                         "is a grey rectangle and donates nothing")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("grey_candidate_garments", args.verbose)
    from PIL import Image
    from make_reference_pack import (GARMENT, IDENTITY_ONLY, LBL, Parser,
                                     mask_to_identity)

    if args.scope == "head":
        keep = set(IDENTITY_ONLY)
    else:
        # Inverse rule: anything the parser does not call a garment, an
        # accessory or background. Derived from the label map rather than
        # listed, so a change to GARMENT cannot leave a stale set behind.
        keep = {n for n in LBL.values()
                if n not in GARMENT and n not in {"background", "sunglasses"}}

    files = sorted(f for f in args.src.rglob("*") if f.suffix.lower() in IMG_EXT)
    if not files:
        log.error("No images under %s", args.src)
        return 1
    out_root = args.out or (args.src.parent / (args.src.name + "_identity_only"))
    log.info("%d candidate(s); scope=%s keeps %s, everything else goes grey",
             len(files), args.scope, sorted(keep))

    parser = Parser(log)
    manifest = {"source": str(args.src), "scope": args.scope,
                "kept_labels": sorted(keep),
                "feather": args.feather, "images": [], "rejected": []}
    for f in files:
        im = Image.open(f).convert("RGB")
        labels = parser.parse(im)
        greyed, kept = mask_to_identity(im, labels, feather=args.feather,
                                        keep=keep)
        rel = f.relative_to(args.src)
        if kept < args.min_kept:
            manifest["rejected"].append({"file": str(rel), "kept": round(kept, 5)})
            log.info("%-42s REJECTED, only %.2f%% survives the split",
                     str(rel), 100 * kept)
            continue
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        greyed.save(dst)
        manifest["images"].append({"file": str(rel), "kept": round(kept, 5)})
        log.info("%-42s %.2f%% kept", str(rel), 100 * kept)

    (out_root / "identity_only.json").write_text(json.dumps(manifest, indent=2))
    log.info("=" * 62)
    log.info("%d written, %d rejected -> %s", len(manifest["images"]),
             len(manifest["rejected"]), out_root)
    log.info("These carry %s on neutral grey. They are identity evidence and a "
             "donor for those regions; they are not, and cannot become, a "
             "garment donor.", ", ".join(sorted(keep)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
