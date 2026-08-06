#!/usr/bin/env python
"""Phase 11c - strip the invented attribute out of generated candidates.

A full-extent candidate is safe as match evidence and dangerous as a texture
donor. The LoRA behind it was trained on tightly framed anchor crops, so it knows
the anchor region and nothing else: everything outside that region is the base
model inventing attributes, and they are not the attributes in the source
interval. docs/STATE.md is explicit that the attribute in the source is the sole
ground truth, and this project has already measured what an invented one looks
like.

A reference-based super-resolution model cannot be told to take the anchor and
leave the attribute - it transfers whatever it matched. So the invented attribute is
removed before the image can ever reach one.

This deliberately reuses `make_reference_pack.mask_to_match` rather than
implementing a second version. That function is where the authority split lives,
including the detail that matters most here: the feather ramps INWARD, because a
plain outward blur was measured leaving up to 83/255 of the original pixel in a
ring around the anchor region - and what lies immediately around it is
attribute-bearing, which is exactly the hint that would instruct a model about
coverage.

SCOPE. Two rules are available and they disagree on the periphery.

  `anchor`        The anchor region only, exactly the rule make_reference_pack.py
                  enforces, on the reasoning recorded there: how much of a
                  peripheral extent is visible is a fact about the attribute the
                  candidate carries, so a reference showing more of one instructs
                  a generative model to uncover the source's.
  `not_attributes`  Everything the parser does not call an attribute, an accessory
                  or background - the anchor region plus the exposed peripheral
                  ones. The default here, at the user's instruction.

The coverage hazard is real but belongs to a path that is now closed: it was
about CONDITIONING a model that regenerates the attribute, and VACE is out - the
plate supplies the attribute and nothing regenerates it. What remains is the
transfer hazard: a RefSR step matching across the whole subject could paste an
exposed region from the reference where the source is covered. The mitigation is
to restrict the transfer region rather than to blind the reference, which is why
this default is safe HERE and would not be safe in the reference pack.

    scripts/grey_candidate_attributes.py --in DIR [--scope not_attributes|anchor]
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
    ap.add_argument("--scope", choices=["not_attributes", "anchor"],
                    default="not_attributes",
                    help="not_attributes keeps the exposed regions too; anchor "
                         "keeps the anchor region only, the reference-pack rule")
    ap.add_argument("--feather", type=int, default=3)
    ap.add_argument("--min-kept", type=float, default=0.01,
                    help="Reject a candidate keeping less than this fraction: "
                         "if the parser found almost no anchor region, the greyed image "
                         "is a grey rectangle and donates nothing")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("grey_candidate_attributes", args.verbose)
    from PIL import Image
    from make_reference_pack import (ATTRIBUTE, MATCH_ONLY, LBL, Parser,
                                     mask_to_match)

    if args.scope == "anchor":
        keep = set(MATCH_ONLY)
    else:
        # Inverse rule: anything the parser does not call an attribute, an
        # accessory or background. Derived from the label map rather than
        # listed, so a change to ATTRIBUTE cannot leave a stale set behind.
        keep = {n for n in LBL.values()
                if n not in ATTRIBUTE and n not in {"background", "sunglasses"}}

    files = sorted(f for f in args.src.rglob("*") if f.suffix.lower() in IMG_EXT)
    if not files:
        log.error("No images under %s", args.src)
        return 1
    out_root = args.out or (args.src.parent / (args.src.name + "_match_only"))
    log.info("%d candidate(s); scope=%s keeps %s, everything else goes grey",
             len(files), args.scope, sorted(keep))

    parser = Parser(log)
    manifest = {"source": str(args.src), "scope": args.scope,
                "kept_labels": sorted(keep),
                "feather": args.feather, "images": [], "rejected": []}
    for f in files:
        im = Image.open(f).convert("RGB")
        labels = parser.parse(im)
        greyed, kept = mask_to_match(im, labels, feather=args.feather,
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

    (out_root / "match_only.json").write_text(json.dumps(manifest, indent=2))
    log.info("=" * 62)
    log.info("%d written, %d rejected -> %s", len(manifest["images"]),
             len(manifest["rejected"]), out_root)
    log.info("These carry %s on neutral grey. They are match evidence and a "
             "donor for those regions; they are not, and cannot become, a "
             "attribute donor.", ", ".join(sorted(keep)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
