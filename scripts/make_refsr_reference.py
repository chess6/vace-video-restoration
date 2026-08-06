#!/usr/bin/env python
"""Phase 11 - build the reference a RefSR model can actually use.

A reference-based super-resolution model transfers REAL texture from a reference
image into a low-resolution frame. It does that by finding correspondences
between the two, so the reference has to resemble the frame in the ways the
matcher cares about: head pose, scale, and roughly the framing. The seventeen
photographs in this project were chosen for identity evidence, not for that -
they are a different outfit, different lighting and whatever pose the camera
caught. Handing a matcher an angled portrait to align against a 23-pixel
side-on face is how RefSR quietly degrades to plain single-image SR.

So this picks, per target frame, the reference whose head pose is closest, and
re-crops it to that frame's head geometry:

  * yaw from 5-point landmarks, exactly as make_reference_pack.py measures it,
    so "closest pose" means the same thing in both places;
  * scale normalised by inter-ocular distance, times the SR factor, so the
    reference carries the detail the model is being asked to synthesise rather
    than being upsampled itself;
  * ties broken by face pixel count - between two equally-posed references the
    one with more real detail wins.

It selects on IDENTITY-verified references only (identity.resolve_targets, run
exclusions honoured), because a reference of the wrong person would transfer
that person's texture into the frame, which is worse than transferring nothing.

Rule 2b: this reads pixels to measure pose and crop, which is what the automated
stages already do. It reports geometry and scores, never what is depicted, and
writes every artefact to disk (rule 1).

    scripts/make_refsr_reference.py --target VIDEO --frames 0,40,80 --scale 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, setup_logging  # noqa: E402


def eye_distance(kps) -> float:
    """Inter-ocular distance in pixels; the scale a face matcher aligns on."""
    if kps is None or len(kps) < 2:
        return 0.0
    return float(np.linalg.norm(np.asarray(kps[0]) - np.asarray(kps[1])))


def head_crop(im, kps, box, out_eye_px: float, margin: float = 2.6):
    """Crop around the head and scale so the eyes sit `out_eye_px` apart.

    Normalising on the eyes rather than the box makes the reference's detail
    directly comparable to the target's: at scale 4 the reference carries four
    times the pixels per face feature, which is the only reason it can supply
    anything the frame does not already have.
    """
    from PIL import Image
    eye = eye_distance(kps)
    if eye <= 1:
        return None, 0.0
    cx = float(np.mean([p[0] for p in kps[:2]]))
    cy = float(np.mean([p[1] for p in kps[:2]]))
    half = eye * margin
    L, T = int(round(cx - half)), int(round(cy - half))
    R, B = int(round(cx + half)), int(round(cy + half))
    L, T = max(0, L), max(0, T)
    R, B = min(im.width, R), min(im.height, B)
    if R - L < 16 or B - T < 16:
        return None, 0.0
    crop = im.crop((L, T, R, B))
    factor = out_eye_px / eye
    if factor <= 0:
        return None, 0.0
    w, h = max(16, round(crop.width * factor)), max(16, round(crop.height * factor))
    # Never upsample the reference to fake detail it does not have: an upscaled
    # reference transfers its own interpolation, and the whole point is real
    # texture. Report the shortfall instead and let the caller judge.
    resample = Image.LANCZOS
    return crop.resize((w, h), resample), factor


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path, required=True,
                    help="Video whose frames are being restored")
    ap.add_argument("--frames", default="0",
                    help="Comma-separated frame indices to build references for")
    ap.add_argument("--scale", type=float, default=4.0,
                    help="SR factor: the reference is built at this multiple of "
                         "the target's face scale")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("make_refsr_reference", args.verbose)
    import cv2
    from PIL import Image, ImageOps
    import track_subject as T
    from identity import load_exclusions, reference_files, resolve_targets
    from make_reference_pack import face_detail, face_yaw

    models = T.Models(log)
    res = resolve_targets(reference_files(), models, log, load_exclusions(log))
    per = res.get("per_image") or {}
    if not per:
        log.error("No identity-verified reference. Refusing to build a RefSR "
                  "reference from an unverified face.")
        return 1

    # Measure every candidate once: pose, scale, and how much real detail it has.
    cands = []
    for name, v in per.items():
        f = Path(v["file"])
        im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        emb, kps, frac = face_detail(models, im)
        if kps is None:
            continue
        cands.append({"name": name, "im": im, "kps": kps,
                      "yaw": face_yaw(kps), "eye": eye_distance(kps),
                      "face_px": int(v.get("face_pixels") or 0),
                      "agreement": float(v.get("agreement") or 0.0)})
    if not cands:
        log.error("No verified reference yielded landmarks.")
        return 1
    log.info("%d verified reference(s) measured for pose and scale", len(cands))

    out = args.out or (P.intermediate / "refsr_references")
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.target))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        log.error("%s decoded 0 frames", args.target)
        return 1

    manifest = {"target": args.target.name, "scale": args.scale, "pairs": []}
    for idx in [int(x) for x in args.frames.split(",") if x.strip()]:
        if idx >= len(frames):
            log.warning("frame %d beyond the target's %d", idx, len(frames))
            continue
        rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        emb, kps, frac = face_detail(models, pil)
        if kps is None:
            log.warning("frame %d: no face found, so there is nothing to match "
                        "a reference against; skipped", idx)
            continue
        t_yaw, t_eye = face_yaw(kps), eye_distance(kps)

        # Closest pose wins; more real detail breaks the tie.
        def cost(c):
            dy = abs((c["yaw"] or 0.0) - (t_yaw or 0.0))
            return (round(dy, 2), -c["face_px"])
        best = sorted(cands, key=cost)[0]
        want_eye = t_eye * args.scale
        crop, factor = head_crop(best["im"], best["kps"], None, want_eye)
        if crop is None:
            log.warning("frame %d: %s could not be cropped", idx, best["name"])
            continue
        dst = out / f"ref_frame{idx:04d}.png"
        crop.save(dst)
        rec = {"frame": idx, "reference": best["name"],
               "target_yaw": round(float(t_yaw or 0), 3),
               "reference_yaw": round(float(best["yaw"] or 0), 3),
               "target_eye_px": round(t_eye, 1),
               "reference_eye_px": round(best["eye"], 1),
               "scale_applied": round(factor, 3),
               "reference_upsampled": bool(factor > 1.0),
               "agreement": round(best["agreement"], 4),
               "crop_px": list(crop.size), "path": str(dst)}
        manifest["pairs"].append(rec)
        log.info("frame %4d  <- %-30s yaw %+0.2f vs %+0.2f | eyes %.0f -> %.0f px "
                 "| x%.2f%s", idx, best["name"], t_yaw or 0, best["yaw"] or 0,
                 t_eye, want_eye, factor,
                 "  UPSAMPLED (reference has less detail than asked for)"
                 if factor > 1.0 else "")

    (out / "refsr_references.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %d reference(s) and %s", len(manifest["pairs"]),
             out / "refsr_references.json")
    if any(p["reference_upsampled"] for p in manifest["pairs"]):
        log.warning("At least one reference had to be upsampled to reach the "
                    "requested scale. RefSR cannot transfer detail a reference "
                    "does not have; expect that frame to gain little.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
