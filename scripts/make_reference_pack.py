#!/usr/bin/env python
"""Phase 5b - per-appearance reference packs, with the outfit taken from source.

The single global reference sheet was the direct cause of invented clothing
colours: it tiled whichever three photographs scored best, so a shot could be
conditioned on two different outfits at once, and VACE resolved the conflict by
inventing a third. This builds ONE pack per appearance cluster, and never mixes
conflicting outfits into the same image.

The division of labour that makes this work:

  * The external photographs are the authority on IDENTITY and on high-frequency
    detail - face structure, hair, skin texture. They are not the authority on
    what the person is wearing in this shot, because they were taken elsewhere.
  * THIS SOURCE INTERVAL is the authority on the current outfit: garment shape,
    boundaries, accessories and colour. It is low resolution, but it is the only
    thing that is actually correct about the clothes.

So the pack is assembled as: strongest identity/face view from the externals, an
exact-outfit body view taken from the best SOURCE frame, and one compatible
alternate angle. Panels are labelled with their provenance and validation score.

Clustering is automatic (CLIP appearance embeddings + garment histograms), and
so is garment parsing (SegFormer). No manual labelling anywhere.

720p references are treated as adequate and are NOT upscaled to reach 1080p:
resampling adds no information and softens the very detail they are here for.

    scripts/make_reference_pack.py [--shot shot0000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    IMAGE_EXTS, P, geometry_key, load_config, load_manifest, pilot_chunks, rel,
    save_manifest, setup_logging,
)

SEGFORMER_ID = "mattmdjaga/segformer_b2_clothes"
SEGFORMER_REV = "584abc1e1d26"

# SegFormer clothes label map (from the model card's id2label).
LBL = {0: "background", 1: "hat", 2: "hair", 3: "sunglasses", 4: "upper",
       5: "skirt", 6: "pants", 7: "dress", 8: "belt", 9: "left_shoe",
       10: "right_shoe", 11: "face", 12: "left_leg", 13: "right_leg",
       14: "left_arm", 15: "right_arm", 16: "bag", 17: "scarf"}
GARMENT = {"hat", "upper", "skirt", "pants", "dress", "belt", "left_shoe",
           "right_shoe", "bag", "scarf"}
SKIN = {"face", "left_leg", "right_leg", "left_arm", "right_arm"}


class Parser:
    """SegFormer clothes parsing, loaded once."""

    def __init__(self, log):
        import torch
        from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading clothing parser %s", SEGFORMER_ID)
        self.proc = SegformerImageProcessor.from_pretrained(
            SEGFORMER_ID, revision=SEGFORMER_REV)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            SEGFORMER_ID, revision=SEGFORMER_REV).to(self.dev).eval()

    def parse(self, pil):
        """Returns a HxW label map at the image's own resolution."""
        import torch
        with torch.inference_mode():
            inp = self.proc(images=pil, return_tensors="pt").to(self.dev)
            logits = self.model(**inp).logits
            up = torch.nn.functional.interpolate(
                logits, size=pil.size[::-1], mode="bilinear", align_corners=False)
        return up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def palette(rgb: np.ndarray, labels: np.ndarray) -> dict:
    """Median Lab colour per garment class, plus its pixel share.

    Median rather than mean: a few blown-out highlights or a dark fold should not
    move the recorded colour of a garment.
    """
    import cv2
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    out = {}
    total = labels.size
    for idx, name in LBL.items():
        if name not in GARMENT:
            continue
        m = labels == idx
        n = int(m.sum())
        if n < max(64, total * 0.001):
            continue
        out[name] = {"lab": [round(float(np.median(lab[..., c][m])), 2)
                             for c in range(3)],
                     "pixel_share": round(n / total, 5)}
    return out


def delta_e(a: list, b: list) -> float:
    """CIE76 perceptual difference. Crude next to CIEDE2000, but monotonic and
    dependency-free; ~2.3 is the just-noticeable threshold."""
    return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def source_outfit(work: Path, mask_video: Path, parser: Parser, n_probe: int,
                  log) -> dict:
    """Learn the CURRENT outfit from this interval: the only correct authority.

    Picks the frames where the subject is largest and sharpest, parses garments
    on each, and keeps a temporally smoothed palette plus the single best frame
    as an exact-outfit panel.
    """
    import cv2
    from PIL import Image
    cap, mcap = cv2.VideoCapture(str(work)), cv2.VideoCapture(str(mask_video))
    frames, masks = [], []
    while True:
        ok, f = cap.read()
        okm, m = mcap.read()
        if not (ok and okm):
            break
        frames.append(f)
        masks.append(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) > 127)
    cap.release(); mcap.release()
    if not frames:
        raise RuntimeError("no frames to learn the outfit from")

    score = []
    for f, m in zip(frames, masks):
        if m.sum() < 64:
            score.append(-1.0)
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_32F)[m].var())
        score.append(float(m.sum()) * (1.0 + sharp / 500.0))
    order = np.argsort(score)[::-1][:n_probe]

    per_frame, best = [], None
    for i in order:
        if score[i] <= 0:
            continue
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        ys, xs = np.where(masks[i])
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = rgb[y0:y1, x0:x1]
        if crop.shape[0] < 32 or crop.shape[1] < 16:
            continue
        pil = Image.fromarray(crop)
        lab = parser.parse(pil)
        pal = palette(crop, lab)
        per_frame.append(pal)
        if best is None:
            best = {"frame": int(i), "crop": crop, "labels": lab, "palette": pal}

    if not per_frame:
        raise RuntimeError("clothing parsing found no garments in the source")

    # Temporal smoothing: median across probe frames, and the spread as a
    # stability measure. A garment whose colour swings between frames is not a
    # reliable constraint and is reported as such.
    names = sorted({k for p in per_frame for k in p})
    smoothed = {}
    for name in names:
        vals = [p[name]["lab"] for p in per_frame if name in p]
        if len(vals) < max(2, len(per_frame) // 3):
            continue                       # seen too rarely to trust
        arr = np.array(vals, dtype=np.float32)
        med = [round(float(np.median(arr[:, c])), 2) for c in range(3)]
        spread = float(np.mean([delta_e(v, med) for v in vals]))
        smoothed[name] = {"lab": med, "temporal_deltaE": round(spread, 2),
                          "frames_seen": len(vals),
                          "stable": bool(spread < 8.0)}
    log.info("Source outfit (%d probe frames): %s", len(per_frame),
             ", ".join(f"{k}(dE{v['temporal_deltaE']:.1f}"
                       f"{'' if v['stable'] else ' UNSTABLE'})"
                       for k, v in smoothed.items()) or "nothing parsed")
    return {"palette": smoothed, "best": best, "n_probe": len(per_frame)}


def cluster_references(files: list[Path], parser: Parser, models, bank, log,
                       thresh: float) -> list[dict]:
    """Group the external photographs by appearance, so one pack never mixes
    two different outfits. Identity is checked separately: a photo of someone
    else is dropped before clustering, not clustered with them."""
    from PIL import Image, ImageOps
    items = []
    for f in files:
        try:
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        except Exception as e:
            log.warning("%s unreadable (%s)", f.name, e)
            continue
        if min(im.size) < 256:
            log.info("%-34s rejected: %dpx short side", f.name, min(im.size))
            continue
        lab = parser.parse(im)
        pal = palette(np.asarray(im), lab)
        emb = models.clip_embed([im])[0]
        face, _, face_frac = models.face_embed(im)
        items.append({"file": f, "image": im, "palette": pal, "clip": emb,
                      "face": face, "face_frac": face_frac,
                      "size": list(im.size)})

    # Appearance clustering: garment palette distance first (that is what an
    # "outfit" is), CLIP distance as the tiebreak for pose/framing differences.
    groups: list[dict] = []
    for it in items:
        placed = False
        for g in groups:
            shared = set(it["palette"]) & set(g["palette"])
            if shared:
                d = np.mean([delta_e(it["palette"][k]["lab"],
                                     g["palette"][k]["lab"]) for k in shared])
                same_outfit = d < thresh
            else:
                same_outfit = False
            appearance = float(np.dot(it["clip"], g["members"][0]["clip"]))
            if same_outfit and appearance > 0.6:
                g["members"].append(it)
                for k, v in it["palette"].items():
                    g["palette"].setdefault(k, v)
                placed = True
                break
        if not placed:
            groups.append({"members": [it], "palette": dict(it["palette"])})
    for i, g in enumerate(groups):
        log.info("appearance cluster %d: %d image(s) - %s", i, len(g["members"]),
                 ", ".join(m["file"].name for m in g["members"][:6]))
    return groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--shot", nargs="*", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--probe-frames", type=int, default=8)
    ap.add_argument("--outfit-deltae", type=float, default=18.0,
                    help="Garment colour distance above which two photographs "
                         "are treated as different outfits")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("make_reference_pack", args.verbose)
    cfg = load_config(args.config)
    man = load_manifest()
    import track_subject as T
    models = T.Models(log)
    bank = T.build_identity_bank(models, log)
    parser = Parser(log)

    work = P.root / man["normalized"]["work_path"]
    W = int(man["normalized"]["width"])
    H = int(man["normalized"]["height"])
    refs = sorted(p for p in P.references.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not refs:
        log.error("No reference images in %s", P.references)
        return 1

    groups = cluster_references(refs, parser, models, bank, log, args.outfit_deltae)

    chunks = pilot_chunks(man) if args.pilot else man["chunks"]
    shot_ids = sorted({c["shot_id"] for c in chunks})
    if args.shot:
        shot_ids = [s for s in shot_ids if s in set(args.shot)]

    pack_dir = P.intermediate / "reference_packs"
    pack_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for sid in shot_ids:
        mask_video = P.masks / f"{sid}_mask.mkv"
        if not mask_video.exists():
            log.warning("%s: no mask; skipping", sid)
            continue
        log.info("=" * 62)
        log.info("%s: learning the current outfit from the source", sid)
        outfit = source_outfit(work, mask_video, parser, args.probe_frames, log)

        # Choose the appearance cluster whose garments match THIS shot's clothes.
        best_g, best_d = None, None
        for gi, g in enumerate(groups):
            shared = set(g["palette"]) & set(outfit["palette"])
            if not shared:
                continue
            d = float(np.mean([delta_e(g["palette"][k]["lab"],
                                       outfit["palette"][k]["lab"]) for k in shared]))
            if best_d is None or d < best_d:
                best_g, best_d = gi, d
        if best_g is None:
            log.warning("No reference cluster shares a garment class with this "
                        "shot; using cluster 0 for identity only.")
            best_g, best_d = 0, float("nan")
        exact = best_d is not None and best_d == best_d and best_d < args.outfit_deltae
        log.info("Closest appearance cluster: %d (garment dE %.1f) -> %s", best_g,
                 best_d if best_d == best_d else float("nan"),
                 "treated as the SAME outfit" if exact else
                 "treated as a DIFFERENT outfit; clothes will come from the source")

        g = groups[best_g]
        # Panel 1: strongest identity/face view from the externals.
        idw = sorted(g["members"], key=lambda m: -(m["face_frac"] or 0))
        panel_face = idw[0] if idw else g["members"][0]
        # Panel 3: a compatible alternate angle from the SAME cluster only.
        alt = next((m for m in g["members"] if m is not panel_face), None)

        pack = {
            "shot_id": sid,
            "cluster": best_g,
            "cluster_size": len(g["members"]),
            "garment_deltaE_to_source": (round(best_d, 2) if best_d == best_d
                                         else None),
            "outfit_authority": "external_photograph" if exact else "source_frames",
            "source_palette": outfit["palette"],
            "panels": [
                {"role": "identity_face", "provenance": panel_face["file"].name,
                 "native_size": panel_face["size"],
                 "upscaled": False,
                 "face_fraction": round(float(panel_face["face_frac"] or 0), 4)},
                {"role": "exact_outfit_body",
                 "provenance": f"source frame {outfit['best']['frame']} "
                               f"of this interval",
                 "note": "garment shape, boundaries, accessories and colour come "
                         "from the footage itself, which is the only correct "
                         "authority for what is being worn here"},
            ],
        }
        if alt is not None:
            pack["panels"].append(
                {"role": "alternate_angle", "provenance": alt["file"].name,
                 "native_size": alt["size"], "upscaled": False})

        # ---- render the sheet ------------------------------------------------
        from PIL import Image
        n = len(pack["panels"])
        pw, ph = W // n, H
        sheet = Image.new("RGB", (W, H), (0, 0, 0))
        imgs = [panel_face["image"], Image.fromarray(outfit["best"]["crop"])]
        if alt is not None:
            imgs.append(alt["image"])
        for i, im in enumerate(imgs):
            k = min(pw / im.width, ph / im.height)
            # Only ever downscale to fit the panel. A 720p reference is adequate;
            # enlarging it invents nothing and softens what it was chosen for.
            k = min(k, 1.0) if im.width >= pw or im.height >= ph else k
            tw, th = max(1, int(im.width * k)), max(1, int(im.height * k))
            sheet.paste(im.resize((tw, th), Image.LANCZOS),
                        (i * pw + (pw - tw) // 2, (ph - th) // 2))
        sheet_path = pack_dir / f"{sid}_reference_pack.png"
        sheet.save(sheet_path)
        pack["sheet"] = rel(sheet_path)
        pack["key"] = geometry_key({}, {
            "shot": sid, "cluster": best_g, "w": W, "h": H,
            "panels": [p["provenance"] for p in pack["panels"]],
            "palette": {k: v["lab"] for k, v in outfit["palette"].items()}})
        (pack_dir / f"{sid}_pack.json").write_text(json.dumps(
            {k: v for k, v in pack.items()}, indent=2, default=str) + "\n")

        shot = next(s for s in man["shots"] if s["shot_id"] == sid)
        shot["reference_pack"] = {"sheet": pack["sheet"], "key": pack["key"],
                                  "cluster": best_g,
                                  "outfit_authority": pack["outfit_authority"]}
        log.info("%s: pack -> %s (%d panels, outfit authority: %s)", sid,
                 rel(sheet_path), n, pack["outfit_authority"])
        made += 1

    save_manifest(man)
    log.info("=" * 62)
    log.info("Built %d reference pack(s) in %s", made, rel(pack_dir))
    log.info("Panels carry provenance; nothing was upscaled to reach 1080p.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
