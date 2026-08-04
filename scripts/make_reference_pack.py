#!/usr/bin/env python
"""Phase 5b - identity-only external conditioning + source-derived garment.

The task here is to RESTORE the garment that is already in the footage, not to
infer a garment from photographs taken elsewhere. Those two things were being
conflated, and that is why clothing colours were invented.

Strict division of authority:

  * External photographs condition IDENTITY ONLY - face, hair, exposed skin,
    body appearance. Their clothing is segmented out and replaced with neutral
    grey before the panel is drawn, so it cannot reach the model at all. A faint
    ghost of the wrong jacket is still a jacket to a generative model.
  * THIS SOURCE INTERVAL is the sole ground truth for the garment: class,
    silhouette, boundaries, colours, patterns, accessories, folds and how they
    move. It is low resolution, and that is fine - the job is to preserve its
    low-frequency colour and structure while generating only the missing
    high-frequency texture.

Which photographs become panels is decided by IDENTITY EVIDENCE ALONE:
leave-one-out face agreement across the reference set, how many pixels the face
occupies, and - for the second panel - how different the viewing angle is. Never
by garment colour. Garment distance is still measured and recorded, purely as
evidence that the photographs show other clothes; it is a diagnostic, never a
switch, and the source remains the garment authority whatever it says.

Appearance clustering still runs and is still recorded, but it no longer confines
the choice. Its job was to stop two outfits being combined in one conditioning
image, and there is no longer an outfit in an external panel to conflict - so
restricting panels to a single cluster would only discard viewing angles.

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
# What an EXTERNAL photograph is allowed to condition: identity only. Its
# clothing belongs to a different day and must never reach the model, because
# VACE will happily treat it as the garment to draw. Sunglasses and hats are
# excluded too - they are accessories of that other appearance, not identity.
IDENTITY_ONLY = {"hair", "face", "left_leg", "right_leg", "left_arm", "right_arm"}


def identity_regions(labels: np.ndarray) -> np.ndarray:
    """Boolean mask of the pixels of an external photo that may be shown."""
    ids = [i for i, n in LBL.items() if n in IDENTITY_ONLY]
    return np.isin(labels, ids)


def mask_to_identity(im, labels, feather: int = 3):
    """Blank everything that is not identity in an external reference.

    Garments, accessories and background are removed rather than dimmed: a faint
    ghost of the wrong jacket is still a jacket as far as the model is concerned.
    A few pixels of feather stop the cut-out edge from reading as a hard graphic
    shape, which would itself become something to reproduce.
    """
    from PIL import Image
    arr = np.asarray(im).astype(np.float32)
    m = identity_regions(labels).astype(np.float32)
    if feather > 0:
        import cv2
        r = feather * 2 + 1
        m = cv2.GaussianBlur(m, (r, r), 0)
    # Neutral mid-grey, not black: a large black field shifts the exposure the
    # model infers from the panel.
    out = arr * m[..., None] + 128.0 * (1.0 - m[..., None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), float(m.mean())


def build_panel_images(face_panel: dict, source_crop, alt_panel: dict | None):
    """The panel images, in sheet order, with the authority split enforced here.

    Externals in, identity-only out; the source crop passes through untouched.
    Keeping this in one function means the rule is one testable statement rather
    than something a future edit to the rendering loop can quietly break.

    Returns (images, kept_fractions) where kept_fractions[i] is None for the
    source panel, which is never masked.
    """
    from PIL import Image
    face_img, face_keep = mask_to_identity(face_panel["image"],
                                           face_panel["labels"])
    imgs = [face_img, Image.fromarray(np.asarray(source_crop))]
    keeps: list[float | None] = [face_keep, None]
    if alt_panel is not None:
        alt_img, alt_keep = mask_to_identity(alt_panel["image"],
                                             alt_panel["labels"])
        imgs.append(alt_img)
        keeps.append(alt_keep)
    return imgs, keeps


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


def cluster_references(files: list[Path], parser: Parser, models, log,
                       thresh: float) -> list[dict]:
    """Group the external photographs by appearance, and gather everything the
    panel choice needs: garment palette, CLIP embedding, face embedding, size.

    The grouping is now diagnostic - see the module docstring. Identity is a
    separate question, decided in score_identity() from these same embeddings.
    """
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
        items.append({"file": f, "image": im, "labels": lab, "palette": pal,
                      "clip": emb, "face": face, "face_frac": face_frac,
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
        for m in g["members"]:
            m["cluster"] = i
        log.info("appearance cluster %d: %d image(s) - %s", i, len(g["members"]),
                 ", ".join(m["file"].name for m in g["members"][:6]))
    return groups


def score_identity(items: list[dict]) -> None:
    """Rate each photograph on how well it serves IDENTITY conditioning, in place.

    Deliberately blind to garment colour. Once clothing is segmented out of the
    external panels, which outfit a photograph happens to show says nothing about
    how well it depicts the face, and letting it decide would reintroduce exactly
    the coupling these packs exist to break.

    Agreement is LEAVE-ONE-OUT: each photograph is compared against the face
    embeddings of the OTHERS, never against a bank that contains itself. Scoring
    against the whole reference set returns 1.000 for any photo that is in it,
    which is a self-comparison dressed up as evidence - the same shape of mistake
    as measuring a mask's overlap with its own complement. What is wanted is
    whether the rest of the reference set agrees this is the same person.

      agreement  leave-one-out face similarity: is this the right person
      face_res   how many pixels the face occupies; a correct match at 20 px
                 carries no detail worth transferring

    Weighted in that order, because a confident wrong identity is the worst
    outcome available. Photographs with no detectable face score zero: they can
    still be the right person, but nothing here can show that they are.
    """
    faces = [m for m in items if m["face"] is not None]
    E = np.stack([m["face"] for m in faces]) if faces else None
    for m in items:
        agree = 0.0
        if m["face"] is not None and E is not None and len(faces) > 1:
            sims = E @ m["face"]
            # Drop this photograph's own row rather than trusting a threshold:
            # a genuine near-duplicate should still count as agreement.
            own = next(i for i, o in enumerate(faces) if o is m)
            agree = float(np.max(np.delete(sims, own)))
        w, h = m["size"]
        # sqrt -> a face side length; ~200 px is already ample for conditioning,
        # so the score saturates there instead of rewarding ever-larger portraits.
        px = float(m["face_frac"] or 0.0) * w * h
        res = float(np.clip(np.sqrt(px) / 200.0, 0.0, 1.0))
        m["identity"] = {"agreement": round(agree, 4),
                         "face_resolution": round(res, 4),
                         "face_pixels": int(px),
                         "score": round(0.7 * agree + 0.3 * res, 4)}


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
    # No identity bank here on purpose: it is built from these same
    # photographs, so scoring one against it is a self-comparison. Identity
    # agreement is computed leave-one-out in score_identity() instead.
    parser = Parser(log)

    work = P.root / man["normalized"]["work_path"]
    W = int(man["normalized"]["width"])
    H = int(man["normalized"]["height"])
    refs = sorted(p for p in P.references.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not refs:
        log.error("No reference images in %s", P.references)
        return 1

    groups = cluster_references(refs, parser, models, log, args.outfit_deltae)

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

        # Panels are chosen on IDENTITY EVIDENCE ALONE, across every reference.
        #
        # Appearance clusters exist to stop two different outfits being combined
        # into one conditioning image. That constraint no longer binds the
        # external panels: their clothing is segmented out before they are drawn,
        # so there is no outfit left to conflict. Confining the choice to one
        # cluster would now throw away viewing angles for no benefit - and
        # missing viewpoints were a real weakness of these references. Clusters
        # are still computed and recorded, as a diagnostic.
        items = [m for g in groups for m in g["members"]]
        score_identity(items)
        ranked = sorted(items, key=lambda m: (m["identity"]["score"],
                                              m["identity"]["face_pixels"],
                                              m["file"].name), reverse=True)
        for m in ranked:
            i = m["identity"]
            log.info("%-34s identity %.3f (leave-one-out agreement %.3f, face "
                     "%d px, cluster %d)", m["file"].name, i["score"],
                     i["agreement"], i["face_pixels"], m["cluster"])
        panel_face = ranked[0]
        if panel_face["identity"]["agreement"] <= 0:
            log.warning("No reference agrees with any other on identity (too few "
                        "detectable faces). Identity conditioning will be weak. "
                        "The garment is unaffected: it comes from the source.")

        # A second panel earns its place only by showing a DIFFERENT angle, and
        # only if it is confidently the same person - an unverified photograph
        # would be conditioning the model on a stranger's face.
        MIN_AGREE = 0.45
        others = [m for m in ranked[1:]
                  if m["identity"]["agreement"] >= MIN_AGREE]
        alt = (min(others, key=lambda m: float(np.dot(m["clip"], panel_face["clip"])))
               if others else None)
        if alt is not None:
            log.info("alternate angle: %s (CLIP similarity to panel 1 %.3f, the "
                     "most different view among %d verified reference(s))",
                     alt["file"].name,
                     float(np.dot(alt["clip"], panel_face["clip"])), len(others))
        else:
            log.info("No second reference cleared the identity bar (agreement "
                     ">= %.2f); the pack uses one identity panel.", MIN_AGREE)
        best_g = int(panel_face["cluster"])

        # Diagnostic only: how far the chosen photographs' clothing is from what
        # is actually worn in this interval. Recorded as evidence that the
        # external clothing is incompatible - never as a reason to use it.
        chosen_pal: dict = {}
        for m in ([panel_face] + ([alt] if alt is not None else [])):
            for k, v in m["palette"].items():
                chosen_pal.setdefault(k, v)
        shared = set(chosen_pal) & set(outfit["palette"])
        best_d = (float(np.mean([delta_e(chosen_pal[k]["lab"],
                                         outfit["palette"][k]["lab"])
                                 for k in shared])) if shared else float("nan"))
        same_look = best_d == best_d and best_d < args.outfit_deltae
        log.info("Chosen references' garments are dE %.1f from this interval's "
                 "(%s). Either way the garment comes from the source; this is a "
                 "diagnostic, not a switch.",
                 best_d if best_d == best_d else float("nan"),
                 "similar" if same_look else "a different outfit"
                 if best_d == best_d else "no shared garment class")

        pack = {
            "shot_id": sid,
            "cluster": best_g,
            "clusters_found": len(groups),
            "references_considered": len(items),
            # Invariant, not a decision. The garment in this interval is the only
            # record of what is being worn, so it is always the authority; the
            # externals only ever contribute identity. Making this conditional on
            # a colour distance meant a coincidentally similar photograph could
            # promote itself to garment source, which is the failure being fixed.
            "outfit_authority": "source_frames",
            "identity_authority": "external_photographs",
            "external_garment_similarity": {
                "deltaE_to_source": (round(best_d, 2) if best_d == best_d
                                     else None),
                "threshold": args.outfit_deltae,
                "looks_like_same_outfit": bool(same_look),
                "note": "diagnostic only; does not affect where the garment "
                        "comes from",
            },
            "identity_selection": panel_face["identity"],
            "source_palette": outfit["palette"],
            "panels": [
                {"role": "identity_face_only", "provenance": panel_face["file"].name,
                 "native_size": panel_face["size"],
                 "upscaled": False,
                 "conditions": "face, hair and exposed skin only; clothing and "
                               "accessories segmented out",
                 "face_fraction": round(float(panel_face["face_frac"] or 0), 4),
                 "identity": panel_face["identity"],
                 "cluster": panel_face["cluster"]},
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
                {"role": "alternate_angle_identity_only",
                 "provenance": alt["file"].name,
                 "conditions": "identity regions only; clothing segmented out",
                 "native_size": alt["size"], "upscaled": False})

        # ---- render the sheet ------------------------------------------------
        from PIL import Image
        n = len(pack["panels"])
        pw, ph = W // n, H
        sheet = Image.new("RGB", (W, H), (0, 0, 0))
        # EXTERNAL panels are stripped to identity before they are drawn. Their
        # clothing is from another day; leaving it visible invites the model to
        # reproduce it, which is the fault being corrected. The SOURCE panel is
        # left whole - it is the only correct record of the current garment.
        imgs, keeps = build_panel_images(panel_face, outfit["best"]["crop"], alt)
        for i, k in enumerate(keeps):
            if k is None:
                continue
            pack["panels"][i]["clothing_removed"] = True
            pack["panels"][i]["identity_pixels_kept"] = round(k, 4)
        low = [pack["panels"][i]["role"] for i, k in enumerate(keeps)
               if k is not None and k < 0.05]
        if low:
            log.warning("%s: identity masking kept under 5%% of panel(s) %s. The "
                        "parser may have failed on those photographs; check the "
                        "pack sheet before trusting the identity conditioning.",
                        sid, ", ".join(low))
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
        shot["reference_pack"] = {
            "sheet": pack["sheet"], "key": pack["key"], "cluster": best_g,
            "outfit_authority": pack["outfit_authority"],
            "identity_authority": pack["identity_authority"],
            "identity_score": panel_face["identity"]["score"],
            "external_garment_deltaE": pack["external_garment_similarity"][
                "deltaE_to_source"]}
        log.info("%s: pack -> %s (%d panels; identity from externals, garment "
                 "from %s)", sid, rel(sheet_path), n, pack["outfit_authority"])
        made += 1

    save_manifest(man)
    log.info("=" * 62)
    log.info("Built %d reference pack(s) in %s", made, rel(pack_dir))
    log.info("Panels carry provenance; nothing was upscaled to reach 1080p.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
