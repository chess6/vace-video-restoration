#!/usr/bin/env python
"""Phase 5 - build a VACE reference sheet from inputs/references/.

Why a single composited sheet rather than a list of images:
WanVaceToVideo in the installed ComfyUI does

    reference_image = common_upscale(reference_image[:1] ...)

i.e. it consumes exactly ONE image and rescales it to the generation WxH.
Passing three separate references is therefore impossible; the useful views have
to be tiled into one frame. That is what this script produces.

What it does:
  * reads PNG / JPEG / WebP (and BMP/TIFF), never modifying the originals
  * applies EXIF orientation
  * rejects corrupt, tiny, and duplicate images (perceptual hash)
  * detects people and faces, embeds faces, and clusters them so that photos of
    OTHER people are dropped instead of contaminating the reference
  * classifies each usable image as face / upper / full-body / detail view
  * scores by sharpness, subject coverage and resolution, and picks up to 3
    complementary views (prefers: one full body, one face, one alternate angle)
  * composites a clean sheet: no captions, no watermarks, no borders, minimal
    empty space (panels are cover-cropped, not letterboxed)
  * writes a separate contact sheet for inspection, and a JSON provenance record

Nothing is ever displayed on screen; everything is written to disk.

    scripts/prepare_references.py [--config ...] [--max-refs 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMAGE_EXTS, P, load_config, setup_logging  # noqa: E402

MIN_SIDE = 256          # below this a reference adds noise, not identity
MIN_SHARPNESS = 12.0    # Laplacian variance on the subject crop


# ---------------------------------------------------------------------------
# basic image analysis
# ---------------------------------------------------------------------------

def load_exif_upright(path: Path) -> Image.Image | None:
    try:
        im = Image.open(path)
        im.load()
        im = ImageOps.exif_transpose(im)      # correct EXIF rotation
        return im.convert("RGB")
    except Exception:
        return None


def phash(im: Image.Image, size: int = 16) -> int:
    """Perceptual hash via DCT-free mean-threshold on a small grayscale image."""
    g = np.asarray(im.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)
    bits = (g > g.mean()).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def sharpness(im: Image.Image) -> float:
    import cv2
    g = np.asarray(im.convert("L"), dtype=np.uint8)
    if max(g.shape) > 1024:
        s = 1024 / max(g.shape)
        g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# detection / identity
# ---------------------------------------------------------------------------

def get_face_app(log):
    try:
        from insightface.app import FaceAnalysis
        import onnxruntime as ort
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0 if "CUDA" in providers[0] else -1, det_size=(640, 640))
        log.info("Face analysis: insightface buffalo_l (%s)", providers[0])
        return app
    except Exception as e:
        log.warning("insightface unavailable (%s). Falling back to person-box "
                    "heuristics only; identity clustering will be skipped.", e)
        return None


def detect_person_boxes(images: list[Image.Image], log) -> list[list[tuple]]:
    """Open-vocabulary person detection with Grounding DINO.

    Returns, per image, a list of (x0, y0, x1, y1, score).
    """
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from common import require_cuda

    dev = require_cuda(log)
    mid = "IDEA-Research/grounding-dino-base"
    log.info("Loading detector %s", mid)
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        mid, dtype=torch.float16).to(dev).eval()

    out: list[list[tuple]] = []
    with torch.inference_mode():
        for im in images:
            inputs = proc(images=im, text="a person. a human. a man. a woman.",
                          return_tensors="pt").to(dev)
            res = model(**inputs)
            post = proc.post_process_grounded_object_detection(
                res, inputs.input_ids, threshold=0.30, text_threshold=0.25,
                target_sizes=[(im.height, im.width)])[0]
            boxes = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s))
                     for b, s in zip(post["boxes"].cpu(), post["scores"].cpu())]
            boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
            out.append(boxes)
    del model
    torch.cuda.empty_cache()
    return out


def classify_view(box: tuple | None, face_box: tuple | None, im: Image.Image) -> str:
    W, H = im.size
    if box is None:
        return "detail"
    bw, bh = box[2] - box[0], box[3] - box[1]
    cover_h = bh / H
    if face_box is not None:
        fh = face_box[3] - face_box[1]
        if fh / max(bh, 1) > 0.45:
            return "face"
    if cover_h > 0.80 and bh / max(bw, 1) > 1.7:
        return "full_body"
    if cover_h > 0.55:
        return "upper_body"
    return "detail"


# ---------------------------------------------------------------------------
# compositing
# ---------------------------------------------------------------------------

def cover_crop(im: Image.Image, box: tuple | None, tw: int, th: int,
               pad_frac: float = 0.06) -> Image.Image:
    """Crop around the subject then scale to exactly (tw, th) with NO empty space.

    Uses cover semantics: fill the panel and trim the overflow, so the sheet has
    no letterbox bars that VACE could mistake for scene content.
    """
    W, H = im.size
    if box is not None:
        x0, y0, x1, y1 = box[:4]
        pw, ph = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
        x0, y0, x1, y1 = x0 - pw, y0 - ph, x1 + pw, y1 + ph
    else:
        x0, y0, x1, y1 = 0, 0, W, H

    # widen the crop to the panel aspect where the image allows it
    target_ar = tw / th
    cw, ch = x1 - x0, y1 - y0
    if cw / ch < target_ar:
        need = ch * target_ar
        cx = (x0 + x1) / 2
        x0, x1 = cx - need / 2, cx + need / 2
    else:
        need = cw / target_ar
        cy = (y0 + y1) / 2
        y0, y1 = cy - need / 2, cy + need / 2

    # clamp into the image, preserving size where possible
    if x1 - x0 > W:
        x0, x1 = 0, W
    else:
        if x0 < 0: x1, x0 = x1 - x0, 0
        if x1 > W: x0, x1 = x0 - (x1 - W), W
        x0 = max(0.0, x0)
    if y1 - y0 > H:
        y0, y1 = 0, H
    else:
        if y0 < 0: y1, y0 = y1 - y0, 0
        if y1 > H: y0, y1 = y0 - (y1 - H), H
        y0 = max(0.0, y0)

    crop = im.crop((int(x0), int(y0), int(round(x1)), int(round(y1))))
    # final cover-resize
    cw, ch = crop.size
    scale = max(tw / cw, th / ch)
    crop = crop.resize((max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))),
                       Image.LANCZOS)
    left = (crop.width - tw) // 2
    top = (crop.height - th) // 2
    return crop.crop((left, top, left + tw, top + th))


def layout_panels(n: int, W: int, H: int) -> list[tuple[int, int, int, int]]:
    """Panel rectangles that tile the sheet completely (no gaps, no borders)."""
    if n <= 1:
        return [(0, 0, W, H)]
    if n == 2:
        half = W // 2
        return [(0, 0, half, H), (half, 0, W - half, H)]
    # 3: a tall primary panel (full body) on the left, two stacked on the right
    left = int(W * 0.5)
    top_h = H // 2
    return [(0, 0, left, H), (left, 0, W - left, top_h), (left, top_h, W - left, H - top_h)]


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--refs-dir", type=Path, default=None)
    ap.add_argument("--max-refs", type=int, default=3)
    ap.add_argument("--identity-threshold", type=float, default=0.28,
                    help="Cosine similarity below which a face is treated as a different person")
    ap.add_argument("--no-detect", action="store_true",
                    help="Skip Grounding DINO (faster; uses whole-image panels)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("prepare_references", args.verbose)
    cfg = load_config(args.config)
    W, H = cfg["video"]["width"], cfg["video"]["height"]
    refs_dir = args.refs_dir or P.references

    files = sorted(p for p in refs_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not files:
        log.error("No reference images in %s. Supported: %s",
                  refs_dir, ", ".join(sorted(IMAGE_EXTS)))
        return 1
    log.info("Found %d candidate reference image(s) in %s", len(files), refs_dir)

    # ---- load + basic rejection ---------------------------------------------
    recs: list[dict] = []
    for f in files:
        im = load_exif_upright(f)
        if im is None:
            log.warning("REJECT %s: unreadable or corrupt", f.name)
            continue
        if min(im.size) < MIN_SIDE:
            log.warning("REJECT %s: %dx%d is below the %dpx minimum side",
                        f.name, im.width, im.height, MIN_SIDE)
            continue
        recs.append({"path": f, "im": im, "phash": phash(im),
                     "sharp": sharpness(im), "w": im.width, "h": im.height})

    if not recs:
        log.error("No usable reference images survived validation.")
        return 1

    # ---- duplicate rejection -------------------------------------------------
    kept: list[dict] = []
    for r in recs:
        dup = next((k for k in kept if hamming(k["phash"], r["phash"]) <= 12), None)
        if dup:
            # keep the sharper / larger of the pair
            if r["sharp"] * r["w"] * r["h"] > dup["sharp"] * dup["w"] * dup["h"]:
                log.info("DUPLICATE %s ~ %s: keeping %s (sharper/larger)",
                         r["path"].name, dup["path"].name, r["path"].name)
                kept[kept.index(dup)] = r
            else:
                log.info("DUPLICATE %s ~ %s: dropping %s",
                         r["path"].name, dup["path"].name, r["path"].name)
            continue
        kept.append(r)
    log.info("%d image(s) after duplicate removal", len(kept))

    # ---- person detection ----------------------------------------------------
    if args.no_detect:
        for r in kept:
            r["box"] = None
    else:
        try:
            boxes = detect_person_boxes([r["im"] for r in kept], log)
            for r, bs in zip(kept, boxes):
                r["boxes"] = bs
                r["box"] = bs[0] if bs else None
                r["n_people"] = len(bs)
                if len(bs) > 1:
                    log.warning("%s contains %d detected people. Only the largest "
                                "is used; verify the contact sheet.", r["path"].name, len(bs))
        except Exception as e:
            log.warning("Detection failed (%s); using whole-image panels.", e)
            for r in kept:
                r["box"] = None

    # ---- face embedding + identity clustering -------------------------------
    app = get_face_app(log)
    if app is not None:
        import cv2
        for r in kept:
            bgr = cv2.cvtColor(np.asarray(r["im"]), cv2.COLOR_RGB2BGR)
            faces = app.get(bgr)
            if faces:
                f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                r["face_box"] = tuple(float(v) for v in f.bbox)
                emb = np.asarray(f.normed_embedding, dtype=np.float32)
                r["face_emb"] = emb
            else:
                r["face_box"] = None
                r["face_emb"] = None

        embs = [r for r in kept if r.get("face_emb") is not None]
        if len(embs) >= 2:
            # dominant identity = the face most similar to all the others
            M = np.stack([r["face_emb"] for r in embs])
            sim = M @ M.T
            medoid = int(np.argmax(sim.sum(axis=1)))
            ref_emb = M[medoid]
            for r in embs:
                r["id_sim"] = float(ref_emb @ r["face_emb"])
            outliers = [r for r in embs if r["id_sim"] < args.identity_threshold]
            for r in outliers:
                log.warning("REJECT %s: face similarity %.3f to the dominant identity "
                            "is below %.2f - this looks like a different person.",
                            r["path"].name, r["id_sim"], args.identity_threshold)
            kept = [r for r in kept if r not in outliers]
            log.info("Identity clustering: dominant face from %s, %d image(s) kept",
                     embs[medoid]["path"].name, len(kept))
        else:
            log.info("Fewer than 2 faces detected; skipping identity clustering.")
    else:
        for r in kept:
            r["face_box"] = None
            r["face_emb"] = None

    if not kept:
        log.error("All reference images were rejected.")
        return 1

    # ---- classify + score ----------------------------------------------------
    for r in kept:
        r["view"] = classify_view(r.get("box"), r.get("face_box"), r["im"])
        cover = 1.0
        if r.get("box"):
            b = r["box"]
            cover = ((b[2] - b[0]) * (b[3] - b[1])) / (r["w"] * r["h"])
        r["cover"] = cover
        res_score = min(1.0, (min(r["w"], r["h"]) / 1024.0))
        sharp_score = min(1.0, r["sharp"] / 200.0)
        r["score"] = 0.45 * sharp_score + 0.30 * res_score + 0.25 * min(1.0, cover * 2.5)
        log.info("%-32s view=%-10s sharp=%7.1f cover=%.2f score=%.3f",
                 r["path"].name, r["view"], r["sharp"], cover, r["score"])
        if r["sharp"] < MIN_SHARPNESS:
            log.warning("%s is very soft (Laplacian var %.1f); it may blur the reference.",
                        r["path"].name, r["sharp"])

    # ---- pick up to N complementary views ------------------------------------
    # Priority order deliberately covers body first: the brief says non-facial
    # identity (clothing, silhouette, accessories) matters as much as the face.
    selected: list[dict] = []
    for want in ("full_body", "face", "upper_body", "detail"):
        pool = [r for r in kept if r["view"] == want and r not in selected]
        if pool and len(selected) < args.max_refs:
            selected.append(max(pool, key=lambda r: r["score"]))
    for r in sorted(kept, key=lambda r: -r["score"]):
        if len(selected) >= args.max_refs:
            break
        if r not in selected:
            selected.append(r)
    # primary panel should be the most body-revealing view
    order = {"full_body": 0, "upper_body": 1, "face": 2, "detail": 3}
    selected.sort(key=lambda r: (order.get(r["view"], 9), -r["score"]))
    log.info("Selected %d view(s): %s", len(selected),
             ", ".join(f"{r['path'].name}({r['view']})" for r in selected))

    # ---- composite the sheet -------------------------------------------------
    P.reference_sheets.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (W, H), (0, 0, 0))
    panels = layout_panels(len(selected), W, H)
    for r, (px, py, pw, ph) in zip(selected, panels):
        # For a face view, prefer the face box so the panel is not mostly torso.
        box = r.get("face_box") if r["view"] == "face" and r.get("face_box") else r.get("box")
        sheet.paste(cover_crop(r["im"], box, pw, ph), (px, py))

    sheet_path = P.reference_sheets / "reference_sheet.png"
    sheet.save(sheet_path)
    log.info("Reference sheet -> %s (%dx%d, %d panel(s), no captions/borders)",
             sheet_path, W, H, len(selected))

    # ComfyUI's LoadImage reads from its input directory
    P.comfy_input.mkdir(parents=True, exist_ok=True)
    sheet.save(P.comfy_input / "reference_sheet.png")

    # ---- contact sheet for inspection (written, never displayed) -------------
    cols = min(4, max(1, len(kept)))
    rows = (len(kept) + cols - 1) // cols
    cw, ch = 320, 320
    contact = Image.new("RGB", (cols * cw, rows * ch), (18, 18, 18))
    for i, r in enumerate(sorted(kept, key=lambda r: -r["score"])):
        thumb = r["im"].copy()
        thumb.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
        x = (i % cols) * cw + (cw - thumb.width) // 2
        y = (i // cols) * ch + (ch - thumb.height) // 2
        contact.paste(thumb, (x, y))
    contact_path = P.reference_sheets / "contact_sheet.png"
    contact.save(contact_path)
    log.info("Contact sheet -> %s", contact_path)

    # ---- provenance ----------------------------------------------------------
    prov = {
        "sheet": str(sheet_path.relative_to(P.root)),
        "sheet_size": [W, H],
        "contact_sheet": str(contact_path.relative_to(P.root)),
        "selected": [{
            "file": str(r["path"]), "view": r["view"], "score": round(r["score"], 4),
            "sharpness": round(r["sharp"], 2), "size": [r["w"], r["h"]],
            "person_box": r.get("box"), "face_box": r.get("face_box"),
            "identity_similarity": round(r["id_sim"], 4) if "id_sim" in r else None,
        } for r in selected],
        "considered": [str(r["path"]) for r in kept],
        "rejected_note": "See logs/prepare_references.log for per-file rejection reasons.",
        "note": ("WanVaceToVideo consumes exactly one reference image "
                 "(reference_image[:1]), which is why these views are tiled into a "
                 "single sheet at the generation resolution."),
    }
    (P.reference_sheets / "reference_provenance.json").write_text(json.dumps(prov, indent=2))
    log.info("Provenance -> %s", P.reference_sheets / "reference_provenance.json")
    log.info("Originals in %s were not modified.", refs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
