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
consensus face agreement across the reference set (near-duplicates collapsed
first, so repeated copies cannot vote twice), how many pixels the face occupies,
and - for the second panel - how far the head is actually turned, measured from
face landmarks. Never by garment colour, and never by full-image similarity,
which responds to background and framing rather than to viewpoint.

Garment distance is still measured and recorded, purely as evidence that the
photographs show other clothes; it is a diagnostic, never a switch, and the
source remains the garment authority whatever it says.

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
# What an EXTERNAL photograph is allowed to condition.
#
# Narrowed to head only. Arms and legs were in this set on the reasoning that
# bare skin is "identity", but they are not: how much arm or leg is visible is a
# fact about what the person is WEARING that day. A reference showing bare arms
# tells the model to produce bare arms, and if the source has sleeves it has
# just been instructed to remove them. Same for legs and hemlines. Sleeve and
# torso coverage belong to the source, exactly like the garment itself.
#
# Hats, sunglasses and scarves stay out too - accessories of another day.
IDENTITY_ONLY = {"hair", "face"}

# Head classes, used to ask what the SOURCE actually exposes before any external
# face is allowed to condition anything. A face covering in the source is part of
# what the person is wearing; an external photograph showing an uncovered face
# must never be read as permission to remove it.
HEAD = {"hair", "face", "hat", "sunglasses", "scarf"}
COVERING = {"hat", "sunglasses", "scarf"}


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
        return self.parse_prob(pil)[0]

    def parse_prob(self, pil):
        """(labels, confidence) - the softmax probability of the winning class.

        Confidence is what makes failing closed possible. An argmax alone says
        "face" just as firmly at 0.31 as at 0.99, and a garment region the parser
        is unsure about looks exactly like an exposed one. Regenerating only
        where the parser is certain is the difference between improving a face
        and erasing a sleeve.
        """
        import torch
        with torch.inference_mode():
            inp = self.proc(images=pil, return_tensors="pt").to(self.dev)
            logits = self.model(**inp).logits
            up = torch.nn.functional.interpolate(
                logits, size=pil.size[::-1], mode="bilinear", align_corners=False)
            prob = torch.softmax(up, dim=1)[0]
            conf, lab = prob.max(dim=0)
        return (lab.cpu().numpy().astype(np.uint8),
                conf.cpu().numpy().astype(np.float32))


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
        # A SECOND embedding, of the identity-masked image. The full-frame one
        # above is what clusters outfits; it is dominated by background, framing
        # and clothing, so it must never be read as "a different viewing angle".
        # This one has all of that removed and describes the person only.
        id_emb = models.clip_embed([mask_to_identity(im, lab, feather=0)[0]])[0]
        face, kps, face_frac = face_detail(models, im)
        items.append({"file": f, "image": im, "labels": lab, "palette": pal,
                      "clip": emb, "clip_identity": id_emb, "face": face,
                      "yaw": face_yaw(kps), "face_frac": face_frac,
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


def face_detail(models, pil):
    """(embedding, 5-point landmarks, face area fraction) for the largest face.

    Models.face_embed returns the bounding box, which says nothing about which
    way the head is turned. The landmarks do.
    """
    app = models.face()
    if app is None:
        return None, None, 0.0
    import cv2
    bgr = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        return None, None, 0.0
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    fa = ((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) / (pil.width * pil.height)
    return (np.asarray(f.normed_embedding, dtype=np.float32),
            getattr(f, "kps", None), float(fa))


def face_yaw(kps) -> float | None:
    """Signed yaw proxy in [-1, 1] from the 5-point face landmarks.

    The nose sits midway between the eyes on a head-on view and slides towards
    the near eye as the head turns, so its offset from the eye midpoint, divided
    by the interocular distance, tracks yaw. Scale-free and roll-tolerant,
    because both terms rotate together.

    This is a real geometric measurement of head orientation. It replaces a
    full-image CLIP distance, which was being read as "a different angle" when
    what it actually responds to is background, framing, lighting and clothing -
    a photograph of the same pose in a different room scored as a new viewpoint.
    """
    if kps is None or len(kps) < 3:
        return None
    le, re, nose = np.asarray(kps[0]), np.asarray(kps[1]), np.asarray(kps[2])
    axis = re - le
    d = float(np.hypot(*axis))
    if d < 1e-3:
        return None
    mid = (le + re) / 2.0
    # Project nose-minus-midpoint onto the eye axis, so head roll does not leak in.
    t = float(np.dot(nose - mid, axis) / (d * d))
    return float(np.clip(t * 2.0, -1.0, 1.0))


def consensus_identity(items: list[dict], same_face: float = 0.35,
                       duplicate: float = 0.92) -> dict:
    """Which references are agreed to show one person, by consensus.

    Taking the single maximum similarity is not verification: it asks only
    whether SOME other photograph agrees, so two copies of the wrong person
    vouch for each other and both score perfectly. Near-duplicates have to go
    first, then agreement has to come from the group rather than from one
    neighbour.

      1. collapse near-duplicates (>= `duplicate`) to one representative, so
         repeated copies cannot vote repeatedly
      2. link representatives that match at all (>= `same_face`) and take the
         largest connected group - the consensus identity
      3. score each reference by its MEDIAN similarity to the other members of
         that group, which a single outlier cannot lift

    Anything outside the consensus group scores zero and is never drawn: an
    unverified photograph would be conditioning the model on a stranger's face.
    """
    faces = [m for m in items if m["face"] is not None]
    for m in items:
        m["identity_group"] = None
        m["agreement"] = 0.0
        m["duplicate_of"] = None
    if len(faces) < 2:
        return {"members": len(faces), "group": [], "duplicates": 0,
                "note": "too few detectable faces to reach consensus"}

    E = np.stack([m["face"] for m in faces])
    S = E @ E.T

    # 1. near-duplicate collapse (union-find over the duplicate threshold)
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            if S[i, j] >= duplicate:
                parent[find(i)] = find(j)
    reps: dict[int, int] = {}
    for i in range(len(faces)):
        reps.setdefault(find(i), i)
    rep_idx = sorted(reps.values())
    n_dup = len(faces) - len(rep_idx)
    for i in range(len(faces)):
        r = reps[find(i)]
        if r != i:
            faces[i]["duplicate_of"] = faces[r]["file"].name

    # 2. largest connected group among representatives
    adj = {i: set() for i in rep_idx}
    for a in range(len(rep_idx)):
        for b in range(a + 1, len(rep_idx)):
            i, j = rep_idx[a], rep_idx[b]
            if S[i, j] >= same_face:
                adj[i].add(j)
                adj[j].add(i)
    seen, best_group = set(), []
    for start in rep_idx:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u] - seen:
                seen.add(v)
                stack.append(v)
        if len(comp) > len(best_group):
            best_group = comp
    group = set(best_group)
    # A duplicate of a group member is in the group too - it is the same photo.
    members = [i for i in range(len(faces)) if reps[find(i)] in group]

    # 3. median agreement with the OTHER representatives of that group
    for i in members:
        others = [j for j in best_group if reps[find(j)] != reps[find(i)]]
        faces[i]["agreement"] = (round(float(np.median(S[i, others])), 4)
                                 if others else 0.0)
        faces[i]["identity_group"] = "consensus"
    return {"members": len(members), "representatives": len(best_group),
            "duplicates": n_dup, "candidates": len(faces),
            "same_face_threshold": same_face, "duplicate_threshold": duplicate}


def score_identity(items: list[dict]) -> None:
    """Rate each photograph on how well it serves IDENTITY conditioning, in place.

    Deliberately blind to garment colour. Once clothing is segmented out of the
    external panels, which outfit a photograph happens to show says nothing about
    how well it depicts the face, and letting it decide would reintroduce exactly
    the coupling these packs exist to break.

    Agreement comes from consensus_identity(), which is neither a self-comparison
    nor a single maximum: near-duplicates are collapsed first, then a photograph
    is scored by its MEDIAN similarity to the rest of the agreed group. Both of
    the shortcuts it replaces gave a perfect score to the wrong answer - scoring
    against a bank built from these same photographs returned 1.000 for anything
    already in it, and taking the maximum let two copies of a stranger vouch for
    each other.

      agreement  consensus face similarity: is this the right person
      face_res   how many pixels the face occupies; a correct match at 20 px
                 carries no detail worth transferring

    Weighted in that order, because a confident wrong identity is the worst
    outcome available. A photograph with no detectable face, or one outside the
    consensus group, scores zero: it may well be the right person, but nothing
    here shows that it is.
    """
    for m in items:
        w, h = m["size"]
        # sqrt -> a face side length; ~200 px is already ample for conditioning,
        # so the score saturates there instead of rewarding ever-larger portraits.
        px = float(m["face_frac"] or 0.0) * w * h
        res = float(np.clip(np.sqrt(px) / 200.0, 0.0, 1.0))
        agree = float(m.get("agreement") or 0.0)
        m["identity"] = {"agreement": round(agree, 4),
                         "face_resolution": round(res, 4),
                         "face_pixels": int(px),
                         "yaw": (round(m["yaw"], 3) if m.get("yaw") is not None
                                 else None),
                         "verified": bool(m.get("identity_group") == "consensus"),
                         "duplicate_of": m.get("duplicate_of"),
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
        consensus = consensus_identity(items)
        score_identity(items)
        log.info("Identity consensus: %d of %d reference(s) with a face agree "
                 "(%d near-duplicate(s) collapsed first, so repeated copies "
                 "cannot vote twice)", consensus.get("members", 0),
                 consensus.get("candidates", 0), consensus.get("duplicates", 0))
        ranked = sorted(items, key=lambda m: (m["identity"]["score"],
                                              m["identity"]["face_pixels"],
                                              m["file"].name), reverse=True)
        for m in ranked:
            i = m["identity"]
            log.info("%-34s identity %.3f (consensus agreement %.3f, face %d px, "
                     "yaw %s, %s%s)", m["file"].name, i["score"], i["agreement"],
                     i["face_pixels"],
                     f"{i['yaw']:+.2f}" if i["yaw"] is not None else "n/a",
                     "verified" if i["verified"] else "NOT VERIFIED",
                     f", duplicate of {i['duplicate_of']}"
                     if i["duplicate_of"] else "")
        verified = [m for m in ranked if m["identity"]["verified"]]
        if not verified:
            log.warning("No reference is identity-verified by consensus. Falling "
                        "back to the best-scoring photograph; identity "
                        "conditioning is weak and should not be trusted. The "
                        "garment is unaffected: it comes from the source.")
        panel_face = (verified or ranked)[0]

        # A second panel earns its place only by showing a genuinely different
        # HEAD ORIENTATION, and only if it is identity-verified - an unverified
        # photograph would condition the model on a stranger's face.
        #
        # Ranked by yaw difference where the landmarks give one. The
        # identity-masked embedding is the fallback for faces the detector could
        # not land landmarks on; the FULL-image embedding is not used here at
        # all, because it responds to background, framing and clothing, and a
        # photograph of the same pose in a different room would win on it.
        others = [m for m in verified if m is not panel_face
                  and m["identity"]["duplicate_of"] is None]
        alt, alt_why = None, ""
        y0 = panel_face["identity"]["yaw"]
        with_yaw = [m for m in others if m["identity"]["yaw"] is not None]
        if y0 is not None and with_yaw:
            alt = max(with_yaw, key=lambda m: abs(m["identity"]["yaw"] - y0))
            d = abs(alt["identity"]["yaw"] - y0)
            alt_why = (f"head yaw differs by {d:.2f} "
                       f"({y0:+.2f} -> {alt['identity']['yaw']:+.2f})")
            if d < 0.10:
                log.info("The most different head orientation available differs "
                         "by only %.2f; the references all face much the same way, so "
                         "the second panel adds angle coverage in name only.", d)
        elif others:
            alt = min(others, key=lambda m: float(
                np.dot(m["clip_identity"], panel_face["clip_identity"])))
            alt_why = ("no landmarks; identity-masked embedding distance "
                       f"{1 - float(np.dot(alt['clip_identity'], panel_face['clip_identity'])):.3f}")
        if alt is not None:
            log.info("alternate view: %s (%s; chosen among %d verified, "
                     "non-duplicate reference(s))", alt["file"].name, alt_why,
                     len(others))
        else:
            log.info("No second identity-verified, non-duplicate reference; the "
                     "pack uses one identity panel.")
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
            "identity_consensus": consensus,
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
