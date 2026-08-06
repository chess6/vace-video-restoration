#!/usr/bin/env python
"""Phase 5b - match-only external conditioning + source-derived attribute.

The task here is to RESTORE the attribute that is already in the footage, not to
infer an attribute from references captured elsewhere. Those two things were being
conflated, and that is why attribute colours were invented.

Strict division of authority:

  * External references condition MATCH ONLY - the anchor region and the exposed
    regions around it. Their attributes are segmented out and replaced with neutral
    grey before the panel is drawn, so they cannot reach the model at all. A faint
    ghost of the wrong attribute is still an attribute to a generative model.
  * THIS SOURCE INTERVAL is the sole ground truth for the attribute: class,
    silhouette, boundaries, colours, patterns, accessories, folds and how they
    move. It is low resolution, and that is fine - the job is to preserve its
    low-frequency colour and structure while generating only the missing
    high-frequency texture.

Which references become panels is decided by MATCH EVIDENCE ALONE:
consensus anchor agreement across the reference set (near-duplicates collapsed
first, so repeated copies cannot vote twice), how many pixels the anchor occupies,
and - for the second panel - how far the anchor is actually turned, measured from
its keypoints. Never by attribute colour, and never by full-image similarity,
which responds to background and framing rather than to viewpoint.

Attribute distance is still measured and recorded, purely as evidence that the
references show other attributes; it is a diagnostic, never a switch, and the
source remains the attribute authority whatever it says.

Appearance clustering still runs and is still recorded, but it no longer confines
the choice. Its job was to stop two appearances being combined in one conditioning
image, and there is no longer an appearance in an external panel to conflict - so
restricting panels to a single cluster would only discard viewing angles.

Clustering is automatic (CLIP appearance embeddings + attribute histograms), and
so is attribute parsing (SegFormer). No manual labelling anywhere.

The parser's label strings below are its own, read from the model card's id2label
and indexed by string, so they stay verbatim (the dependency floor of rule 2a).
Everything built ON them - set names, record keys, prose - uses role words.

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

# SegFormer attributes label map (from the model card's id2label).
LBL = {0: "background", 1: "hat", 2: "hair", 3: "sunglasses", 4: "upper",
       5: "skirt", 6: "pants", 7: "dress", 8: "belt", 9: "left_shoe",
       10: "right_shoe", 11: "face", 12: "left_leg", 13: "right_leg",
       14: "left_arm", 15: "right_arm", 16: "bag", 17: "scarf"}
ATTRIBUTE = {"hat", "upper", "skirt", "pants", "dress", "belt", "left_shoe",
           "right_shoe", "bag", "scarf"}
EXPOSED = {"face", "left_leg", "right_leg", "left_arm", "right_arm"}
# What an EXTERNAL reference is allowed to condition.
#
# Narrowed to the anchor region only. The peripheral extents were in this set on
# the reasoning that an exposed region is "match", but they are not: how much of a
# peripheral extent is visible is a fact about the attribute the candidate carries
# that day. A reference showing more of one instructs the model to expose more, and
# if the source covers it, the model has just been told to uncover it. Coverage
# belongs to the source, exactly like the attribute itself.
#
# Accessory classes stay out too - they are attributes of another day.
MATCH_ONLY = {"hair", "face"}

# The anchor region's classes, used to ask what the SOURCE actually exposes before
# any external anchor is allowed to condition anything. A covering in the source is
# part of the attribute the candidate presents; an external reference showing it
# uncovered must never be read as permission to remove it.
ANCHOR_REGION = {"hair", "face", "hat", "sunglasses", "scarf"}
COVERING = {"hat", "sunglasses", "scarf"}


def match_regions(labels: np.ndarray, allowed: set | None = None,
                     keep: set | None = None) -> np.ndarray:
    """Boolean mask of the pixels of an external reference that may be shown.

    `allowed` narrows MATCH_ONLY further for one shot. It is how a covered
    source anchor is protected: with the anchor class removed, an external
    reference showing it uncovered contributes the surrounding class and nothing
    else, so it cannot instruct the model to take the covering off. It can only
    ever narrow - intersecting is the safety property, so a caller passing a wider
    set gets the narrow one.

    `keep` REPLACES the set instead of narrowing it, and exists for callers
    outside the reference pack that mean something different by match -
    grey_candidate_attributes.py keeps the exposed regions as well as the anchor
    region, on the user's instruction, for a transfer path where nothing
    regenerates the attribute. Anything on the pack's own path passes `allowed`
    and cannot widen anything.
    """
    if keep is not None:
        names = set(keep)
    else:
        names = MATCH_ONLY if allowed is None else (set(allowed) & MATCH_ONLY)
    ids = [i for i, n in LBL.items() if n in names]
    return np.isin(labels, ids)


def mask_to_match(im, labels, feather: int = 3, allowed: set | None = None,
                     keep: set | None = None):
    """Blank everything that is not match in an external reference.

    Attributes, accessories and background are removed rather than dimmed: a faint
    ghost of the wrong attribute is still an attribute as far as the model is
    concerned.
    A few pixels of feather stop the cut-out edge from reading as a hard graphic
    shape, which would itself become something to reproduce.
    """
    from PIL import Image
    arr = np.asarray(im).astype(np.float32)
    hard = match_regions(labels, allowed, keep)
    m = hard.astype(np.float32)
    if feather > 0:
        import cv2
        r = feather * 2 + 1
        # Erode first, blur, then re-multiply by the hard mask. The ramp then
        # lies entirely INSIDE the match region and no pixel outside it keeps
        # any of its original value.
        #
        # A plain blur ramps OUTWARDS. Measured on a real reference, that let up
        # to 83/255 of the true pixel survive in a ring around the anchor region -
        # and what lies immediately around it is attribute-bearing. On a reference
        # where that ring is uncovered and the footage shows it covered, the ring
        # is a hint to uncover it. Same one-sided reasoning as the occluder
        # boundary in composite_subject.py: the ramp may only ever give ground
        # inwards.
        k = np.ones((2 * feather + 1,) * 2, np.uint8)
        inner = cv2.erode(hard.astype(np.uint8), k).astype(np.float32)
        m = cv2.GaussianBlur(inner, (r, r), 0) * m
    # Neutral mid-grey, not black: a large black field shifts the exposure the
    # model infers from the panel.
    out = arr * m[..., None] + 128.0 * (1.0 - m[..., None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), float(m.mean())


def build_panel_images(anchor_panel: dict, source_crop, alt_panel: dict | None,
                       allowed: set | None = None):
    """The panel images, in sheet order, with the authority split enforced here.

    Externals in, match-only out; the source crop passes through untouched.
    Keeping this in one function means the rule is one testable statement rather
    than something a future edit to the rendering loop can quietly break.

    Returns (images, kept_fractions) where kept_fractions[i] is None for the
    source panel, which is never masked.
    """
    from PIL import Image
    anchor_img, anchor_keep = mask_to_match(anchor_panel["image"],
                                           anchor_panel["labels"], allowed=allowed)
    imgs = [anchor_img, Image.fromarray(np.asarray(source_crop))]
    keeps: list[float | None] = [anchor_keep, None]
    if alt_panel is not None:
        alt_img, alt_keep = mask_to_match(alt_panel["image"],
                                             alt_panel["labels"], allowed=allowed)
        imgs.append(alt_img)
        keeps.append(alt_keep)
    return imgs, keeps


class Parser:
    """SegFormer attributes parsing, loaded once."""

    def __init__(self, log):
        import torch
        from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading attributes parser %s", SEGFORMER_ID)
        self.proc = SegformerImageProcessor.from_pretrained(
            SEGFORMER_ID, revision=SEGFORMER_REV)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            SEGFORMER_ID, revision=SEGFORMER_REV).to(self.dev).eval()

    def parse(self, pil):
        """Returns a HxW label map at the image's own resolution."""
        return self.parse_prob(pil)[0]

    def parse_prob(self, pil):
        """(labels, confidence) - the softmax probability of the winning class.

        Confidence is what makes failing closed possible. An argmax alone names a
        class just as firmly at 0.31 as at 0.99, and an attribute region the parser
        is unsure about looks exactly like an exposed one. Regenerating only
        where the parser is certain is the difference between improving the anchor
        and erasing part of an attribute.
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
    """Median Lab colour per attribute class, plus its pixel share.

    Median rather than mean: a few blown-out highlights or a dark fold should not
    move the recorded colour of an attribute.
    """
    import cv2
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    out = {}
    total = labels.size
    for idx, name in LBL.items():
        if name not in ATTRIBUTE:
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


def source_appearance(work: Path, mask_video: Path, parser: Parser, n_probe: int,
                  log) -> dict:
    """Learn the CURRENT appearance from this interval: the only correct authority.

    Picks the frames where the subject is largest and sharpest, parses attributes
    on each, and keeps a temporally smoothed palette plus the single best frame
    as an exact-appearance panel.
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
        raise RuntimeError("no frames to learn the appearance from")

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
        raise RuntimeError("attributes parsing found no attributes in the source")

    # Temporal smoothing: median across probe frames, and the spread as a
    # stability measure. An attribute whose colour swings between frames is not a
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
    log.info("Source appearance (%d probe frames): %s", len(per_frame),
             ", ".join(f"{k}(dE{v['temporal_deltaE']:.1f}"
                       f"{'' if v['stable'] else ' UNSTABLE'})"
                       for k, v in smoothed.items()) or "nothing parsed")
    return {"palette": smoothed, "best": best, "n_probe": len(per_frame)}


def cluster_references(files: list[Path], parser: Parser, models, log,
                       thresh: float, verified: dict | None = None) -> list[dict]:
    """Group the external references by appearance, and gather everything the
    panel choice needs: attribute palette, CLIP embedding, anchor embedding, size.

    The grouping is now diagnostic - see the module docstring. Match is a
    separate question, decided in score_match() from these same embeddings.

    `verified` maps a resolved path to the target instance that
    reference_match.resolve_targets already agreed on. It is not optional in practice:
    without it this function called anchor_detail(), which returns the LARGEST
    anchor in the reference, silently discarding the verified target. In an image
    where the subject is not the biggest anchor - a group shot, a non-target nearer
    the camera - the panel choice, the orientation and the anchor fraction would
    then all describe the wrong candidate, in a function whose whole job is to prepare
    match conditioning. main() filters `files` to verified images, so only
    WHICH anchor was taken was wrong, not which files; that is quiet enough to
    have survived this long.
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
        # A SECOND embedding, of the match-masked image. The full-frame one
        # above is what clusters appearances; it is dominated by background, framing
        # and attributes, so it must never be read as "a different viewing angle".
        # This one has all of that removed and describes the candidate only.
        id_emb = models.clip_embed([mask_to_match(im, lab, feather=0)[0]])[0]
        v = (verified or {}).get(str(f)) or (verified or {}).get(f)
        if v is not None:
            anchor, kps = v["anchor"], v.get("keypoints")
            anchor_frac = float(v.get("anchor_frac") or 0.0)
        else:
            log.warning("%-34s no verified target instance; falling back to the "
                        "largest anchor", f.name)
            anchor, kps, anchor_frac = anchor_detail(models, im)
        items.append({"file": f, "image": im, "labels": lab, "palette": pal,
                      "clip": emb, "clip_match": id_emb, "anchor": anchor,
                      "orientation": anchor_orientation(kps), "anchor_frac": anchor_frac,
                      "size": list(im.size)})

    # Appearance clustering: attribute palette distance first (that is what an
    # "appearance" is), CLIP distance as the tiebreak for pose/framing differences.
    groups: list[dict] = []
    for it in items:
        placed = False
        for g in groups:
            shared = set(it["palette"]) & set(g["palette"])
            if shared:
                d = np.mean([delta_e(it["palette"][k]["lab"],
                                     g["palette"][k]["lab"]) for k in shared])
                same_appearance = d < thresh
            else:
                same_appearance = False
            appearance = float(np.dot(it["clip"], g["members"][0]["clip"]))
            if same_appearance and appearance > 0.6:
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


def anchor_detail(models, pil):
    """(embedding, 5-point keypoints, anchor area fraction) for the largest anchor.

    Models.anchor_embed returns the bounding box, which says nothing about which
    way the anchor is turned. The keypoints do.
    """
    app = models.anchor()
    if app is None:
        return None, None, 0.0
    import cv2
    bgr = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    anchors = app.get(bgr)
    if not anchors:
        return None, None, 0.0
    f = max(anchors, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    fa = ((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) / (pil.width * pil.height)
    return (np.asarray(f.normed_embedding, dtype=np.float32),
            getattr(f, "kps", None), float(fa))


def anchor_orientation(kps) -> float | None:
    """Signed turn proxy in [-1, 1] from the detector's 5-point anchor keypoints.

    Keypoints 0 and 1 are a symmetric pair and 2 lies on the axis of symmetry
    between them. Square-on, 2 sits at their midpoint; as the anchor turns it
    slides towards the nearer of the pair. Its offset from that midpoint, divided
    by the span of the pair, therefore tracks the turn. Scale-free and tolerant of
    in-plane rotation, because both terms rotate together.

    This is a real geometric measurement. It replaces a full-image CLIP distance,
    which was being read as "a different angle" when what it actually responds to
    is background, framing, lighting and attributes - a reference at the same angle
    in a different setting scored as a new viewpoint.
    """
    if kps is None or len(kps) < 3:
        return None
    a, b, mid_pt = np.asarray(kps[0]), np.asarray(kps[1]), np.asarray(kps[2])
    axis = b - a
    d = float(np.hypot(*axis))
    if d < 1e-3:
        return None
    mid = (a + b) / 2.0
    # Project onto the pair's axis, so in-plane rotation does not leak in.
    t = float(np.dot(mid_pt - mid, axis) / (d * d))
    return float(np.clip(t * 2.0, -1.0, 1.0))


def consensus_match(items: list[dict], same_anchor: float = 0.35,
                       duplicate: float = 0.92) -> dict:
    """Which references are agreed to show one candidate, by consensus.

    Taking the single maximum similarity is not verification: it asks only
    whether SOME other reference agrees, so two copies of the wrong candidate
    vouch for each other and both score perfectly. Near-duplicates have to go
    first, then agreement has to come from the group rather than from one
    neighbour.

      1. collapse near-duplicates (>= `duplicate`) to one representative, so
         repeated copies cannot vote repeatedly
      2. link representatives that match at all (>= `same_anchor`) and take the
         largest connected group - the consensus match
      3. score each reference by its MEDIAN similarity to the other members of
         that group, which a single outlier cannot lift

    Anything outside the consensus group scores zero and is never drawn: an
    unverified reference would be conditioning the model on a non-target's anchor.
    """
    anchors = [m for m in items if m["anchor"] is not None]
    for m in items:
        m["match_group"] = None
        m["agreement"] = 0.0
        m["duplicate_of"] = None
    if len(anchors) < 2:
        return {"members": len(anchors), "group": [], "duplicates": 0,
                "note": "too few detectable anchors to reach consensus"}

    E = np.stack([m["anchor"] for m in anchors])
    S = E @ E.T

    # 1. near-duplicate collapse (union-find over the duplicate threshold)
    parent = list(range(len(anchors)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            if S[i, j] >= duplicate:
                parent[find(i)] = find(j)
    reps: dict[int, int] = {}
    for i in range(len(anchors)):
        reps.setdefault(find(i), i)
    rep_idx = sorted(reps.values())
    n_dup = len(anchors) - len(rep_idx)
    for i in range(len(anchors)):
        r = reps[find(i)]
        if r != i:
            anchors[i]["duplicate_of"] = anchors[r]["file"].name

    # 2. largest connected group among representatives
    adj = {i: set() for i in rep_idx}
    for a in range(len(rep_idx)):
        for b in range(a + 1, len(rep_idx)):
            i, j = rep_idx[a], rep_idx[b]
            if S[i, j] >= same_anchor:
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
    # A duplicate of a group member is in the group too - it is the same image.
    members = [i for i in range(len(anchors)) if reps[find(i)] in group]

    # 3. median agreement with the OTHER representatives of that group
    for i in members:
        others = [j for j in best_group if reps[find(j)] != reps[find(i)]]
        anchors[i]["agreement"] = (round(float(np.median(S[i, others])), 4)
                                 if others else 0.0)
        anchors[i]["match_group"] = "consensus"
    return {"members": len(members), "representatives": len(best_group),
            "duplicates": n_dup, "candidates": len(anchors),
            "same_anchor_threshold": same_anchor, "duplicate_threshold": duplicate}


def score_match(items: list[dict]) -> None:
    """Rate each reference on how well it serves MATCH conditioning, in place.

    Deliberately blind to attribute colour. Once attributes are segmented out of the
    external panels, which appearance a reference happens to show says nothing about
    how well it depicts the anchor, and letting it decide would reintroduce exactly
    the coupling these packs exist to break.

    Agreement comes from consensus_match(), which is neither a self-comparison
    nor a single maximum: near-duplicates are collapsed first, then a reference
    is scored by its MEDIAN similarity to the rest of the agreed group. Both of
    the shortcuts it replaces gave a perfect score to the wrong answer - scoring
    against a bank built from these same references returned 1.000 for anything
    already in it, and taking the maximum let two copies of a non-target vouch for
    each other.

      agreement  consensus anchor similarity: is this the right candidate
      anchor_res   how many pixels the anchor occupies; a correct match at 20 px
                 carries no detail worth transferring

    Weighted in that order, because a confident wrong match is the worst
    outcome available. A reference with no detectable anchor, or one outside the
    consensus group, scores zero: it may well be the right candidate, but nothing
    here shows that it is.
    """
    for m in items:
        w, h = m["size"]
        # sqrt -> an anchor side length; ~200 px is already ample for conditioning,
        # so the score saturates there instead of rewarding ever-tighter framing.
        px = float(m["anchor_frac"] or 0.0) * w * h
        res = float(np.clip(np.sqrt(px) / 200.0, 0.0, 1.0))
        agree = float(m.get("agreement") or 0.0)
        m["match"] = {"agreement": round(agree, 4),
                         "anchor_resolution": round(res, 4),
                         "anchor_pixels": int(px),
                         "orientation": (round(m["orientation"], 3)
                                         if m.get("orientation") is not None
                                         else None),
                         "verified": bool(m.get("match_group") == "consensus"),
                         "duplicate_of": m.get("duplicate_of"),
                         "score": round(0.7 * agree + 0.3 * res, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--shot", nargs="*", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--probe-frames", type=int, default=8)
    ap.add_argument("--appearance-deltae", type=float, default=18.0,
                    help="Attribute colour distance above which two references "
                         "are treated as different appearances")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("make_reference_pack", args.verbose)
    cfg = load_config(args.config)
    man = load_manifest()
    import track_subject as T
    models = T.Models(log)
    # No match bank here on purpose: it is built from these same
    # references, so scoring one against it is a self-comparison. Match
    # agreement is computed leave-one-out in score_match() instead.
    parser = Parser(log)

    work = P.root / man["normalized"]["work_path"]
    W = int(man["normalized"]["width"])
    H = int(man["normalized"]["height"])
    refs = sorted(p for p in P.references.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not refs:
        log.error("No reference images in %s", P.references)
        return 1

    # Who the target is comes from the shared resolver, so tracking, this pack
    # and evaluation cannot disagree about it. It also drops images that contain
    # a second candidate that cannot be told apart from the target - the failure
    # that put a non-target through a full generation.
    from reference_match import load_exclusions, resolve_targets
    targets = resolve_targets(refs, models, log, load_exclusions(log))
    if not targets["per_image"]:
        log.error("No reference image yields a verified target anchor. Refusing to "
                  "build a pack: conditioning on an unverified anchor is how the "
                  "wrong candidate gets drawn.")
        return 1
    usable = [Path(v["file"]) for v in targets["per_image"].values()]
    if len(usable) != len(refs):
        log.info("Pack will use %d of %d reference image(s); the rest are "
                 "excluded, unverified or ambiguous.", len(usable), len(refs))
    verified = {str(Path(v["file"])): v["instance"]
                for v in targets["per_image"].values()}
    groups = cluster_references(usable, parser, models, log, args.appearance_deltae,
                                verified=verified)

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
        log.info("%s: learning the current appearance from the source", sid)
        appearance = source_appearance(work, mask_video, parser, args.probe_frames, log)

        # Panels are chosen on MATCH EVIDENCE ALONE, across every reference.
        #
        # Appearance clusters exist to stop two different appearances being combined
        # into one conditioning image. That constraint no longer binds the
        # external panels: their attributes are segmented out before they are drawn,
        # so there is no appearance left to conflict. Confining the choice to one
        # cluster would now throw away viewing angles for no benefit - and
        # missing viewpoints were a real weakness of these references. Clusters
        # are still computed and recorded, as a diagnostic.
        items = [m for g in groups for m in g["members"]]
        # Verified match comes from the shared resolver above, not from a
        # second opinion computed here.
        for m in items:
            rec = targets["per_image"].get(m["file"].name)
            m["agreement"] = (rec or {}).get("agreement", 0.0)
            m["match_group"] = "consensus" if rec else None
            m["duplicate_of"] = None
            m["other_candidates"] = (rec or {}).get("other_candidates", 0)
        consensus = {"members": len(targets["per_image"]),
                     "candidates": targets["instances"],
                     "images": targets["images"],
                     "rejected": len(targets["rejected"]),
                     "source": "reference_match.resolve_targets"}
        score_match(items)
        log.info("Match consensus: %d of %d reference(s) with an anchor agree "
                 "(%d near-duplicate(s) collapsed first, so repeated copies "
                 "cannot vote twice)", consensus.get("members", 0),
                 consensus.get("candidates", 0), consensus.get("duplicates", 0))
        ranked = sorted(items, key=lambda m: (m["match"]["score"],
                                              m["match"]["anchor_pixels"],
                                              m["file"].name), reverse=True)
        for m in ranked:
            i = m["match"]
            log.info("%-34s match %.3f (consensus agreement %.3f, anchor %d px, "
                     "turn %s, %s%s)", m["file"].name, i["score"], i["agreement"],
                     i["anchor_pixels"],
                     f"{i['orientation']:+.2f}"
                     if i["orientation"] is not None else "n/a",
                     "verified" if i["verified"] else "NOT VERIFIED",
                     f", duplicate of {i['duplicate_of']}"
                     if i["duplicate_of"] else "")
        verified = [m for m in ranked if m["match"]["verified"]]
        if not verified:
            log.warning("No reference is match-verified by consensus. Falling "
                        "back to the best-scoring reference; match "
                        "conditioning is weak and should not be trusted. The "
                        "attribute is unaffected: it comes from the source.")
        panel_anchor = (verified or ranked)[0]

        # A second panel earns its place only by showing a genuinely different
        # ANCHOR ORIENTATION, and only if it is match-verified - an unverified
        # reference would condition the model on a non-target's anchor.
        #
        # Ranked by orientation difference where the keypoints give one. The
        # match-masked embedding is the fallback for anchors the detector could
        # not land keypoints on; the FULL-image embedding is not used here at
        # all, because it responds to background, framing and attributes, and a
        # reference at the same angle in a different setting would win on it.
        others = [m for m in verified if m is not panel_anchor
                  and m["match"]["duplicate_of"] is None]
        alt, alt_why = None, ""
        y0 = panel_anchor["match"]["orientation"]
        with_turn = [m for m in others if m["match"]["orientation"] is not None]
        if y0 is not None and with_turn:
            alt = max(with_turn, key=lambda m: abs(m["match"]["orientation"] - y0))
            d = abs(alt["match"]["orientation"] - y0)
            alt_why = (f"anchor orientation differs by {d:.2f} "
                       f"({y0:+.2f} -> {alt['match']['orientation']:+.2f})")
            if d < 0.10:
                log.info("The most different anchor orientation available differs "
                         "by only %.2f; the references are all oriented much the "
                         "same way, so the second panel adds angle coverage in "
                         "name only.", d)
        elif others:
            alt = min(others, key=lambda m: float(
                np.dot(m["clip_match"], panel_anchor["clip_match"])))
            alt_why = ("no keypoints; match-masked embedding distance "
                       f"{1 - float(np.dot(alt['clip_match'], panel_anchor['clip_match'])):.3f}")
        if alt is not None:
            log.info("alternate view: %s (%s; chosen among %d verified, "
                     "non-duplicate reference(s))", alt["file"].name, alt_why,
                     len(others))
        else:
            log.info("No second match-verified, non-duplicate reference; the "
                     "pack uses one match panel.")
        best_g = int(panel_anchor["cluster"])

        # Diagnostic only: how far the chosen references' attributes are from what
        # is actually present in this interval. Recorded as evidence that the
        # external attributes are incompatible - never as a reason to use them.
        chosen_pal: dict = {}
        for m in ([panel_anchor] + ([alt] if alt is not None else [])):
            for k, v in m["palette"].items():
                chosen_pal.setdefault(k, v)
        shared = set(chosen_pal) & set(appearance["palette"])
        best_d = (float(np.mean([delta_e(chosen_pal[k]["lab"],
                                         appearance["palette"][k]["lab"])
                                 for k in shared])) if shared else float("nan"))
        same_look = best_d == best_d and best_d < args.appearance_deltae
        log.info("Chosen references' attributes are dE %.1f from this interval's "
                 "(%s). Either way the attribute comes from the source; this is a "
                 "diagnostic, not a switch.",
                 best_d if best_d == best_d else float("nan"),
                 "similar" if same_look else "a different appearance"
                 if best_d == best_d else "no shared attribute class")

        # Does the source show a covered anchor? If so, an external reference
        # showing an uncovered one must not condition the anchor: that would remove
        # an attribute the source actually carries. make_protected_mask.py decides this;
        # absent its verdict we fail closed and assume covered.
        _shot = next(x for x in man["shots"] if x["shot_id"] == sid)
        prot = _shot.get("protected_mask") or {}
        anchor_ok = bool(prot.get("anchor_conditioning_allowed"))
        if not prot:
            anchor_ok = False
            log.warning("%s: no protected-mask analysis, so it is unknown whether "
                        "the source anchor is covered. Failing closed: external "
                        "ANCHOR conditioning is disabled and only the surrounding "
                        "class is used. Run scripts/make_protected_mask.py first.",
                        sid)
        elif not anchor_ok:
            log.warning("%s: the source anchor is COVERED. External anchor "
                        "conditioning is disabled; only the surrounding class is "
                        "conditioned, so an uncovered reference cannot instruct "
                        "the model to take the covering off.", sid)
        allowed = MATCH_ONLY if anchor_ok else (MATCH_ONLY - {"face"})
        log.info("%s: external references may condition %s (source anchor %s)",
                 sid, "+".join(sorted(allowed)) or "NOTHING",
                 "exposed" if anchor_ok else "covered/unknown")

        pack = {
            "shot_id": sid,
            "cluster": best_g,
            "clusters_found": len(groups),
            "references_considered": len(items),
            # Invariant, not a decision. The attribute in this interval is the only
            # record of the attribute present, so it is always the authority; the
            # externals only ever contribute match. Making this conditional on
            # a colour distance meant a coincidentally similar reference could
            # promote itself to attribute source, which is the failure being fixed.
            "appearance_authority": "source_frames",
            "match_authority": "external_references",
            "external_attribute_similarity": {
                "deltaE_to_source": (round(best_d, 2) if best_d == best_d
                                     else None),
                "threshold": args.appearance_deltae,
                "looks_like_same_appearance": bool(same_look),
                "note": "diagnostic only; does not affect where the attribute "
                        "comes from",
            },
            "match_selection": panel_anchor["match"],
            "match_consensus": consensus,
            "external_conditioning": {
                "regions": sorted(allowed),
                "anchor_allowed": bool(anchor_ok),
                "source_anchor_covered": bool(prot.get("source_anchor_covered", True)),
                "note": "peripheral extents are never conditioned externally: "
                        "how much of the extent is visible is attribute coverage, "
                        "which belongs to the source",
            },
            "source_palette": appearance["palette"],
            "panels": [
                {"role": "match_anchor_only", "provenance": panel_anchor["file"].name,
                 "native_size": panel_anchor["size"],
                 "upscaled": False,
                 "conditions": "the anchor region only; attributes and "
                               "accessories segmented out",
                 "anchor_fraction": round(float(panel_anchor["anchor_frac"] or 0), 4),
                 "match": panel_anchor["match"],
                 "cluster": panel_anchor["cluster"]},
                {"role": "exact_appearance_extent",
                 "provenance": f"source frame {appearance['best']['frame']} "
                               f"of this interval",
                 "note": "attribute shape, boundaries, accessories and colour come "
                         "from the footage itself, which is the only correct "
                         "authority for the attribute present here"},
            ],
        }
        if alt is not None:
            pack["panels"].append(
                {"role": "alternate_angle_match_only",
                 "provenance": alt["file"].name,
                 "conditions": "match regions only; attributes segmented out",
                 "native_size": alt["size"], "upscaled": False})

        # ---- render the sheet ------------------------------------------------
        from PIL import Image
        n = len(pack["panels"])
        pw, ph = W // n, H
        sheet = Image.new("RGB", (W, H), (0, 0, 0))
        # EXTERNAL panels are stripped to match before they are drawn. Their
        # attributes are from another day; leaving them visible invites the model to
        # reproduce it, which is the fault being corrected. The SOURCE panel is
        # left whole - it is the only correct record of the current attribute.
        imgs, keeps = build_panel_images(panel_anchor, appearance["best"]["crop"],
                                         alt, allowed=allowed)
        for i, k in enumerate(keeps):
            if k is None:
                continue
            pack["panels"][i]["attributes_removed"] = True
            pack["panels"][i]["match_pixels_kept"] = round(k, 4)
        low = [pack["panels"][i]["role"] for i, k in enumerate(keeps)
               if k is not None and k < 0.05]
        if low:
            log.warning("%s: match masking kept under 5%% of panel(s) %s. The "
                        "parser may have failed on those references; check the "
                        "pack sheet before trusting the match conditioning.",
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
            "palette": {k: v["lab"] for k, v in appearance["palette"].items()}})
        (pack_dir / f"{sid}_pack.json").write_text(json.dumps(
            {k: v for k, v in pack.items()}, indent=2, default=str) + "\n")

        shot = next(s for s in man["shots"] if s["shot_id"] == sid)
        shot["reference_pack"] = {
            "sheet": pack["sheet"], "key": pack["key"], "cluster": best_g,
            "appearance_authority": pack["appearance_authority"],
            "match_authority": pack["match_authority"],
            "match_score": panel_anchor["match"]["score"],
            "external_attribute_deltaE": pack["external_attribute_similarity"][
                "deltaE_to_source"]}
        log.info("%s: pack -> %s (%d panels; match from externals, attribute "
                 "from %s)", sid, rel(sheet_path), n, pack["appearance_authority"])
        made += 1

    save_manifest(man)
    log.info("=" * 62)
    log.info("Built %d reference pack(s) in %s", made, rel(pack_dir))
    log.info("Panels carry provenance; nothing was upscaled to reach 1080p.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
