#!/usr/bin/env python
"""Who the target is. One answer, shared by tracking, packing and evaluation.

A reference image can contain more than one candidate, and the candidate that
happens to be largest in it is not necessarily the one being restored. The
previous approach took the biggest detected candidate and the biggest detected anchor
per image, embedded the whole crop with CLIP, and let that drive selection. Every
step of that is wrong when a second candidate is present:

  * the largest candidate may be the wrong candidate, and at a distance the target's
    own anchor can be too small to detect at all, so the "largest anchor" is
    a non-target's;
  * CLIP over a whole crop responds to attributes, background and framing, so a
    reference showing the wrong candidate with similar attributes scores as a match;
  * a wrong-candidate image with no detectable anchor still contributed its
    appearance embedding, which then dominated because the anchor term is
    down-weighted when the anchor is small - exactly the situation in low
    resolution footage.

The result was a confident, unflagged track of the wrong candidate.

What replaces it:

  * every anchor and every candidate box in each reference is detected;
  * consensus is formed across anchor INSTANCES, not one anchor per image, so a
    reference containing both candidates contributes both and neither is assumed;
  * the dominant cluster is chosen by how many DISTINCT references support it,
    so a candidate appearing in two images cannot outvote one appearing in twelve;
  * the target anchor is associated with the candidate box containing it, which is
    what tracking actually needs;
  * an image where two different anchors both match the target, or where the
    target cannot be resolved, is REJECTED rather than guessed at.

Match here is anchor embeddings only. Attributes-sensitive appearance embeddings
are deliberately absent: they are what let the wrong candidate in.

The exclusion list is read from `intermediate/reference_exclusions.txt` - one
filename per line, `#` for comments. It is git-ignored by design, because it
names the user's files (rule 2a).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMAGE_EXTS, P  # noqa: E402

EXCLUSIONS = "reference_exclusions.txt"

# Cosine similarity on normalised ArcFace embeddings.
SAME_ANCHOR = 0.35        # two instances are plausibly the same candidate
DUPLICATE = 0.92        # ...and so alike they are effectively the same shot
AMBIGUOUS = 0.28        # a second anchor this close to the target makes the image
                        # unusable: we cannot say which one is meant


def load_exclusions(log=None) -> set[str]:
    """Filenames to keep out of match work for this run.

    Untracked on purpose. A tracked exclusion list would publish the names of
    the user's files, and worse, would imply something about who is in them.
    """
    p = P.intermediate / EXCLUSIONS
    if not p.exists():
        return set()
    names = set()
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    if names and log is not None:
        log.info("Excluding %d reference image(s) from match work "
                 "(see %s; untracked by design)", len(names), EXCLUSIONS)
    return names


def _contains(box, pt) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def anchor_instances(models, pil, detect_candidates: bool = True) -> list[dict]:
    """EVERY anchor in one reference, each tied to the candidate box containing it.

    Returns one record per anchor: embedding, box, area fraction, keypoints, and
    the smallest candidate box that contains the anchor centre (the candidate box is
    what a tracker needs to seed from, and the smallest containing one is a single
    candidate rather than a group).
    """
    import cv2
    app = models.anchor()
    if app is None:
        return []
    bgr = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    anchors = app.get(bgr)
    if not anchors:
        return []
    candidates = models.detect_candidates(pil) if detect_candidates else []
    out = []
    for f in anchors:
        b = [float(v) for v in f.bbox]
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        holding = [p for p in candidates if _contains(p, (cx, cy))]
        candidate = min(holding, key=lambda p: (p[2] - p[0]) * (p[3] - p[1])) \
            if holding else None
        out.append({
            "anchor": np.asarray(f.normed_embedding, dtype=np.float32),
            "box": b,
            # `kps` is the detector's own attribute name; the record key is ours.
            "keypoints": getattr(f, "kps", None),
            "det_score": float(getattr(f, "det_score", 0.0)),
            "anchor_frac": (b[2] - b[0]) * (b[3] - b[1]) / (pil.width * pil.height),
            "candidate_box": [float(v) for v in candidate[:4]] if candidate else None,
            "candidate_score": float(candidate[4]) if candidate else None,
        })
    return out


def dominant_cluster(instances: list[dict]) -> dict:
    """Group anchor instances and pick the match the reference set is ABOUT.

    Chosen by how many distinct source images support a group, not by how many
    instances it has: two references showing one candidate can contain several
    anchors each, and instance counts would let a small set of images dominate.
    """
    if not instances:
        return {"members": [], "images": 0, "groups": []}
    E = np.stack([i["anchor"] for i in instances])
    S = E @ E.T
    n = len(instances)

    adj = {i: set() for i in range(n)}
    for a in range(n):
        for b in range(a + 1, n):
            if S[a, b] >= SAME_ANCHOR:
                adj[a].add(b)
                adj[b].add(a)
    seen, groups = set(), []
    for start in range(n):
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
        groups.append(sorted(comp))

    def support(g):
        return (len({instances[i]["file"] for i in g}), len(g))

    groups.sort(key=support, reverse=True)
    best = groups[0]
    return {"members": best, "images": len({instances[i]["file"] for i in best}),
            "groups": [{"instances": len(g),
                        "images": len({instances[i]["file"] for i in g})}
                       for g in groups],
            "similarity": S}


def resolve_targets(files: list[Path], models, log,
                    exclusions: set[str] | None = None) -> dict:
    """The verified target anchor in each usable reference.

    Returns the shared match bank plus, per image, the target anchor instance
    and its candidate box. Images where two anchors both plausibly match the target
    are rejected: with a second candidate present, guessing is how the wrong one
    gets tracked.
    """
    from PIL import Image, ImageOps
    exclusions = exclusions or set()

    instances, skipped = [], []
    for f in files:
        if f.name in exclusions:
            skipped.append((f.name, "excluded for this run"))
            continue
        try:
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        except Exception as e:
            skipped.append((f.name, f"unreadable ({e})"))
            continue
        found = anchor_instances(models, im)
        if not found:
            skipped.append((f.name, "no detectable anchor"))
            continue
        for inst in found:
            inst["file"] = f
            inst["size"] = list(im.size)
        instances.extend(found)
        if len(found) > 1:
            log.info("%-34s %d anchors detected", f.name, len(found))

    if not instances:
        return {"bank": None, "per_image": {}, "skipped": skipped,
                "rejected": [], "instances": 0, "images": 0}

    dom = dominant_cluster(instances)
    S = dom["similarity"]
    member = set(dom["members"])

    per_image: dict[str, dict] = {}
    rejected: list[tuple[str, str]] = []
    by_file: dict[Path, list[int]] = {}
    for idx, inst in enumerate(instances):
        by_file.setdefault(inst["file"], []).append(idx)

    for f, idxs in by_file.items():
        hits = [i for i in idxs if i in member]
        if not hits:
            skipped.append((f.name, "contains no anchor from the target match"))
            continue
        if len(hits) > 1:
            rejected.append((f.name, f"{len(hits)} anchors match the target"))
            continue
        target = hits[0]
        # A second, DIFFERENT anchor that is still close to the target makes the
        # image unusable: the embeddings are not separating the two candidates, so
        # any choice here is a guess.
        rivals = [i for i in idxs if i != target
                  and float(S[target, i]) >= AMBIGUOUS]
        if rivals:
            rejected.append((f.name, "another anchor is too close to the target "
                                     "to tell them apart"))
            continue
        inst = instances[target]
        others = [i for i in idxs if i != target]
        per_image[f.name] = {
            "file": f,
            "instance": inst,
            "anchors_in_image": len(idxs),
            "other_candidates": len(others),
            # Confidence the target is the RIGHT anchor: agreement with the rest
            # of the cluster, excluding this image's own instances so a
            # reference cannot vouch for itself.
            "agreement": _agreement(S, target, member, by_file[f]),
            "candidate_box": inst["candidate_box"],
            "anchor_pixels": int(inst["anchor_frac"] * inst["size"][0] * inst["size"][1]),
        }

    bank = np.stack([per_image[k]["instance"]["anchor"] for k in per_image]) \
        if per_image else None

    log.info("Target match: %d instance(s) across %d image(s) form the "
             "dominant cluster; %d image(s) usable, %d rejected as ambiguous, "
             "%d skipped", len(dom["members"]), dom["images"], len(per_image),
             len(rejected), len(skipped))
    for name, why in rejected:
        log.warning("REJECTED %-30s %s", name, why)
    if len(dom["groups"]) > 1:
        log.info("Other match group(s) present in the references: %s",
                 ", ".join(f"{g['instances']} anchor(s) in {g['images']} image(s)"
                           for g in dom["groups"][1:]))

    return {"bank": bank, "per_image": per_image, "skipped": skipped,
            "rejected": rejected, "instances": len(instances),
            "images": dom["images"], "groups": dom["groups"]}


def _agreement(S, target: int, member: set, same_file: list[int]) -> float:
    """Median similarity to target-cluster instances from OTHER references."""
    others = [i for i in member if i not in same_file]
    if not others:
        return 0.0
    return round(float(np.median(S[target, others])), 4)


def reference_files() -> list[Path]:
    return sorted(p for p in P.references.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
