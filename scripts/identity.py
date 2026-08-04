#!/usr/bin/env python
"""Who the target is. One answer, shared by tracking, packing and evaluation.

A reference photograph can contain more than one person, and the person who
happens to be largest in it is not necessarily the one being restored. The
previous approach took the biggest detected person and the biggest detected face
per image, embedded the whole crop with CLIP, and let that drive selection. Every
step of that is wrong when a second person is present:

  * the largest person may be the wrong person, and at a distance the target's
    own face can be too small to detect at all, so the "largest face" is
    somebody else's;
  * CLIP over a whole crop responds to clothing, background and framing, so a
    photograph of the wrong person in similar clothes scores as a match;
  * a wrong-person image with no detectable face still contributed its
    appearance embedding, which then dominated because the face term is
    down-weighted when the face is small - exactly the situation in low
    resolution footage.

The result was a confident, unflagged track of the wrong person.

What replaces it:

  * every face and every person box in each photograph is detected;
  * consensus is formed across face INSTANCES, not one face per image, so a
    photograph containing both people contributes both and neither is assumed;
  * the dominant cluster is chosen by how many DISTINCT photographs support it,
    so a person appearing in two images cannot outvote one appearing in twelve;
  * the target face is associated with the person box containing it, which is
    what tracking actually needs;
  * an image where two different faces both match the target, or where the
    target cannot be resolved, is REJECTED rather than guessed at.

Identity here is face embeddings only. Clothing-sensitive appearance embeddings
are deliberately absent: they are what let the wrong person in.

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
SAME_FACE = 0.35        # two instances are plausibly the same person
DUPLICATE = 0.92        # ...and so alike they are effectively the same shot
AMBIGUOUS = 0.28        # a second face this close to the target makes the image
                        # unusable: we cannot say which one is meant


def load_exclusions(log=None) -> set[str]:
    """Filenames to keep out of identity work for this run.

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
        log.info("Excluding %d reference image(s) from identity work "
                 "(see %s; untracked by design)", len(names), EXCLUSIONS)
    return names


def _contains(box, pt) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def face_instances(models, pil, detect_people: bool = True) -> list[dict]:
    """EVERY face in one photograph, each tied to the person box containing it.

    Returns one record per face: embedding, box, area fraction, landmarks, and
    the smallest person box that contains the face centre (the person box is
    what a tracker needs to seed from, and the smallest containing one is the
    individual rather than a group).
    """
    import cv2
    app = models.face()
    if app is None:
        return []
    bgr = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        return []
    people = models.detect_people(pil) if detect_people else []
    out = []
    for f in faces:
        b = [float(v) for v in f.bbox]
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        holding = [p for p in people if _contains(p, (cx, cy))]
        person = min(holding, key=lambda p: (p[2] - p[0]) * (p[3] - p[1])) \
            if holding else None
        out.append({
            "face": np.asarray(f.normed_embedding, dtype=np.float32),
            "box": b,
            "kps": getattr(f, "kps", None),
            "det_score": float(getattr(f, "det_score", 0.0)),
            "face_frac": (b[2] - b[0]) * (b[3] - b[1]) / (pil.width * pil.height),
            "person_box": [float(v) for v in person[:4]] if person else None,
            "person_score": float(person[4]) if person else None,
        })
    return out


def dominant_cluster(instances: list[dict]) -> dict:
    """Group face instances and pick the identity the reference set is ABOUT.

    Chosen by how many distinct source images support a group, not by how many
    instances it has: two photographs of one person can contain several faces
    each, and instance counts would let a small set of images dominate.
    """
    if not instances:
        return {"members": [], "images": 0, "groups": []}
    E = np.stack([i["face"] for i in instances])
    S = E @ E.T
    n = len(instances)

    adj = {i: set() for i in range(n)}
    for a in range(n):
        for b in range(a + 1, n):
            if S[a, b] >= SAME_FACE:
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
    """The verified target face in each usable photograph.

    Returns the shared identity bank plus, per image, the target face instance
    and its person box. Images where two faces both plausibly match the target
    are rejected: with a second person present, guessing is how the wrong one
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
        found = face_instances(models, im)
        if not found:
            skipped.append((f.name, "no detectable face"))
            continue
        for inst in found:
            inst["file"] = f
            inst["size"] = list(im.size)
        instances.extend(found)
        if len(found) > 1:
            log.info("%-34s %d faces detected", f.name, len(found))

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
            skipped.append((f.name, "contains no face from the target identity"))
            continue
        if len(hits) > 1:
            rejected.append((f.name, f"{len(hits)} faces match the target"))
            continue
        target = hits[0]
        # A second, DIFFERENT face that is still close to the target makes the
        # image unusable: the embeddings are not separating the two people, so
        # any choice here is a guess.
        rivals = [i for i in idxs if i != target
                  and float(S[target, i]) >= AMBIGUOUS]
        if rivals:
            rejected.append((f.name, "another face is too close to the target "
                                     "to tell them apart"))
            continue
        inst = instances[target]
        others = [i for i in idxs if i != target]
        per_image[f.name] = {
            "file": f,
            "instance": inst,
            "faces_in_image": len(idxs),
            "other_people": len(others),
            # Confidence the target is the RIGHT face: agreement with the rest
            # of the cluster, excluding this image's own instances so a
            # photograph cannot vouch for itself.
            "agreement": _agreement(S, target, member, by_file[f]),
            "person_box": inst["person_box"],
            "face_pixels": int(inst["face_frac"] * inst["size"][0] * inst["size"][1]),
        }

    bank = np.stack([per_image[k]["instance"]["face"] for k in per_image]) \
        if per_image else None

    log.info("Target identity: %d instance(s) across %d image(s) form the "
             "dominant cluster; %d image(s) usable, %d rejected as ambiguous, "
             "%d skipped", len(dom["members"]), dom["images"], len(per_image),
             len(rejected), len(skipped))
    for name, why in rejected:
        log.warning("REJECTED %-30s %s", name, why)
    if len(dom["groups"]) > 1:
        log.info("Other identity group(s) present in the references: %s",
                 ", ".join(f"{g['instances']} face(s) in {g['images']} image(s)"
                           for g in dom["groups"][1:]))

    return {"bank": bank, "per_image": per_image, "skipped": skipped,
            "rejected": rejected, "instances": len(instances),
            "images": dom["images"], "groups": dom["groups"]}


def _agreement(S, target: int, member: set, same_file: list[int]) -> float:
    """Median similarity to target-cluster instances from OTHER photographs."""
    others = [i for i in member if i not in same_file]
    if not others:
        return 0.0
    return round(float(np.median(S[target, others])), 4)


def reference_files() -> list[Path]:
    return sorted(p for p in P.references.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
