#!/usr/bin/env python
"""Phase 5e - did the subject LoRA move identity, measured so it can fail.

The identity bank is built from the reference photographs, and the LoRA is
trained on those same photographs. Scoring its output against the whole bank
therefore returns a high number whatever the LoRA learned - the "self-comparison
as evidence" trap in docs/STATE.md. So the bank here is built from the HELD-OUT
images alone, which make_lora_dataset.py reserved and no training step ever saw.

Three numbers are printed, and the middle one is the point:

  ceiling   real photographs of this person, the training crops' own verified
            faces, scored against the held-out bank. Same person, same camera
            era, no generation involved. This is what "as good as a photograph"
            is worth on this metric - not 1.0, and knowing it is the difference
            between reading 0.31 as a failure and reading it as most of the way.
  measured  the generated media, scored against the held-out bank.
  invalid   the same generated media scored against the bank of TRAINING faces.
            Printed only to show the gap the trap would have hidden. Never quote
            it as a result.

identity.py treats 0.35 as the threshold for a plausible match, and the 1.3B
pilot scored 0.17-0.21 without a LoRA (reports/pilot_results.md).

Rule 2b: this decodes frames to run a face model over them, which is what the
automated stages already do. It reports numbers, never a description of what is
depicted, and displays nothing (rule 1).

    scripts/score_lora_identity.py --media DIR_OR_FILE [DIR_OR_FILE ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, setup_logging  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
VID_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
THRESHOLD = 0.35


def _eye_px(kps) -> float:
    """Inter-ocular distance: how much real detail per facial feature this image
    carries. Identity says whether it is the right person; this says whether it
    is worth anything as a reference, because a reference can only donate detail
    it actually has."""
    if kps is None or len(kps) < 2:
        return 0.0
    return float(np.linalg.norm(np.asarray(kps[0]) - np.asarray(kps[1])))


def probe_media(path: Path, models, n_frames: int,
                mask_path: Path | None = None) -> list[np.ndarray]:
    """Face embeddings found in one image or video.

    With a mask video, each frame is cropped to the tracked subject's bounding
    box before the face model sees it. face_detail() takes the LARGEST face, and
    this shot has another person in it - at full frame the metric would happily
    score the wrong one and report it as identity.
    """
    from PIL import Image, ImageOps
    from make_reference_pack import face_detail

    if path.suffix.lower() in IMG_EXT:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        emb, kps, _ = face_detail(models, im)
        return [(emb, _eye_px(kps))] if emb is not None else []

    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return []

    masks = []
    if mask_path is not None:
        mcap = cv2.VideoCapture(str(mask_path))
        while True:
            ok, m = mcap.read()
            if not ok:
                break
            masks.append(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) > 127)
        mcap.release()
        if len(masks) < len(frames):
            raise SystemExit(
                f"{mask_path.name} has {len(masks)} frame(s) for "
                f"{path.name}'s {len(frames)}. Cropping frame i by mask j is a "
                f"measurement of nothing; refusing to guess the alignment.")

    idx = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int)
    out = []
    for i in idx:
        frame = frames[i]
        if masks:
            if masks[i].sum() < 64:
                continue
            ys, xs = np.where(masks[i])
            frame = frame[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        emb, kps, _ = face_detail(models, Image.fromarray(rgb))
        if emb is not None:
            out.append((emb, _eye_px(kps)))
    return out


def members(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(f for f in path.rglob("*")
                      if f.suffix.lower() in IMG_EXT | VID_EXT)
    return [path]


def stats(sims: list[float]) -> dict:
    if not sims:
        return {"faces": 0}
    a = np.asarray(sims, dtype=np.float32)
    return {"faces": int(a.size),
            "median": round(float(np.median(a)), 4),
            "best": round(float(a.max()), 4),
            "worst": round(float(a.min()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--media", type=Path, nargs="+", required=True,
                    help="Files or directories. A directory is one group, named "
                         "after itself, so one checkpoint per directory reads as "
                         "one row.")
    ap.add_argument("--dataset", type=Path,
                    default=P.intermediate / "lora_dataset",
                    help="Where make_lora_dataset.py wrote dataset.json")
    ap.add_argument("--frames", type=int, default=8,
                    help="Frames probed per video")
    ap.add_argument("--mask", type=Path, default=None,
                    help="Subject mask video. Videos are cropped to it before "
                         "face detection, so another person in frame cannot be "
                         "scored as the subject. Ignored for still images.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("score_lora_identity", args.verbose)
    manifest_path = args.dataset / "dataset.json"
    if not manifest_path.exists():
        log.error("No %s. Run scripts/make_lora_dataset.py first: without its "
                  "split there is no held-out set, and without a held-out set "
                  "this measurement is circular.", manifest_path)
        return 1
    manifest = json.loads(manifest_path.read_text())
    hold_names = [e["file"] for e in manifest.get("holdout", [])]
    train_names = [e["file"] for e in manifest.get("train", [])]
    if not hold_names:
        log.error("The dataset reserved no held-out images. Re-export with "
                  "--holdout >= 1; scoring against the training set alone "
                  "cannot answer the question.")
        return 1

    import track_subject as T
    from identity import load_exclusions, reference_files, resolve_targets

    models = T.Models(log)
    # Consensus over EVERY reference, exactly as tracking and the pack resolve
    # it - then the bank is subset. Re-resolving identity from three photographs
    # alone would be a different, weaker consensus, and the held-out images were
    # verified as this person by the full set.
    res = resolve_targets(reference_files(), models, log, load_exclusions(log))
    per = res.get("per_image") or {}
    if not per:
        log.error("No verified reference identity; nothing to score against.")
        return 1

    def bank_of(names):
        embs = [per[n]["instance"]["face"] for n in names if n in per]
        return np.stack(embs) if embs else None

    hold_bank = bank_of(hold_names)
    train_bank = bank_of(train_names)
    if hold_bank is None:
        log.error("None of the held-out images resolved a face this run.")
        return 1
    log.info("Held-out bank: %d face(s). Training bank: %d face(s).",
             len(hold_bank), 0 if train_bank is None else len(train_bank))

    # Ceiling: real photographs of the same person, disjoint from the bank.
    ceiling = stats([float(np.max(hold_bank @ per[n]["instance"]["face"]))
                     for n in train_names if n in per])
    eyes = []  # reset per group below

    groups = {}
    for path in args.media:
        if not path.exists():
            log.warning("%s does not exist; skipped", path)
            continue
        files = members(path)
        if not files:
            log.warning("%s holds no image or video; skipped", path)
            continue
        sims, sims_train, eyes, n_probed = [], [], [], 0
        for f in files:
            for emb, eye in probe_media(f, models, args.frames, args.mask):
                n_probed += 1
                sims.append(float(np.max(hold_bank @ emb)))
                eyes.append(eye)
                if train_bank is not None:
                    sims_train.append(float(np.max(train_bank @ emb)))
        groups[path.name] = {"items": len(files),
                             "measured": stats(sims),
                             "eye_px_median": round(float(np.median(eyes)), 1) if eyes else None,
                             "invalid_train_bank": stats(sims_train)}
        if not n_probed:
            log.warning("%-28s no face detected in any item - a LoRA that "
                        "produces no findable face is not a high score, it is "
                        "no measurement", path.name)

    w = max([len(k) for k in groups] + [12])
    log.info("=" * (w + 46))
    log.info("%-*s %6s %8s %8s %8s %8s   %s", w, "group", "faces", "median",
             "best", "worst", "eye px", "vs train bank (INVALID)")
    log.info("%-*s %6d %8s %8s %8s %8s   %s", w, "CEILING (real photos)",
             ceiling.get("faces", 0), ceiling.get("median", "-"),
             ceiling.get("best", "-"), ceiling.get("worst", "-"), "-", "-")
    for name, g in groups.items():
        m, iv = g["measured"], g["invalid_train_bank"]
        log.info("%-*s %6d %8s %8s %8s %8s   %s", w, name, m.get("faces", 0),
                 m.get("median", "-"), m.get("best", "-"), m.get("worst", "-"),
                 g.get("eye_px_median", "-"), iv.get("median", "-"))
    log.info("=" * (w + 46))
    log.info("Threshold for a plausible match: %.2f. The 1.3B pilot measured "
             "0.17-0.21 with no LoRA.", THRESHOLD)
    log.info("The 'vs train bank' column is what scoring against the training "
             "images would have told you. It is not a result.")

    out = args.out or (P.intermediate / "lora_eval" / "identity_scores.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Under intermediate/, which .gitignore denies wholesale: this names the
    # user's reference files and must never reach a remote (rule 2a).
    out.write_text(json.dumps(
        {"holdout_images": len(hold_bank), "training_images": len(train_names),
         "ceiling_real_photographs": ceiling, "threshold": THRESHOLD,
         "groups": groups}, indent=2))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
