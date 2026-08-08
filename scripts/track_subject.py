#!/usr/bin/env python
"""Phase 6 - match-aware automatic subject detection and full-subject tracking.

Fully automatic under normal conditions. No manual boxes or clicks are required;
the script only asks for help on genuinely ambiguous shots.

Per shot:
  1. Sample representative frames.
  2. Detect every candidate with Grounding DINO (open-vocabulary).
  3. Score each candidate against the target's verified reference anchors
     (scripts/reference_match.py) using anchor-embedding similarity ONLY.

     Attributes-sensitive appearance similarity used to stand in when no anchor was
     visible. That is backwards: when the anchor cannot be seen is exactly when
     attributes similarity is least able to tell two candidates apart. A candidate
     with no resolvable anchor now scores nothing and the shot is flagged for a
     human rather than tracked on a guess.
  4. Take the highest-confidence candidate, initialise SAM 2.1 from its generated
     box, and track the WHOLE SUBJECT (not just the anchor) through the shot.
  5. Re-detect and re-seed after tracking-confidence loss, disappearance,
     reappearance or a sudden mask-area jump.
  6. Emit frame-aligned grayscale mask videos, review contact sheets and
     per-shot confidence scores.

Mask semantics are NOT assumed. Verified by reading
ComfyUI/comfy_extras/nodes_wan.py::WanVaceToVideo:
      inactive = control_video * (1 - mask)
      reactive = control_video * mask
so WHITE (255) = the region VACE regenerates = the subject,
   BLACK (0)   = preserved original pixels.
Run scripts/verify_mask_polarity.py for an end-to-end proof of this.

Runs as its own process, after ComfyUI work rather than alongside it, so SAM 2
and VACE never contend for VRAM.

Nothing is ever displayed on screen; review artefacts are written to disk.

Typical use (automatic):
    scripts/track_subject.py

Correct one shot without touching the others:
    scripts/track_subject.py --shot shot0003 --force
    scripts/track_subject.py --shot shot0003 --init-box 120,40,300,470 --force
    scripts/track_subject.py --shot shot0003 --init-points 210,150,+ 260,300,+ 40,40,- --force
    scripts/track_subject.py --shot shot0003 --init-mask inputs/subject_seeds/shot0003.png --force
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    IMAGE_EXTS, P, human_time, load_config, load_manifest, probe_frames,
    prompt_overlay, rel, require_cuda, require_tools, run, save_manifest,
    setup_logging, slice_frames,
)

DINO_ID = "IDEA-Research/grounding-dino-base"
CLIP_ID = "openai/clip-vit-large-patch14"
SAM2_ID = "facebook/sam2.1-hiera-large"

# Exact Hugging Anchor revisions. Without these, `from_pretrained` follows the
# repo's default branch, so the same code silently picks up different weights on
# a later machine and the run stops being reproducible. Recorded in
# reports/versions.md.
DINO_REV = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
CLIP_REV = "32bd64288804d66eefd0ccbe215aa642df71cc41"
SAM2_REV = "665f8e2ad61cf5f53d65644ff27c8ee525124610"

# Grounding DINO carries a strong prior for the framing its training set is
# dominated by. A generic class phrase finds a subject matching that prior
# confidently and one departing from it weakly or not at all - the box has an
# unexpected aspect and scores against the phrase poorly. Where a shot departs
# from the prior the bias is not a small ranking effect: it can mean the only
# candidate offered is a non-target.
#
# So the prompt should name, alongside the generic phrases, whatever framings the
# source actually contains, giving a subject the generic phrases under-score its
# own way of being named. Those phrases are load-bearing and they are the one
# thing a detector prompt cannot state without stating what the subject is, so
# the whole prompt lives in the untracked overlay and this file does not record
# which phrases any particular source needed. See common.py::prompt_overlay.
def detect_prompt() -> str:
    return prompt_overlay("detect_prompt")

# Grounding DINO box threshold. 0.30 suits clean references; a heavily degraded
# low-resolution source scores the same true detection lower, so when the primary
# pass finds nothing at all the shot is retried at DETECT_THRESHOLD_MIN before
# being given up on. A weak *detection* is not the same as a weak *match*
# match: whether the shot is trusted is still decided by the fused anchor and
# appearance score below, and a fallback detection is recorded in the report.
DETECT_THRESHOLD = 0.30
DETECT_THRESHOLD_MIN = 0.18

# Decision thresholds. Deliberately conservative: it is cheaper to ask about one
# shot than to silently restore the wrong candidate for 40 seconds.
AUTO_ACCEPT = 0.42     # fused score at/above which we proceed with no questions
AMBIGUOUS_MARGIN = 0.08  # top-1 must beat top-2 by this much
MIN_TRACK_SCORE = 0.0   # SAM2 object score below which a frame is "lost"
AREA_JUMP = 3.0         # frame-to-frame mask area ratio treated as tracking loss
MIN_AREA_FRAC = 0.0008  # below this fraction of frame the subject is "gone"


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class Models:
    def __init__(self, log):
        import torch
        self.torch = torch
        self.dev = require_cuda(log)
        self.log = log
        self._dino = self._dino_proc = None
        self._clip = self._clip_proc = None
        self._anchor = None
        self._sam = None

    # -- lazily loaded so a stage that does not need a model never pays for it
    def dino(self):
        if self._dino is None:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self.log.info("Loading detector %s", DINO_ID)
            self._dino_proc = AutoProcessor.from_pretrained(DINO_ID, revision=DINO_REV)
            # fp32 weights, not fp16: Grounding DINO fuses a fp32 text branch into
            # the vision features, and a half-precision load fails inside the
            # encoder with "mat1 and mat2 must have the same dtype". Speed comes
            # from autocast in detect_candidates() instead. ~0.9 GiB of VRAM.
            self._dino = AutoModelForZeroShotObjectDetection.from_pretrained(
                DINO_ID, revision=DINO_REV, dtype=self.torch.float32).to(self.dev).eval()
        return self._dino, self._dino_proc

    def clip(self):
        if self._clip is None:
            from transformers import CLIPModel, CLIPImageProcessor
            self.log.info("Loading appearance encoder %s", CLIP_ID)
            self._clip_proc = CLIPImageProcessor.from_pretrained(CLIP_ID, revision=CLIP_REV)
            self._clip = CLIPModel.from_pretrained(
                CLIP_ID, revision=CLIP_REV, dtype=self.torch.float16).to(self.dev).eval()
        return self._clip, self._clip_proc

    def anchor(self):
        # The backend is resolved by ROLE through scripts/backends.py, which
        # reads the binding from an untracked config (CLAUDE.md rules 2a, 2c).
        # A pretrained model is named after what it was trained to find, so the
        # model id used to state the subject's category right here.
        # THIS USED TO FALL BACK to "flag the shot for manual seeding", which
        # sounds conservative and is not: a missing binding is a configuration
        # error the operator can fix in seconds, and converting it into a
        # per-shot data condition buries it in a warning line inside a stage
        # that then keeps running. Every shot gets flagged, the flags look like
        # ordinary tracking difficulty, and the actual cause — no backend — is
        # indistinguishable from a hard shot. It raises now.
        #
        # An ordinary per-image detection failure is a different thing and stays
        # non-fatal: the backend loaded, ran, and found no anchor in this frame.
        # That is data. This is configuration.
        if self._anchor is None:
            import backends      # BackendUnavailable propagates, deliberately
            try:
                import onnxruntime as ort
            except ImportError as e:
                raise backends.BackendUnavailable(
                    f"onnxruntime is not installed, so match cannot be "
                    f"established for any shot: {e}") from e
            cuda_ok = "CUDAExecutionProvider" in ort.get_available_providers()
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if cuda_ok else ["CPUExecutionProvider"])
            self._anchor = backends.anchor_embedder(providers=providers,
                                                    log=self.log)
        return self._anchor

    def sam(self):
        if self._sam is None:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            self.log.info("Loading tracker %s", SAM2_ID)
            self._sam = SAM2VideoPredictor.from_pretrained(SAM2_ID, revision=SAM2_REV, device=self.dev)
        return self._sam

    def release(self, *names: str):
        for n in names:
            setattr(self, f"_{n}", None)
        self.torch.cuda.empty_cache()

    # -- embeddings ---------------------------------------------------------
    def clip_embed(self, pil_images: list) -> np.ndarray:
        model, proc = self.clip()
        with self.torch.inference_mode():
            inp = proc(images=pil_images, return_tensors="pt").to(self.dev)
            inp["pixel_values"] = inp["pixel_values"].half()
            f = model.get_image_features(**inp).float()
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()

    def anchor_embed(self, pil_image):
        """Returns (embedding, anchor_box, anchor_area_fraction) or (None, None, 0)."""
        app = self.anchor()
        if app is None:
            return None, None, 0.0
        import cv2
        bgr = cv2.cvtColor(np.asarray(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        anchors = app.get(bgr)
        if not anchors:
            return None, None, 0.0
        f = max(anchors, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        fa = ((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) / \
             (pil_image.width * pil_image.height)
        return np.asarray(f.normed_embedding, dtype=np.float32), \
            tuple(float(v) for v in f.bbox), float(fa)

    def detect_candidates(self, pil_image, threshold=DETECT_THRESHOLD) -> list[tuple]:
        model, proc = self.dino()
        with self.torch.inference_mode():
            inputs = proc(images=pil_image, text=detect_prompt(),
                          return_tensors="pt").to(self.dev)
            # autocast rather than a manual .half(): the model is fp32 (see
            # dino()) and autocast casts only the ops that are safe in half,
            # leaving the text branch alone.
            with self.torch.autocast("cuda", dtype=self.torch.float16):
                res = model(**inputs)
            post = proc.post_process_grounded_object_detection(
                res, inputs.input_ids, threshold=threshold, text_threshold=0.25,
                target_sizes=[(pil_image.height, pil_image.width)])[0]
        out = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s))
               for b, s in zip(post["boxes"].cpu(), post["scores"].cpu())]
        return sorted(out, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))


# ---------------------------------------------------------------------------
# match bank from the reference images
# ---------------------------------------------------------------------------

def build_match_bank(models: Models, log) -> dict:
    """Anchor embeddings of the ONE candidate being restored.

    Delegates to scripts/reference_match.py so tracking, pack selection and evaluation
    all mean the same thing by "the target". See that module for why the old
    largest-candidate/largest-anchor-per-image approach let a second candidate in.

    There is deliberately no appearance bank any more. It was a CLIP embedding
    of the largest candidate crop, which responds to attributes, background and
    framing rather than to who someone is - and because the anchor term is
    down-weighted when the anchor is small, that appearance term DOMINATED at low
    resolution. Two references of a different candidate, neither with a detectable
    target anchor, were enough to carry the selection.
    """
    from reference_match import load_exclusions, reference_files, resolve_targets

    files = reference_files()
    if not files:
        raise RuntimeError(
            f"No reference images in {P.references}. Match-aware selection "
            "needs at least one. Add references and run "
            "scripts/prepare_references.py.")
    res = resolve_targets(files, models, log, load_exclusions(log))
    if res["bank"] is None:
        raise RuntimeError(
            "No usable target anchor in any reference image. Match cannot be "
            "established, and tracking on appearance alone is what put the "
            "wrong candidate in the last run. Add a reference showing the target's "
            "anchor, or seed the shot manually with --init-box / --init-points.")
    log.info("Match bank: %d verified target anchor(s) from %d image(s)",
             len(res["bank"]), len(res["per_image"]))
    return {"anchor": res["bank"],
            "files": [str(v["file"]) for v in res["per_image"].values()],
            "per_image": res["per_image"],
            "rejected": res["rejected"]}


def score_candidate(models: Models, bank: dict, crop) -> dict:
    """Is this crop the target? Anchor evidence only.

    Appearance similarity used to fill in when no anchor was resolvable, which
    sounds like graceful degradation and is really the opposite: it is precisely
    when the anchor cannot be seen that attribute similarity is least trustworthy
    and most likely to pick a non-target. A candidate with no usable anchor now
    scores nothing and the shot is flagged for a human, rather than tracked on a
    guess.
    """
    femb, _, anchor_frac = models.anchor_embed(crop)
    if femb is None or bank.get("anchor") is None:
        return {"fused": 0.0, "anchor_sim": None, "anchor_frac": anchor_frac,
                "reason": "no anchor resolvable in this candidate"}
    anchor_sim = float(np.max(bank["anchor"] @ femb))
    # Confidence is the similarity itself, discounted when the anchor is so small
    # that the embedding is mostly interpolation. Never promoted by anything
    # else: there is nothing else here to promote it.
    scale = float(np.clip(anchor_frac / 0.005, 0.0, 1.0))
    return {"fused": float(anchor_sim * scale), "anchor_sim": anchor_sim,
            "anchor_frac": anchor_frac, "anchor_scale": scale}


# ---------------------------------------------------------------------------
# frame IO
# ---------------------------------------------------------------------------

def extract_shot_frames(work: Path, out_dir: Path, start: int, end: int,
                        log) -> list[Path]:
    """SAM 2's video predictor wants a directory of JPEGs named %05d.jpg."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(work),
         "-vf", f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS",
         "-vsync", "0", "-q:v", "2", "-start_number", "0",
         str(out_dir / "%05d.jpg")], log)
    frames = sorted(out_dir.glob("*.jpg"))
    if len(frames) != end - start:
        raise RuntimeError(f"Extracted {len(frames)} frames, expected {end - start}")
    return frames


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------

def track_shot(models: Models, frames_dir: Path, n_frames: int, seed_frame: int,
               seed: dict, log, window: int = 240, reseed=None,
               max_recoveries: int = 8, min_gap: int = 4
               ) -> tuple[np.ndarray, list[float], list[tuple[int, int]]]:
    """Propagate a seed through the shot.

    Returns (masks[T,H,W] uint8, scores, absent_ranges), where absent_ranges are
    [a, b) spans in which re-detection found no one matching the match bank.

    Long shots are processed in overlapping windows because SAM 2's video state
    grows with frame count and would otherwise exhaust VRAM on a 12 GB card.
    Each new window is re-seeded from the last confident mask of the previous one.

    `reseed(frame_idx) -> seed dict | None` is called to restart the track after a
    loss. Passing None disables recovery (used by the manual-seed path, where the
    user has already said where the subject is).
    """
    import torch
    predictor = models.sam()
    from PIL import Image
    probe = Image.open(frames_dir / "00000.jpg")
    W, H = probe.size

    masks = np.zeros((n_frames, H, W), dtype=np.uint8)
    scores = [0.0] * n_frames
    filled = np.zeros(n_frames, dtype=bool)

    def run_window(w_start: int, w_end: int, local_seed_idx: int, seed_obj: dict):
        """One SAM2 pass over [w_start, w_end)."""
        # SAM2 needs its own directory view; symlink the slice to keep it cheap.
        sub = frames_dir.parent / f"{frames_dir.name}_w{w_start:05d}"
        if sub.exists():
            shutil.rmtree(sub)
        sub.mkdir(parents=True)
        for i in range(w_start, w_end):
            (sub / f"{i - w_start:05d}.jpg").symlink_to(frames_dir / f"{i:05d}.jpg")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(sub),
                                         offload_video_to_cpu=True,
                                         offload_state_to_cpu=True)
            predictor.reset_state(state)
            rel = local_seed_idx - w_start
            if "mask" in seed_obj:
                predictor.add_new_mask(state, frame_idx=rel, obj_id=1,
                                       mask=seed_obj["mask"])
            elif "points" in seed_obj:
                pts = np.array([p[:2] for p in seed_obj["points"]], dtype=np.float32)
                lbl = np.array([p[2] for p in seed_obj["points"]], dtype=np.int32)
                predictor.add_new_points_or_box(state, frame_idx=rel, obj_id=1,
                                                points=pts, labels=lbl)
            else:
                predictor.add_new_points_or_box(
                    state, frame_idx=rel, obj_id=1,
                    box=np.array(seed_obj["box"][:4], dtype=np.float32))

            # forward then backward so the whole window is covered from the seed
            for reverse in (False, True):
                for f_idx, obj_ids, logits in predictor.propagate_in_video(
                        state, start_frame_idx=rel, reverse=reverse):
                    g = w_start + f_idx
                    if g >= n_frames:
                        continue
                    m = (logits[0] > 0.0).cpu().numpy().squeeze()
                    masks[g] = (m.astype(np.uint8) * 255)
                    scores[g] = float(logits[0].max().item())
                    filled[g] = True
        shutil.rmtree(sub, ignore_errors=True)

    def present_at(t: int) -> bool:
        return mask_present(masks, t)

    def extend_forward(end: int, limit: int) -> int:
        """Chain windows forward from `end` while there is a mask to re-seed on."""
        while end < limit:
            anchor = end - 1
            if not present_at(anchor):
                return end
            nw_end = min(limit, end + window - 1)
            run_window(anchor, nw_end, anchor, {"mask": masks[anchor] > 127})
            if nw_end <= end:
                return end
            end = nw_end
        return end

    def extend_backward(start: int, limit: int) -> int:
        while start > limit:
            if not present_at(start):
                return start
            nw_start = max(limit, start - window + 1)
            run_window(nw_start, start + 1, start, {"mask": masks[start] > 127})
            if nw_start >= start:
                return start
            start = nw_start
        return start

    def empty_runs() -> list[tuple[int, int]]:
        return mask_gaps(masks[:n_frames])

    # ---- first window ------------------------------------------------------
    w_start = max(0, min(seed_frame - window // 2, n_frames - window))
    w_start = max(0, w_start)
    w_end = min(n_frames, w_start + window)
    run_window(w_start, w_end, seed_frame, seed)

    w_end = extend_forward(w_end, n_frames)
    w_start = extend_backward(w_start, 0)

    # ---- recovery ------------------------------------------------------------
    # Propagation stops wherever the subject leaves frame or SAM 2 loses the
    # track, which used to end the shot: everything past that point stayed black
    # and was merely reported as an event. Instead, re-detect inside each hole and
    # restart propagation there. A hole where re-detection finds nobody matching
    # the match bank is genuine absence, and is returned as such so the caller
    # can treat it as intentional pass-through rather than a failure.
    absent: list[tuple[int, int]] = []
    if reseed is not None:
        tried: set[tuple[int, int]] = set()
        for _ in range(max_recoveries):
            holes = [r for r in empty_runs()
                     if r[1] - r[0] >= min_gap and r not in tried]
            if not holes:
                break
            a, b = max(holes, key=lambda r: r[1] - r[0])
            tried.add((a, b))
            probe = (a + b) // 2
            new_seed = reseed(probe)
            if new_seed is None:
                log.info("Frames %d-%d: no matching subject found; treating as "
                         "absent (nothing to regenerate there).", a, b)
                absent.append((a, b))
                continue
            log.info("Frames %d-%d: re-seeded at frame %d, resuming the track.",
                     a, b, probe)
            ws = max(a, min(probe - window // 2, b - window))
            ws = max(a, ws)
            run_window(ws, min(b, ws + window), probe, new_seed)
            extend_forward(min(b, ws + window), b)
            extend_backward(ws, a)

    return masks, scores, absent


def mask_present(masks: np.ndarray, t: int) -> bool:
    """Is the subject masked in frame t at all?"""
    h, w = masks.shape[1], masks.shape[2]
    return bool((masks[t] > 127).sum() / float(h * w) > MIN_AREA_FRAC)


def mask_gaps(masks: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [a, b) ranges with no subject mask - the holes recovery targets."""
    runs: list[tuple[int, int]] = []
    a = None
    for t in range(len(masks)):
        if not mask_present(masks, t):
            a = t if a is None else a
        elif a is not None:
            runs.append((a, t))
            a = None
    if a is not None:
        runs.append((a, len(masks)))
    return runs


def skip_shot_chunks(man: dict, shot_id: str, reason: str) -> int:
    """Mark a shot's chunks as intentionally skipped.

    A shot with no subject in it has nothing to regenerate. Marking its chunks
    `skipped` keeps them out of the generation queue and out of the full-run
    preflight, and assembly then passes the original frames through at their
    original positions.
    """
    n = 0
    for c in man.get("chunks", []):
        if c["shot_id"] == shot_id and c.get("status") in (None, "", "pending"):
            c["status"] = "skipped"
            c["error"] = reason
            n += 1
    return n


def diagnose(masks: np.ndarray, scores: list[float], log) -> list[dict]:
    """Flag tracking loss, disappearance/reappearance and sudden area jumps."""
    T, H, W = masks.shape
    areas = (masks > 127).reshape(T, -1).sum(axis=1) / float(H * W)
    events: list[dict] = []
    present = areas > MIN_AREA_FRAC
    for t in range(T):
        if not present[t]:
            if t == 0 or present[t - 1]:
                events.append({"frame": int(t), "type": "disappeared",
                               "area": float(areas[t])})
            continue
        if t > 0 and not present[t - 1]:
            events.append({"frame": int(t), "type": "reappeared", "area": float(areas[t])})
        if t > 0 and present[t - 1] and areas[t - 1] > 0:
            ratio = areas[t] / areas[t - 1]
            if ratio > AREA_JUMP or ratio < 1.0 / AREA_JUMP:
                events.append({"frame": int(t), "type": "area_jump",
                               "ratio": round(float(ratio), 2),
                               "area": float(areas[t])})
        if scores[t] < MIN_TRACK_SCORE:
            events.append({"frame": int(t), "type": "low_confidence",
                           "score": round(scores[t], 3)})
    return events


def write_mask_video(masks: np.ndarray, out: Path, fps: int, grow: int,
                     feather: int, log) -> None:
    """Grayscale mask video: WHITE = regenerate (the subject), BLACK = preserve.

    Dilation is applied here, because it is a decision about how much of the
    silhouette edge and contact shadow belongs to the subject. Feathering is NOT:
    the workflow's FeatherMask node softens the edge with the configured radius
    just before conditioning. Doing it in both places softened twice, so the
    configured radius did not describe the edge the model actually saw, and the
    result was pixels regenerated outside the intended boundary.

    Holes inside the subject (e.g. a gap enclosed by its own outline) are left as
    SAM 2 produced them; no fill-holes is applied, so background visible through
    the subject stays preserved.
    """
    import cv2
    T, H, W = masks.shape
    out.parent.mkdir(parents=True, exist_ok=True)
    if feather:
        log.debug("Feather %d px is applied by the workflow, not here.", feather)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray", "-an", str(out)],
        stdin=subprocess.PIPE)
    k = None
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    for t in range(T):
        m = masks[t]
        if k is not None:
            m = cv2.dilate(m, k)
        ff.stdin.write(m.astype(np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    if ff.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {out}")


def write_contact_sheet(frames_dir: Path, masks: np.ndarray, out: Path,
                        title_frames: list[int], log) -> None:
    """Overlay grid for review. Written to disk only; never displayed."""
    from PIL import Image
    import cv2
    # Always sample the whole shot, then add the notable frames on top of that.
    # This used to be `title_frames or <spaced>`, so any non-empty title_frames
    # suppressed the spread entirely - and title_frames is never empty, because
    # the seed frame is always prepended. A shot with no events therefore got a
    # ONE-frame review sheet: the reviewer saw a single moment out of the whole
    # shot at exactly the point where the whole shot is what needs judging.
    n = len(masks)
    want = [int(i) for i in (title_frames or []) if 0 <= i < n]
    for i in np.linspace(0, n - 1, 12).astype(int):
        if len(set(want)) >= 12:
            break
        want.append(int(i))
    idxs = sorted(set(want))[:12]
    tiles = []
    for i in idxs:
        f = frames_dir / f"{i:05d}.jpg"
        if not f.exists():
            continue
        im = np.asarray(Image.open(f).convert("RGB")).copy()
        m = masks[i] > 127
        # red tint inside the regenerated region + white contour
        im[m] = (im[m] * 0.55 + np.array([255, 60, 60]) * 0.45).astype(np.uint8)
        cnts, _ = cv2.findContours((m.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnts, -1, (255, 255, 255), 1)
        cv2.putText(im, f"f{i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(Image.fromarray(im))
    if not tiles:
        log.warning("%s: no extracted frames available, no review sheet written",
                    out.name)
        return
    if len(tiles) < len(idxs):
        # Silence here once produced a sheet that looked like a rendering fault.
        log.warning("%s: only %d of %d requested frames were on disk; the sheet "
                    "covers less of the shot than it should", out.name,
                    len(tiles), len(idxs))
    # Never pad the grid with empty cells: a blank tile reads as a broken render.
    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    tw = 320
    th = int(tw * tiles[0].height / tiles[0].width)
    sheet = Image.new("RGB", (cols * tw, rows * th), (12, 12, 12))
    for i, t in enumerate(tiles):
        sheet.paste(t.resize((tw, th), Image.LANCZOS), ((i % cols) * tw, (i // cols) * th))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


# ---------------------------------------------------------------------------

def parse_points(specs: list[str]) -> list[tuple[float, float, int]]:
    pts = []
    for s in specs:
        parts = s.split(",")
        if len(parts) != 3:
            raise ValueError(f"--init-points expects x,y,+ or x,y,- (got {s!r})")
        x, y, sign = parts
        pts.append((float(x), float(y), 1 if sign.strip() in ("+", "1") else 0))
    return pts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--shot", nargs="*", default=None, help="Limit to these shot ids")
    ap.add_argument("--force", action="store_true", help="Retrack even if masks exist")
    ap.add_argument("--window", type=int, default=240, help="SAM2 propagation window")
    ap.add_argument("--keep-frames", action="store_true", help="Keep extracted JPEGs")
    ap.add_argument("--no-recover", action="store_true",
                    help="Do not re-detect and restart the track after a loss")
    ap.add_argument("--detect-threshold", type=float, default=DETECT_THRESHOLD,
                    help=f"Grounding DINO box threshold (default {DETECT_THRESHOLD})")
    ap.add_argument("--detect-threshold-min", type=float, default=DETECT_THRESHOLD_MIN,
                    help="Retry threshold used only when the primary pass finds "
                         f"nothing at all (default {DETECT_THRESHOLD_MIN}). "
                         "Set equal to --detect-threshold to disable the retry.")
    # manual seeds (only needed for shots flagged needs_user)
    ap.add_argument("--init-box", default=None, help="x0,y0,x1,y1 in working-stream pixels")
    ap.add_argument("--init-points", nargs="*", default=None, help="x,y,+ x,y,- ...")
    ap.add_argument("--init-mask", type=Path, default=None,
                    help="First-frame mask PNG (white = subject)")
    ap.add_argument("--seed-frame", type=int, default=0,
                    help="Frame WITHIN the shot that the manual seed refers to")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("track_subject", args.verbose)
    require_tools("ffmpeg", "ffprobe")
    cfg = load_config(args.config)
    man = load_manifest()

    work = P.root / man["normalized"]["work_path"]
    W = man["normalized"]["width"]
    H = man["normalized"]["height"]
    fps = man["normalized"]["fps"]
    grow = int(cfg["mask"]["grow"])
    feather = int(cfg["mask"]["feather"])

    if not work.exists():
        log.error("Working stream missing: %s. Run preprocess_source.py first.", work)
        return 1

    manual = bool(args.init_box or args.init_points or args.init_mask)
    if manual and (not args.shot or len(args.shot) != 1):
        log.error("Manual seeding applies to exactly one shot: pass --shot shotXXXX")
        return 1

    shots = man["shots"]
    if args.shot:
        want = set(args.shot)
        shots = [s for s in shots if s["shot_id"] in want]
        if not shots:
            log.error("No matching shots. Available: %s",
                      ", ".join(s["shot_id"] for s in man["shots"][:20]))
            return 1

    models = Models(log)
    bank = None if manual else build_match_bank(models, log)

    P.masks.mkdir(parents=True, exist_ok=True)
    review_dir = P.masks / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = P.intermediate / "_frames"
    tmp_root.mkdir(parents=True, exist_ok=True)

    report: list[dict] = []
    needs_user: list[str] = []
    t_all = time.time()

    for si, shot in enumerate(shots):
        sid = shot["shot_id"]
        n = shot["end_frame"] - shot["start_frame"]
        mask_video = P.masks / f"{sid}_mask.mkv"

        if mask_video.exists() and not args.force and probe_frames(mask_video) == n:
            log.info("[%d/%d] %s: masks already present (%d frames), skipping",
                     si + 1, len(shots), sid, n)
            continue

        log.info("[%d/%d] %s: %d frames (%.2fs)", si + 1, len(shots), sid, n, n / fps)
        frames_dir = tmp_root / sid
        frames = extract_shot_frames(work, frames_dir, shot["start_frame"],
                                     shot["end_frame"], log)

        from PIL import Image

        # ---- decide the seed -------------------------------------------------
        seed: dict
        conf = 0.0
        detail: dict = {}
        seed_frame = 0

        if manual:
            seed_frame = max(0, min(args.seed_frame, n - 1))
            if args.init_mask:
                m = np.asarray(Image.open(args.init_mask).convert("L").resize((W, H)))
                seed = {"mask": m > 127}
                detail = {"source": "manual --init-mask", "file": str(args.init_mask)}
            elif args.init_points:
                seed = {"points": parse_points(args.init_points)}
                detail = {"source": "manual --init-points", "points": seed["points"]}
            else:
                seed = {"box": [float(v) for v in args.init_box.split(",")]}
                detail = {"source": "manual --init-box", "box": seed["box"]}
            conf = 1.0
            log.info("%s: manual seed at shot-frame %d (%s)", sid, seed_frame,
                     detail["source"])
        else:
            # representative frames spread across the shot
            probes = sorted(set(int(x) for x in np.linspace(0, n - 1, min(6, n))))

            def probe_pass(thr: float) -> tuple[dict | None, list[dict]]:
                best_c, cands = None, []
                for pf in probes:
                    im = Image.open(frames[pf]).convert("RGB")
                    for b in models.detect_candidates(im, threshold=thr)[:5]:
                        crop = im.crop(tuple(int(v) for v in b[:4]))
                        if crop.width < 12 or crop.height < 24:
                            continue
                        sc = score_candidate(models, bank, crop)
                        cand = {"frame": pf, "box": list(b[:4]), "det_score": b[4], **sc}
                        cands.append(cand)
                        if best_c is None or cand["fused"] > best_c["fused"]:
                            best_c = cand
                return best_c, cands

            det_thr = args.detect_threshold
            best, all_cands = probe_pass(det_thr)

            # "Found somebody" is not "found the subject". The retry below used
            # to fire only when the primary pass found NOTHING, which means a
            # confident detection of the wrong candidate suppressed it entirely.
            # That is what happened once: the detector is trained on upright
            # candidates, so a target whose pose sits far from that prior scored
            # below threshold while a non-target scored above it, and the
            # non-target was the only candidate ever offered.
            #
            # So the retry also fires when candidates exist but NONE of them
            # carries match evidence: no resolvable anchor anywhere is exactly
            # the symptom of the right candidate never having been surfaced.
            if (best is not None
                    and not any(c.get("anchor_sim") is not None for c in all_cands)
                    and args.detect_threshold_min < det_thr):
                log.warning("%s: %d candidate(s) at %.2f but not one has a "
                            "resolvable anchor. Retrying at %.2f before trusting "
                            "any of them - an unusual pose scores low, and the "
                            "subject may simply never have been surfaced.",
                            sid, len(all_cands), det_thr, args.detect_threshold_min)
                lo_best, lo_cands = probe_pass(args.detect_threshold_min)
                if any(c.get("anchor_sim") is not None for c in lo_cands):
                    log.warning("%s: the lower threshold surfaced a candidate "
                                "WITH match evidence. Using it.", sid)
                    det_thr, best, all_cands = (args.detect_threshold_min,
                                                lo_best, lo_cands)
                else:
                    # Keep the wider candidate set anyway: the ambiguity report
                    # and the review sheet should show what was actually there.
                    all_cands = lo_cands or all_cands

            if best is None and args.detect_threshold_min < det_thr:
                det_thr = args.detect_threshold_min
                log.warning("%s: nothing above the %.2f detection threshold; "
                            "retrying at %.2f (degraded sources score lower)",
                            sid, args.detect_threshold, det_thr)
                best, all_cands = probe_pass(det_thr)
                if best is not None:
                    log.warning("%s: seeded from a WEAK detection (score %.3f). "
                                "Match scoring still decides whether this shot "
                                "is trusted; check its review sheet.",
                                sid, best["det_score"])
            if best is None:
                # Noextent matching anywhere in the shot. That is a fact about the
                # footage, not a failure to be fixed: the shot is passed through
                # unrestored. It no longer blocks the full-run preflight.
                log.info("%s: no candidate detected in any probe frame. Marked "
                         "subject_absent; the shot passes through unrestored.", sid)
                shot["subject_status"] = "subject_absent"
                shot["subject_note"] = "no candidate detected in any probe frame"
                skip_shot_chunks(man, sid, "subject_absent")
                report.append({"shot_id": sid, "status": "subject_absent",
                               "reason": "no candidate detected", "confidence": 0.0})
                if not args.keep_frames:
                    shutil.rmtree(frames_dir, ignore_errors=True)
                continue

            # Ambiguity is a per-frame question: within one frame, does a second
            # candidate score nearly as well as the winner? Comparing across frames
            # would compare the subject with itself. The worst frame governs,
            # because an ambiguous frame anywhere can capture the wrong subject -
            # only the winning frame used to be examined.
            per_frame: dict[int, list[float]] = {}
            for c in all_cands:
                per_frame.setdefault(c["frame"], []).append(c["fused"])
            margins = {fr: (sorted(v, reverse=True)[0] - sorted(v, reverse=True)[1])
                       for fr, v in per_frame.items() if len(v) > 1}
            win_top = sorted(per_frame[best["frame"]], reverse=True)
            runner_up = win_top[1] if len(win_top) > 1 else 0.0
            worst_frame = min(margins, key=margins.get) if margins else best["frame"]
            margin = margins.get(worst_frame, best["fused"] - runner_up)
            conf = best["fused"]
            seed = {"box": best["box"]}
            seed_frame = best["frame"]
            detail = {"source": "auto", "n_candidates": len(all_cands),
                      "detect_threshold": det_thr,
                      "weak_detection": det_thr < args.detect_threshold,
                      "runner_up": round(runner_up, 4), "margin": round(margin, 4),
                      "margin_worst_frame": int(worst_frame),
                      "margin_in_seed_frame": round(best["fused"] - runner_up, 4),
                      "frames_probed": len(per_frame),
                      **{k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in best.items() if k != "box"}}
            n_anchord = sum(1 for c in all_cands if c.get("anchor_sim") is not None)
            log.info("%s: best candidate frame=%d confidence=%.3f (anchor match %s, "
                     "anchor %.2f%% of crop) worst margin=%.3f at frame %d over %d "
                     "candidate(s), %d with a resolvable anchor",
                     sid, seed_frame, conf,
                     f"{best['anchor_sim']:.3f}" if best.get("anchor_sim") is not None
                     else "NONE", 100 * (best.get("anchor_frac") or 0.0), margin,
                     worst_frame, len(all_cands), n_anchord)
            if n_anchord == 0:
                log.warning("%s: NO candidate had a resolvable anchor. There is no "
                            "match evidence in this shot at all. Attributes "
                            "similarity is not used as a substitute - that is "
                            "what tracked the wrong candidate before.", sid)

            if conf < AUTO_ACCEPT or margin < AMBIGUOUS_MARGIN or n_anchord == 0:
                reason = ("no match evidence: no candidate had a resolvable anchor"
                          if n_anchord == 0 else
                          "low confidence" if conf < AUTO_ACCEPT else
                          "ambiguous: two candidates score alike")
                log.warning("%s: %s (conf=%.3f margin=%.3f). A mask is still "
                            "written so you have something to LOOK AT, but it is "
                            "flagged needs_user and generation will refuse to "
                            "run on it. Seed it with --init-box or --init-points.",
                            sid, reason, conf, margin)
                shot["subject_status"] = "needs_user"
                shot["subject_note"] = reason
                needs_user.append(sid)

        # ---- track ------------------------------------------------------------
        def reseed_at(frame_idx: int, _sid=sid) -> dict | None:
            """Re-detect the subject at one frame, for restarting a lost track.

            Returns a box seed only if the best candidate clears the same match
            bar used for auto-acceptance, so a lost track is never silently
            resumed on the wrong candidate; a weaker match leaves the gap empty.
            """
            im = Image.open(frames[frame_idx]).convert("RGB")
            found = None
            for thr in (args.detect_threshold, args.detect_threshold_min):
                boxes = models.detect_candidates(im, threshold=thr)
                for b in boxes[:5]:
                    crop = im.crop(tuple(int(v) for v in b[:4]))
                    if crop.width < 12 or crop.height < 24:
                        continue
                    sc = score_candidate(models, bank, crop)
                    if found is None or sc["fused"] > found["fused"]:
                        found = {"box": list(b[:4]), **sc}
                if found is not None or thr == args.detect_threshold_min:
                    break
            if found is None or found["fused"] < AUTO_ACCEPT:
                return None
            return {"box": found["box"]}

        t0 = time.time()
        masks, scores, absent = track_shot(
            models, frames_dir, n, seed_frame, seed, log, window=args.window,
            reseed=None if args.no_recover else reseed_at)
        events = diagnose(masks, scores, log)
        area_frac = float((masks > 127).mean())
        elapsed = time.time() - t0
        log.info("%s: tracked %d frames in %s (%.1f fps), mean mask area %.2f%%, "
                 "%d event(s)", sid, n, human_time(elapsed), n / max(elapsed, 1e-6),
                 100 * area_frac, len(events))
        for e in events[:8]:
            log.warning("  %s @ shot-frame %s %s", e["type"], e["frame"],
                        {k: v for k, v in e.items() if k not in ("type", "frame")})
        if len(events) > 8:
            log.warning("  ... and %d more (see the JSON report)", len(events) - 8)

        absent_frames = sum(b - a for a, b in absent)
        if absent:
            log.info("%s: %d frame(s) in %d span(s) have no matching subject; "
                     "they are preserved as-is.", sid, absent_frames, len(absent))

        if area_frac < MIN_AREA_FRAC:
            if absent_frames >= n:
                # Re-detection looked and found nobody anywhere: absence, not a
                # tracking failure, so it does not need the user.
                log.info("%s: subject absent throughout; the shot passes through "
                         "unrestored.", sid)
                shot["subject_status"] = "subject_absent"
                shot["subject_note"] = "subject absent throughout"
                skip_shot_chunks(man, sid, "subject_absent")
            else:
                log.error("%s: tracked region is essentially empty (%.4f%%). "
                          "Flagging needs_user.", sid, 100 * area_frac)
                shot["subject_status"] = "needs_user"
                shot["subject_note"] = "tracked mask empty"
                if sid not in needs_user:
                    needs_user.append(sid)

        write_mask_video(masks, mask_video, fps, grow, feather, log)
        got = probe_frames(mask_video)
        if got != n:
            log.error("%s: mask video has %d frames, expected %d", sid, got, n)
            return 1

        write_contact_sheet(frames_dir, masks, review_dir / f"{sid}_review.png",
                            [seed_frame] + [e["frame"] for e in events[:6]], log)

        shot["subject_confidence"] = round(float(conf), 4)
        if manual:
            # A manual seed IS the answer to the question needs_user was asking,
            # so it clears that flag. Leaving it set meant a stale flag from an
            # earlier automatic attempt outlived the human's correction and kept
            # the shot blocked forever with no way to unblock it.
            #
            # It does NOT stand in for approval: someone still has to look at the
            # review sheet and confirm the mask follows the right subject for the
            # whole shot, which is a different question from who to start from.
            shot["subject_status"] = "manual"
            shot.pop("subject_note", None)
        elif shot.get("subject_status") not in ("needs_user", "subject_absent"):
            shot["subject_status"] = "auto"
        report.append({
            "shot_id": sid, "status": shot["subject_status"],
            "confidence": round(float(conf), 4), "seed_frame": int(seed_frame),
            "seed": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in seed.items() if k != "mask"},
            "selection": detail, "mean_mask_area_frac": round(area_frac, 5),
            "absent_ranges": [[int(a), int(b)] for a, b in absent],
            "absent_frames": int(absent_frames),
            "events": events, "mask_video": rel(mask_video),
            "review_sheet": rel(review_dir / f"{sid}_review.png"),
            "track_seconds": round(elapsed, 1),
        })

        if not args.keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)

    # ---- per-chunk mask slices ------------------------------------------------
    log.info("Slicing per-chunk masks...")
    made = 0
    for c in man["chunks"]:
        shot_id = c["shot_id"]
        sm = P.masks / f"{shot_id}_mask.mkv"
        if not sm.exists():
            continue
        shot = next(s for s in man["shots"] if s["shot_id"] == shot_id)
        dst = (P.root / c["mask_path"]).with_suffix(".mkv")
        if dst.exists() and not args.force and probe_frames(dst) == c["n_frames"]:
            c["mask_path"] = str(dst.relative_to(P.root))
            continue
        # chunk frames are absolute; shot mask starts at the shot's first frame
        rel_start = c["start_frame"] - shot["start_frame"]
        slice_frames(sm, dst, rel_start, rel_start + c["n_frames"], c["fps"], log,
                     lossless=True, gray=True)
        if probe_frames(dst) != c["n_frames"]:
            log.error("%s: mask slice frame count mismatch", c["chunk_id"])
            return 1
        c["mask_path"] = str(dst.relative_to(P.root))
        made += 1

    save_manifest(man)

    # Merge into any existing report rather than replacing it: correcting a
    # single shot with --shot must not erase the results for every other shot.
    P.reports.mkdir(parents=True, exist_ok=True)
    report_path = P.reports / "tracking_report.json"
    merged: dict[str, dict] = {}
    if report_path.exists():
        try:
            prev = json.loads(report_path.read_text())
            merged = {s["shot_id"]: s for s in prev.get("shots", [])}
        except Exception as e:
            log.warning("Could not read the previous tracking report (%s); "
                        "starting a fresh one.", e)
    for s in report:
        merged[s["shot_id"]] = s
    # A shot only still "needs user" if its LATEST result says so.
    all_needs = sorted(sid for sid, s in merged.items()
                       if s.get("status") == "needs_user")
    all_shots = sorted(merged.values(), key=lambda s: s["shot_id"])

    report_path.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mask_polarity": "white=regenerate (subject), black=preserve",
        "polarity_source": "ComfyUI/comfy_extras/nodes_wan.py::WanVaceToVideo "
                           "(reactive = control_video * mask)",
        "thresholds": {"auto_accept": AUTO_ACCEPT, "ambiguous_margin": AMBIGUOUS_MARGIN,
                       "area_jump": AREA_JUMP, "min_area_frac": MIN_AREA_FRAC},
        "shots_in_this_run": [s["shot_id"] for s in report],
        "shots": all_shots,
        "needs_user": all_needs,
        # Deliberate pass-through, not an open question: no one matching the
        # match bank appears in these shots, so there is nothing to regenerate.
        "subject_absent": sorted(sid for sid, s in merged.items()
                                 if s.get("status") == "subject_absent"),
    }, indent=2))
    needs_user = [s for s in all_needs
                  if next((x for x in man["shots"] if x["shot_id"] == s), {})
                  .get("subject_status") != "manual"]
    absent_shots = sorted(sid for sid, s in merged.items()
                          if s.get("status") == "subject_absent")

    log.info("=" * 62)
    log.info("Tracked %d shot(s) in %s; %d chunk mask slice(s) written",
             len(report), human_time(time.time() - t_all), made)
    log.info("Review sheets: %s", review_dir)
    log.info("Report       : reports/tracking_report.json")
    if absent_shots:
        log.info("%d shot(s) have no subject and pass through unrestored: %s",
                 len(absent_shots), ", ".join(absent_shots[:8])
                 + (" ..." if len(absent_shots) > 8 else ""))
    if needs_user:
        log.warning("%d shot(s) need your input: %s", len(needs_user),
                    ", ".join(needs_user))
        log.warning("Inspect the review sheet, then re-seed just that shot, e.g.:")
        log.warning("  scripts/track_subject.py --shot %s --init-box x0,y0,x1,y1 --force",
                    needs_user[0])
    else:
        log.info("All shots resolved automatically; no manual selection needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
