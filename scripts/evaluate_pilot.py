#!/usr/bin/env python
"""Phase 10b - measure the pilot variants against each other.

Every number here is computed from pixels with numpy/OpenCV. Nothing is opened,
displayed or eyeballed (CLAUDE.md rules 1 and 2b): the script reports statistics,
and the human judgement about whether the restoration is *good* stays with the
person reading reports/pilot_results.md.

Metrics, and what each is evidence for:

  temporal_stability     mean |frame_t - frame_{t-1}| after compensating for the
                         source's own motion, measured OUTSIDE the subject mask.
                         Lower = a steadier background. This is what catches
                         flicker and background drift between chunks.
  bg_drift_vs_source     mean |variant - source| outside the mask, low-frequency
                         only. Large values mean the environment moved away from
                         the original scene - fine for a restoration, a warning
                         when it should have been preserved.
  bg_preserved_within_N  fraction of background pixels within N grey levels of
                         the plate the variant was supposed to preserve. Reported
                         at several tolerances because every variant shares a
                         lossy container: at tolerance 0 the container noise
                         dominates and the ranking inverts.
  bg_drift_vs_plate      mean |variant - plate| outside the mask. The robust form
                         of the same question, and the one to trust.
  subject_sharpness      Laplacian variance inside the mask
  bg_sharpness           Laplacian variance outside the mask
  sharpness_balance      subject / background. Near 1.0 means the figure and the
                         environment look like they belong to the same image; a
                         value far from 1 is the pasted-on look.
  edge_halo              mean brightness step across a narrow ring straddling the
                         mask edge, relative to the same ring in the source.
                         Above 1 means the composite introduced a rim.
  detail_gain            high-frequency energy vs the source, outside the mask.
                         >1 = environmental detail added.
  hf_invention           high-frequency energy added where the source had none.
                         A proxy for hallucinated texture, altered signage and
                         mutated distant faces: those all show up as new detail
                         in regions that were flat.
  chunk_boundary_jump    frame-to-frame difference at chunk seams vs the median
                         elsewhere. >1 means a visible seam.

Garment fidelity is measured against the SOURCE interval, never against the
reference photographs - they show a different outfit, so agreeing with them
would be the failure, not the goal. The brief pulls in two directions and the
metrics are kept apart accordingly:

Read them in that order: garment CLASS and coverage first, then boundaries,
then accessories, and colour only if `colour_is_meaningful` - once the
silhouette has moved, a colour offset is not the defect, and chroma correction
would only match a missing garment to the palette of the one it replaced.

  garment_deltaE         area-weighted colour distance from the source, measured
                         on the SAME frame over regions the source's own parse
                         defines, so both are read from identical pixels. The
                         source measured against itself must score ~0; that
                         control is what caught the earlier sample-dependent
                         version scoring 39.6 on it.
  garment_temporal_dE    how much that colour crawls between frames
  garment_iou            silhouette overlap with the source's parsed garment
  garment_boundary_f     edge agreement at 2 px, which IoU is too blunt to see
  garment_lowfreq_drift  blurred difference inside the garment. Must stay SMALL:
                         this is the structure that was supposed to survive.
  garment_hf_gain        high-frequency energy vs the source. Should be ABOVE 1
                         for a restoration - but an increase ALONE does not show
                         texture was restored: ringing, added noise and an
                         invented weave raise it too. Reported as
                         high_frequency_increased, never as "texture restored".
  garment_pattern_chi2   gradient-orientation histogram distance - a changed
                         weave or print rather than a sharper one
  accessories_*          accessory classes invented or dropped, by name

    scripts/evaluate_pilot.py --report reports/pilot_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, load_manifest, probe_frames, rel, setup_logging  # noqa: E402


def read_gray(path: Path, limit: int | None = None) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok or (limit and len(out) >= limit):
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    return np.stack(out) if out else np.zeros((0, 1, 1), np.float32)


def read_rgb(path: Path, limit: int | None = None) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok or (limit and len(out) >= limit):
            break
        out.append(f.astype(np.float32))
    cap.release()
    return np.stack(out) if out else np.zeros((0, 1, 1, 3), np.float32)


def lowpass(a: np.ndarray, k: int = 9) -> np.ndarray:
    import cv2
    return np.stack([cv2.GaussianBlur(f, (k, k), 0) for f in a])


def highfreq(a: np.ndarray) -> np.ndarray:
    return a - lowpass(a)


def edge_ring(mask: np.ndarray, width: int = 4) -> np.ndarray:
    """Narrow ring straddling the subject silhouette."""
    import cv2
    m = (mask > 127).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return (cv2.dilate(m, k) - cv2.erode(m, k)).astype(bool)


def metrics(variant: np.ndarray, source: np.ndarray, masks: np.ndarray,
            plate: np.ndarray | None) -> dict:
    """All metrics for one variant. Arrays are (T,H,W) grayscale float32."""
    T = min(len(variant), len(source), len(masks))
    variant, source, masks = variant[:T], source[:T], masks[:T]
    inside = masks > 127
    outside = ~inside
    eps = 1e-6

    def sel_mean(a, sel):
        v = [a[t][sel[t]].mean() for t in range(len(a)) if sel[t].any()]
        return float(np.mean(v)) if v else float("nan")

    # temporal stability outside the subject, relative to the source's own motion
    dv = np.abs(np.diff(variant, axis=0))
    ds = np.abs(np.diff(source, axis=0))
    out2 = outside[1:]
    tv = sel_mean(dv, out2)
    ts = sel_mean(ds, out2)

    lo_v, lo_s = lowpass(variant), lowpass(source)
    hf_v, hf_s = np.abs(highfreq(variant)), np.abs(highfreq(source))

    import cv2
    lap_in, lap_out = [], []
    for t in range(T):
        lap = cv2.Laplacian(variant[t], cv2.CV_32F)
        if inside[t].any():
            lap_in.append(float(lap[inside[t]].var()))
        if outside[t].any():
            lap_out.append(float(lap[outside[t]].var()))
    s_in = float(np.mean(lap_in)) if lap_in else float("nan")
    s_out = float(np.mean(lap_out)) if lap_out else float("nan")

    # halo: brightness step across the silhouette ring, vs the same ring in source
    rings = [edge_ring(masks[t]) for t in range(T)]
    ring_v = sel_mean(np.abs(np.stack([cv2.Laplacian(f, cv2.CV_32F) for f in variant])),
                      np.stack(rings))
    ring_s = sel_mean(np.abs(np.stack([cv2.Laplacian(f, cv2.CV_32F) for f in source])),
                      np.stack(rings))

    # invention: new high frequency where the source was flat
    flat = (hf_s < np.percentile(hf_s, 25)) & outside
    inv = sel_mean(hf_v - hf_s, flat)

    m = {
        "frames": int(T),
        "temporal_stability": round(tv, 4),
        "temporal_stability_vs_source": round(tv / (ts + eps), 4),
        "bg_drift_vs_source": round(sel_mean(np.abs(lo_v - lo_s), outside), 4),
        "subject_sharpness": round(s_in, 2),
        "bg_sharpness": round(s_out, 2),
        "sharpness_balance": round(s_in / (s_out + eps), 4),
        "edge_halo": round(ring_v / (ring_s + eps), 4),
        "detail_gain": round(sel_mean(hf_v, outside) / (sel_mean(hf_s, outside) + eps), 4),
        "hf_invention": round(inv, 4),
    }
    if plate is not None:
        pl = plate[:T]
        # Tolerance, not equality. Every variant is written through the same
        # lossy yuv420p container, and the composited ones make one extra
        # RGB->YUV->RGB round trip on the way, which shifts almost every pixel by
        # about one level. A bit-exact test therefore scores the container rather
        # than the pipeline - it ranked the composite path, whose background is
        # copied verbatim, BELOW the path that regenerates it through a VAE.
        for tol in (1.0, 2.0, 4.0):
            same = [float((np.abs(variant[t] - pl[t])[outside[t]] <= tol).mean())
                    for t in range(T) if outside[t].any()]
            m[f"bg_preserved_within_{int(tol)}"] = (
                round(float(np.mean(same)), 4) if same else None)
        # The robust one: mean absolute low-frequency difference from the plate.
        # Unlike a threshold it does not care where the container noise sits.
        m["bg_drift_vs_plate"] = round(sel_mean(np.abs(lowpass(variant) - lowpass(pl)),
                                                outside), 4)
        m["bg_max_drift_vs_plate"] = round(float(np.max(
            [np.abs(variant[t] - pl[t])[outside[t]].max()
             for t in range(T) if outside[t].any()])), 2)
    return m


def garment_metrics(path: Path, mask_path: Path, source_palette: dict,
                    n_probe: int, log) -> dict:
    """Did the garments keep the colour the SOURCE says they are?

    Parses clothing on the variant itself and compares each garment class with
    the palette learned from the original footage. Two separate failures are
    distinguished, because they have different causes:

      garment_deltaE      distance from the source colour. Large = the model
                          repainted the clothes, which is the fault reported on
                          the earlier build.
      garment_temporal_dE spread of that colour across frames. Large = the
                          colour crawls even if its average is right.

    ~2.3 is a just-noticeable difference; 10+ is an obviously different colour.
    """
    import cv2
    from PIL import Image
    from make_reference_pack import Parser, palette, delta_e

    if not source_palette:
        return {}
    parser = Parser(log)
    cap, mcap = cv2.VideoCapture(str(path)), cv2.VideoCapture(str(mask_path))
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
        return {}

    idx = np.linspace(0, len(frames) - 1, min(n_probe, len(frames))).astype(int)
    per_frame = []
    for i in idx:
        if masks[i].sum() < 64:
            continue
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        ys, xs = np.where(masks[i])
        crop = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if crop.shape[0] < 32 or crop.shape[1] < 16:
            continue
        per_frame.append(palette(crop, parser.parse(Image.fromarray(crop))))
    if not per_frame:
        return {}

    drift, temporal, seen = [], [], []
    for name, ref in source_palette.items():
        vals = [p[name]["lab"] for p in per_frame if name in p]
        if not vals:
            continue
        med = [float(np.median([v[c] for v in vals])) for c in range(3)]
        drift.append(delta_e(med, ref["lab"]))
        temporal.append(float(np.mean([delta_e(v, med) for v in vals])))
        seen.append(name)
    if not drift:
        return {"garment_classes_found": 0}
    return {"garment_classes_found": len(seen),
            "garment_classes": seen,
            # SECONDARY, and sample-dependent: the pack palette was computed on
            # the sharpest frames of the interval while this is computed on
            # evenly spaced ones, so the two never see identical lighting or
            # pose. Measuring the source against itself this way scores ~40,
            # which is why the primary garment_deltaE is measured same-frame in
            # garment_structure() instead. Kept only as a cross-check.
            "garment_deltaE_vs_pack_palette": round(float(np.mean(drift)), 2),
            "garment_deltaE_vs_pack_palette_worst": round(float(np.max(drift)), 2),
            # This one is sound on its own terms: it is the spread WITHIN this
            # variant across frames, so no cross-sample comparison is involved.
            "garment_temporal_dE": round(float(np.mean(temporal)), 2)}


ACCESSORY = {"hat", "bag", "belt", "scarf", "left_shoe", "right_shoe"}


def _probe_pairs(var_path: Path, src_path: Path, mask_path: Path, n_probe: int):
    """Frame-aligned (source crop, variant crop) pairs around the subject.

    Both are cropped with the SAME box from the same mask, so the two label maps
    that come out of the parser are directly comparable pixel for pixel. Nothing
    is resampled: a resize here would blur exactly the boundary being measured.
    """
    import cv2
    cap, vcap = cv2.VideoCapture(str(src_path)), cv2.VideoCapture(str(var_path))
    src, var, msk = [], [], []
    mcap = cv2.VideoCapture(str(mask_path))
    while True:
        oks, s = cap.read()
        okv, v = vcap.read()
        okm, m = mcap.read()
        if not (oks and okv and okm):
            break
        src.append(s); var.append(v); msk.append(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) > 127)
    for c in (cap, vcap, mcap):
        c.release()
    if not src:
        return []
    idx = np.linspace(0, len(src) - 1, min(n_probe, len(src))).astype(int)
    out = []
    for i in idx:
        if msk[i].sum() < 64 or src[i].shape != var[i].shape:
            continue
        ys, xs = np.where(msk[i])
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if y1 - y0 < 32 or x1 - x0 < 16:
            continue
        out.append((int(i),
                    cv2.cvtColor(src[i][y0:y1, x0:x1], cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(var[i][y0:y1, x0:x1], cv2.COLOR_BGR2RGB)))
    return out


def _boundary_f(a: np.ndarray, b: np.ndarray, tol: int = 2) -> float:
    """Symmetric boundary F-score at a `tol` pixel tolerance.

    Region IoU is dominated by the interior and barely moves when an edge slides
    a few pixels, which is precisely the failure that reads as a redrawn garment.
    This measures the edges themselves.
    """
    import cv2
    k = np.ones((3, 3), np.uint8)
    ea = (cv2.morphologyEx(a.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0)
    eb = (cv2.morphologyEx(b.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0)
    if not ea.any() or not eb.any():
        return float("nan")
    da = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 3)
    prec = float((db[ea] <= tol).mean())      # source edge explained by variant
    rec = float((da[eb] <= tol).mean())       # variant edge justified by source
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def _orient_hist(gray: np.ndarray, region: np.ndarray, bins: int = 16):
    """Magnitude-weighted gradient orientation histogram - a cheap, resolution
    tolerant description of pattern (stripes, print, weave direction)."""
    import cv2
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)[region]
    ang = (np.arctan2(gy, gx)[region] % np.pi) / np.pi * bins
    h = np.bincount(np.clip(ang.astype(int), 0, bins - 1), weights=mag,
                    minlength=bins).astype(np.float64)
    return h / max(1e-6, h.sum())


def garment_structure(var_path: Path, src_path: Path, mask_path: Path,
                      n_probe: int, log) -> dict:
    """Everything about the garment that colour alone cannot catch.

    The brief for this restoration is specific: keep the source's low-frequency
    colour and structure, and add only the missing high-frequency texture. That
    is two measurements pulling in opposite directions, so they are reported
    separately rather than fused into one score.

      garment_iou           overlap of the parsed garment region with the
                            source's. Low = the model redrew the silhouette.
      garment_boundary_f    edge agreement at 2 px. Catches an edge that slid
                            while the region stayed roughly the same size.
      garment_lowfreq_drift mean |variant - source| inside the garment after
                            heavy blur. This must stay SMALL: it is the
                            structure and colour that were supposed to survive.
      garment_hf_gain       high-frequency energy vs the source inside the
                            garment. This should be ABOVE 1: it is the texture
                            the restoration exists to add. Below 1 means the
                            pass smoothed the clothes instead of restoring them.
      garment_pattern_chi2  distance between gradient-orientation histograms.
                            Large = the weave or print changed direction, which
                            is a different fabric rather than a sharper one.
      accessories_*         accessory classes invented or dropped, by name. An
                            invented bag is not a texture detail.
    """
    import cv2
    from PIL import Image
    from make_reference_pack import LBL, GARMENT, Parser

    pairs = _probe_pairs(var_path, src_path, mask_path, n_probe)
    if not pairs:
        return {}
    parser = Parser(log)
    gids = [i for i, n in LBL.items() if n in GARMENT]

    ious, bfs, lows, hfs, chis = [], [], [], [], []
    drift_w, drift_worst, cls_drift = [], [], {}
    per_class: dict[str, list] = {}
    invented: dict[str, int] = {}
    dropped: dict[str, int] = {}
    for _, s_rgb, v_rgb in pairs:
        ls = parser.parse(Image.fromarray(s_rgb))
        lv = parser.parse(Image.fromarray(v_rgb))
        gs, gv = np.isin(ls, gids), np.isin(lv, gids)
        if not gs.any():
            continue
        union = int((gs | gv).sum())
        ious.append(int((gs & gv).sum()) / max(1, union))
        bfs.append(_boundary_f(gs, gv))

        for idx, name in LBL.items():
            if name not in GARMENT:
                continue
            a, b = (ls == idx), (lv == idx)
            if a.any() or b.any():
                per_class.setdefault(name, []).append(
                    int((a & b).sum()) / max(1, int((a | b).sum())))
            if name in ACCESSORY:
                # Judged by pixel share, not presence: a stray dozen pixels from
                # the parser is not an invented handbag.
                sa, sb = a.mean(), b.mean()
                if sb > 0.01 and sa < 0.002:
                    invented[name] = invented.get(name, 0) + 1
                elif sa > 0.01 and sb < 0.002:
                    dropped[name] = dropped.get(name, 0) + 1

        # Garment colour drift, measured on THIS frame over regions the SOURCE's
        # own parse defines, and read from both images at the same pixels.
        #
        # Comparing against the stored pack palette instead was invalid three
        # ways, and the source-vs-itself control proved it by scoring 39.6 when
        # it must score 0: the two palettes came from DIFFERENT frame samples
        # (the pack takes the sharpest frames, the evaluator evenly spaced ones),
        # a class present in one frame of six carried the same weight as one
        # present in all six, and SegFormer moves the same physical garment
        # between `dress`, `upper` and `skirt` from frame to frame, so per-class
        # comparison was not comparing the same cloth.
        s_lab = cv2.cvtColor(s_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        v_lab = cv2.cvtColor(v_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        per_class_dE, weights = [], []
        for idx, name in LBL.items():
            if name not in GARMENT:
                continue
            reg = ls == idx
            n_px = int(reg.sum())
            if n_px < max(64, int(0.005 * gs.size)):
                continue                     # too small to be a stable region
            a = [float(np.median(s_lab[..., ch][reg])) for ch in range(3)]
            b = [float(np.median(v_lab[..., ch][reg])) for ch in range(3)]
            dE = float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))
            per_class_dE.append(dE)
            weights.append(n_px)
            cls_drift.setdefault(name, []).append(dE)
        if per_class_dE:
            w = np.asarray(weights, np.float64)
            drift_w.append(float(np.average(per_class_dE, weights=w)))
            drift_worst.append(float(np.max(per_class_dE)))

        sg = cv2.cvtColor(s_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        vg = cv2.cvtColor(v_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lo_s = cv2.GaussianBlur(sg, (0, 0), 4.0)
        lo_v = cv2.GaussianBlur(vg, (0, 0), 4.0)
        lows.append(float(np.mean(np.abs(lo_v - lo_s)[gs])))
        hs = float(cv2.Laplacian(sg, cv2.CV_32F)[gs].var())
        hv = float(cv2.Laplacian(vg, cv2.CV_32F)[gs].var())
        hfs.append(hv / max(1e-6, hs))
        ha, hb = _orient_hist(sg, gs), _orient_hist(vg, gs)
        chis.append(float(0.5 * np.sum((ha - hb) ** 2 / (ha + hb + 1e-9))))

    if not ious:
        return {}
    valid_bf = [b for b in bfs if b == b]
    out = {
        "garment_frames_probed": len(ious),
        "garment_iou": round(float(np.median(ious)), 4),
        "garment_boundary_f": (round(float(np.median(valid_bf)), 4)
                               if valid_bf else None),
        "garment_lowfreq_drift": round(float(np.median(lows)), 2),
        "garment_hf_gain": round(float(np.median(hfs)), 3),
        "garment_pattern_chi2": round(float(np.median(chis)), 4),
        "garment_class_iou": {k: round(float(np.median(v)), 4)
                              for k, v in sorted(per_class.items())},
        "accessories_invented": invented,
        "accessories_dropped": dropped,
        # Area-weighted, so a class covering a handful of pixels cannot dominate.
        "garment_deltaE": (round(float(np.median(drift_w)), 2)
                           if drift_w else None),
        "garment_deltaE_worst_class": (round(float(np.median(drift_worst)), 2)
                                       if drift_worst else None),
        "garment_deltaE_by_class": {
            k: round(float(np.median(v)), 2)
            for k, v in sorted(cls_drift.items())
            # Only classes the source parse found on most probe frames. A class
            # seen once is a parser flicker, not a garment.
            if len(v) >= max(2, len(ious) // 2)},
        "outfit_mutation": bool(drift_w and float(np.median(drift_w)) > 10.0),
    }
    # The two directions of the brief, stated as one boolean each so a report
    # cannot quietly average them into "fine".
    out["structure_preserved"] = bool(out["garment_iou"] > 0.75
                                      and out["garment_lowfreq_drift"] < 8.0)
    # Colour is the LAST question, and only a meaningful one while the garment
    # is still there. Once the silhouette has moved this far, a colour offset is
    # not the defect and correcting it would only make a missing garment better
    # matched to the palette of the one it replaced.
    out["colour_is_meaningful"] = bool(out["garment_iou"] > 0.60
                                       and out["garment_lowfreq_drift"] < 12.0)
    out["chroma_correction_advised"] = bool(
        out["colour_is_meaningful"] and (out.get("garment_deltaE") or 0) > 2.3)
    # Deliberately NOT called "texture_restored". All this measures is that
    # high-frequency energy went up, and ringing, added noise and a hallucinated
    # weave all raise it just as reliably as genuine restored fabric does.
    # Calling it "restored" would be asserting the conclusion the metric cannot
    # reach. Pattern agreement (garment_pattern_chi2) and temporal stability are
    # reported alongside it; a restoration claim needs all three to hold.
    out["high_frequency_increased"] = bool(out["garment_hf_gain"] > 1.0)
    return out


def reference_identity(log):
    """Face embeddings of the consensus-verified references, once per run.

    Delegates to identity.resolve_targets so that "the person the references
    agree on" means exactly the same thing here, in the pack and in tracking.

    It previously walked inputs/references itself, took the LARGEST face in each
    photograph via face_detail(), and never consulted the exclusion list. Both
    are the failure modes identity.py exists to prevent: the largest face in a
    photograph is not necessarily the target, and a reference the user excluded
    for this run could still enter the bank and then be used to judge whether a
    generated frame looked like the target.
    """
    import track_subject as T
    from identity import load_exclusions, reference_files, resolve_targets

    models = T.Models(log)
    res = resolve_targets(reference_files(), models, log, load_exclusions(log))
    bank = res.get("bank")
    if bank is None:
        log.warning("No verified reference identity; identity metrics skipped")
    return models, bank


def covering_metrics(var_path: Path, src_path: Path, mask_path: Path,
                     n_probe: int, log) -> dict:
    """Did the pass remove a face covering, or invent an uncovered face?

    A covering is apparel. Taking it off is not an improvement that scored badly
    on some axis - it is the model dressing the person differently, and no
    colour or sharpness metric notices it at all. So it is asked directly:
    compare the head region's covering and skin composition against the source.
    """
    import cv2
    from PIL import Image
    from make_reference_pack import COVERING, HEAD, LBL, Parser

    pairs = _probe_pairs(var_path, src_path, mask_path, n_probe)
    if not pairs:
        return {}
    parser = Parser(log)
    ids = {n: i for i, n in LBL.items()}
    cov_ids = [ids[n] for n in COVERING]
    head_ids = [ids[n] for n in HEAD]
    face_id = ids["face"]

    s_cov, v_cov, s_face, v_face = [], [], [], []
    for _, s_rgb, v_rgb in pairs:
        ls = parser.parse(Image.fromarray(s_rgb))
        lv = parser.parse(Image.fromarray(v_rgb))
        hs, hv = np.isin(ls, head_ids), np.isin(lv, head_ids)
        if not hs.any():
            continue
        s_cov.append(float(np.isin(ls, cov_ids).sum()) / max(1, int(hs.sum())))
        v_cov.append(float(np.isin(lv, cov_ids).sum()) / max(1, int(hv.sum()) or 1))
        s_face.append(float((ls == face_id).sum()) / max(1, int(hs.sum())))
        v_face.append(float((lv == face_id).sum()) / max(1, int(hv.sum()) or 1))
    if not s_cov:
        return {}
    sc, vc = float(np.median(s_cov)), float(np.median(v_cov))
    sf, vf = float(np.median(s_face)), float(np.median(v_face))
    covered = sc > sf or sf < 0.15
    # Removed: the source's head was substantially covered and the variant's is
    # not, with bare face appearing where there was none.
    removed = bool(covered and vc < 0.5 * sc and vf > sf + 0.10)
    return {"source_head_covering_fraction": round(sc, 4),
            "variant_head_covering_fraction": round(vc, 4),
            "source_head_face_fraction": round(sf, 4),
            "variant_head_face_fraction": round(vf, 4),
            "source_face_covered": bool(covered),
            "covering_removed": removed,
            "covering_retention": round(vc / sc, 3) if sc > 1e-6 else None}


def identity_metrics(path: Path, mask_path: Path, models, bank, n_probe: int,
                     log) -> dict:
    """How much the generated face agrees with the external references.

    This is the ONLY thing the external photographs are allowed to influence, so
    it is measured and reported entirely apart from garment fidelity. Comparing
    the same number for the source says whether conditioning improved identity
    or merely changed it - a variant that agrees less than the 240p source did
    has made the face worse, however sharp it looks.
    """
    import cv2
    from PIL import Image
    from make_reference_pack import face_detail
    if bank is None:
        return {}
    cap, mcap = cv2.VideoCapture(str(path)), cv2.VideoCapture(str(mask_path))
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
        return {}
    idx = np.linspace(0, len(frames) - 1, min(n_probe, len(frames))).astype(int)
    sims, found = [], 0
    for i in idx:
        if masks[i].sum() < 64:
            continue
        ys, xs = np.where(masks[i])
        crop = frames[i][ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        emb, _, _ = face_detail(models, Image.fromarray(rgb))
        if emb is None:
            continue
        found += 1
        sims.append(float(np.max(bank @ emb)))
    if not sims:
        return {"identity_face_frames": 0,
                "identity_note": "no face was detectable in the subject region"}
    return {"identity_face_frames": found,
            "identity_frames_probed": int(len(idx)),
            "identity_similarity": round(float(np.median(sims)), 4),
            "identity_similarity_worst": round(float(np.min(sims)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", type=Path, default=None,
                    help="JSON mapping variant name -> video path "
                         "(default: reports/pilot_variants.json)")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--no-garment", action="store_true",
                    help="Skip clothing parsing (faster; loses colour drift)")
    ap.add_argument("--garment-frames", type=int, default=6)
    ap.add_argument("--no-identity", action="store_true",
                    help="Skip face identity scoring (faster; loses the "
                         "separate identity-from-references measurement)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("evaluate_pilot", args.verbose)
    man = load_manifest()
    vpath = args.variants or (P.reports / "pilot_variants.json")
    if not vpath.exists():
        log.error("No variant list at %s. Run make_comparisons.py first.", vpath)
        return 1
    spec = json.loads(vpath.read_text())
    variants = spec["variants"]

    pilot = man.get("pilot")
    # The interval, not one chunk: pilot_compare assembles every variant across
    # all intersecting chunks and hands over a matching interval-wide mask.
    source = read_gray(P.root / spec["source"])
    mask_path = P.root / spec["mask"]
    masks = (read_gray(mask_path) if mask_path.exists()
             else np.zeros_like(source, dtype=np.float32))
    if not mask_path.exists():
        log.warning("No interval mask at %s; subject/background metrics will be "
                    "meaningless.", mask_path)
    # The palette the SOURCE says this outfit is. Everything is measured against
    # this, not against the reference photographs, which showed a different one.
    source_palette = {}
    pack_dir = P.intermediate / "reference_packs"
    if pack_dir.exists():
        packs = sorted(pack_dir.glob("*_pack.json"))
        if packs:
            source_palette = json.loads(packs[0].read_text()).get("source_palette", {})
            log.info("Garment reference: %d class(es) from the source - %s",
                     len(source_palette), ", ".join(source_palette))
    iv = spec.get("interval", {})
    log.info("Pilot interval %s-%s across %d chunk(s): %d source frame(s), "
             "%d mask frame(s), %d variant(s)",
             iv.get("start_frame"), iv.get("end_frame"),
             len(spec.get("chunks", [])), len(source), len(masks), len(variants))
    if len(masks) != len(source):
        log.warning("Mask has %d frames but the interval has %d; metrics use the "
                    "shorter.", len(masks), len(source))

    id_models, id_bank = (None, None)
    if not args.no_identity:
        try:
            id_models, id_bank = reference_identity(log)
        except Exception as e:
            log.warning("Identity scoring unavailable (%s); garment and "
                        "background metrics are unaffected.", e)

    out = {}
    for name, entry in variants.items():
        path = P.root / entry["path"]
        if not path.exists():
            log.warning("%-28s MISSING (%s)", name, path)
            continue
        plate = None
        if entry.get("plate"):
            pp = P.root / entry["plate"]
            plate = read_gray(pp) if pp.exists() else None
        v = read_gray(path)
        m = metrics(v, source, masks, plate)
        if source_palette and not args.no_garment and mask_path.exists():
            try:
                m.update(garment_metrics(path, mask_path, source_palette,
                                         args.garment_frames, log))
            except Exception as e:
                log.warning("%s: garment colour metrics unavailable (%s)", name, e)
        if not args.no_garment and mask_path.exists():
            try:
                m.update(garment_structure(path, P.root / spec["source"],
                                           mask_path, args.garment_frames, log))
            except Exception as e:
                log.warning("%s: garment structure metrics unavailable (%s)",
                            name, e)
        if not args.no_garment and mask_path.exists():
            try:
                m.update(covering_metrics(path, P.root / spec["source"], mask_path,
                                          args.garment_frames, log))
            except Exception as e:
                log.warning("%s: covering metrics unavailable (%s)", name, e)
        if id_bank is not None and mask_path.exists():
            try:
                im_ = identity_metrics(path, mask_path, id_models, id_bank,
                                       args.garment_frames, log)
                # If the SOURCE face is covered, agreement with an uncovered
                # reference photograph is not a score to maximise - a high value
                # would mean the covering came off. It is reported as
                # unobservable, and the covering check is what matters instead.
                if m.get("source_face_covered"):
                    im_ = {"identity_similarity_unobservable": True,
                           "identity_note": "the source face is covered; "
                                            "external-reference agreement cannot "
                                            "be observed and must not be "
                                            "maximised - see covering_removed",
                           "identity_similarity_if_measured":
                               im_.get("identity_similarity")}
                m.update(im_)
            except Exception as e:
                log.warning("%s: identity metrics unavailable (%s)", name, e)
        m["path"] = rel(path)
        m["describes"] = entry.get("describes", "")
        out[name] = m
        gm = (f"  garmentdE={m['garment_deltaE']:.1f}"
              f"{'/MUTATED' if m.get('outfit_mutation') else ''}"
              if m.get("garment_deltaE") is not None else "")
        log.info("%-28s stability=%.3f (x%.2f src)  bal=%.2f  halo=%.2f  "
                 "detail=%.2f  invent=%.3f%s" + gm, name, m["temporal_stability"],
                 m["temporal_stability_vs_source"], m["sharpness_balance"],
                 m["edge_halo"], m["detail_gain"], m["hf_invention"],
                 f"  plate_drift={m['bg_drift_vs_plate']:.2f} "
                 f"within2={m['bg_preserved_within_2']:.3f}"
                 if m.get("bg_drift_vs_plate") is not None else "")
        if m.get("covering_removed"):
            log.warning("%-28s FACE COVERING REMOVED: the source head is %.0f%% "
                        "covering and this variant is %.0f%%, with bare face "
                        "rising %.0f%%->%.0f%%. The model undressed the subject.",
                        "", 100 * m["source_head_covering_fraction"],
                        100 * m["variant_head_covering_fraction"],
                        100 * m["source_head_face_fraction"],
                        100 * m["variant_head_face_fraction"])
        elif m.get("source_face_covered"):
            log.info("%-28s covering retained (%.2f of the source's); identity "
                     "agreement is UNOBSERVABLE here and is not scored", "",
                     m.get("covering_retention") or 0.0)
        if m.get("identity_similarity") is not None:
            log.info("%-28s identity: agreement with the external references "
                     "%.3f (worst %.3f) on %d/%d probed frame(s)", "",
                     m["identity_similarity"], m["identity_similarity_worst"],
                     m["identity_face_frames"], m["identity_frames_probed"])
        elif m.get("identity_face_frames") == 0:
            log.info("%-28s identity: no face detectable in the subject region",
                     "")
        if m.get("garment_iou") is not None:
            log.info("%-28s garment: iou=%.3f boundaryF=%s lowfreq=%.1f "
                     "hf_gain=%.2f pattern=%.3f  -> structure %s, %s%s",
                     "", m["garment_iou"],
                     ("%.3f" % m["garment_boundary_f"]
                      if m.get("garment_boundary_f") is not None else "n/a"),
                     m["garment_lowfreq_drift"], m["garment_hf_gain"],
                     m["garment_pattern_chi2"],
                     "PRESERVED" if m["structure_preserved"] else "MOVED",
                     "HF UP" if m["high_frequency_increased"] else "HF DOWN",
                     ("  accessories invented: "
                      + ", ".join(m["accessories_invented"])
                      if m.get("accessories_invented") else "")
                     + ("" if m.get("colour_is_meaningful", True) else
                        "  [colour not meaningful: the garment structure has "
                        "gone, so dE and chroma correction say nothing]"))

    rp = args.report or (P.reports / "pilot_metrics.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    # A pilot is an INTERVAL and can span several chunks, so the report names all
    # of them. This used to reference an undefined `cid`, which raised NameError
    # at the very last statement - after every metric had been computed - so the
    # report was never written at all and the whole evaluation was lost.
    rp.write_text(json.dumps({"pilot": pilot,
                              "chunks": list(spec.get("chunks", [])),
                              "interval": iv,
                              "variants": out}, indent=2) + "\n")
    if not out:
        log.warning("No variant produced metrics; the report records an empty "
                    "set rather than pretending the comparison happened.")
    log.info("Metrics for %d variant(s) -> %s", len(out), rel(rp))
    log.info("These are statistics, not a verdict. Open the comparison videos "
             "yourself and fill in reports/pilot_results.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
