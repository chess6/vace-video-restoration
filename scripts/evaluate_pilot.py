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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", type=Path, default=None,
                    help="JSON mapping variant name -> video path "
                         "(default: reports/pilot_variants.json)")
    ap.add_argument("--report", type=Path, default=None)
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
    cid = pilot["chunks"][0]
    c = next(x for x in man["chunks"] if x["chunk_id"] == cid)
    source = read_gray(P.root / spec["source"])
    masks = read_gray(P.root / c["mask_path"])
    log.info("Pilot %s: %d source frame(s), %d mask frame(s), %d variant(s)",
             cid, len(source), len(masks), len(variants))

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
        m["path"] = rel(path)
        m["describes"] = entry.get("describes", "")
        out[name] = m
        log.info("%-28s stability=%.3f (x%.2f src)  bal=%.2f  halo=%.2f  "
                 "detail=%.2f  invent=%.3f%s", name, m["temporal_stability"],
                 m["temporal_stability_vs_source"], m["sharpness_balance"],
                 m["edge_halo"], m["detail_gain"], m["hf_invention"],
                 f"  plate_drift={m['bg_drift_vs_plate']:.2f} "
                 f"within2={m['bg_preserved_within_2']:.3f}"
                 if m.get("bg_drift_vs_plate") is not None else "")

    rp = args.report or (P.reports / "pilot_metrics.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps({"pilot": pilot, "chunk": cid, "variants": out},
                             indent=2) + "\n")
    log.info("Metrics -> %s", rel(rp))
    log.info("These are statistics, not a verdict. Open the comparison videos "
             "yourself and fill in reports/pilot_results.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
