#!/usr/bin/env python
"""Phase 9c - build side-by-side comparison videos and frame grids.

Everything is written to outputs/comparisons/ as files. Nothing is played or
displayed on screen.

Panels, when the corresponding input exists:
  1. original 240p input, enlarged for viewing (nearest-neighbour so you can see
     exactly what the source really contains, not an interpolated flatter version)
  2. depth-controlled VACE with reference sheet + subject mask
  3. the same without reference conditioning (ablation)
  4. a second seed of the reference-conditioned version
  5. mask overlay on the source
  6. the reference sheet

    scripts/make_comparisons.py [--pilot]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, load_config, load_manifest, probe_dims_fps, probe_frames, require_tools,
    run, setup_logging,
)


def label(img: np.ndarray, text: str) -> np.ndarray:
    import cv2
    out = img.copy()
    h = max(22, out.shape[0] // 18)
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (8, int(h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                h / 42.0, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def read_frames(path: Path, n: int | None = None) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    while n is None or len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out) if out else np.zeros((0, 1, 1, 3), np.uint8)


def resize_to(frames: np.ndarray, w: int, h: int, nearest=False) -> np.ndarray:
    import cv2
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LANCZOS4
    return np.stack([cv2.resize(f, (w, h), interpolation=interp) for f in frames])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--pilot", action="store_true", default=True)
    ap.add_argument("--grid-frames", type=int, default=6)
    args = ap.parse_args()

    log = setup_logging("make_comparisons")
    require_tools("ffmpeg")
    cfg = load_config(args.config)
    man = load_manifest()
    fps = man["normalized"]["fps"]
    P.comparisons.mkdir(parents=True, exist_ok=True)

    pilot_chunks = [c for c in man["chunks"] if c.get("is_pilot")]
    if not pilot_chunks:
        log.error("No pilot chunks marked. Run scripts/extract_pilot.py.")
        return 1
    cid = pilot_chunks[0]["chunk_id"]
    W, H = pilot_chunks[0]["width"], pilot_chunks[0]["height"]

    # ---- gather whatever variants exist -------------------------------------
    variants: list[tuple[str, Path, bool]] = []   # (label, path, nearest)

    src_chunk = (P.root / pilot_chunks[0]["control_path"]).with_suffix(".mp4")
    if src_chunk.exists():
        variants.append(("1. ORIGINAL 240p (enlarged, no interpolation)", src_chunk, True))

    main_out = P.restored_480p / f"{cid}.mp4"
    if main_out.exists():
        variants.append(("2. VACE + depth + reference + mask", main_out, False))

    for tag, lbl in (("noref", "3. VACE without reference conditioning"),
                     ("seedB", "4. VACE reference-conditioned, second seed")):
        p = P.restored_480p / f"{cid}_{tag}.mp4"
        if p.exists():
            variants.append((lbl, p, False))

    if len(variants) < 2:
        log.error("Need at least the source and one restored output. Found: %s",
                  [v[0] for v in variants])
        return 1
    log.info("Comparing %d variant(s)", len(variants))

    n = min(probe_frames(p) for _, p, _ in variants)
    stacks = []
    for lbl, p, nearest in variants:
        f = read_frames(p, n)
        f = resize_to(f, W, H, nearest=nearest)
        stacks.append((lbl, f))

    # ---- mask overlay panel ---------------------------------------------------
    mask_p = (P.root / pilot_chunks[0]["mask_path"])
    if mask_p.exists():
        import cv2
        m = read_frames(mask_p, n)
        m = resize_to(m, W, H)
        base = stacks[0][1]
        ov = base.astype(np.float32).copy()
        sel = (m[..., 0:1].astype(np.float32) / 255.0)
        tint = np.array([255, 60, 60], np.float32)
        ov = ov * (1 - 0.45 * sel) + tint * (0.45 * sel)
        stacks.append(("5. Tracked subject mask (red = regenerated)",
                       ov.astype(np.uint8)))

    # ---- side-by-side video ---------------------------------------------------
    cols = 2 if len(stacks) <= 4 else 3
    rows = (len(stacks) + cols - 1) // cols
    grid_w, grid_h = W * cols, H * rows
    import subprocess
    sbs = P.comparisons / f"{cid}_side_by_side.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{grid_w}x{grid_h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium",
         "-pix_fmt", "yuv420p", str(sbs)], stdin=subprocess.PIPE)
    for t in range(n):
        canvas = np.zeros((grid_h, grid_w, 3), np.uint8)
        for i, (lbl, f) in enumerate(stacks):
            r, c = divmod(i, cols)
            canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = label(f[t], lbl)
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    log.info("Side-by-side video -> %s", sbs.relative_to(P.root))

    # ---- frame grids -----------------------------------------------------------
    from PIL import Image
    idxs = sorted(set(int(x) for x in np.linspace(0, n - 1, args.grid_frames)))
    tile_w = 420
    tile_h = int(tile_w * H / W)
    sheet = Image.new("RGB", (tile_w * len(idxs), tile_h * len(stacks)), (10, 10, 10))
    for r, (lbl, f) in enumerate(stacks):
        for c, ti in enumerate(idxs):
            im = Image.fromarray(label(f[ti], f"{lbl}  [f{ti}]")).resize(
                (tile_w, tile_h), Image.LANCZOS)
            sheet.paste(im, (c * tile_w, r * tile_h))
    grid = P.comparisons / f"{cid}_frame_grid.png"
    sheet.save(grid)
    log.info("Frame grid -> %s", grid.relative_to(P.root))

    # ---- reference sheet copy --------------------------------------------------
    ref = P.reference_sheets / "reference_sheet.png"
    if ref.exists():
        import shutil
        shutil.copy2(ref, P.comparisons / "reference_sheet.png")
        log.info("Reference sheet copied into outputs/comparisons/")

    log.info("=" * 62)
    log.info("Comparison artefacts are FILES in %s. Nothing was displayed.",
             P.comparisons)
    log.info("Review them, then record your judgement in reports/pilot_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
