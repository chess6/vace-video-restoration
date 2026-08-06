#!/usr/bin/env python
"""Contact sheet with coordinate axes burned in, for re-seeding a track.

`track_subject.py --init-box x0,y0,x1,y1` needs numbers in working-stream
pixels, and its own review sheet has no axes - so a reviewer has to count grid
squares to produce them, which is slow and easy to get wrong. This burns the
coordinates into every tile.

Written for exactly that job. On this pilot the automatic track locked onto
static scenery twice; both shots were re-seeded from boxes read off a sheet like
this one.

Geometry is probed from the input rather than assumed, so the same script serves
any working geometry.

Writes one PNG to disk and prints its path. Displays nothing (CLAUDE.md rule 1);
the reviewer opens it themselves, if and when they choose.

    scripts/make_labeled_sheet.py <video> [mask|none] <out.png> [n_tiles]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

STEP = 80          # gridline spacing, in source pixels


def probe(path: str) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"])


def frames(path: str, w: int, h: int, pix: str, chan: int):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", f"format={pix}",
         "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pix, "-"],
        capture_output=True, check=True).stdout
    fs = w * h * chan
    n = len(raw) // fs
    shape = (h, w, chan) if chan > 1 else (h, w)
    return [np.frombuffer(raw[i * fs:(i + 1) * fs], np.uint8).reshape(shape)
            for i in range(n)]


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    import cv2

    work, maskv, out = sys.argv[1], sys.argv[2], sys.argv[3]
    ntiles = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    W, H = probe(work)

    vid = frames(work, W, H, "bgr24", 3)
    msk = None
    if maskv.lower() != "none":
        mw, mh = probe(maskv)
        if (mw, mh) != (W, H):
            print(f"mask is {mw}x{mh} but video is {W}x{H}; refusing to guess")
            return 1
        msk = frames(maskv, W, H, "gray", 1)

    n = len(vid)
    idxs = sorted(set(int(i) for i in np.linspace(0, n - 1, min(ntiles, n))))
    print(f"{n} frames; tiling {idxs}")

    tiles = []
    for i in idxs:
        im = vid[i].copy()
        if msk is not None and i < len(msk):
            m = msk[i] > 127
            im[m] = (im[m] * 0.55 + np.array([60, 60, 255]) * 0.45).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, cs, -1, (255, 255, 255), 1)
        for x in range(0, W + 1, STEP):
            cv2.line(im, (x, 0), (x, H), (255, 255, 0), 1)
            if x < W:
                for col, th in (((0, 0, 0), 3), ((255, 255, 0), 1)):
                    cv2.putText(im, str(x), (x + 3, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th, cv2.LINE_AA)
        for y in range(0, H + 1, STEP):
            cv2.line(im, (0, y), (W, y), (255, 255, 0), 1)
            if y > 0:
                for col, th in (((0, 0, 0), 3), ((255, 255, 0), 1)):
                    cv2.putText(im, str(y), (3, y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th, cv2.LINE_AA)
        for col, th in (((0, 0, 0), 4), ((0, 255, 255), 2)):
            cv2.putText(im, f"f{i}", (W - 70, H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)
        tiles.append(im)

    cols = min(3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    pad = 6
    sheet = np.full((pad + rows * (H + pad), pad + cols * (W + pad), 3), 20, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        y0, x0 = pad + r * (H + pad), pad + c * (W + pad)
        sheet[y0:y0 + H, x0:x0 + W] = t
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, sheet)
    print(f"wrote {out}  {sheet.shape[1]}x{sheet.shape[0]}  ({len(tiles)} tiles of "
          f"{W}x{H}, gridlines every {STEP}px, labels in frame coordinates)")
    print("Nothing was displayed; open it yourself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
