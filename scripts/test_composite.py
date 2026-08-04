#!/usr/bin/env python
"""Checks on the compositing layer order, on synthetic masks.

No models, no video, no CUDA.

    venv/bin/python scripts/test_composite.py

What it proves:

  * the foreground boundary is genuinely ONE-SIDED - no pixel outside the
    occluder is ever dimmed, so the generated figure can never spread over
    whoever is in front of it
  * the occluder's core stays fully opaque; only a narrow rim ramps
  * carrying part of the previous frame's alpha reduces the frame-to-frame
    crawl of a jittering silhouette, and does so without touching a static one
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from composite_subject import occluder_alpha  # noqa: E402

BAND = 2


class Failures(list):
    def check(self, cond: bool, msg: str) -> bool:
        if not cond:
            self.append(msg)
        return cond


def disc(size=120, cx=60, cy=60, r=25, dx=0):
    yy, xx = np.mgrid[0:size, 0:size]
    return ((xx - cx - dx) ** 2 + (yy - cy) ** 2 <= r * r)


def as_u8(m):
    return (m.astype(np.uint8) * 255)


def test_one_sided(f: Failures) -> None:
    occ = disc()
    a = occluder_alpha(as_u8(occ), BAND)

    f.check(bool((a[~occ] == 1.0).all()),
            "the foreground ramp dims pixels OUTSIDE the occluder; the figure "
            "would be blended over something in front of it")

    # A few pixels in from the silhouette the occluder must be fully opaque.
    import cv2
    core = cv2.erode(occ.astype(np.uint8),
                     np.ones((2 * BAND + 3, 2 * BAND + 3), np.uint8)).astype(bool)
    f.check(bool((a[core] == 0.0).all()),
            "the occluder core is not fully opaque; the figure shows through it")

    ramp = (a > 0.0) & (a < 1.0)
    f.check(bool(ramp.any()), "no ramp at all - the edge is still fully hard")
    f.check(bool((~occ)[ramp].sum() == 0),
            "part of the ramp lies outside the occluder, so it is not one-sided")
    # And it stays narrow: a wide ramp is the halo this is meant to avoid.
    f.check(ramp.sum() < occ.sum() * 0.5,
            f"the ramp covers {ramp.sum() / occ.sum():.0%} of the occluder; it "
            f"is supposed to be a rim, not a gradient across the whole shape")


def test_degenerate_cases(f: Failures) -> None:
    empty = np.zeros((60, 60), bool)
    f.check(bool((occluder_alpha(as_u8(empty), BAND) == 1.0).all()),
            "with no occluder the figure must be untouched")

    occ = disc(size=60, cx=30, cy=30, r=12)
    hard = occluder_alpha(as_u8(occ), 0)
    f.check(bool((hard[occ] == 0.0).all() and (hard[~occ] == 1.0).all()),
            "band 0 must fall back to the hard edge, with nothing in between")


def test_temporal_smoothing(f: Failures) -> None:
    """A silhouette that jitters by a pixel per frame, as a re-segmented one does."""
    def run(smooth: float, jitter: bool):
        prev, crawl = None, []
        for i in range(12):
            occ = disc(dx=(i % 2) if jitter else 0)
            a = occluder_alpha(as_u8(occ), BAND)
            if smooth > 0 and prev is not None:
                a = (1.0 - smooth) * a + smooth * prev
            if prev is not None:
                crawl.append(float(np.abs(a - prev).mean()))
            prev = a
        return float(np.mean(crawl))

    raw = run(0.0, jitter=True)
    smoothed = run(0.35, jitter=True)
    f.check(smoothed < raw,
            f"smoothing did not steady a jittering edge ({smoothed:.5f} vs "
            f"{raw:.5f})")
    f.check(run(0.35, jitter=False) == 0.0,
            "smoothing introduced movement into a static occluder")


def main() -> int:
    f = Failures()
    for t in (test_one_sided, test_degenerate_cases, test_temporal_smoothing):
        t(f)
    if f:
        print(f"FAILED: {len(f)} check(s)")
        for m in f:
            print(f"  - {m}")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
