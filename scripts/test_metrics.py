#!/usr/bin/env python
"""Checks on the attribute-fidelity primitives, on synthetic shapes.

No models, no video, no CUDA. These numbers end up in a report that decides
whether a restoration is judged good, so they are worth proving on inputs whose
answer is known in advance rather than trusting because they look plausible.

    venv/bin/python scripts/test_metrics.py

What it proves:

  * the boundary score is 1.0 for an identical edge, stays high for a shift
    inside its tolerance, and collapses for a shift well outside it - so it
    actually distinguishes "sharper in the same place" from "redrawn elsewhere"
  * region IoU alone does NOT distinguish those, which is why the boundary
    score exists
  * the pattern descriptor separates stripe directions and ignores brightness
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_pilot import _boundary_f, _orient_hist  # noqa: E402


class Failures(list):
    def check(self, cond: bool, msg: str) -> bool:
        if not cond:
            self.append(msg)
        return cond


def square(size=200, x=50, y=50, w=100, h=100):
    a = np.zeros((size, size), bool)
    a[y:y + h, x:x + w] = True
    return a


def iou(a, b):
    return float((a & b).sum()) / max(1, int((a | b).sum()))


def test_boundary_score(f: Failures) -> None:
    a = square()
    f.check(abs(_boundary_f(a, a) - 1.0) < 1e-9,
            "an identical edge did not score 1.0")

    # Shifted DIAGONALLY, so the whole boundary moves. A purely horizontal shift
    # leaves the top and bottom edges lying exactly on top of the originals, and
    # the score stays middling for a perfectly good reason - half the boundary
    # really did not move.
    near = square(x=51, y=51)                    # 1 px, inside the 2 px tolerance
    far = square(x=58, y=58)                     # 8 px, plainly a different edge
    b_near, b_far = _boundary_f(a, near), _boundary_f(a, far)
    f.check(b_near > 0.9, f"a 1 px shift scored {b_near:.3f}; within tolerance "
                          f"it should stay near 1")
    f.check(b_far < 0.4, f"a 15 px shift scored {b_far:.3f}; a redrawn edge must "
                         f"score low")
    f.check(b_near > b_far, "the boundary score is not monotonic in edge error")

    # The point of having it at all: IoU barely reacts to the shift that matters.
    f.check(iou(a, far) > 0.5 and b_far < 0.4,
            f"IoU {iou(a, far):.2f} vs boundaryF {b_far:.2f}: the two metrics no "
            f"longer disagree on a shifted edge, so one of them is broken")

    empty = np.zeros_like(a)
    f.check(_boundary_f(a, empty) != _boundary_f(a, empty) or True,
            "unreachable")           # documents that an empty side returns NaN
    f.check(np.isnan(_boundary_f(a, empty)),
            "an empty region should report NaN, not a score")


def test_pattern_descriptor(f: Failures) -> None:
    reg = np.ones((120, 120), bool)
    vert = np.zeros((120, 120), np.float32)
    vert[:, ::6] = 255.0                          # vertical stripes
    horiz = vert.T.copy()                         # same pattern, rotated 90 deg

    def chi2(a, b):
        ha, hb = _orient_hist(a, reg), _orient_hist(b, reg)
        return float(0.5 * np.sum((ha - hb) ** 2 / (ha + hb + 1e-9)))

    f.check(chi2(vert, vert) < 1e-9, "identical patterns did not score 0")
    d_rot = chi2(vert, horiz)
    f.check(d_rot > 0.5, f"a 90 deg rotation scored only {d_rot:.3f}; the "
                         f"descriptor cannot tell stripe directions apart")

    # Brightness must not read as a pattern change: a darker attribute is not a
    # different fabric, and the histogram is normalised so it cannot say so.
    d_bright = chi2(vert, vert * 0.5)
    f.check(d_bright < 1e-6,
            f"halving brightness moved the pattern distance to {d_bright:.4f}")


def main() -> int:
    f = Failures()
    for t in (test_boundary_score, test_pattern_descriptor):
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
