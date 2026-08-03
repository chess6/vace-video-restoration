#!/usr/bin/env python
"""Generate synthetic 240p media so the pipeline can be validated end to end
before any real footage exists.

Produces:
  * a 240p, 4:3, 24 fps clip with a moving figure, a textured background,
    a hard scene cut in the middle, and an audio tone (for A/V sync checks)
  * three "reference" stills of the same figure at higher resolution

This is a rig for exercising the plumbing: normalization, scene detection,
chunking, depth, SAM 2 tracking, generation, assembly and audio remux. The
identity-matching stage cannot be meaningfully validated on a synthetic figure
and is exercised separately with a manual seed.

Writes nothing outside the paths given on the command line, and displays nothing.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H = 320, 240
FPS = 24


def background(seed: int, w: int, h: int) -> np.ndarray:
    """Deterministic textured background so depth has something to work with."""
    rng = np.random.default_rng(seed)
    small = rng.integers(40, 210, size=(h // 20, w // 20, 3), dtype=np.uint8)
    bg = np.asarray(Image.fromarray(small).resize((w, h), Image.BICUBIC)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    bg += 18 * np.sin(xx / 9.0 + seed)[:, :, None]
    bg += 12 * np.cos(yy / 7.0 - seed)[:, :, None]
    return np.clip(bg, 0, 255).astype(np.uint8)


def draw_figure(img: Image.Image, cx: float, cy: float, scale: float, phase: float,
                turn: float) -> None:
    """A crude but consistently-coloured humanoid: head, torso, arms, legs."""
    d = ImageDraw.Draw(img)
    s = scale
    skin = (226, 190, 158)
    shirt = (40, 90, 190)
    trousers = (35, 40, 55)
    hair = (60, 40, 30)

    # torso
    d.rounded_rectangle([cx - 11 * s, cy - 16 * s, cx + 11 * s, cy + 14 * s],
                        radius=4 * s, fill=shirt)
    # head
    hx = cx + 3 * s * turn
    d.ellipse([hx - 8 * s, cy - 34 * s, hx + 8 * s, cy - 16 * s], fill=skin)
    d.chord([hx - 8 * s, cy - 34 * s, hx + 8 * s, cy - 22 * s], 180, 360, fill=hair)
    # arms swing out of phase with the legs
    a = math.sin(phase) * 0.6
    for side in (-1, 1):
        ax = cx + side * 11 * s
        ay = cy - 12 * s
        ex = ax + side * 9 * s * math.cos(a * side)
        ey = ay + 22 * s * abs(math.cos(a * side * 0.5)) + 4 * s
        d.line([ax, ay, ex, ey], fill=shirt, width=int(max(2, 5 * s)))
        d.ellipse([ex - 3 * s, ey - 3 * s, ex + 3 * s, ey + 3 * s], fill=skin)
    # legs
    for side in (-1, 1):
        lx = cx + side * 5 * s
        ly = cy + 14 * s
        ex = lx + side * 4 * s + 8 * s * math.sin(phase) * side
        ey = ly + 26 * s
        d.line([lx, ly, ex, ey], fill=trousers, width=int(max(3, 7 * s)))
    # a small accessory: a bright bag, to test accessory preservation
    d.rectangle([cx + 6 * s, cy - 6 * s, cx + 16 * s, cy + 6 * s],
                fill=(230, 170, 40))


def render_clip(out: Path, seconds: float, cut_at: float) -> None:
    n = int(seconds * FPS)
    cut = int(cut_at * FPS)
    bg_a = background(1, W, H)
    bg_b = background(9, W, H)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-crf", "30", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
         "-shortest", str(out)], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        after = i >= cut
        base = bg_b if after else bg_a
        # slow camera drift so background motion exists
        shift = int(6 * math.sin(t * 0.5))
        frame = np.roll(base, shift, axis=1).copy()
        img = Image.fromarray(frame)
        # the figure walks across, and turns after the cut
        prog = (i - cut) / max(n - cut, 1) if after else i / max(cut, 1)
        cx = 60 + prog * (W - 120)
        cy = H * 0.62 + 4 * math.sin(t * 2.0)
        scale = 1.5 + 0.25 * math.sin(t * 0.7)
        draw_figure(img, cx, cy, scale, t * 6.0, -1.0 if after else 1.0)
        ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    if ff.returncode != 0:
        raise RuntimeError("ffmpeg failed rendering the self-test clip")


def render_references(out_dir: Path) -> list[Path]:
    """Higher-resolution stills of the same figure: full body, upper, side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    specs = [("full_body", 900, 1200, 4.2, 0.0), ("upper", 900, 900, 8.0, 0.4),
             ("side", 800, 1100, 4.0, -1.0)]
    for name, w, h, scale, turn in specs:
        img = Image.fromarray(background(4, w, h))
        draw_figure(img, w * 0.5, h * (0.42 if name == "upper" else 0.55),
                    scale, 0.5, turn)
        p = out_dir / f"_selftest_ref_{name}.png"
        img.save(p)
        made.append(p)
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--refs-dir", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--cut-at", type=float, default=10.0)
    args = ap.parse_args()

    args.video.parent.mkdir(parents=True, exist_ok=True)
    render_clip(args.video, args.seconds, args.cut_at)
    refs = render_references(args.refs_dir)
    print(f"video : {args.video} ({args.seconds:g}s, {W}x{H}@{FPS}, cut at "
          f"{args.cut_at:g}s, with audio)")
    for r in refs:
        print(f"ref   : {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
