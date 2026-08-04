#!/usr/bin/env python
"""Exhaustive checks of chunk planning and assembly planning.

Pure arithmetic: no CUDA, no ComfyUI, no video decoding, so it runs in seconds
and can be run on any machine before committing.

    venv/bin/python scripts/test_chunking.py [--max-frames 1000]

What it proves, over every shot length in the swept range and over many
missing-chunk patterns:

  * every emitted window is a legal VACE length (4n+1) and fits inside its shot
  * shots never span a detected cut, so no chunk can either
  * the chunks of a shot cover the shot completely
  * assembly planning covers [span_start, span_end) exactly once, in order,
    for every subset of chunks - including subsets with holes, which must be
    filled from the source rather than closing the gap

These are the cases that broke before: a final window snapped backwards can
overlap the previous one by more than the previous chunk has left after its own
seam, and non-adjacent chunks were stitched as though adjacent.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble import op_frames, plan_stitch  # noqa: E402
from common import P, Shot  # noqa: E402
from preprocess_source import build_shots, chunk_shot  # noqa: E402

P_ROOT = P.root

CHUNK_FRAMES = 81
OVERLAP = 8
MIN_SHOT = 17


class Failures(list):
    def check(self, cond: bool, msg: str) -> bool:
        if not cond:
            self.append(msg)
        return cond


def shot_of(n: int) -> Shot:
    return Shot(shot_id="shot0000", start_frame=0, end_frame=n,
                src_start_sec=0.0, src_end_sec=0.0, n_frames=n)


def check_windows(n: int, f: Failures) -> list[tuple[int, int, int]]:
    """Window legality and coverage for a shot of n frames."""
    w = chunk_shot(shot_of(n), CHUNK_FRAMES, OVERLAP)
    if n < 5:
        f.check(w == [], f"n={n}: expected no window below the 5-frame minimum")
        return w
    if not f.check(bool(w), f"n={n}: no windows emitted"):
        return w

    for st, en, _ov in w:
        f.check((en - st - 1) % 4 == 0, f"n={n}: window {st}-{en} is not 4n+1")
        f.check(en - st <= CHUNK_FRAMES, f"n={n}: window {st}-{en} exceeds chunk size")
        f.check(0 <= st < en <= n, f"n={n}: window {st}-{en} escapes the shot")

    covered = set()
    for st, en, _ in w:
        covered |= set(range(st, en))
    f.check(covered == set(range(n)),
            f"n={n}: windows cover {len(covered)}/{n} frames")
    return w


def check_plan(spans: list[tuple[int, int]], f: Failures, label: str) -> None:
    """Assembly planning must tile [span_start, span_end) exactly once, in order."""
    ops = plan_stitch(spans)
    span_start = min(s for s, _ in spans)
    span_end = max(s + n for s, n in spans)

    total = sum(op_frames(o) for o in ops)
    if not f.check(total == span_end - span_start,
                   f"{label}: plan emits {total} frames, span needs "
                   f"{span_end - span_start}"):
        return

    # Walk the plan and confirm each op reads frames that actually exist in the
    # chunk it names, and lands where it should on the absolute timeline.
    pos = span_start
    for o in ops:
        if o[0] == "skip":
            continue
        if o[0] == "copy":
            _, i, off, n = o
            s, ln = spans[i]
            f.check(0 <= off and off + n <= ln,
                    f"{label}: copy {off}+{n} outside chunk {i} of length {ln}")
            f.check(s + off == pos,
                    f"{label}: copy from chunk {i} lands at {s + off}, expected {pos}")
        elif o[0] == "blend":
            _, i, off_i, j, off_j, n = o
            si, li = spans[i]
            sj, lj = spans[j]
            f.check(0 <= off_i and off_i + n <= li,
                    f"{label}: blend reads {off_i}+{n} outside chunk {i} ({li})")
            f.check(0 <= off_j and off_j + n <= lj,
                    f"{label}: blend reads {off_j}+{n} outside chunk {j} ({lj})")
            # both sides must describe the SAME absolute frames, or the dissolve
            # would cross-fade two different moments
            f.check(si + off_i == sj + off_j == pos,
                    f"{label}: blend sides at {si + off_i} and {sj + off_j}, "
                    f"expected {pos}")
        else:
            _, a0, n = o
            f.check(a0 == pos, f"{label}: source fill at {a0}, expected {pos}")
        pos += op_frames(o)

    f.check(pos == span_end, f"{label}: plan ends at {pos}, expected {span_end}")


def stamp(idx: int, w: int = 64, h: int = 48):
    """A uniform frame whose colour encodes its absolute index.

    Two channels at 16 spacing give 256 distinguishable indices with enough
    separation to survive an ffv1 yuv420p round trip, so a decoded frame can be
    mapped back to the index it was written with.
    """
    import numpy as np
    f = np.zeros((h, w, 3), np.uint8)
    f[:, :, 0] = (idx // 16) % 16 * 16 + 8
    f[:, :, 1] = (idx % 16) * 16 + 8
    return f


def decode_idx(frame) -> tuple[int, int]:
    import numpy as np
    hi = int(round((float(np.median(frame[:, :, 0])) - 8) / 16)) % 16
    lo = int(round((float(np.median(frame[:, :, 1])) - 8) / 16)) % 16
    return hi, lo


def check_stitch_io(f: Failures) -> None:
    """Run the real stitch_shot over synthetic chunks and verify placement."""
    import logging
    import shutil
    import tempfile

    import numpy as np

    from assemble import FrameWriter, read_frames, stitch_shot

    log = logging.getLogger("test_stitch")
    log.addHandler(logging.NullHandler())
    tmp = Path(tempfile.mkdtemp(prefix="stitch_test_"))
    try:
        n = 300
        windows = chunk_shot(shot_of(n), CHUNK_FRAMES, OVERLAP)

        # the normalized source, used to fill holes
        work = tmp / "work.mkv"
        wr = FrameWriter(work, 64, 48, 16)
        wr.write(np.stack([stamp(i) for i in range(n)]))
        wr.close()

        chunks = []
        for k, (st, en, ov) in enumerate(windows):
            p = tmp / f"c{k:03d}.mkv"
            wr = FrameWriter(p, 64, 48, 16)
            wr.write(np.stack([stamp(i) for i in range(st, en)]))
            wr.close()
            # absolute: `P.root / "/abs/path"` yields the absolute path unchanged,
            # so the loader finds these without them living in the project tree
            chunks.append({"chunk_id": f"shot0000_c{k:03d}", "start_frame": st,
                           "end_frame": en, "n_frames": en - st,
                           "overlap_prev": ov, "output_path": str(p)})

        for label, subset in (("complete", list(range(len(chunks)))),
                              ("missing middle", [0, 2, 3]),
                              ("missing tail", [0, 1]),
                              ("only last", [len(chunks) - 1])):
            cs = [chunks[i] for i in subset]
            out = tmp / f"stitched_{label.replace(' ', '_')}.mkv"
            start, count = stitch_shot(cs, 16, out, log, work=work)
            want_start = min(c["start_frame"] for c in cs)
            want_end = max(c["start_frame"] + c["n_frames"] for c in cs)
            if not f.check(start == want_start and count == want_end - want_start,
                           f"{label}: got span {start}+{count}, want "
                           f"{want_start}+{want_end - want_start}"):
                continue
            frames = read_frames(out)
            if not f.check(len(frames) == count,
                           f"{label}: decoded {len(frames)} frames, wrote {count}"):
                continue
            bad = [i for i in range(count)
                   if decode_idx(frames[i]) != ((start + i) // 16 % 16,
                                                (start + i) % 16)]
            f.check(not bad,
                    f"{label}: {len(bad)} frame(s) at the wrong index, first at "
                    f"output position {bad[0] if bad else -1}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-frames", type=int, default=1000)
    ap.add_argument("--subset-limit", type=int, default=64,
                    help="Max missing-chunk subsets to try per shot length")
    ap.add_argument("--with-video", action="store_true",
                    help="Also encode and decode real video to check placement")
    args = ap.parse_args()

    f = Failures()
    lengths = range(1, args.max_frames + 1)

    # ---- 1. window legality and full coverage --------------------------------
    all_windows: dict[int, list] = {}
    for n in lengths:
        all_windows[n] = check_windows(n, f)
    print(f"1. windows legal and complete for shot lengths 1-{args.max_frames}: "
          f"{'FAIL' if f else 'ok'}")

    # ---- 2. full assembly of every shot length -------------------------------
    before = len(f)
    for n in lengths:
        w = all_windows[n]
        if not w:
            continue
        check_plan([(st, en - st) for st, en, _ in w], f, f"full n={n}")
    print(f"2. assembly plans for every complete shot: "
          f"{'FAIL' if len(f) > before else 'ok'}")

    # ---- 3. assembly with missing chunks -------------------------------------
    # Every non-empty subset for small chunk counts; a capped sample beyond that.
    before = len(f)
    for n in lengths:
        w = all_windows[n]
        if len(w) < 2:
            continue
        spans = [(st, en - st) for st, en, _ in w]
        idxs = range(len(spans))
        subsets = [s for r in idxs for s in itertools.combinations(idxs, r + 1)]
        for sub in subsets[:args.subset_limit]:
            check_plan([spans[i] for i in sub], f, f"n={n} subset={sub}")
    print(f"3. assembly plans with missing chunks: "
          f"{'FAIL' if len(f) > before else 'ok'}")

    # ---- 4. shots never span a cut -------------------------------------------
    before = len(f)
    for total in (40, 100, 160, 200, 500):
        for cuts in ([10], [10, 20], [5, 6, 7], [1, total - 1], [3, 90],
                     list(range(10, total, 17))):
            cuts = [c for c in cuts if 0 < c < total]
            shots = build_shots(cuts, total, MIN_SHOT)
            bounds = sorted(set([0] + cuts + [total]))
            f.check([s.start_frame for s in shots] == bounds[:-1]
                    and [s.end_frame for s in shots] == bounds[1:],
                    f"total={total} cuts={cuts}: shot boundaries {[(s.start_frame, s.end_frame) for s in shots]} "
                    f"do not match the cuts {bounds}")
            for s in shots:
                for c in cuts:
                    f.check(not (s.start_frame < c < s.end_frame),
                            f"total={total} cuts={cuts}: shot "
                            f"{s.start_frame}-{s.end_frame} spans cut {c}")
                for st, en, _ in chunk_shot(s, CHUNK_FRAMES, OVERLAP):
                    for c in cuts:
                        f.check(not (st < c < en),
                                f"total={total} cuts={cuts}: chunk {st}-{en} "
                                f"spans cut {c}")
    print(f"4. no shot or chunk spans a detected cut: "
          f"{'FAIL' if len(f) > before else 'ok'}")

    # ---- 5. execute a real stitch and verify every frame's position ----------
    before = len(f)
    if args.with_video:
        check_stitch_io(f)
        print(f"5. real stitch puts every frame at its original index: "
              f"{'FAIL' if len(f) > before else 'ok'}")
    else:
        print("5. real stitch (skipped; pass --with-video to decode and check)")

    print("-" * 62)
    if f:
        print(f"FAILED: {len(f)} problem(s). First 20:")
        for m in f[:20]:
            print(f"  {m}")
        return 1
    print("All chunking and assembly checks PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
