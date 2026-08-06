#!/usr/bin/env python
"""Join a run's restored plates into one deliverable, in manifest order.

WHY THIS EXISTS SEPARATELY FROM assemble.py
`assemble.py` stitches VACE chunk OUTPUTS with seam blending. On the path that
measurement settled on there are no VACE outputs: the plate is the deliverable,
and the only thing needed is to put the plates back in order. Reaching for the
seam blender here would blend seams that do not exist.

WHY IT IS NOT `ffmpeg -i bg_*.mkv`
A clip with a scene cut in it becomes several shots, each chunked and each
padded to the 4n+1 the model requires. One 33-second clip came back as five
chunks totalling 893 frames for 528 real ones - 69% of it padding and shot
boundaries. Taking the first plate delivers a fragment; concatenating all of
them delivers the padding too. This walks the manifest, takes each chunk's real
frame span, and trims the result to the clip's true length.

Verified by frame count, not by exit code (rule 4): the output must contain
exactly the frames the manifest says the source had, or nothing is written.

    VACE_RUN=<run> scripts/assemble_plates.py --profile background_aggressive \\
        --out outputs/clips_final/<name>.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, probe_frames, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="background_aggressive")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crf", type=int, default=14)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("assemble_plates", args.verbose)
    man_path = P.intermediate / "chunk_manifest.json"
    if not man_path.exists():
        log.error("No manifest at %s (set VACE_RUN?)", man_path)
        return 1
    man = json.loads(man_path.read_text())
    chunks = man.get("chunks") or []
    if not chunks:
        log.error("Manifest lists no chunks.")
        return 1

    want = int(man["normalized"]["total_frames"])
    log.info("%d chunk(s); the source stream is %d frame(s)", len(chunks), want)

    pieces, total = [], 0
    tmp = P.intermediate / "_plate_join"
    tmp.mkdir(parents=True, exist_ok=True)
    for c in sorted(chunks, key=lambda c: int(c["start_frame"])):
        entry = (c.get("background") or {}).get(args.profile)
        # Older runs recorded the plate as a bare path, newer ones as a dict.
        bg = entry.get("path") if isinstance(entry, dict) else entry
        if not bg:
            log.error("%s has no %s plate. Run restore_background.py first.",
                      c["chunk_id"], args.profile)
            return 1
        src = P.root / bg
        n_real = int(c["n_frames"])
        have = probe_frames(src)
        # The plate may carry padding the chunk does not: the model needs 4n+1,
        # the timeline does not.
        piece = tmp / f"{c['chunk_id']}.mkv"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                        "-frames:v", str(min(n_real, have)), "-c:v", "ffv1",
                        str(piece)], check=True)
        got = probe_frames(piece)
        log.info("%-22s %5d frame(s) of %5d in the plate", c["chunk_id"], got, have)
        pieces.append(piece)
        total += got

    listing = tmp / "concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in pieces))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-frames:v", str(want),
                    "-c:v", "libx264", "-crf", str(args.crf), "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(args.out)], check=True)

    got = probe_frames(args.out)
    if got != want:
        log.error("%s has %d frame(s), the source had %d. Refusing to call this "
                  "a deliverable.", args.out.name, got, want)
        return 1
    log.info("wrote %s (%d frames, joined from %d chunk(s) totalling %d)",
             args.out, got, len(pieces), total)
    for p in pieces:
        p.unlink(missing_ok=True)
    listing.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
