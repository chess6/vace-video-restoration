# inputs/source/

Put **one** source video file here.

Recognised extensions: `.mp4 .mkv .mov .avi .webm .m4v .mpg .mpeg .ts .wmv .flv`

If your file has a different extension, either rename a copy or pass an explicit
path:

```bash
venv/bin/python scripts/inspect_source.py    --source /full/path/to/video.xyz
venv/bin/python scripts/preprocess_source.py --source /full/path/to/video.xyz
```

## This file is never modified

The pipeline opens your original **read-only**. It is never re-encoded, moved,
trimmed or overwritten. All work happens on copies:

- `intermediate/normalized/source_cfr.mp4` — constant-frame-rate working copy at
  the original resolution
- `intermediate/normalized/work_<W>x<H>_16fps.mp4` — the stream VACE actually
  sees: scaled and padded (never stretched) to a multiple of 16, resampled to the
  model's 16 fps

Audio is remuxed at assembly time from **this original file**, not from any
intermediate, so audio never inherits generation loss or timing drift.

## Then run

```bash
venv/bin/python scripts/inspect_source.py --exact-frames
venv/bin/python scripts/preprocess_source.py --auto-aspect
```

`--auto-aspect` picks generation dimensions matching your true display aspect
ratio instead of padding to 16:9 and wasting capacity on the bars down the sides.
It reports the dimensions it chose; both axes are always multiples of 16.

Read `reports/source_info.md` afterwards — it flags variable frame rate,
interlacing, non-square pixels and rotation metadata, all of which change what
the preprocessing needs to do.
