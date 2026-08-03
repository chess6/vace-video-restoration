# Actions that genuinely require you

Everything that could be installed, configured, downloaded, generated and
verified without your media has been done. What follows are only the things that
cannot be done on your behalf.

---

## 1. Provide the source video  — **blocking**

```bash
cp /path/to/your_video.mp4 inputs/source/
```

One video file. It is opened read-only; the pipeline never modifies, moves or
re-encodes your original. Everything downstream works on copies in
`intermediate/`.

If your file is not one of `.mp4 .mkv .mov .avi .webm .m4v .mpg .ts`, either
rename it or pass `--source /full/path` to `inspect_source.py` and
`preprocess_source.py`.

---

## 2. Provide the reference images  — **blocking**

```bash
cp /path/to/photos/*.jpg inputs/references/
```

PNG, JPEG or WebP. What actually helps, in priority order:

1. **A full-body shot** — the brief says non-facial identity matters as much as
   the face, and the full body is what carries clothing, silhouette and
   proportions. This is the single most valuable reference.
2. **A clear face shot** — as sharp and front-on as you have.
3. **An alternate angle** — side or back view, so the model has something to work
   from when the figure turns.

Practical notes:

- Minimum 256 px on the short side. Smaller images are rejected automatically.
- Photos containing **other people** are handled: the script clusters faces and
  drops images whose face does not match the dominant identity. But a photo where
  the main figure is small and someone else is large may still pick the wrong
  person, so prefer photos where your subject is the clear subject.
- Near-duplicates are detected and the sharper copy kept; you do not need to
  curate them yourself.
- Only **three** views end up on the sheet, because VACE consumes a single
  reference image. More photos are still useful: they widen the pool the selector
  chooses from and strengthen the identity match used for tracking.

---

## 3. Inspect two artefacts before the pilot  — **your judgement needed**

After running the reference and tracking stages, open these files yourself.
Nothing is displayed automatically.

| File | What to check |
|---|---|
| `intermediate/reference_sheets/reference_sheet.png` | Is this your subject, in useful views, with no other people, no captions and no wasted space? |
| `intermediate/reference_sheets/contact_sheet.png` | Did anything good get rejected, or anything bad get kept? |
| `intermediate/masks/review/*.png` | Does the red region cover the **whole figure** — hair, hands, feet, bag — and not the background? |
| `reports/tracking_report.json` | Any shots listed under `needs_user`? |

---

## 4. Re-seed any shot flagged `needs_user`  — **only if flagged**

Tracking is automatic. It asks for help only when a shot is genuinely ambiguous
(nobody detected, or two people scoring alike). If `reports/tracking_report.json`
lists shots under `needs_user`, look at that shot's review sheet and re-seed just
that shot:

```bash
venv/bin/python scripts/track_subject.py --shot shot0003 --force \
    --init-box x0,y0,x1,y1
```

Coordinates are in working-stream pixels (the size reported in the manifest under
`normalized`). No other shot is touched.

---

## 5. Judge the pilot  — **your judgement needed**

Command exit codes prove the pipeline ran. They prove nothing about whether the
restoration is good. After the pilot, open:

- `outputs/comparisons/*_side_by_side.mp4`
- `outputs/comparisons/*_frame_grid.png`
- `outputs/final/pilot_master.mp4`

and fill in `reports/pilot_results.md`, which has a scoring template ready.

The comparison includes the reference-conditioned output, a no-reference
ablation and a second seed, so you can see what the reference is actually
contributing.

---

## 6. Approve the full run  — **explicit approval required**

Nothing processes the full 30 minutes until you run:

```bash
scripts/run_full.sh --confirm-full-run
```

Without the flag it prints the measured time and disk estimates and exits.
**Read those estimates first** — see `reports/benchmark.json`. On this GPU the
full job is measured in days, not hours, which is a genuine argument for doing
production on the cloud 14B profile instead.

---

## Optional

**A neural upscaler for the final resize.** Nothing is downloaded, because
Lanczos may well win and the brief says not to assume otherwise. To evaluate one:

```bash
# put e.g. RealESRGAN_x2plus.pth here
cp RealESRGAN_x2plus.pth ComfyUI/models/upscale_models/
venv/bin/python scripts/compare_upscalers.py --target 720p
```

It runs strictly on **already-restored** output, never on the raw 240p source.

**Free some RAM for long runs.** This machine has 15 GiB total and the desktop
session was using a large share of it at inspection time. Closing Firefox and
Cursor before a multi-hour run is worth several GiB.

---

## No `sudo` is required

Every system dependency (`ffmpeg`, `ffprobe`, `git`, `curl`, `python3.12-venv`,
the NVIDIA driver) was already present. Nothing was installed system-wide,
the NVIDIA driver was not touched, no system CUDA toolkit was installed
(PyTorch ships its own), and your Miniconda environment was not modified.
