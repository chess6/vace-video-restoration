# VACE reference-conditioned video restoration

Reference-conditioned generative restoration of long-form, low-resolution video
using **Wan2.1-VACE-1.3B** inside **ComfyUI**, driven by depth control, a tracked
full-subject mask and a reference sheet built from higher-quality stills of the
main subject.

This is deliberately **not** conventional super-resolution. The premise is that
detail-hallucinating upscalers applied directly to a degraded source cannot
recover match, because the information is not there to recover. Real-ESRGAN
appears here only as an optional *final* resize of already-restored output, and
only after being compared against a plain Lanczos resize.

Nothing in this project ever opens a viewer, player or image window. Every visual
artefact is written to disk for you to open yourself.

---

## How it works

Two models, each doing the job it is actually good at: **SeedVR2 3B** restores the
whole frame, and **VACE 1.3B** replaces only the tracked main subject on top of
that, using the reference sheet.

```
Original video
├── scene cuts, timing, depth, masks and subject motion   <- ALWAYS from the original
└── SeedVR2 full-frame restoration  (background_conservative | background_aggressive)
      └── VACE replaces only the masked main subject using reference images
            └── blend / composite   (in_vace | composite)
                  └── exact-timeline assembly with the original audio
```

The rule that keeps this honest: **nothing structural is ever derived from
SeedVR2 output.** Scene cuts, chunk timing, depth, masks and tracking all come
from the original normalized stream. SeedVR2 supplies pixels and nothing else, so
switching background profile - or turning the stage off - cannot move a chunk
boundary, shift the tracked subject or change what the model is told to follow.

```
inputs/source/your.mp4  (never modified)
        │
        ├─ inspect_source.py ........ ffprobe: codec, fps, VFR, SAR, colour, audio
        │
        └─ preprocess_source.py ..... CFR working copy at native size
                                      VACE stream: scale+pad to W×H, resample to 16 fps
                                      PySceneDetect cuts → shots
                                      shots → 81-frame chunks with overlap
                                      → intermediate/chunk_manifest.json
                                              │
inputs/references/*.jpg ──► prepare_references.py
        (never modified)      EXIF fix, dedupe, reject corrupt/small,
                              match-cluster to drop other candidates,
                              pick ≤3 complementary views,
                              tile into ONE clean sheet
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                        make_depth.py   track_subject.py   (manifest)
                        Depth Anything  Grounding DINO →
                        V2 per frame    anchor embed + ReID  →
                                        SAM 2.1 full-subject track
                              │               │
                              └───────┬───────┘
                                      ▼
                              run_chunks.py  ──► ComfyUI ──► VACE 1.3B
                                      │
                                      ▼
                              assemble.py: overlap dedupe, seam dissolve,
                                           audio remux from the ORIGINAL,
                                           A/V sync verification
                                      ▼
                              outputs/final/restored_master_480p.mp4
```

### The one structural detail that matters most

`WanVaceToVideo` in the installed ComfyUI computes:

```python
inactive = control_video * (1 - mask)   # the PRESERVED region
reactive = control_video * mask         # the REGENERATED region
```

Two consequences that the whole design follows from:

1. **White = regenerate, black = preserve.** Not assumed — proven end to end by
   `scripts/verify_mask_polarity.py`, which generates with a half-white mask and
   measures that the white side changes ~19× more than the black side.

2. **The control video must not be pure depth.** Because the preserved region's
   pixels come out of `control_video` itself, feeding depth everywhere would
   replace your background with a depth map. The workflow therefore composites
   **original RGB outside the mask, depth inside it** using the native
   `ImageCompositeMasked` node.

A third constraint, also read from the node rather than guessed:
`reference_image` is indexed `[:1]`, so VACE consumes **exactly one** reference
image. That is why `prepare_references.py` tiles your best views into a single
sheet instead of passing a list.

Valid chunk lengths are `4n+1` (`length` has `step=4` from `min=1`), and width and
height must be multiples of 16. `scripts/common.py` enforces both and refuses to
run on a config that violates them.

---

## Quick start (once you have your files)

```bash
cd vace-video-restoration

# 0. put your media in place
cp /path/to/your_video.mp4        inputs/source/
cp /path/to/reference_stills/*    inputs/references/

# 1. sanity-check the machine
scripts/verify_env.sh

# 2. start ComfyUI (background; never opens a browser)
scripts/start_comfyui.sh --daemon

# 3. look at the source
venv/bin/python scripts/inspect_source.py --exact-frames

# 4. normalize, detect cuts, build the chunk manifest
#    --auto-aspect picks dimensions matching your true aspect ratio
venv/bin/python scripts/preprocess_source.py --auto-aspect

# 5. build the reference sheet, then INSPECT it yourself
venv/bin/python scripts/prepare_references.py
#    -> intermediate/reference_sheets/reference_sheet.png
#    -> intermediate/reference_sheets/contact_sheet.png

# 6. depth control
venv/bin/python scripts/make_depth.py

# 7. automatic match-aware subject tracking
venv/bin/python scripts/track_subject.py
#    -> intermediate/masks/review/*.png   (check these)
#    -> reports/tracking_report.json      (confidence per shot)

# 8. pick and run a representative pilot
venv/bin/python scripts/extract_pilot.py --seconds 8
venv/bin/python scripts/run_chunks.py --pilot

# 9. controlled comparison: no-reference ablation + a second seed
#    --no-reference drops ONLY the reference sheet and keeps the tracked mask,
#    so the ablation changes exactly one variable against the baseline.
#    Each variant is recorded separately in the manifest under runs.<tag>, so
#    these neither skip as "already done" nor overwrite the baseline's output.
venv/bin/python scripts/run_chunks.py --pilot --no-reference --tag noref
venv/bin/python scripts/run_chunks.py --pilot --seed 987654 --tag seedB

# 10. build comparison artefacts and assemble the pilot
venv/bin/python scripts/make_comparisons.py
venv/bin/python scripts/assemble.py --pilot

# 11. judge the result, write it down, and only then consider the full run
#     reports/pilot_results.md
```

**The pipeline stops here by design.** Nothing processes the whole video until
you explicitly run:

```bash
scripts/run_full.sh                      # prints estimates, then refuses
scripts/run_full.sh --confirm-full-run   # actually starts
```

---

## Command reference

| Task | Command |
|---|---|
| Recreate everything from a fresh clone | `scripts/bootstrap.sh` (`--check` to only report) |
| Verify environment | `scripts/verify_env.sh` |
| Chunking + assembly tests (no GPU) | `venv/bin/python scripts/test_chunking.py --with-video` |
| Sample a short clip from each source | `venv/bin/python scripts/sample_clips.py --seconds 5` |
| Start / stop / check ComfyUI | `scripts/start_comfyui.sh --daemon` / `--stop` / `--status` |
| Re-download + checksum models | `scripts/download_models.sh` |
| Install auxiliary models | `scripts/download_aux_models.sh` |
| Inspect the source | `venv/bin/python scripts/inspect_source.py` |
| Normalize + chunk | `venv/bin/python scripts/preprocess_source.py --auto-aspect` |
| Prepare references | `venv/bin/python scripts/prepare_references.py` |
| Generate depth | `venv/bin/python scripts/make_depth.py` |
| Download SeedVR2 3B (fp8) | `scripts/download_seedvr2.sh` |
| Restore backgrounds | `venv/bin/python scripts/restore_background.py --all-profiles --pilot` |
| VACE over a restored plate | `venv/bin/python scripts/run_chunks.py --pilot --background background_conservative` |
| Composite subject onto a plate | `venv/bin/python scripts/composite_subject.py --chunk <id> --subject <mp4> --background <mkv> --out <mkv>` |
| Build the 8-way pilot comparison | `venv/bin/python scripts/pilot_compare.py` |
| Measure the variants | `venv/bin/python scripts/evaluate_pilot.py` |
| **Zip everything needing review** | `venv/bin/python scripts/make_review_bundle.py` |
| Track the subject | `venv/bin/python scripts/track_subject.py` |
| Review sheet with coordinates burned in | `venv/bin/python scripts/make_labeled_sheet.py <video> <mask|none> <out.png> [tiles]` |
| Re-seed one shot | `venv/bin/python scripts/track_subject.py --shot shot0003 --init-box x0,y0,x1,y1 --force` |
| Prove mask polarity | `venv/bin/python scripts/verify_mask_polarity.py` |
| Rebuild workflows | `venv/bin/python scripts/build_workflows.py` |
| Smoke test | `venv/bin/python scripts/smoke_test.py` |
| Extract a pilot | `venv/bin/python scripts/extract_pilot.py --seconds 8` |
| Run the pilot | `venv/bin/python scripts/run_chunks.py --pilot` |
| Build comparisons | `venv/bin/python scripts/make_comparisons.py` |
| Assemble | `venv/bin/python scripts/assemble.py [--pilot] [--deliver 720p]` |
| Compare upscalers | `venv/bin/python scripts/compare_upscalers.py --target 720p` |
| Benchmark + estimate | `venv/bin/python scripts/benchmark.py` |
| Full run (guarded) | `scripts/run_full.sh --confirm-full-run` |
| Resume failed chunks only | `venv/bin/python scripts/run_chunks.py --resume-failed` |
| End-to-end self-test | `scripts/selftest.sh` |
| Record pinned versions | `scripts/record_versions.sh` |
| Enforce the no-display rule | `scripts/check_no_display.sh` |
| Verify nothing private is committed | `scripts/check_repo_clean.sh` |

### Repository hygiene

Two guards enforce the rules in `CLAUDE.md` and both run automatically on
`git push` via `.git/hooks/pre-push`:

- **`check_no_display.sh`** — fails if any code calls a viewer or player, if the
  ComfyUI launcher could open a browser, if the GUI build of OpenCV is installed,
  or if matplotlib has an interactive backend.
- **`check_repo_clean.sh`** — fails if any media, archive or compiled Python is
  tracked, if anything but the two placeholder docs is tracked under `inputs/`,
  or if any tracked file's *contents* mention a filename that currently exists in
  `inputs/`. The forbidden-word list is derived at run time from whatever is
  actually in `inputs/` — including entries inside archives, listed without
  extracting them — so it keeps working for future material without being told
  anything about it.

Everything under `inputs/` is ignored wholesale, along with every common media
and archive extension project-wide, and every report derived from source media
(`source_info.*`, `tracking_report.json`, `assembly*.json`).

Every long-running script logs to `logs/<name>.log`, records per-chunk status in
`intermediate/chunk_manifest.json`, and is resumable by re-running it.

---

## Correcting a bad shot

`track_subject.py` is automatic. It only asks for help when a shot is genuinely
ambiguous, and it flags those in `reports/tracking_report.json` under
`needs_user`. To fix one shot without touching any other:

```bash
# look at intermediate/masks/review/shot0003_review.png first
venv/bin/python scripts/track_subject.py --shot shot0003 --force \
    --init-box 120,40,300,470

# or with clicks-as-coordinates: x,y,+ for the subject, x,y,- for not-subject
venv/bin/python scripts/track_subject.py --shot shot0003 --force \
    --init-points 210,150,+ 260,300,+ 40,40,-

# or hand-paint a first-frame mask (white = subject)
venv/bin/python scripts/track_subject.py --shot shot0003 --force \
    --init-mask inputs/subject_seeds/shot0003.png
```

Then regenerate only that shot's chunks:

```bash
venv/bin/python scripts/run_chunks.py --only shot0003_c000 shot0003_c001 --redo
```

---

## Layout

```
ComfyUI/                    official ComfyUI (pinned commit, see reports/versions.md)
venv/                       isolated Python 3.12 environment
configs/local_1p3b.yaml     the local 480p validation profile
configs/cloud_14b.yaml      cloud 720p 14B profile (nothing downloaded locally)
inputs/source/              YOUR video          - read-only to this pipeline
inputs/references/          YOUR stills         - read-only to this pipeline
inputs/subject_seeds/       optional hand-painted first-frame masks
intermediate/               working streams, depth, masks, chunk manifest
workflows/                  ComfyUI graphs, UI format + _api.json format
scripts/                    everything above
outputs/pilots|comparisons|restored_480p|final
reports/                    environment, source info, tracking, pilot results
logs/
```

## Measured performance on this machine

Benchmarked with a real 832×480 × 81-frame generation (`reports/benchmark.json`,
`reports/runtime_estimates.md`):

| Quantity | Measured |
|---|---|
| 81-frame chunk, 25 steps | **16 m 19 s** |
| Per generated frame | **12.10 s** |
| Peak VRAM | **11 854 MiB** of 12 288 MiB |
| Fixed cost per chunk | ~52 s (VAE encode/decode + conditioning) |
| Marginal cost per step | ~37.1 s |

Extrapolated to a **30-minute reference workload** at 16 fps
(395 chunks, 31 995 generated frames) — substitute your own duration:

| Steps | Reference workload |
|---:|---:|
| 15 | 67 h |
| 20 | 87 h |
| **25 (baseline)** | **107 h** |

**At that scale the job is ~4.5 days of continuous compute on this GPU.** The GPU sits at
100 % utilisation, so this is compute-bound and not fixable by memory tuning: at
this shape the Wan latent is ~33 000 tokens after patchifying and attention is
quadratic in that, with CFG doubling it again.

This is the strongest practical argument for validating locally and running
production from `configs/cloud_14b.yaml`. See `reports/runtime_estimates.md` for
the full analysis and the options.

## Tuning

`configs/local_1p3b.yaml` holds seed, steps, cfg, sampler, dimensions, frame
count, overlap, mask feather/grow, prompts and output fps. After changing
dimensions, frame count or prompts, rebuild the workflows:

```bash
venv/bin/python scripts/build_workflows.py
```

Acceleration (CausVid, TeaCache, aggressive quantization) is deliberately absent
from the baseline. Add it only after the baseline output is judged correct, and
compare against the baseline pilot when you do.

## Moving to the cloud for 720p

`configs/cloud_14b.yaml` targets Wan2.1-VACE-**14B** at 1280×720. The 14B weights
are **not** downloaded here and will not fit this GPU. The same chunk manifest,
reference sheet, prompts, masks and depth videos carry over; on the rented box,
re-run `preprocess_source.py`, `make_depth.py` and `track_subject.py` with
`--config configs/cloud_14b.yaml` to regenerate the control streams at 720p from
the same normalized source.
