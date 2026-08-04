# Project state

Durable context for this repository: the things that are expensive to
rediscover, and the standing instructions that outlive any one session.

**Read this before starting work.** See CLAUDE.md rule 0.

**Size limit: 200 lines / 12 KB**, enforced by `scripts/check_repo_clean.sh`.
It is a working memory, not a changelog. When it is full, delete the oldest
resolved entries — git history already has them. Anything that is now enforced
by a test or a guard script belongs in that test, not here.

**Rule 2a applies to this file.** It is tracked, so it must never name an input
file, a person, an interval, a duration or a resolution of the user's media.
Concrete values live in the untracked run manifest. Refer to things by role
("the pilot interval", "the chosen reference cluster"), never by name.

---

## What this pipeline is

Reference-conditioned generative video restoration. Wan2.1-VACE-1.3B plus
SeedVR2, driven through a pinned ComfyUI, on a 12 GB RTX 3060.

Stage order, and who owns what:

1. **SeedVR2** restores the **full frame** — the background plate. Reduced
   precision, VAE tiling, temporal batching to fit VRAM. Cached by interval and
   config hash.
2. **Controls** (depth, pose, masks) are derived from the **original** footage
   only, never from the restored plate.
3. **VACE** regenerates the subject over that preserved plate.
4. **Compositing** layers: restored plate, then generated figure, then preserved
   foreground occluders from the original.

## Authority split — the rule that keeps being violated

External reference photographs show a **different outfit**. They may condition
**identity only**: face, hair, exposed skin, body appearance. Their clothing and
accessories are segmented out and replaced with neutral grey before the panel is
drawn. A faint ghost of the wrong jacket is still a jacket to a generative model.

The garment in the **source interval being restored** is the sole ground truth:
class, silhouette, boundaries, colour, pattern, accessories, folds, motion. It is
low resolution and that is fine — preserve its low-frequency colour and structure,
generate only the missing high-frequency texture.

Consequences that are easy to get wrong:

- `outfit_authority` is the constant `"source_frames"`. Never conditional.
- Garment colour distance to the externals is a **diagnostic**, never a switch,
  and never a selection criterion.
- Which photographs become panels is decided by **identity evidence alone**:
  leave-one-out face agreement, face pixel resolution, viewing-angle difference.
- Appearance clustering still runs, but no longer confines the choice. Its job
  was to stop two outfits being combined in one image; with clothing segmented
  out of every external panel there is no outfit left to conflict, so restricting
  panels to one cluster would only discard viewing angles.
- Never describe a good result as a "reference-pack garment fix". Report three
  separate effects: identity improvement from the externals; garment fidelity
  from source-derived conditioning; any effect from chroma correction.

Enforced by `scripts/test_reference_pack.py`.

## Model facts, read from the installed source

See CLAUDE.md rule 6 for the full list and how they were proven. The ones that
bite most often:

- `length` must be **4n+1**; width/height multiples of **16**.
- **White = regenerate, black = preserve** (`reactive = control_video * mask`).
- `reference_image` is indexed `[:1]` — exactly **one** image, hence the
  composited sheet.
- VACE **centre-crops** the reference image, so build the sheet at the
  manifest's dimensions, not the config's.
- Encode control streams into ComfyUI's input dir with `-qp 0`. Lossy
  re-encoding rounds mask edges outward and regenerates pixels outside the
  tracked boundary.
- Use `VAEEncodeTiled`. Untiled encode peaks near 12 GB and OOMs.
- SeedVR2 is **native** to the pinned ComfyUI (`comfy_extras/nodes_seedvr.py`).
  No custom node pack. `frames_per_chunk` must be 4n+1; `temporal_overlap` is in
  **latent** frames.
- Dynamic combos serialise as `<parent>.<child>` under the input name
  (`comfy_api/latest/_io.py::finalize_prefix`).

## Provenance and staleness

`generation_key` content-hashes everything that determines a chunk's pixels:
reference pack, masks, occluders, controls, ROI transform, prompts, seed, model
and sampler settings, background plate. A result whose key no longer matches is
**not** a result — it must regenerate before anything is compared.

The key is captured **before** staging and re-read **after** inference. If they
disagree, the inputs changed during the ~16-minute run and the output settles as
`stale`: the file is kept for inspection but never counted as current. Recording
the post-run key would mark stale pixels current forever.

The manifest carries a per-object revision counter (`_loaded_rev`). A stage that
loaded an old copy cannot silently overwrite a newer one.

Run namespacing: `VACE_RUN=<name>` → `runs/<name>/{intermediate,outputs,reports,logs}`.

## Measurement discipline

Past mistakes worth not repeating — each cost a wrong conclusion:

- **Circular measurement.** Occluders were defined as `people & ~dilate(subject)`
  and then tested for overlap with `dilate(subject)`. Empty by construction. The
  reported "0.0000% OK" was a tautology, not a result. Define occluders against
  the subject at true extent; apply dilation only *after* the set is fixed.
- **Self-comparison as evidence.** The identity bank is built from the reference
  photographs, so scoring a photograph against it returns 1.000 for anything
  already in it. Score leave-one-out, against the others.
- **Container noise read as signal.** An exact-equality background metric
  measured encoder noise and inverted the ranking. Use a tolerance.
- **Piping a build through `head`/`tail`.** SIGPIPE killed a generator mid-write
  and left a stale graph; `tail` masked a non-zero exit. Do not pipe long-running
  builds through either.
- Prove masks are produced **independently** before reporting an overlap between
  them, and report the frames where occlusion actually occurs.

## Standing instructions from the user

- Cut short sample clips rather than processing whole videos. Nothing runs the
  full video without explicit confirmation (rule 5).
- Every time work needs the user's eyes, hand over **one zip** from
  `scripts/make_review_bundle.py` (rule 1).
- Prepare the 14B cloud path, but **do not download 14B locally**.
- State separate effects, separate runtimes, separate VRAM and disk figures.
  Never a single blended claim.

## Open work

Tracked in the task list; kept here only as orientation.

- Clothing fidelity beyond mean colour: garment class, boundaries, patterns,
  accessories. Chroma correction only if measurable drift remains, with the
  uncorrected output kept as a separate comparison.
- Whole-body pose control: generate a pose-controlled variant and compare it
  against the depth-controlled one under identical references, background,
  prompt and seed.
- Compositing layer 3 should take **plate** pixels by default, with an opaque
  core and a narrow, one-sided, temporally stable boundary — not a hard edge.
- Multi-chunk assembly is exercised on a synthetic longer interval, separately
  from the pilot, which is padded to a single inference.
