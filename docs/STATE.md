# Project state

Durable context: what is expensive to rediscover, and the standing instructions
that outlive a session. **Read before starting work** (CLAUDE.md rule 0).

**Size limit: 200 lines / 12 KB**, enforced by `scripts/check_repo_clean.sh`.
A working memory, not a changelog: when full, delete the oldest resolved
entries. Anything a test or guard now enforces belongs there, not here.

**Rule 2a applies here.** Tracked, so never name an input file, a person, or an
interval/duration/resolution of the user's media. Refer to things by role.

---

## What this pipeline is

Reference-conditioned generative video restoration. Wan2.1-VACE plus SeedVR2,
through a pinned ComfyUI. Validated locally on 12 GB; run on rented GPUs.

Stage order, and who owns what:

1. **SeedVR2** restores the full frame — the plate. Reduced precision, VAE
   tiling, temporal batching, cached by interval + config hash. **This stage
   produces essentially all of the measured quality.**
2. **Controls** (depth, pose, masks) come from the **original** only, never the
   plate.
3. **VACE** regenerates the subject over that preserved plate.
4. **Compositing**: plate, generated figure, then preserved foreground occluders
   from the original.

## Who the target is

Reference photographs can contain more than one person. Identity is resolved by
`scripts/identity.py` and **nothing else may decide it**:

- Detect **every** face and person box per image; form consensus across face
  **instances**, not one largest face per image.
- The dominant identity is the one supported by the most **distinct images**.
- Tie the target face to the person box containing it — a tracker seeds from it.
- **Reject** an image where two faces cannot be told apart. Never guess.
- Tracking, pack selection and evaluation share this one bank.

**Never use clothing-sensitive or whole-image embeddings for target identity.**
A CLIP crop embedding responds to clothing and framing and *dominated* at low
resolution — two photographs of another person carried the selection. No
resolvable face scores **zero** and flags the shot.

Generation refuses to start unless `subject_status` is settled **and** a human
has approved the track, bound to the mask's content hash — which also means an
approval never survives a re-track or a change of machine. Run-specific
exclusions live in an untracked `intermediate/reference_exclusions.txt`.

## Authority split — the rule that keeps being violated

External reference photographs show a **different outfit**. They may condition
**identity only**: face, hair, exposed skin, body appearance. Their clothing is
segmented out and replaced with neutral grey before the panel is drawn — a faint
ghost of the wrong jacket is still a jacket. The garment in the **source
interval** is the sole ground truth: class, silhouette, boundaries, colour,
pattern, accessories, folds, motion. Low resolution is fine — preserve its
low-frequency colour and structure, generate only high-frequency texture.

Consequences that are easy to get wrong:

- `outfit_authority` is the constant `"source_frames"`. Garment colour distance
  to the externals is a **diagnostic**, never a switch or a selection criterion.
- Which photographs become panels is decided by **identity evidence alone**:
  consensus agreement, face pixel resolution, and head yaw from landmarks.
- `IDENTITY_ONLY` is **head only** (`hair`, `face`). Arms and legs are apparel:
  how much limb is visible is sleeve and hemline coverage. A reference with bare
  arms instructs the model to remove the source's sleeves.
- If the **source face is covered**, external face conditioning is disabled —
  only hair is used. Fail closed: with no covering analysis available, assume
  covered.
- The **protected-apparel submask** is what VACE regenerates: confidently
  exposed head regions only, eroded from every garment boundary, required to
  persist across frames. Everything else is black and comes from the plate.
- Never describe a good result as a "reference-pack garment fix". Report three
  separate effects: identity improvement from the externals; garment fidelity
  from source-derived conditioning; any effect from chroma correction.
- **Two builders exist and the wrong one is easy to run.** `prepare_references.py`
  (Phase 5) tiles whole photographs — background, wardrobe, watermarks. Only
  `make_reference_pack.py` (5b) applies the split above. `run_chunks.py` prints
  `Reference conditioning: global` or `pack`; **that line is the check.**

Enforced by `scripts/test_reference_pack.py`.

## What the 1.3B pilot settled — do not re-run these

Numbers in `reports/pilot_results.md`. Score identity with
`identity.resolve_targets`, never `evaluate_pilot.py`.

- **Reference-conditioned VACE 1.3B did not improve this pilot.** Removing the
  sheet moved the face 6.6/255; swapping whole-photo for identity-only moved it
  1.89 — on this shot and seed the model read a reference's *presence* more than
  its content. Not established beyond this pilot.
- **The plate beat every VACE variant here** (all under the 0.35 identity
  threshold). **Aggressive SeedVR2 is the main lever**: +36.9% vs +20.8%, ~45 s;
  7B adds +57.3% luma detail but ~3x the chroma noise.
- Composite in `gbrp`, not `yuv420p`; after `--protected` pass `--mask` the
  protected submask, else the garment arrives VAE-degraded.
- Re-assert `unet_name` at run time: the graph bakes it in at build time, so a
  config change alone runs the old checkpoint while recording the new one.

## Model facts, read from the installed source

See CLAUDE.md rule 6 for the full list and how they were proven. The ones that
bite most often:

- `length` must be **4n+1**; width/height multiples of **16**.
- **White = regenerate, black = preserve** (`reactive = control_video * mask`).
- `reference_image` is indexed `[:1]` — exactly **one** image, hence the sheet,
  and VACE **centre-crops** it, so build it at the manifest's dimensions.
- Encode control streams into ComfyUI's input dir with `-qp 0`; lossy re-encoding
  rounds mask edges outward.
- Use `VAEEncodeTiled`. Untiled encode peaks near 12 GB and OOMs.
- SeedVR2 is **native** to the pinned ComfyUI (`comfy_extras/nodes_seedvr.py`),
  no custom node pack. `frames_per_chunk` must be 4n+1; `temporal_overlap` is in
  **latent** frames.
- Dynamic combos serialise as `<parent>.<child>` under the input name
  (`comfy_api/latest/_io.py::finalize_prefix`).

## Provenance and staleness

**Two keys, not one.** `vace_key` hashes what reaches the sampler (reference
sheet, plate, control, mask, ROI streams, prompts, seed, model, sampler);
`composite_key` what reaches the compositor. As one key, widening an alpha ramp
— seconds of CPU — marked a finished generation stale and demanded ~18 GPU
minutes to reproduce identical pixels.

Hash file **contents**, never a config or geometry key: a rebuilt plate keeps
its filename and geometry while its pixels change completely. A result whose key
no longer matches is **not** a result — though a video *container* rehashes even
when its decoded pixels do not, so a re-run can invalidate an intact approval.

The key is captured **before** staging and re-read **after** inference; if they
disagree the output settles as `stale`, kept for inspection but never current.
Recording the post-run key would mark stale pixels current forever.

The manifest carries a per-object revision counter (`_loaded_rev`). A stage that
loaded an old copy cannot silently overwrite a newer one.

Run namespacing: `VACE_RUN=<name>` → `runs/<name>/{intermediate,outputs,reports,logs}`.

## Measurement discipline

Past mistakes worth not repeating — each cost a wrong conclusion:

- **Circular measurement.** Never test a set against something derived by
  subtracting it. Occluders defined as `people & ~dilate(subject)` then tested
  against `dilate(subject)` are empty by construction; "0.0000% OK" was a
  tautology.
- **Self-comparison as evidence.** The bank is built from the reference photos,
  so scoring one against it returns 1.000. Verify by consensus: collapse
  near-duplicates, score by median agreement with the rest. A single maximum
  lets two copies of the wrong person vouch for each other.
- **A proxy standing in for the thing itself.** CLIP distance is not a viewing
  angle (use face landmarks); proximity is not occlusion (verify depth order);
  a rise in high-frequency energy is not restored texture — ringing and noise
  raise it too. Name the measurement after what it measures.
- **A detector's training bias read as absence.** Grounding DINO is trained on
  upright people; a reclining subject scored ≤0.27 against an upright
  bystander's 0.33–0.36, so the only candidate offered was the wrong person.
  "Found somebody" is not "found the subject": retry lower whenever no candidate
  has identity evidence. Never score a subject by height; pose makes it wrong.
- **Container noise read as signal.** An exact-equality metric measured encoder
  noise and inverted the ranking. Use a tolerance.
- **A static mask is architecture, not a subject.** A bbox that barely moves has
  locked onto scenery. Correlate mask occupancy with per-pixel temporal variance:
  the doorway track scored r=+0.03 against motion, a correct one +0.58.

## Standing instructions from the user

- Cut short sample clips; nothing runs the full video without confirmation.
- Every time work needs the user's eyes, hand over **one zip** from
  `scripts/make_review_bundle.py` (rule 1).
- Prepare the 14B cloud path, but **do not download 14B locally**.
- State separate effects, separate runtimes, separate VRAM and disk figures.
  Never a single blended claim.
- The macOS laptop has no CUDA and the lockfile is CUDA-only, so there is **no
  local pipeline there**. From that host, all GPU work runs on the rented box.
- **Take the GPU pod down whenever it is not actively needed**, including while
  the user reviews, without asking. `runpodctl stop pod` if it will be used
  again; `remove pod` when done. Either way the volume persists.

## Cloud

`docs/CLOUD_RUNBOOK.md`: connection quirks, transfer allowlist, gates, teardown.
Protected-regenerable fraction: 1.57% at 240p, 4.42% at 720p — small either way.

## Open work

Tracked in the task list; kept here only as orientation.

- `mask.grow=4` puts 4.51% of the dilated subject mask onto another person.
  Reduce it or rely on the occluder layer.
- If the source face is covered, reference identity agreement is **unobservable**,
  not a score to maximise. `covering_removed` is the metric that matters.
- Untested, in order: SeedVR2 **7B**, VACE-14B, a subject LoRA. Pose-vs-depth
  control matters only if VACE earns its place at a larger size.
