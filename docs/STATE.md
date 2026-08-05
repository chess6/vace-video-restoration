# Project state

Durable context: what is expensive to rediscover, and the standing instructions
that outlive a session. **Read before starting work** (CLAUDE.md rule 0).

**Size limit: 200 lines / 12 KB** (`check_repo_clean.sh`). A working memory, not
a changelog: when full, delete resolved entries and let anything a guard now
enforces live in that guard.

**Rule 2a applies here.** Tracked, so never name an input file, a person, or an
interval/duration/resolution of the user's media. Refer to things by role.

---

## What this pipeline is

Reference-conditioned generative video restoration. Wan2.1-VACE plus SeedVR2,
through a pinned ComfyUI. Validated locally on 12 GB; run on rented GPUs. Stage
order, and who owns what:

1. **SeedVR2** restores the full frame — the plate, cached by interval + config
   hash. **This stage produces essentially all of the measured quality.**
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
- Tracking, pack selection, LoRA crops and evaluation share this one bank.

**Never use clothing-sensitive or whole-image embeddings for target identity.**
A CLIP crop embedding responds to clothing and framing and *dominated* at low
resolution — two photographs of another person carried the selection. No
resolvable face scores **zero** and flags the shot.

The approval gate binds to the mask's content hash, so an approval never
survives a re-track or a change of machine. Run-specific exclusions live in an
untracked `intermediate/reference_exclusions.txt`.

## Authority split — the rule that keeps being violated

External reference photographs show a **different outfit**. They condition
**identity only**: face, hair, exposed skin, body appearance. Their clothing is
segmented out and replaced with neutral grey before the panel is drawn — a faint
ghost of the wrong jacket is still a jacket. The garment in the **source
interval** is the sole ground truth: class, silhouette, boundaries, colour,
pattern, accessories, folds, motion. Low resolution is fine — preserve its
low-frequency colour and structure, generate only high-frequency texture.
Never call a good result a "reference-pack garment fix": report identity gain
from the externals, garment fidelity from source conditioning, and any chroma
effect as three separate effects.

Consequences that are easy to get wrong:

- `outfit_authority` is the constant `"source_frames"`. Garment colour distance
  to the externals is a **diagnostic**, never a switch or a selection criterion.
- Which photographs become panels is decided by **identity evidence alone**:
  consensus agreement, face pixel resolution, and head yaw from landmarks.
- `IDENTITY_ONLY` is **head only** (`hair`, `face`). Arms and legs are apparel:
  visible limb is sleeve and hemline coverage, so a reference with bare arms
  instructs the model to remove the source's sleeves.
- If the **source face is covered**, external face conditioning is disabled —
  only hair is used. Fail closed: with no covering analysis, assume covered.
- The **protected-apparel submask** is what VACE regenerates: confidently
  exposed head regions only, eroded from every garment boundary, required to
  persist across frames. Everything else is black and comes from the plate.
- **Two builders exist and the wrong one is easy to run.** `prepare_references.py`
  (Phase 5) tiles whole photographs — background, wardrobe, watermarks. Only
  `make_reference_pack.py` (5b) applies the split above. `run_chunks.py` prints
  `Reference conditioning: global` or `pack`; **that line is the check.**

Enforced by `scripts/test_reference_pack.py`.

## What is settled at 1.3B — do not re-run these

Numbers in `reports/pilot_results.md` and `lora_results.md`. Score identity
through `identity.resolve_targets`, never `evaluate_pilot.py`'s own bank.

- **The plate beat every VACE variant** (all under the 0.35 identity threshold);
  **3B aggressive is the lever and the plate to ship**: sharpness 15.2 source →
  **65.9** (3B) vs 50.3 (7B) vs 22.0 (7B at denoise 0.75 + lab, which kept
  neither the detail nor the quiet — that experiment is closed). A `background`
  key names the **profile, not the model**: one key can hold either's pixels.
- **In the pixels VACE regenerates it is at or below a plain Lanczos upscale**:
  on the 3B plate, baseline 9.7, plate 14.2 (+47%), VACE 8.0 / 8.4 with the
  LoRA. It discards the plate's work in the one region a viewer looks at.
  Measure **in-mask, never the bounding box** — the box is 20x the submask and
  reports +120%, the plate read back as VACE's (`scripts/compare_720p.py`).
- **Sharpness and chroma point opposite ways**: the plate is sharpest and
  noisiest (+36%), the LoRA arm cleanest (−12%). Report both or the metric
  contradicts the eye — it did, for a session.
- **A subject LoRA learns the face and barely moves the pipeline.** 0.023 →
  **0.5167** trained (ceiling 0.7454), yet matched arms give 0.1682 vs 0.1612,
  0.1263 at double strength, on the plate 0.2015 vs 0.1769: **the protected path
  has no room for an identity prior**. It is still the better VACE arm on the 3B
  plate (in-mask 8.4 vs 8.0, chroma −12% vs −1%). Score only against the
  held-out split; the trigger token is load-bearing; musubi's merge trap is in
  `prepare_musubi_dit.py`.
- Composite in `gbrp`, not `yuv420p`; after `--protected` pass `--mask` the
  submask, else the garment arrives VAE-degraded.

## Model facts, read from the installed source

CLAUDE.md rule 6 carries the constraints themselves (4n+1, multiples of 16, mask
polarity, the single reference image) and how each was proven. What it omits:

- VACE **centre-crops** the reference image; build the sheet at the manifest's
  dimensions.
- Encode control streams into ComfyUI's input dir with `-qp 0`; lossy re-encoding
  rounds mask edges outward.
- Use `VAEEncodeTiled`. Untiled encode peaks near 12 GB and OOMs.
- SeedVR2 is **native** to the pinned ComfyUI (`comfy_extras/nodes_seedvr.py`),
  no node pack. `frames_per_chunk` is 4n+1; `temporal_overlap` is in **latent**
  frames.
- Dynamic combos serialise as `<parent>.<child>` under the input name
  (`comfy_api/latest/_io.py::finalize_prefix`).

## Provenance and staleness

**Two keys, not one.** `vace_key` hashes what reaches the sampler (reference
sheet, plate, control, mask, ROI streams, prompts, seed, model incl. LoRA,
sampler); `composite_key` what reaches the compositor. As one key, widening an
alpha ramp — seconds of CPU — marked a finished generation stale and demanded
~18 GPU minutes to reproduce identical pixels.

Hash file **contents**, never a config or geometry key: a rebuilt plate keeps
its filename and geometry while its pixels change completely. A result whose key
no longer matches is **not** a result — though a video *container* rehashes even
when its decoded pixels do not, so a re-run can invalidate an intact approval.

The key is captured **before** staging and re-read **after** inference; if they
disagree the output settles as `stale`, kept for inspection but never current.
Recording the post-run key would mark stale pixels current forever.

The manifest carries a per-object revision counter (`_loaded_rev`). A stage that
loaded an old copy cannot silently overwrite a newer one. Run namespacing:
`VACE_RUN=<name>` → `runs/<name>/{intermediate,outputs,reports,logs}`.

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
  angle (use face landmarks); proximity is not occlusion (verify depth order); a
  rise in high-frequency energy is not restored texture — ringing raises it too.
  Name the measurement after what it measures.
- **A detector's training bias read as absence.** Grounding DINO is trained on
  upright people; a reclining subject scored ≤0.27 against an upright bystander's
  0.33–0.36, offering only the wrong person. "Found somebody" is not "found the
  subject": retry lower when no candidate has identity evidence, and never score
  a subject by height.
- **"It ran" is not "it did something."** Eight LoRA checkpoints and the
  baseline scored identical to four decimals: the merge had silently no-oped.
- **A static mask is architecture, not a subject.** A bbox that barely moves has
  locked onto scenery. Correlate mask occupancy with per-pixel temporal variance:
  the doorway track scored r=+0.03 against motion, a correct one +0.58.

## Standing instructions from the user

- Every time work needs the user's eyes, hand over **one zip** from
  `scripts/make_review_bundle.py` (rule 1).
- Prepare the 14B cloud path, but **do not download 14B locally**.
- State separate effects, separate runtimes, separate VRAM and disk figures.
  Never a single blended claim.
- The macOS laptop has no CUDA and the lockfile is CUDA-only: **no local
  pipeline there**, so all GPU work runs on the rented box.
- **Take the GPU box down whenever it is not actively needed**, including while
  the user reviews, without asking. Stop, never destroy: Vast's `/workspace` is
  not a volume, and a stopped box cannot restart while another tenant holds the
  GPU — RunPod's **network volume** survives that and reattaches to a new pod.

## Cloud and open work

`docs/CLOUD_RUNBOOK.md`: connection quirks, transfer allowlist, gates, teardown.
Protected-regenerable fraction 4.42% at 720p — that smallness is the measured
reason VACE cannot be steered here. `scripts/state_bundle.sh` carries the
irreplaceable half off a box; run it FIRST, before any work.

- `mask.grow=4` puts 4.51% of the dilated mask onto another person: reduce it or
  rely on the occluder layer.
- Untested: **VACE-14B**. Pose-vs-depth control and any further LoRA work matter
  only if VACE earns its place at a larger size — at 1.3B it has not.
