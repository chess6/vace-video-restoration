# Project state

Durable context: what is expensive to rediscover, and the standing instructions
that outlive a session. **Read before starting work** (CLAUDE.md rule 0).
**Size limit: 240 lines / 16 KB**, enforced by `check_repo_clean.sh`; evict
resolved entries to make room, never a model or architecture fact still in force.
**Rule 2a applies here** — tracked, so never name an input file, a candidate, or
an interval/duration/resolution of the user's media.

## What this pipeline is

Reference-conditioned generative video restoration. Wan2.1-VACE plus SeedVR2,
through a pinned ComfyUI. Validated locally on 12 GB; run on rented GPUs. Stage
order, and who owns what:

1. **SeedVR2** restores the full frame — the plate, cached by interval + config
   hash. **This stage produces essentially all of the measured quality.**
2. **Controls** (depth, pose, masks) come from the **original**, never the plate.
3. **VACE** regenerates the subject over that preserved plate.
4. **Compositing**: plate, generated subject, then foreground occluders from the
   original.

## Who the target is

A reference can contain more than one candidate. Match is resolved by
`scripts/reference_match.py` and **nothing else may decide it** — tracking, pack
selection, LoRA crops and evaluation share that one bank. Detect **every** anchor
and candidate box per image; consensus across anchor **instances**, never one
largest anchor per image; the dominant match is the one supported by the most
**distinct images**; tie the target anchor to the box containing it, so a tracker
can seed from it; **reject** an image whose anchors cannot be told apart. Its
docstring has the reasoning; `test_reference_pack.py` enforces it.

**Never use attributes-sensitive or whole-image embeddings for target match.** A
CLIP crop embedding responds to attributes and framing and *dominated* at low
resolution — two references of another candidate carried the selection. No
resolvable anchor scores **zero** and flags the shot. The approval gate binds to
the mask's content hash, so it never survives a re-track or a change of machine.

## Authority split — the rule that keeps being violated

External references show a **different appearance**. They condition **match
only**: the anchor region and the exposed regions around it. Their attributes are
segmented out and replaced with neutral grey before the panel is drawn — a faint
ghost of the wrong attribute is still an attribute. The attribute in the **source
interval** is the sole ground truth: class, silhouette, boundaries, colour,
pattern, accessories, folds, motion. Low resolution is fine — preserve its
low-frequency colour and structure, generate only high-frequency texture. Never
call a good result a "reference-pack attribute fix": report match gain, attribute
fidelity and any chroma effect separately.

Consequences that are easy to get wrong:

- `appearance_authority` is the constant `"source_frames"`. Attribute colour
  distance to the externals is a **diagnostic**, never a switch or a criterion.
- Panels are chosen by **match evidence alone**: consensus agreement, anchor
  pixel resolution, anchor orientation from keypoints.
- `MATCH_ONLY` is the **anchor region only**. Peripheral extents are
  attribute-bearing: how much is visible is attribute coverage, so an external
  reference with different coverage instructs the model to alter the source's.
- If the **source anchor is covered**, external anchor conditioning is disabled —
  only the surrounding class is. Fail closed: no covering analysis ⇒ covered.
- The **protected-attribute submask** is what VACE regenerates: confidently
  exposed anchor-region pixels only, eroded from every attribute boundary,
  required to persist across frames. The rest is black and comes from the plate.
- **Two builders exist and the wrong one is easy to run.** `prepare_references.py`
  (5) tiles whole references; only `make_reference_pack.py` (5b) applies the split
  above. `run_chunks.py` prints `Reference conditioning: global` or `pack` — **that
  line is the check** (`test_reference_pack.py`).

## What is settled at 1.3B — do not re-run these

Numbers in `reports/pilot_results.md` and `lora_results.md`. Score match
through `match.resolve_targets`, never `evaluate_pilot.py`'s own bank.

- **The plate beat every VACE variant** (all under the 0.35 match threshold);
  **3B aggressive is the lever and the plate to ship**: sharpness 15.2 source →
  **65.9** (3B) vs 50.3 (7B) vs 22.0 (7B at denoise 0.75 + lab — closed). A
  `background` key names the **profile, not the model**: one key holds either's.
- Measure **in-mask, never the bounding box** — the box is 20x the submask and
  reports +120%, the plate read back as VACE's (`scripts/compare_720p.py`).
- **Sharpness and chroma point opposite ways**: the plate is sharpest and
  noisiest (+36%), the LoRA arm cleanest (−12%). Report both, or the metric
  contradicts the eye — it did, for a session.
- **A subject LoRA learns the anchor and barely moves the pipeline.** 0.023 →
  **0.5167** trained (ceiling 0.7454), yet matched arms give 0.1682 vs 0.1612 and
  0.2015 vs 0.1769 on the plate. Score only against the held-out split; the
  trigger is load-bearing; musubi's merge trap: `prepare_musubi_dit.py`.
- **`model.lora` is a stack, not a slot** (`common.lora_stack`): entries chain,
  each is bind-checked separately, contents hash into `vace_key` (keys moved once).
- Composite in `gbrp`, not `yuv420p`; after `--protected` pass `--mask` the
  submask, else the attribute arrives VAE-degraded.

## Three untracked configs, and rule 2c

Anything stating the subject's **category** is withheld from the repo, not merely
from a push. Each is untracked, carried by `state_bundle.sh`, on the denylist,
and **fails loud** rather than defaulting — a stage run against the wrong binding
emits plausible numbers for the wrong thing.

- **`prompt.local.yaml`** — `positive`/`negative` overlay every profile; tracked
  configs keep a category-free default that runs but **will not reproduce a
  recorded number**. `trigger`, `detect_prompt`, `probe_prompt`,
  `candidate_grid`, `loras:` have no default: a generic detector prompt changes
  which candidate is offered, a missing trigger looks like a dead LoRA.
- **`backends.local.yaml`** — a model is named for what it was trained to find.
  Code resolves **roles** (`anchor_embed`, `attribute_parser`) via
  `scripts/backends.py`; label map and groups live there too.
- **`vocab.local.txt`** — checks 6/6b's wordlist was a *negative image* of the
  subject, in the one file every clone carries. Machinery stayed tracked.

**Rule 2c**: agents may not READ `configs/agent_denylist.txt`'s entries —
untracked ≠ unreadable, and that gap is how the category reached a conversation
while every push check passed. `.claude/settings.json` deny rules plus
`scripts/agent_guard.sh` (PreToolUse hook covering Bash, which otherwise walks
around them via `cat`/`ls`/`find`); it over-blocks on purpose. Check 8 stops the
three pieces drifting. Irreducible residue: the lockfile and `bootstrap.sh` must
name a package to install it.

## Model facts, read from the installed source

CLAUDE.md rule 6 carries the constraints themselves (4n+1, multiples of 16, mask
polarity, the single reference image) and how each was proven. What it omits:

- VACE **centre-crops** the reference image; build the sheet at manifest size.
- Encode control streams into ComfyUI's input dir with `-qp 0`; lossy
  re-encoding rounds mask edges outward. Use `VAEEncodeTiled` — untiled peaks
  near 12 GB and OOMs.
- SeedVR2 is **native** to the pinned ComfyUI (`comfy_extras/nodes_seedvr.py`),
  no node pack. `frames_per_chunk` is 4n+1; `temporal_overlap` is in **latent** frames.
- Dynamic combos serialise `<parent>.<child>` (`_io.py::finalize_prefix`).

## Provenance and staleness

**Two keys, not one.** `vace_key` hashes what reaches the sampler (sheet, plate,
control, mask, ROI streams, prompts, seed, model incl. LoRA, sampler);
`composite_key` what reaches the compositor. As one key, widening an alpha ramp —
seconds of CPU — marked a finished generation stale and demanded ~18 GPU minutes
to reproduce identical pixels.

Hash file **contents**, never a config or geometry key: a rebuilt plate keeps its
filename and geometry while its pixels change completely. A result whose key no
longer matches is **not** a result — though a video *container* rehashes even when
its decoded pixels do not, so a re-run can invalidate an intact approval.

The key is captured **before** staging and re-read **after** inference; if they
disagree the output settles as `stale`, kept but never current. Recording the
post-run key would mark stale pixels current forever. The manifest's per-object
`_loaded_rev` stops a stage that loaded an old copy overwriting a newer one. Run
namespacing: `VACE_RUN=<name>` → `runs/<name>/{intermediate,outputs,reports,logs}`.

## Measurement discipline

Past mistakes worth not repeating — each cost a wrong conclusion:

- **Circular measurement.** Never test a set against something derived by
  subtracting it. Occluders defined as `candidates & ~dilate(subject)` and tested
  against `dilate(subject)` are empty by construction; "0.0000% OK" was a tautology.
- **"It ran" is not "it did something."** Eight LoRA checkpoints and the baseline
  scored identical to four decimals: the merge had silently no-oped.
- **Self-comparison as evidence.** The bank is built from the references, so
  scoring one against it returns 1.000. Verify by consensus: collapse
  near-duplicates, score by median agreement with the rest. A single maximum lets
  two copies of the wrong candidate vouch for each other.
- **A proxy standing in for the thing itself.** CLIP distance is not a viewing
  angle (use anchor keypoints); proximity is not occlusion (verify depth order); a
  rise in high-frequency energy is not restored texture — ringing raises it too.
- **A detector's training bias read as absence.** On one shot the target scored
  ≤0.27 against a non-target's 0.33–0.36, offering only the wrong candidate. Retry
  lower when no candidate has match evidence; never score a subject by its extent.
- **A static mask is architecture, not a subject.** A bbox that barely moves has
  locked onto scenery. Correlate mask occupancy with per-pixel temporal variance:
  a scenery-locked track scored r=+0.03 against motion, a correct one +0.58.

## Standing instructions from the user

- Every time work needs the user's eyes, hand over **one zip** from
  `scripts/make_review_bundle.py` (rule 1).
- Prepare the 14B cloud path, but **do not download 14B locally**.
- **All rule-2b inspection grants were revoked 2026-08-06.** The allowlist is
  empty and everything is refused. Never restore an entry from memory or from
  history — only the user, writing it there again, creates a grant.
- State separate effects, runtimes, VRAM and disk numbers. Never one blended claim.
- The macOS laptop has no CUDA and the lockfile is CUDA-only: **no local pipeline
  there**, so all GPU work runs on the rented box.
- **Take the GPU box down whenever it is not actively needed**, including while
  the user reviews, without asking. Stop, never destroy: Vast's `/workspace` is not
  a volume, and a stopped box cannot restart while another tenant holds the GPU —
  RunPod's **network volume** survives that, reattaching to a new pod.

## Cloud and open work

`docs/`: `CLOUD_RUNBOOK.md` (quirks, allowlist, gates, teardown), `MODEL_SWAP.md` (UNet swaps), `CANDIDATE_GENERATION.md` + `LORA_TRAINING.md` (candidates).
`scripts/state_bundle.sh` carries the irreplaceable half off a box; run it FIRST.
Dilation is not free: `mask.grow=4` put 4.51% of the mask onto another candidate — rely on the occluder layer instead.

**VACE-14B is tested and it is worse**: in-mask 7.2 against 1.3B's 8.0, the LoRA's
8.4, the plate's 16.5 and a plain upscale's 9.7. fp16 fits one 80 GB card (peak
59.2 GB, 19.81 s/frame). Capacity was never the constraint — only 4.44% of the
subject is repainted, under a control video pinning every pixel. **VACE is out,
the plate is the deliverable**, and pose-vs-depth is moot.

**The region defect: cause not yet established.** Ledger in
`reports/experiment_ledger.md`; do not re-run completed arms. Excluded on
evidence: prompt interference, the VAE (~44 dB), the seed across 12
images, and **local repair at 16× the area with composition pinned** (33 arms,
none helped, edge energy fell in all 30).

**Precision is EXCLUDED on a measurement** (bundle 20): declared `default` loads
100% bfloat16, `fp8_e4m3fn` 100% float8, and regenerating at `default` reproduced
bundle 18 bit-for-bit. **Probe the loaded model; never trust a directory name or
a loader argument.** **THE ADAPTER PARTIALLY FIXES THE REGION** — base-only at
the same seed, prompt and verified dtype is *worse* there — so aggravation is
refuted, the base is the failing component, and adaptation is the only lever
shown to move it. That points at the dataset, not at a model swap.

Never audited: whether training material holds usable evidence of the region
*after* trainer resize and crop — until it does, "cannot draw it" and "was never
shown it" are indistinguishable from outputs alone.

Three things that generalise beyond this defect:

- **A region box does not transfer between images.** Three sibling candidates'
  boxes differed by up to 68 px and by a third in size, on a region ~60 px tall.
  Ask per image — `mark_region.py` makes that four numbers off a labelled grid —
  and key the box to the arm directory. A single shared box would have been wrong
  on two of three with every downstream check still passing.
- **Under masked img2img the seed stops dominating** (910 vs 920 agree to ~0.1,
  against a spread of ~0.19 across seeds in full-frame work): the input latent
  pins composition, so re-rolling cannot rescue a failed denoise level.
- **Prompt text leaks where it is COPIED, not only where authored** — a review
  bundle's inputs record carried the resolved prompt; denylisted now.

Untried, in cost order: a repair-specific overlay key (`candidate_repair` — the
structural arms ran on the full-frame template, the one confound left in the
result above); a **dedicated inpainting model**, since Chroma is not one and
`SetLatentNoiseMask` is masked img2img rather than trained fill; then step 8's
targeted references. **`gen_key.py`, `body_structure.py`, `assert_dataset.py` and
`compare_arms.sh` are lost** — absent from worktree and volume alike, which is
the standing cost of the Chroma tools being untracked.
