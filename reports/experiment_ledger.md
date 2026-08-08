# Experiment ledger — the region defect

Every completed arm on the candidate path, what it varied, and what it settled.
Built from the review bundles and their provenance sidecars rather than from
memory, so that no arm is re-run and no arm is assumed.

Companion to `pilot_results.md` (restoration path) and `lora_results.md`
(subject adaptation). Architecture in `docs/CANDIDATE_GENERATION.md`.

**Rule 2a applies**: no prompt text, no source filenames, no counts of supplied
material, role words throughout. Per-image artefacts stay in the review bundles.

**Comparability**: numbers compare only *within* one baseline. Resolution, cfg,
flow shift and sampler all move absolute values, so two rows from different
baselines are not an A/B.

---

## The defect

A region of a generated candidate that renders as blob-like or blurred, in a
subject that is otherwise correct. It occupies roughly **0.3% of the frame area**
at the working canvas — small, and that smallness was the leading hypothesis for
most of the investigation.

## Completed arms

| # | Bundle | Varied | Held constant | Arms | Conclusion |
|---|---|---|---|---|---|
| 08 | chroma_front_view | — | no LoRA | 1 | floor established, match 0.0425 |
| 09 | chroma_runs | resolution, cfg | no LoRA | 4 | canvas ceiling identified |
| 10 | review_chroma_lora | checkpoint step | baseline A | 8 | **best measured result: 0.6548 match** |
| 11 | chroma_negative_ab | negative present / stripped | seed | 2 | stripping costs **−0.218 match** |
| 12 | chroma_v3_bodyonly | checkpoint step | baseline A | 8 | 0.6407, indistinguishable from 10 |
| 13 | review_three_arms | LoRA arm × seed | baseline B | 9 | **arms indistinguishable; the seed dominates** |
| 14 | prompt_ablation | negative: current / empty | seed | 6 | **prompt ruled out**; empty is worse |
| 15 | vae_roundtrip | encode→decode, two scales | — | 4 | **VAE ruled out**: ~44 dB, SSIM 0.9995+ |
| 16 | lora_strength | 0.00 / 0.80 / 1.00 / 1.15 / 1.30 | seed, baseline | 12 | adapter **not necessary** for the defect — see below |
| 17 | dit_precision | declared dtype (3 values) | seed | 6 | two quantised settings identical; **dtypes unverified** |
| 18 | seed_precision | 6 seeds × 2 declared dtypes | baseline B | 12 | **no seed renders it cleanly**; dtypes unverified |
| 19 | region_repair | denoise 0.30–0.55 × 2 seeds × 3 images | measured box per image | 33 | **local repair fails** |

Baseline A and baseline B differ in resolution, cfg and flow shift; rows are only
comparable inside one of them.

### Arms that settle more than their row suggests

**16 — the adapter is not NECESSARY, which is not the same as not responsible.**
The `0.00` arm does not set strength to zero; it builds the graph with **no
adapter node at all**, and the defect is present in the bare base model. So the
base produces it unaided and the adapter is not required for it to appear.

What this does **not** show is that the adapter never worsens it. Severity was
never compared with and without the adapter at fixed seed — the sweep varied
strength, and no metric here detects the defect, so "worse" was never measurable.
Adapter aggravation stays a live hypothesis.

**17/18 — precision is NOT closed. The record cannot support the claim.** The
runtime dtype is **not** captured in the provenance sidecars for these bundles,
so the arm labels rest on a directory name and an argument passed to a loader,
never on a measurement.

An earlier revision argued the question was closed anyway, because the arms
produced visibly different output and therefore could not have been the same
dtype. **That inference is invalid** and is withdrawn: a difference in output
shows only that something differed, not which dtype either arm ran at, and
specifically not that the arm labelled unquantised ran unquantised. A loader can
silently fall back; a declared dtype is a request, not a receipt.

This is the unresolved arm. Closing it needs runtime telemetry — loaded
checkpoint digest, declared dtype, actual tensor/compute dtype where the runtime
exposes it, and which loader implementation ran — plus one fixed-seed control at
verified native precision against the existing quantised result with every other
input held. `weight_dtype` is recorded by the newer tooling; the older records
are not retrofittable and are marked incomplete rather than reinterpreted.

**19 — the strongest available test, and it failed.** The region was cropped with
context, resampled up so it reached the sampler at **4× linear / 16× area**,
denoised only inside a measured mask, resampled back and composited into the
otherwise untouched original. Mask polarity was measured on the build first
(ratio 37.50, white = repaint) rather than assumed, and every node class and
input key was verified against the running server. All 33 arms assert zero
changed pixels outside the mask and non-zero inside.

Mean absolute delta inside the mask, against the resampling floor its own control
arm measured:

| image | control | 0.30 | 0.35 | 0.40 | 0.50 | 0.55 |
|---|---|---|---|---|---|---|
| A | 0.79 | 3.38 | 3.96 | 4.59 | 5.98 | 6.74 |
| B | 0.92 | 3.06 | 3.58 | 4.11 | 5.40 | 6.17 |
| C | 1.00 | 3.40 | 4.01 | 4.68 | 6.22 | 7.11 |

The model did 3–7× the null arm's work and none of it helped. **Edge energy fell
in all 30 diffusion arms, monotonically with denoise**: given freedom in that
region the model reaches for smooth rather than structured.

## Arms NOT completed

| Asked for | Status |
|---|---|
| verified native-precision control | **not run — this is now the blocking arm** |
| officially-prepared scaled-FP8 | not run; secondary to the above |
| tighter framing that **contains the region** | **not run** — see the correction below |
| targeted or focus-masked adaptation | not run — blocked on verified evidence of the region |
| rank 16 vs rank 32 at equal updates | not run — correctly gated behind the above |

### Correction: the tighter-framing arm does not exist

An earlier revision of this ledger recorded tighter framing as complete but
unexamined, on the basis that one seed in bundle 13 produced a close composition
in all three arms — much larger subject span, and 7 of 17 keypoints against 17 of
17 elsewhere. **The region is not in frame in any of those images.**

The metric had already said so. `keypoint_completeness` at 7/17 does not mean "a
close shot"; it means **ten keypoints were outside the frame**. Completeness
measures how much is *detectable* and span measures how *large* the subject is —
neither reports *which parts are in shot*, and a closer composition is exactly
where those come apart. The same metric saturated at 17/17 on images reported as
malformed. It has now misled in both directions, and no metric in this project
reports whether the region is in frame.

## Decision

Which of the four candidate causes owns this failure:

| Candidate | Verdict |
|---|---|
| Quantisation | **UNVERIFIED.** Not excluded — see below. |
| The current subject adapter | **Not the sole or necessary cause.** Not "excluded": see below. |
| Base-model capability | **Strongly implicated, one confound outstanding.** |
| Absent training evidence | **Open and never audited.** |

### Two verdicts corrected under review

**Quantisation is unverified, not excluded, and the earlier wording overclaimed.**
The argument was: the two precision arms produced visibly different output, so
they cannot have been the same dtype, so the higher-precision one is a real
control and it shows the defect. The second step does not follow. That the arms
*differed* establishes only that *something* differed — a loader path, a cast
schedule, a fallback inside the loader. It does not establish **which** dtype
either arm actually ran at, and in particular does not establish that the arm
labelled unquantised ran unquantised. No runtime dtype was ever captured; the
labels come from a directory name and an argument passed to a loader.

Inferring a dtype from output that "looks different" is the same class of error
this project has already paid for under other names — a proxy standing in for the
thing itself. The correct status is **unverified**, and it stays unverified until
telemetry records the loaded checkpoint digest, the declared dtype, and the
actual tensor/compute dtype the runtime reports.

**The adapter is not the sole or necessary cause. That is weaker than "excluded".**
The zero arm builds the graph with no adapter node at all and the defect is
present, which proves the base model produces it unaided — so the adapter is not
*required* for the defect to appear. It does **not** prove the adapter never
makes it worse. No arm has ever compared defect severity with and without the
adapter at fixed seed and fixed everything else; the sweep varied strength, and
severity was never measured because no metric here detects the defect at all.
"Adapter aggravation" therefore remains a live hypothesis, and it is one of the
outcomes the pending decision must choose between.

The outstanding confound is narrow and worth stating precisely: bundle 19's
higher-denoise arms — the ones aimed at *structure* rather than appearance — were
conditioned on the full-frame template, which on a magnified crop asks for a
whole subject inside the crop. The arms aimed at appearance (0.30–0.40) are
unaffected, because at that denoise the input latent holds composition. So
"local repair failed" is established for appearance correction and **not yet
clean for structural correction**.

The cheapest way to close it adds no new wording at all: condition the redraw on
the trigger alone, or on nothing, at high denoise. Tooling supports both.

The fourth row is the one nobody has evidence about in either direction. It
cannot be settled by generation — it requires auditing whether verified source
material contains usable evidence of the region *after* trainer resizing and
cropping. Until that audit exists, "the base model cannot draw it" and "the
adaptation was never shown it" are indistinguishable from the outputs.

## Recommended architecture

**Restoration path — unchanged and already settled.** The restored plate ships.
It beat every generative variant, and the generative stage measured below a plain
Lanczos upscale over the pixels it regenerates. Nothing in this investigation
touches that conclusion.

**Candidate path — keep the current stack, with a stated limitation.** Chroma
checkpoint + single T5-XXL encoder + Flux VAE + subject adapter remains the best
measured generator for reference candidates by a wide margin (0.6548 against the
previous generator's 0.3906 on match, and ~2.6× on subject span). Two
recommendations follow from this ledger rather than from preference:

- **Do not add a second text encoder.** Read from the installed source: this
  checkpoint's `clip_target` returns a single tokenizer/encoder and has no
  cross-attention path to receive a second one. Adding one is a no-op, not a
  trade-off.
- **Do not adopt an externally-recommended flow shift.** The two baselines here
  differ in flow shift and the higher-scoring one used the sampler default; a
  third value has no measurement behind it in this project.

**The region is a scoped exception, not a reason to replace the stack.** The
generator is good at everything except one small region. The proportionate next
step is a purpose-built masked-fill model applied to that region alone, keeping
the surrounding image, because the current graph performs masked img2img rather
than trained fill. Replacing the base model is justified only if that also fails,
and any replacement must pass a capability gate on the region — base weights
only, no adapter port — before any training is planned for it.
