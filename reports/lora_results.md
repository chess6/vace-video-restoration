# Subject LoRA — what was trained and what it is worth

Phase 5c–5e. Trainer: musubi-tuner, pinned in `versions.md`. One GPU.

---

## What was trained

Tight anchor crops, taken around the anchor `match.resolve_targets` verified in
each external reference — never the largest anchor, which in a group shot is a
non-target. Tight crops rather than the greyed panels `make_reference_pack.py`
builds: greying the attributes out is right for conditioning, where the model
sees the panel once, but a LoRA would meet a flat grey field in every image and
learn that this candidate arrives with a grey background. Cropping excludes the
attributes by framing instead, so the authority split holds with nothing
artificial to learn.

A few verified references sat just below the default crop floor and one far below
it. The floor was lowered to recover the near misses — the extra training images
are worth more here than the small amount of upscaling they cost — and the one
far below stayed out.

**A held-out split was reserved and never trained on.** The match bank is built
from these same references, so scoring a LoRA against the whole bank returns a
good number whatever it learned. That is the self-comparison trap in
`docs/STATE.md`, and without a reserved split the question is unfalsifiable.

Base: **Wan2.1 T2V 1.3B**, not the VACE checkpoint the pipeline runs — musubi
has no VACE task. The substitution was checked, not assumed: all 825 tensors of
the T2V checkpoint appear in the VACE one under identical names with identical
shapes, and VACE only adds its own blocks. `verify_lora_loads.py` then puts the
trained file through ComfyUI's own mapper against the real checkpoint: **300 of
300 modules bind**.

## How match was measured

Each checkpoint generates four probe images from the base T2V model — no VACE,
no plate, no mask — and the anchor in each is scored against a bank built from the
**held-out references alone**. Two more rows make the middle one readable:

- **Ceiling** — the training crops' own verified anchors against the held-out
  bank. Real references of the same candidate, no generation involved. This is
  what "as good as a real reference" is worth on this metric.
- **Baseline** — the same prompt and seeds with no LoRA. Without it a similarity
  of 0.3 says nothing: the base model also produces a candidate.

`reference_match.py` treats **0.35** as the threshold for a plausible match.

## Results

| checkpoint | steps | match vs held-out | vs train bank (invalid) |
|---|---|---|---|
| ceiling — real references | – | **0.7454** | – |
| baseline, no LoRA | – | 0.0226 | 0.0336 |
| v1 | 180 | 0.0305 | 0.0814 |
| v1 | 360 | 0.1442 | 0.1574 |
| v1 | 540 | 0.1903 | 0.2013 |
| v1 | 720 | 0.2539 | 0.2679 |
| v1 | 900 | 0.3301 | 0.3317 |
| v1 | 1080 | 0.3791 | 0.3875 |
| v1 | 1260 | 0.4258 | 0.4261 |
| v2 | 1440 | 0.3997 | 0.4230 |
| v2 | 1800 | 0.4302 | 0.4456 |
| v2 | 2160 | 0.4862 | 0.4927 |
| **v2 (shipped)** | **2520** | **0.5167** | 0.5403 |

Monotone apart from one dip within seed noise, clearing the 0.35 threshold from
about 900 steps, and **still climbing at 2520** — roughly 70% of the way to what
a held-out reference scores. Training was stopped there rather than at a measured
peak; the metric sees match, not the memorised framing that overfitting would
also produce, so "keep going until it turns over" is not a safe rule.

The last column is what scoring against the training images would have reported.
It tracks the valid column closely here — which is exactly why it is dangerous:
it looks like corroboration and is not evidence at all.

## The trap that ate the first run

The first evaluation returned **0.0226 for every checkpoint and for the no-LoRA
baseline, to four decimal places**. Not a curve — the probe images were
byte-identical. musubi merges a LoRA into the checkpoint *before* it strips the
`model.diffusion_model.` prefix that Comfy-Org's repackaged 1.3B carries, so
every key lookup missed and eight checkpoints' worth of GPU time rendered the
base model. The only symptom was one warning line in a log full of progress
bars.

`prepare_musubi_dit.py` writes the de-prefixed copy. The guard that now refuses
any probe set pixel-identical to the baseline is what keeps it from recurring
quietly. ComfyUI was never affected — its loader handles the prefix, which is
why the binding check passed while generation did nothing. A stage that ran is
not a stage that did anything.

## In the pipeline, it does nothing

Three runs of the pilot chunk, protected path, identical in every respect except
the LoRA — the control config extends the treatment config, so the prompt, seed,
sampler, geometry and reference pack cannot have drifted. All three scored
against the held-out bank, with each frame cropped to the tracked subject before
anchor detection so the other candidate in the shot cannot be scored by mistake.

The first three ran **without `--background`**, so their preserved 95.58% was the
original frame rather than a restored plate. The run log asserted "comes from the
plate" regardless of the flag and was believed; it now names what actually
supplies the region, and `--protected` without `--background` warns before the
GPU is touched. The user caught it in one sentence after watching the clips:
the quality matched the source, which is exactly what it was.

**Repeated on the plate, the verdict is identical.** Two more arms, same configs,
`--background background_aggressive`:

| arm | LoRA | plate | match vs held-out |
|---|---|---|---|
| loraE | none | aggressive | **0.2015** |
| loraD | strength 1.0 | aggressive | **0.1769** |
| loraB | none | none | 0.1682 |
| loraA | strength 1.0 | none | 0.1612 |

Still no improvement, still the wrong way round, on the path that actually ships.
The plate lifts both arms by about 0.03 — restoring the surroundings helps the
anchor a little — and the LoRA costs about 0.02 in both pairs. So the earlier
conclusion holds without the caveat it was carrying.

| arm | LoRA | match vs held-out | frames with an anchor |
|---|---|---|---|
| control | none | **0.1682** | 14/16 |
| treatment | strength 1.0 | **0.1612** | 14/16 |
| strong | strength 2.0 | **0.1263** | 11/16 |

**No improvement at strength 1.0** — the difference is the wrong way round and
within noise. At strength 2.0 match is clearly *worse*, and an anchor was
detectable in three fewer frames, which is what a damaged region looks like
rather than a differently-identified one.

That second arm is what makes this conclusive. A LoRA that scores 0.5167 on
direct probes and 0.16 through the pipeline could mean its influence is too
weak to survive the path; doubling it would then have helped. It did not. **The
protected path leaves no room for an match prior.** VACE regenerates 4.42% of
the subject, and every pixel of it is pinned by the control video and ringed by
plate; there is not enough free signal for an anchor prior to express itself, and
pushing harder degrades what is there.

This is consistent with, and independent of, what the 1.3B pilot already found:
the plate beats every VACE variant on this shot, match included.

**So a subject LoRA does not rescue VACE 1.3B here.** What it does establish is
that the *dataset and match plumbing work*: the LoRA learned this candidate's
anchor from a handful of crops well enough to reproduce them at 70% of a real reference's
score. If VACE is ever given a larger region to regenerate, or a larger model at
a size where its output beats the plate, the LoRA is ready and the harness to
judge it exists.

## What the plate is worth, and what VACE costs on top of it

Median Laplacian variance over 16 frames, whole frame. A sharpness **proxy** —
`docs/STATE.md` warns that high-frequency energy is not the same as restored
detail, since ringing and noise raise it too — so every row is stated against
the source it has to beat, with the plate as the reference point.

| stream | sharpness | vs source |
|---|---|---|
| source, unrestored | 15.2 | — |
| **plate registered as `background_aggressive`** | **50.3** | **+231%** |
| VACE on that plate (loraD / loraE) | 40.6 / 40.7 | +167% |
| VACE with no plate (loraA / loraB) | 14.4 / 14.5 | **−5.6% / −4.9%** |

**That plate is the 7B pass, not 3B.** Verified after the fact: inside the review
bundle it is byte-identical to the clip an earlier session named
`plate_7B_aggressive_720p`. The manifest's background key records the **profile
name, not the model** — 7B is selectable at run time — so `background_aggressive`
does not tell you which checkpoint produced those pixels, and +231% here is a 7B
number that carries 7B's ~3x chroma noise. The 3B plate for this chunk is a
different hash under the same profile directory and was not what these arms sat
on. None of this touches the LoRA comparison, where both arms share one plate.

Two things fall out. The plate-free arms are *softer than the source* — the
first batch did not merely fail to improve the shot, it slightly degraded it,
which is what the user saw. And VACE on the plate lands a third of the way back
down: it takes a +231% plate and returns +167%, because the region it
regenerates comes back softer than what it replaced. That is the pilot's
"the plate beat every VACE variant" finding, reproduced on this configuration
and now with a LoRA in the mix.

### And in the pixels VACE actually paints, it matches a plain upscale

Whole-frame numbers hide the thing being asked about, because 95.58% of the
frame is plate. Measured over the protected submask's **pixels** — not its
bounding box, which is twenty times the mask and mostly plate:

| region | Lanczos 720p | plate | VACE+plate | VACE+plate+LoRA |
|---|---|---|---|---|
| whole frame | 15.2 | 50.3 | 40.7 | 40.6 |
| submask bounding box | 15.3 | 48.2 | 33.7 | 33.1 |
| **submask pixels (regenerated)** | **9.3** | **15.7 (+70%)** | **9.6 (+3.8%)** | **9.4 (+1.2%)** |

**In the region it exists to improve, VACE is within noise of upscaling the
source with Lanczos.** The plate had already restored those pixels by +70%;
VACE discards that and returns something no sharper than the original. The
background looks restored because it is plate, and the anchor does not because it
is not — which is what the user reported before any of this was measured:
"as if I'm watching the same original clip".

Note the middle row. A bounding box around the anchor reports +120% and would have
been read as VACE working. It is measuring the plate.

`scripts/compare_720p.py` produces this table, the 100% crops and the
side-by-sides against the Lanczos baseline. The baseline needed no work:
`preprocess_source.py` already upscales the source to the working geometry with
Lanczos, so the working stream *is* the default upscaler's output, frame-aligned
with every variant.

## Repeated on the 3B plate, and the plate question settled

The arms above sat on a plate that turned out to be the 7B pass. Repeated on a
3B plate, on the RunPod volume, with the same configs and one flag changed:

| stream | frame | in-mask (regenerated) | chroma noise |
|---|---|---|---|
| Lanczos 720p | 15.2 | 9.7 | 1.58 |
| **plate 3B aggressive** | **65.9 (+333%)** | **14.2 (+47%)** | 2.15 (+36%) |
| plate 7B quiet | 22.0 (+44%) | 13.5 (+40%) | **2.29 (+45%)** |
| VACE + plate, no LoRA | 56.2 (+269%) | 8.0 (−17%) | 1.57 (−1%) |
| VACE + plate + LoRA | 55.3 (+264%) | 8.4 (−13%) | **1.40 (−12%)** |

**The LoRA arm is the better of the two VACE arms**, on both axes: sharper in
the regenerated pixels (8.4 vs 8.0) and 12% below baseline chroma where the
control merely matches it. Small, but it is the first measurement that agrees
with the user's own reading, who called VACE+LoRA best "but just slightly"
before any of this was measured. It does not rescue VACE: both arms still land
*below* a plain Lanczos upscale inside the submask.

**It also explains the earlier disagreement.** The plate is the sharpest thing
here and the noisiest in colour; the LoRA arm is the cleanest of everything
measured. Sharpness and cleanliness point in opposite directions, and the metric
had only been reporting the first.

**The 7B quiet experiment fails.** Denoise 0.75 with lab colour matching was
meant to keep 7B's detail without its coloured static. It kept neither: 44%
whole-frame sharpness against 3B's 333%, and *more* chroma noise than 3B, not
less. 3B aggressive is the plate to ship.

## VACE-14B: the last lever, and it goes the wrong way

One pass on the same pilot interval, same masks, same approved track, same
3B plate underneath, same prompt and seed. The only change from the control arm
is the checkpoint — `configs/cloud_720p_14b.yaml` extends that arm rather than
`cloud_14b.yaml`, whose 1280x720 geometry would have invalidated every asset and
forced a re-track. No LoRA: it is welded to 1.3B and its shapes do not fit 14B.

| stream | frame | **in-mask (regenerated)** | chroma |
|---|---|---|---|
| Lanczos 720p | 15.2 | 9.7 | 1.58 |
| **plate 3B** | 65.6 | **16.5 (+70.7%)** | 2.12 |
| VACE 1.3B | 56.2 | 8.0 (−16.9%) | 1.57 |
| VACE 1.3B + LoRA | 55.3 | 8.4 (−12.8%) | 1.40 |
| **VACE 14B** | 59.7 | **7.2 (−25.5%)** | 1.46 |

**14B is the worst VACE arm measured.** It does not merely fail to beat the
plate's +70.7%; in the pixels it repaints it lands 25% *below* a plain Lanczos
upscale, and below 1.3B with or without the LoRA. Its frame-level number is the
highest of the VACE rows (59.7) purely because 95.56% of the frame is plate.

Cost of the answer: 26m 44s of A100 80GB at 19.81 s/frame, peak 59.2 GB — so
fp16 14B does fit one 80 GB card at this geometry, which the runbook had listed
as plausible but undemonstrated. About $1.30 including the 33 GB download.

**This closes the question.** VACE does not earn its place on this footage at any
size worth renting, and no subject LoRA changes that, because the constraint was
never model capacity — it is that the protected path regenerates 4.44% of the
subject under a control video that pins every pixel of it. The deliverable is the
plate.

## Cost

| | |
|---|---|
| dataset export | ~1 min, CPU-bound anchor resolution |
| training, 1260 steps | 12 min at ~1.7 it/s |
| training, 2520 steps | 25 min |
| probe generation + scoring | ~20 s per image, ~1 min to score a set |
| one pilot chunk through VACE | 8 min, peak 20.7 GB VRAM |

All on one RTX 4090. Training peaked well under the card; the 8-minute pipeline
runs are the expensive part, which is why the dry run and the LoRA/graph
consistency check exist.

## Limits

- Two of the four pipeline arms carried **no restored plate**; the plate-backed
  pair is the one that represents the shipped path.
- Match here is measured on **probe generations from a tightly framed prompt**,
  square-on and well lit, which is the friendliest possible case. It says the LoRA
  learned the anchor; it does not say what happens inside the pipeline, where VACE
  regenerates a small anchor region over a plate.
- The held-out split is a thin bank. It is the right bank — the only
  uncontaminated one available — but three images cannot characterise an anchor
  across pose and lighting.
- A handful of training images at one framing. Expect the LoRA to carry that framing's
  lighting and scale with it.
- The trigger token is load-bearing: the captions were the token and nothing
  else. A config that enables the LoRA without putting the token in
  `prompt.positive` produces an unchanged result that reads as a failure.
