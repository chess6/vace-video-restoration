# Subject LoRA — what was trained and what it is worth

Phase 5c–5e. Trainer: musubi-tuner, pinned in `versions.md`. One GPU.

---

## What was trained

Nine head crops, taken around the face `identity.resolve_targets` verified in
each reference photograph — never the largest face, which in a group shot is a
stranger. Head crops rather than the greyed panels `make_reference_pack.py`
builds: greying the wardrobe is right for conditioning, where the model sees the
panel once, but a LoRA would meet a flat grey field in every image and learn
that this person arrives with a grey background. Cropping excludes the wardrobe
by framing instead, so the authority split holds with nothing artificial to
learn.

Of the verified photographs, three sat just below the default crop floor and one
far below it. The floor was lowered to recover the three near misses — nine
training images instead of six is worth more here than the small amount of
upscaling it costs — and the fourth stayed out.

**Three images were held out and never trained on.** The identity bank is built
from these same photographs, so scoring a LoRA against the whole bank returns a
good number whatever it learned. That is the self-comparison trap in
`docs/STATE.md`, and without a reserved split the question is unfalsifiable.

Base: **Wan2.1 T2V 1.3B**, not the VACE checkpoint the pipeline runs — musubi
has no VACE task. The substitution was checked, not assumed: all 825 tensors of
the T2V checkpoint appear in the VACE one under identical names with identical
shapes, and VACE only adds its own blocks. `verify_lora_loads.py` then puts the
trained file through ComfyUI's own mapper against the real checkpoint: **300 of
300 modules bind**.

## How identity was measured

Each checkpoint generates four probe images from the base T2V model — no VACE,
no plate, no mask — and the face in each is scored against a bank built from the
**held-out photographs alone**. Two more rows make the middle one readable:

- **Ceiling** — the training crops' own verified faces against the held-out
  bank. Real photographs of the same person, no generation involved. This is
  what "as good as a photograph" is worth on this metric.
- **Baseline** — the same prompt and seeds with no LoRA. Without it a similarity
  of 0.3 says nothing: the base model also produces a person.

`identity.py` treats **0.35** as the threshold for a plausible match.

## Results

| checkpoint | steps | identity vs held-out | vs train bank (invalid) |
|---|---|---|---|
| ceiling — real photographs | – | **0.7454** | – |
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
a real photograph scores. Training was stopped there rather than at a measured
peak; the metric sees identity, not the memorised framing that overfitting would
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
face detection so the other person in the shot cannot be scored by mistake.

**All three ran WITHOUT `--background`, so the preserved 95.58% is the original
frame, not a restored plate.** The three arms are still matched to each other,
which is what the comparison below rests on, but they are not the shipped path
and they do not look like it: the user's first observation on seeing them was
that the quality matched the source, which is exactly what they were. The run
log asserted "comes from the plate" regardless of the flag and was believed;
that line now names what actually supplies the region, and `--protected` without
`--background` warns before the GPU is touched.

Whether the plate changes the LoRA verdict is untested. It should not — VACE
regenerates the same submask either way, and the constraint identified below is
the size of that submask, not the sharpness of what surrounds it — but "should
not" is not a measurement.

| arm | LoRA | identity vs held-out | frames with a face |
|---|---|---|---|
| control | none | **0.1682** | 14/16 |
| treatment | strength 1.0 | **0.1612** | 14/16 |
| strong | strength 2.0 | **0.1263** | 11/16 |

**No improvement at strength 1.0** — the difference is the wrong way round and
within noise. At strength 2.0 identity is clearly *worse*, and a face was
detectable in three fewer frames, which is what a damaged region looks like
rather than a differently-identified one.

That second arm is what makes this conclusive. A LoRA that scores 0.5167 on
direct probes and 0.16 through the pipeline could mean its influence is too
weak to survive the path; doubling it would then have helped. It did not. **The
protected path leaves no room for an identity prior.** VACE regenerates 4.42% of
the figure, and every pixel of it is pinned by the control video and ringed by
plate; there is not enough free signal for a face prior to express itself, and
pushing harder degrades what is there.

This is consistent with, and independent of, what the 1.3B pilot already found:
the plate beats every VACE variant on this shot, identity included.

**So a subject LoRA does not rescue VACE 1.3B here.** What it does establish is
that the *dataset and identity plumbing work*: the LoRA learned this person's
face from nine crops well enough to reproduce them at 70% of a photograph's
score. If VACE is ever given a larger region to regenerate, or a larger model at
a size where its output beats the plate, the LoRA is ready and the harness to
judge it exists.

## Cost

| | |
|---|---|
| dataset export | ~1 min, CPU-bound face resolution |
| training, 1260 steps | 12 min at ~1.7 it/s |
| training, 2520 steps | 25 min |
| probe generation + scoring | ~20 s per image, ~1 min to score a set |
| one pilot chunk through VACE | 8 min, peak 20.7 GB VRAM |

All on one RTX 4090. Training peaked well under the card; the 8-minute pipeline
runs are the expensive part, which is why the dry run and the LoRA/graph
consistency check exist.

## Limits

- The pipeline arms carried **no restored plate** (above). Matched to each other,
  not representative of output quality.
- Identity here is measured on **probe generations from a portrait prompt**,
  head-on and well lit, which is the friendliest possible case. It says the LoRA
  learned the face; it does not say what happens inside the pipeline, where VACE
  regenerates a small head region over a plate.
- Three held-out photographs is a thin bank. It is the right bank — the only
  uncontaminated one available — but three images cannot characterise a face
  across pose and lighting.
- Nine training images at one framing. Expect the LoRA to carry that framing's
  lighting and scale with it.
- The trigger token is load-bearing: the captions were the token and nothing
  else. A config that enables the LoRA without putting the token in
  `prompt.positive` produces an unchanged result that reads as a failure.
