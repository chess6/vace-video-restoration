# Subject LoRA training

How a subject LoRA is built for the candidate path: environment, dataset, split
discipline, configuration, and how arms are compared.

Companion to `CANDIDATE_GENERATION.md`, which covers generation and judging.
Results live in the review bundles indexed by `outputs/INDEX.md`, never here — a
doc that carries numbers goes stale silently.

Rule 2a applies: tracked, so no input filenames, no counts of what the user
supplied, no prompt text, no category words. Role words throughout — subject,
anchor, anchor region, extent, candidate, match.

---

## What is being trained, and onto what

| | |
|---|---|
| Base | **`Chroma1-HD`** — Flux-derived MMDiT, Apache-2.0 |
| ai-toolkit arch | `chroma` |
| Adapter | LoRA, rank 16 / alpha 16, saved fp16 (~112 MB per checkpoint) |
| Trained | the DiT only — the text encoder is **not** trained |
| Text encoder | T5-XXL, frozen |

**A LoRA is welded to the base it was trained against.** The Chroma adapters will
not load onto the Wan checkpoint and the Wan ones will not load onto Chroma:
different architectures, different tensor names, different shapes. There is no
"convert" here — a base change means retraining. Verify binding before spending
generation time, because a mismatched adapter does not necessarily raise; it can
apply nothing and leave a run that recorded a LoRA it was not influenced by.

**ai-toolkit loads the base through diffusers, not the single-file checkpoint
ComfyUI uses.** They are the same weights in different packaging, so the volume
ends up holding two copies — one for training, one for inference. Worth knowing
before diagnosing disk use, and worth not "tidying" either away.

The text encoder is left frozen deliberately: the vendor config notes that
training it is not expected to work for this architecture, and the subject
signal wanted here is visual, not lexical. The trigger is a handle bound in the
DiT, not a new word taught to the encoder.

## Trainer and environment

`ai-toolkit`, cloned to the volume, **in its own virtualenv**. It is kept
separate from the pipeline venv on purpose: the two want different dependency
sets, and a shared env means an upgrade for one silently breaks the other.

**Pin its torch to the CUDA the fleet actually has.** The env lives on the
volume and is reused by whatever pod attaches next, but the *driver* belongs to
the pod. An env built against a newer CUDA than a later pod's driver supports
fails with "driver too old" on the same volume that worked an hour earlier.

**Keep `torch`, `torchvision` and `torchaudio` on the same build.** Downgrading
one leaves the others linked against a different ABI, and the failure surfaces
as a shared-object load error far from its cause.

Verify with a real kernel launch, not `torch.cuda.is_available()` — availability
returns true in cases where the first actual matmul still fails.

## Dataset pipeline

Three steps, in this order. The order is not cosmetic.

```
scripts/make_lora_dataset.py     anchor crops + the split
        │   (DELETES every png/txt in train/ and holdout/ first)
        ▼
write_captions.py                captions that name the framing
        ▼
add_body_crops.py                whole-subject crops beside the anchor crops
        ▼
pin_holdout.py                   force the held-out set to a fixed list
        ▼
assert_dataset.py                refuse to train if the invariants broke
```

`make_lora_dataset.py` **clears the split directories on every run**, which is
correct — otherwise a re-run at different settings leaves stale crops behind and
the trainer, which globs the directory, trains on both. It also means anything
added afterwards is destroyed by the next rebuild, so the later steps must be
re-run every time.

**Crops come from the verified anchor, never the largest one present.** With a
second candidate in frame, guessing is how the wrong one gets trained in
permanently.

**Whole-subject crops inherit their verification.** The crop is taken around a
box that contains an already-verified anchor, so no new trust is introduced.
Framings with no detectable anchor cannot be verified by this pipeline at all
and are out of scope; they would need an explicit human assertion.

**Detection retries at the lower threshold.** The default threshold finds
nothing on many sources, and the resolver does not retry. Without a retry most
sources yield no box and the whole-subject half of the dataset silently collapses
to a handful of crops.

**Extent is captioned from box-vs-frame geometry.** A box touching a frame edge
means the subject continues past it. Captioning such a crop as complete teaches
the model that the phrase for completeness means a cut-off subject, which
poisons the exact wording needed at inference.

**Resolution buckets are capped to what the crops contain.** Training above that
upscales, and the model learns to reproduce resampling artefacts.

## Split discipline

The held-out split is the measuring instrument. It must not move while the thing
being measured changes.

- **Split by source image, not by crop.** Two crops at different scales can come
  from one source; holding out one while training on the other leaks the source
  into the model.
- **Pin it.** The builder spreads the holdout across a consensus ranking, so
  *adding* sources reshuffles it. That has already moved a previously trained
  source into the bank, which would have made every recorded number
  incomparable while looking entirely normal.
- **Assert before every run.** `assert_dataset.py` fails if any training crop
  derives from a held-out source, or if any crop lacks a caption. An uncaptioned
  crop is worse than a missing one: it trains, and everything in it lands on the
  trigger.

Scoring against training material returns a high number by construction. The
`vs train bank` column exists to show what that mistake would have produced; it
is never a result.

## Captioning

**A LoRA learns whatever the caption does not name.** A bare trigger on tightly
framed crops binds framing, crop tightness and viewing angle into the trigger
along with the subject.

That is measured here, not theoretical. It produced collapse on whole-extent
prompts, and weakening the LoRA afterwards did not free the framing — match fell
sharply and framing did not improve, because framing had never been separable.

So captions name what should stay variable: framing, crop tightness, orientation,
lighting. Whatever is left unnamed is what the trigger absorbs.

Caption wording lives in the untracked overlay and the dataset directory, never
in a tracked file.

## Configuration

The tracked config carries the values. What matters about them:

| Setting | Why |
|---|---|
| `batch_size: 1`, `gradient_accumulation: 1` | makes `steps` equal **optimizer updates** |
| `quantize: true` | quantises the **frozen base** so it fits a 24 GB card; the LoRA still trains in bf16 |
| `gradient_checkpointing: true` | required at this size |
| `noise_scheduler: flowmatch` | must match the sampler used for intermediate samples |
| `save_every` | small enough to capture the curve, not just the end |
| sample prompts | include one at the framing the trigger was *not* trained on, so intermediate samples show whether framing became separable |

## Comparing arms

**At equal optimizer updates, not equal epochs.** `steps` counts updates, so
equal steps already means equal updates at batch size 1 — but changing the
dataset size changes epochs, and comparing at equal epochs would give the larger
dataset more updates.

**Sweep every checkpoint.** The over-fitting curve is measured against the
held-out split, not guessed at. Picking a step a priori has been wrong here:
capacity reductions chosen up front cost match without buying the framing
freedom they were supposed to buy.

**Use several fixed seeds, identical across arms.** Composition varies far more
by seed than by arm, and both match and structure metrics follow composition. A
single seed per arm produces a confident and meaningless winner — at one seed
one arm leads, at another a different one does.

**Change one thing at a time, and say when you did not.** A run that changes
captions and the base model together cannot attribute its own result.

## Arms

Structural descriptions only; scores belong in the bundles.

| Arm | Training material |
|---|---|
| anchor-only | tightly framed anchor-region crops |
| extent-only | whole-subject crops |
| combined | both scales, captioned to distinguish them |

## Failure modes already hit

- A trainer env that worked on one pod and failed on the next: **driver, not the
  volume**.
- Mismatched torch component builds after a partial downgrade.
- A wrapper that printed a success marker while the job had produced nothing,
  because its output was piped through a filter that discarded the error. **Log
  unfiltered; a script that reports done on a failed job is worse than a crash.**
- A rebuild that silently changed the held-out split.
- A rebuild that left the trigger unset — the LoRA then binds to nothing and the
  output is a competent non-target.

## Maintenance

Update this file when the trainer, the dataset steps, the split discipline or
the comparison method changes. Keep results out of it.
