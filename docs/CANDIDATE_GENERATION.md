# Candidate generation

How generated reference candidates are produced, trained and judged.

**Scope.** This is the *candidate* path only — it produces reference material.
It is not the restoration path and not the deliverable. The plate ships, takes no
prompt and no LoRA, and nothing here touches it. Nothing generated here is ever
promoted into `inputs/` automatically.

Rule 2a applies in full: this file is tracked, so it never names an input file,
never states how many the user supplied, never carries prompt text, and never
says what kind of thing the subject is. Role words throughout — subject, anchor,
candidate, extent.

---

## Generator

Still-image path through the pinned ComfyUI, driven over its HTTP API by
`intermediate/tools/gen_chroma.py`.

| Role | File | Note |
|---|---|---|
| DiT | `Chroma1-HD.safetensors` | Flux-derived, loaded fp8_e4m3fn |
| Text encoder | `t5xxl_fp8_e4m3fn.safetensors` | **T5-XXL only — Chroma has no CLIP-L** |
| VAE | `chroma_ae.safetensors` | Flux VAE |
| Subject LoRA | `chroma_subject_v*.safetensors` | trained here, see below |

Read from the installed source, not assumed (rule 6):

- `CLIPType.CHROMA = 15`; `CLIPLoader` accepts `type="chroma"`.
- Chroma's `clip_target` returns the **PixArt T5** tokenizer/encoder — one
  encoder, no pooled CLIP branch. Adding a second text encoder does nothing;
  there is no cross-attention path to receive it.
- Chroma's `latent_format` is **Flux**, so the graph uses
  **`EmptySD3LatentImage`** (16 channel). `EmptyLatentImage` is 4 channel and
  would produce noise.
- Flow shift uses **`ModelSamplingAuraFlow`** (multiplier 1.0 over CONST flow
  sampling), never `ModelSamplingFlux` — the latter derives shift from
  width/height, so it silently means something different at each resolution and
  cannot be held constant across an arm comparison.

```
configs/prompt.local.yaml (UNTRACKED)      lora_dataset/dataset.json
  positive | candidate_prompt                 trigger token
  candidate_grid.<key>                            |
  candidate_negative_chroma                       |
        └───────────────┬───────────────────────┘
                        ▼
                 gen_chroma.py   -> ComfyUI /prompt
                        │
  UNETLoader -> LoraLoaderModelOnly -> ModelSamplingAuraFlow ─┐
  CLIPLoader(type=chroma) ───────────────────────────────────┤-> KSampler
  EmptySD3LatentImage ───────────────────────────────────────┘     │
                                                       VAEDecode(Flux VAE)
                                                                   ▼
                                    ref_candidates_chroma/<variant>/*.png
                                          + *.provenance.json
```

## Conditioning text

No prompt text lives in a tracked file. Keys, and what each does:

| Key | Where | Effect if absent |
|---|---|---|
| `positive` / `candidate_prompt` | overlay | template with `{trigger}` and `{desc}` |
| `candidate_grid.<key>.<variant>` | overlay | one `{desc}` per grid cell |
| `candidate_negative_chroma` | overlay | falls back to the Wan negative, which is the wrong language for this encoder and near-inert |
| `trigger` | `dataset.json` | **the LoRA is inert** — output is not the subject |

`--positive-key` selects which overlay key supplies the template. `positive` is
the restoration profile's prompt and is **not** read here by default: the two
paths want different text, and borrowing one for the other silently conditions a
run on wording nobody chose.

**The trigger is load-bearing and fails loud.** A template lacking it produces a
competent image of a non-target — a failure that reads as a bad generation
rather than a configuration error, so it raises instead.

## Training

`ai-toolkit` in its own venv on the volume, `arch: chroma`, `quantize: true`
(the frozen base is quantised; the LoRA trains in bf16).

- Rank 16 / alpha 16, flowmatch, adamw8bit, EMA — see the tracked config.
- `batch_size: 1`, `gradient_accumulation: 1`, so **`steps` counts optimizer
  updates**. Arms are compared at equal updates, not equal epochs — changing the
  dataset size changes epochs but not updates.
- Checkpoints are saved throughout and **all of them are scored**. The
  over-fitting curve is measured, not guessed at; picking a step a priori has
  been wrong here before.
- Resolution buckets are capped to what the crops actually contain. Training
  above that upscales, which teaches the model to reproduce resampling
  artefacts.

**Its torch must match the pod's driver.** The volume's venv was once built
against a newer CUDA than a later pod's driver supported, and the same env on the
same volume then failed. Pin to the CUDA the fleet reliably has, and keep
`torch`, `torchvision` and `torchaudio` on the same build or the ABI mismatches.

## Dataset

Built by `make_lora_dataset.py`, then two additions:

1. `write_captions.py` — **captions name the framing.** A LoRA learns whatever
   the caption does not name, so a bare trigger on tight crops binds framing,
   crop tightness and viewing angle into the trigger token. That is measured,
   not theoretical: it produced full-extent collapse, and weakening the LoRA then
   destroyed match instead of freeing composition.
2. `add_body_crops.py` — adds whole-subject crops beside the anchor crops, so
   the model has information beyond the anchor region rather than inventing it.
   - Match is **inherited**: the crop is taken around a box containing an
     already-verified anchor, so no new trust is introduced.
   - Detection **retries at the lower threshold** when the default finds
     nothing. Without the retry most references yield no box at all.
   - Extent is captioned from **box-vs-frame geometry**. A box touching a frame
     edge means the subject continues past it, so the crop is captioned as
     partial. Calling such a crop full-extent teaches the model that the phrase
     means a cut-off body, poisoning the exact words needed at inference.

**The split is by source image, and pinned.** `pin_holdout.py` fixes the
held-out set; `assert_dataset.py` fails if any training crop derives from a
held-out source, or if any crop lacks a caption. Both matter because adding
references reshuffles a ranking-based split, which once moved a previously
trained image into the bank and would have made every recorded number
incomparable. The held-out bank is the measuring instrument: it must stay still
while the thing being measured changes.

## Judging

Two independent measurements. Neither alone is sufficient.

**Match** — `score_lora_match.py`, the bound anchor backend + a detector, scored
against the **held-out split only**. Scoring against training material returns a
high number by construction. Report **per candidate**, not group medians.

**Structure** — `body_structure.py`, because match is an anchor embedding and
cannot see a malformed extent. Keypoint completeness, extent symmetry, silhouette
continuity (largest connected component over total foreground — catches a
detached part that keypoints still find), and edge density. Writes a labelled
contact sheet to disk.

`edge_density_in_body` is **named for what it measures**. It is not "detail" and
not "quality": ringing and noise raise it too. High keypoint completeness means
the subject's structure is *detectable*, not well rendered — a soft but
correctly proportioned subject scores full marks.

**Seeds dominate arms.** Composition varies enormously by seed, and both metrics
follow composition. Use several fixed seeds, identical across arms, or a single
seed will yield a confident and meaningless winner.

## Repairing one region

`inpaint_chroma.py`, sequenced by `run_region_plan.sh`. For the case where an
image is right everywhere except one region, so regenerating the whole thing
throws away more than it fixes.

Crop the region with context, resample the crop **up** so the region occupies
more of the canvas, denoise only inside a mask, resample back, composite into the
untouched original. Enlarging the finished pixels cannot add structure that was
never drawn — the region has to be larger *while the sampler is running*. The
same argument drives the other half of `run_region_plan.sh`, a ladder of whole
canvases at a fixed aspect ratio, every rung a multiple of 16 in both axes so
nothing is silently reframed.

Wiring, and why it is this and not something simpler:

- Chroma is not an inpainting checkpoint, so there is no
  `InpaintModelConditioning` path; `SetLatentNoiseMask` over an ordinary
  `VAEEncode` is the available route. Its mask acts at **latent** resolution,
  8× coarser than the pixels, which is exactly why the blend back into the
  original happens in pixel space with its own feather rather than trusting the
  sampler's edge.
- **Two masks.** The sampler is allowed to repaint more than the composite blends
  back, so the blend's ramp lands inside repainted pixels rather than on the
  boundary of them. Asserted at run time, not assumed.
- Denoise ~0.30–0.40 corrects appearance, ~0.45–0.55 corrects structure.
- `ModelSamplingAuraFlow`, never `ModelSamplingFlux` — the latter derives its
  shift from width and height, so on a crop's canvas it means something different
  from what it meant on the full frame, and the repair stops being comparable
  with the thing it repairs.

Three things it measures instead of assuming:

| flag | what it settles |
|---|---|
| `--check-nodes` | reads the node schemas off the running server and fails if any wired class or input key is absent — rule 6 in the form available to a machine with no checkout of its own. A test keeps that declared list covering what the graph actually emits, or the preflight would pass and the run fail on an undeclared key. |
| `--verify-mask-polarity` | one image at denoise 1.0 with a white square, reporting the ratio of change inside to outside. Polarity has been assumed wrongly in this project before. |
| `--control` | crop, resample up, resample down, composite — **no diffusion**. Whatever this arm changes, resampling changed. |

Every arm then asserts that not one pixel outside the mask moved and that
something inside it did. An exit code proves a run happened, not that it did
anything.

Sampler settings default to the **source image's own record** where it has one,
because numbers are comparable only within one baseline; the run prints whether
each value came from a flag, from that record, or from a tool default.

### Where the region is — and why there is one box per image

Read from untracked `intermediate/defect_region.txt`, in descending order of how
much is actually known: an explicit box on the command line, a box measured for
**that specific image**, a generic box, then a square grown around a bare point.
The run prints which it used, because a result from a measured box is a stronger
claim than one from a guess and nothing downstream can tell them apart otherwise.

**A box does not transfer between images, and assuming it does is a real error
rather than a pedantic one.** Composition varies enormously by seed on this
generator — the spread across seeds within one arm measured roughly nine times
the spread between arms — so the same coordinates land somewhere else on a
differently framed subject. Measured per image, the boxes for three sibling
candidates differed by up to 68 px in one axis and by a third in size, on a
region about 60 px tall. One shared box would have been wrong on two of the
three, and every downstream check would still have passed.

So the key is the **arm directory name**: images live one directory per arm and
the boxes are recorded per arm, so the two agree without a mapping table to fall
out of date. `mark_region.py` exists for the reading-off step — it burns a
labelled coordinate grid and the currently-recorded box onto a copy of each
image, magnified with NEAREST so the resampler invents no edge, which turns
"describe where it is" into four checkable numbers.

Context is sized as a fraction of the box **per side**, and a small region needs
proportionally *more* context, not less: at 18% a 30×50 region yields a 40×68
crop, which is a border rather than something to reattach to. The default of 1.5
makes the crop 4× the box in each axis, and the run warns below roughly 48
original pixels on the short side, where the crop is mostly interpolation and the
magnification stops buying structure and starts buying invention.

### What exists, and what was lost

The Chroma tools are untracked in `intermediate/tools/`, and travel by state
bundle rather than by git. That has already cost something: **`gen_key.py`,
`body_structure.py`, `assert_dataset.py` and `compare_arms.sh` no longer exist**
— not in the worktree, and not on the volume, which was checked directly. This
document still describes what they did because the design decisions are worth
keeping; it does not describe files you can run.

What replaces them, where anything does: `inpaint_chroma.py` writes its own
content-addressed record on the same before-the-run contract, and
`run_region_plan.sh bundle` copies `*.provenance.json` beside the images, which
is the specific defect that lost bundle 13's per-image provenance. Nothing
replaces the structure metrics, and on the evidence of the region defect nothing
should without being validated against images already known to be bad first.

## Provenance

`gen_key.py` builds a content-addressed key from everything reaching the
sampler: checkpoint, VAE, text encoder, LoRA **contents**, LoRA strength, prompt
digests, seed, dimensions, steps, cfg, sampler, scheduler, flow shift, dataset
digest. Any input change yields a new key, so an older candidate can never be
reattributed to a newer configuration — which seed-only resume detection allowed.

The record is written **before** the run and travels beside the image as
`*.provenance.json`. Recording it afterwards is how stale pixels get marked
current. **Prompt text is hashed, never stored**, which is what makes the record
safe to keep.

## Comparability

Numbers are only comparable **within one baseline**. Resolution, cfg, flow shift
and sampler all move the absolute values, so the same LoRA reads differently
across bundles. Always compare arms measured under identical settings, and state
the baseline beside the number.

## Ops

`autokill.sh` runs on the pod and deletes the pod: an idle timeout plus a hard
lifetime cap, since a hung process can pin the GPU and defeat an idle-only check.
`wd_restart.sh` verifies the pod id against the API and **refuses to start**
otherwise — a watchdog pointed at the wrong id reports healthy and never fires.

---

## Maintenance

Update this file whenever any of the above changes — a model, a node, a
threshold, a metric, a guard. A stale architecture doc is worse than none,
because it is believed. Keep results out of it: those belong in the review
bundles, indexed in `outputs/INDEX.md`.
