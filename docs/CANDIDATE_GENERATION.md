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

**Match** — `score_lora_match.py`, insightface + Grounding DINO, scored
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
