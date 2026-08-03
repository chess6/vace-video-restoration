# Pilot results

> **Status: NOT YET RUN.** This is the scoring template. It will be filled in
> after the first pilot on real footage. A pilot cannot be run until the source
> video and reference images exist — see `NEEDS_USER.md`.
>
> Do not treat a clean exit code as success. The only thing that matters here is
> what the output actually looks like.

---

## Run metadata

| Field | Value |
|---|---|
| Date | _to fill_ |
| Config | `configs/local_1p3b.yaml` |
| Model | `wan2.1_vace_1.3B_fp16.safetensors` |
| Resolution | _to fill_ |
| Pilot segment | _source seconds, frames_ |
| Chunk(s) | _to fill_ |
| Steps / cfg / sampler | 25 / 5.0 / uni_pc |
| Seeds compared | _primary, secondary_ |
| Seconds per generated frame | _from reports/benchmark.json_ |
| Peak VRAM | _to fill_ |

## Variants generated

| # | Variant | Command | Output |
|---|---|---|---|
| 1 | depth + reference + subject mask | `run_chunks.py --pilot` | `outputs/restored_480p/<chunk>.mp4` |
| 2 | no reference conditioning (ablation) | `run_chunks.py --pilot --no-reference --tag noref` | `…_noref.mp4` |
| 3 | reference-conditioned, second seed | `run_chunks.py --pilot --seed 987654 --tag seedB` | `…_seedB.mp4` |

Comparison artefacts: `outputs/comparisons/<chunk>_side_by_side.mp4`,
`outputs/comparisons/<chunk>_frame_grid.png`.

---

## Evaluation

Score each 1–5 (1 = unusable, 3 = acceptable, 5 = excellent). Judge by watching
the side-by-side, not by reading logs.

| Criterion | Score | Notes |
|---|---|---|
| **Facial identity** — is it recognisably the same person? | | |
| **Non-facial identity** — hair, build, posture, gait | | |
| **Clothing accuracy** — garments, colours, patterns | | |
| **Accessory accuracy** — bags, jewellery, glasses, logos | | |
| **Silhouette and body proportions** | | |
| **Temporal flicker** — does detail crawl or pop between frames? | | |
| **Motion preservation** — does it follow the original motion? | | |
| **Background drift** — is the unmasked scene actually unchanged? | | |
| **Mask-edge halos** — visible seam around the figure? | | |
| **Duplicated limbs / fingers** | | |
| **Invented detail** — plausible but wrong texture or objects | | |
| **Duration and audio sync** | | |

### Specific questions to answer

1. **Does the reference sheet help?** Compare variant 1 against variant 2. If they
   are indistinguishable, native reference conditioning is not contributing and a
   subject LoRA becomes worth considering. If variant 2 drifts in identity or
   clothing while variant 1 holds, the reference is doing real work.

2. **How seed-dependent is it?** Compare variant 1 against variant 3. Large
   differences mean the conditioning is too weak to pin the result down; consider
   raising `vace_strength` or improving the reference sheet before anything else.

3. **Is the background genuinely preserved?** Look at the unmasked region in the
   side-by-side. Some change is expected and unavoidable — the VAE round-trip
   alone shifts pixels slightly (measured at ~4.6/255 mean absolute in
   `reports/mask_polarity.json`). Structural change is not expected and means the
   mask is wrong.

4. **Does the mask cover the whole figure?** Halos or a "floating head" effect
   mean the mask is too tight. Increase `mask.grow` in the config.

---

## Verdict

- [ ] **Good enough — proceed to a full run** (`scripts/run_full.sh --confirm-full-run`)
- [ ] **Good enough locally, but do production on the cloud 14B profile**
- [ ] **Needs tuning first** — record what to change below
- [ ] **Approach is not working** — record why

### If tuning: what to change

| Symptom | First thing to try |
|---|---|
| Identity drifts from the reference | better/sharper full-body reference; raise `vace_strength` |
| Background changes where it should not | check the mask review sheets; the mask is probably inverted or too loose |
| Halo around the figure | raise `mask.feather`, then `mask.grow` |
| Figure looks stiff or detached from the scene | lower `vace_strength` slightly; verify depth is aligned |
| Flicker between chunks | raise `video.chunk_overlap` |
| Too slow | drop `sampling.steps` to 15–20 and re-judge; only then consider an acceleration profile |

---

## Notes
_free text_
