# Pilot results

> **Status: RUN**, at 960x720 on rented 48 GB hardware.
> Filled in from measurement. The agent that ran it may not view the footage
> (rule 1), so perceptual rows are marked **needs human** rather than guessed;
> numeric rows are measured and reproducible. Where the user judged something by
> eye, that is attributed to them.

---

## Run metadata

| Field | Value |
|---|---|
| Date | 2026-08-05 |
| Config | `configs/cloud_720p_1p3b.yaml` (960x720, exact 4:3, no padding) |
| Models | `wan2.1_vace_1.3B_fp16` + SeedVR2 3B fp8 plate |
| Pilot segment | the five-second pilot interval, one shot, 1 x 81-frame chunk |
| Steps / cfg / sampler | 25 / 5.0 / uni_pc |
| Seed | 20260803 (single seed; no second-seed run) |
| Hardware | 48 GB card, 128 vCPU, 503 GB RAM |

## Measured results

### Plate: the profile choice dominates everything else

Gradient magnitude, normalised to the unrestored working stream. Named for what
it measures: ringing and noise raise it too.

| stream | gradient | vs source |
|---|---|---|
| source | 2.694 | - |
| conservative, old geometry, resampled up | 3.254 | +20.8% |
| conservative, native 960x720 | 3.285 | +21.9% |
| **aggressive, native 960x720** | **3.690** | **+36.9%** |

Native geometry is nearly free (1m26s vs 1m28s, same VRAM) because
`target_short_edge` is already 720 and the result was previously being resampled
to a 544 short edge and then scaled up again at delivery. But it recovered only
+1.1 points. **The aggressive profile is the real lever**, at +45 s.

User's judgement on the four-way: aggressive is clearly the best; the result
reads as "between 480p and 720p", plausible surface detail, **eyes unresolved**.

### VACE: measurably negative at 1.3B

Everything below shares one plate, mask, seed and geometry. Only conditioning
differs. Head submask = the 4.42% VACE regenerates.

| variant | gradient in head | identity (cosine to the bank) |
|---|---|---|
| **plate alone (no VACE)** | **1.618** | **0.2071** |
| source, unrestored | - | 0.1972 |
| VACE, no reference | 1.311 | 0.1897 |
| VACE, whole-photo sheet | 1.239 | 0.1717 |
| VACE, identity-only pack | 1.263 | 0.1742 |

**The plate alone wins on both metrics.** Every VACE variant is softer than what
it replaces and scores *lower* on identity than the untouched source.

Caveat: `identity.py` treats 0.35 as the threshold for a plausible match. Every
value here is far below it, so no variant resolves identity at all - these are
degrees of failure, not of success. The ordering is nonetheless consistent
across two independent metrics.

### The conditioning ablation

Mean absolute difference inside the head, out of 255:

| pair | difference |
|---|---|
| no-reference vs identity pack | 6.61 |
| no-reference vs whole-photo sheet | 6.58 |
| **whole-photo sheet vs identity pack** | **1.89** |

Removing the reference entirely moves the face by 6.6/255. Exchanging two
radically different sheets moves it by 1.89 — the model responded to a
reference's presence far more than to its content.

**Scope.** One shot, one seed, one model size. The defensible claim is that
reference-conditioned VACE 1.3B did not improve *this* pilot, not that better
references can never pay. Testing that properly needs more shots and seeds.

### Regenerating the garment is wrong under any conditioning

| garment region | gradient | vs plate |
|---|---|---|
| plate (ground truth) | 1.837 | - |
| full-subject, whole-photo sheet | 2.011 | 33.11 |
| full-subject, identity-only pack | 1.203 | 35.82 |

With the wardrobe left in the references the model invents confident, wrong
detail (user: "totally hallucinated"). With it segmented out the model has no
garment information at all and produces mush, 35% under the plate, and lands
*further* from the plate than before. The authority split was right: the garment
must come from the source, never be regenerated.

---

## Evaluation

| Criterion | Score | Notes |
|---|---|---|
| Facial identity | 1 | measured 0.17-0.21 cosine, below the 0.35 threshold, worse than the source |
| Non-facial identity | needs human | |
| Clothing accuracy | n/a on the retained path | garment never regenerated; 95.58% comes from the plate |
| Silhouette | n/a | plate-supplied |
| Temporal flicker | needs human | no frozen or dropped frames |
| Motion preservation | 4 | mask tracks motion, r=+0.72 against temporal variance |
| Background drift | 5 | `bg_preserved_exact` = 1.0000 after the `gbrp` fix |
| Mask-edge halos | needs human | 3 px centred band |
| Duration and audio sync | 5 | 81 frames at 16 fps, decoded and counted |

### The integration decision - settled, Path B

| | Path A (`in_vace`) | Path B (`composite`) |
|---|---|---|
| `bg_preserved_exact` | 0.034 | **1.0000** |
| gradient vs plate | -5.8% | -1.5% |
| edge cost in the 3 px band | none | 2.50 / 255 |

Path A degrades the whole frame through the VAE round trip to regenerate a few
percent of one figure. Path B's background is bit-exact and its only cost is a
three-pixel band.

---

## Verdict

- [x] **Ship the restored plate alone for this shot**
- [ ] Good enough - proceed to a full run with VACE
- [ ] Do production on the cloud 14B profile
- [ ] Approach is not working

The aggressive plate is +36.9% detail over the source, scores best on identity,
invents no garment, and costs 2.5 minutes against 10. Adding VACE at 1.3B makes
the face softer and less like the references, whatever it is conditioned on.

### What is genuinely untested

Only **model capacity**. Everything above is 1.3B VACE and 3B SeedVR2.

1. **SeedVR2 7B** - no training, no identity risk, targets the unresolved fine
   facial detail, and fits 48 GB (3B peaked at 12.7 GB). Do this first.
2. **VACE-14B** - now has a correctly built identity pack waiting for it.
3. **A subject LoRA** - the only route that would genuinely use the reference
   photographs, since conditioning demonstrably does not.

---

## Three defects found and fixed during this pilot

1. **The wrong reference builder was used throughout.** `prepare_references.py`
   (Phase 5) tiles whole photographs - environment, wardrobe, watermarks and
   all. `make_reference_pack.py` (Phase 5b) is the one implementing the
   authority split. `run_chunks.py` logs `Reference conditioning: global` or
   `pack`; that line is the check, and it read `global` for every early run.
2. **`composite_subject.py` wrote `yuv420p`**, destroying the RGB compositing on
   the way to disk. `bg_preserved_exact` was 0.034; with `gbrp` it is 1.0000.
   4:4:4 alone is not enough - 8-bit RGB->YUV is not reversible.
3. **The composite used the subject mask as alpha.** After a `--protected` run
   that takes the whole figure from VACE's VAE round trip, costing 8.0% of the
   garment's detail for nothing. The new `--mask` flag points it at the
   protected submask instead; garment detail returns to parity (1.850 vs 1.837).

## Notes

Tracking needed three attempts. The automatic seed locked onto architecture -
correlation with per-pixel temporal variance r=+0.03, the signature of a static
structure. A motion-seeded re-track scored r=+0.72. Two defects the user
identified by eye remain, in one unstable vertical strip; both were localised
by differencing those frames against the median of the others.

Tracking is deterministic on a given machine (IoU 100.00%, zero disagreeing
samples) but differs by 0.14% across machines, so a content-hash approval can
never survive a hardware change.
