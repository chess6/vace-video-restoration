# Runtime estimates — measured on this machine

Generated: 2026-08-03. Source: two real generations at the production shape
(832×480, 81 frames) plus `reports/benchmark.json`.

## Measured

| Quantity | Value |
|---|---|
| Benchmark job | 832×480 × 81 frames, 25 steps, uni_pc, VACE 1.3B fp16 |
| Wall clock | **16 m 19 s** per 81-frame chunk |
| Per generated frame | **12.10 s** |
| Peak VRAM | **11 854 MiB** of 12 288 MiB |
| GPU utilisation during sampling | 100 % (compute-bound, not I/O-bound) |

Two datapoints at the same shape (4 steps = 200.3 s, 25 steps = 979 s) separate
the fixed and marginal costs cleanly:

- **fixed cost per chunk: ~52 s** — VAE encode of the control video, text
  conditioning, VAE decode of the result
- **marginal cost per step: ~37.1 s**

## Step count vs total runtime

Worked for a **30-minute reference workload** at 16 fps = 28 800 frames; scale
linearly for any other duration. With 81-frame chunks and 8 frames
of overlap that is **395 chunks** and 31 995 generated frames (11 % overlap
overhead).

| Steps | Per chunk | s / frame | Reference workload |
|---:|---:|---:|---:|
| 10 | 7.0 min | 5.22 | **46 h** |
| 15 | 10.1 min | 7.51 | **67 h** |
| 20 | 13.2 min | 9.80 | **87 h** |
| **25 (baseline)** | **16.3 min** | **12.09** | **107 h** |
| 30 | 19.4 min | 14.38 | **128 h** |

## What this means

**At that scale it is a multi-day job on this GPU: about 4.5 days of continuous
compute at the baseline 25 steps.** That is not a configuration
mistake, and it is not fixable by memory tuning — the GPU sits at 100 %
utilisation throughout, so it is genuinely compute-bound.

The cost is inherent to the shape of the work. At 832×480×81 frames the Wan
latent is 21 × 30 × 52 ≈ 33 000 tokens after patchifying, and attention is
quadratic in that. Classifier-free guidance at cfg 5.0 doubles it again by
running conditional and unconditional passes every step.

### Options, in the order worth trying

1. **Judge the pilot first.** Everything above is about throughput, not quality.
   If the 1.3B model at 480p does not preserve match convincingly, runtime is
   irrelevant. Run the pilot and fill in `reports/pilot_results.md`.

2. **Drop steps to 15–20** and compare against the 25-step pilot on the same
   seed. uni_pc converges quickly; if 15 steps is visually indistinguishable
   that is a 40 % saving for free. This is a legitimate baseline tuning, not an
   acceleration hack.

3. **Do production on the cloud 14B profile.** This is what
   `configs/cloud_14b.yaml` exists for, and the numbers above are the strongest
   argument for it: the same manifest, references, masks and depth videos carry
   over unchanged.

   What is **not** established is which rented GPU it needs. The fp16 14B
   checkpoint is ~34.7 GB on its own, and the official VACE example presents
   14B/720p across eight GPUs, so "one 48 GB card is enough" is a hypothesis to
   test, not a plan to budget against. Rent by the hour, run
   `scripts/benchmark.py` and one pilot chunk first, and only then size the job.
   Every subject in this document was measured on the 1.3B/480p profile on this
   machine; none of it transfers to 14B/720p by arithmetic.

4. **Only then** consider acceleration (CausVid / TeaCache / fp8). Deliberately
   excluded from the baseline, per the brief, until the baseline output is known
   to be correct.

### Reducing scope is also legitimate

395 chunks assumes restoring the whole reference workload. If only the shots where the main
subject is prominent actually matter, `intermediate/chunk_manifest.json` can be
filtered and the rest passed through untouched — `assemble.py` already fills
unrestored ranges from the normalized source and preserves total duration.

## Disk

Disk is not a constraint here: **590 GiB free**, against an estimated 10–25 GiB
of intermediates and outputs for the full job. The per-frame constants used in
`scripts/benchmark.py` were measured on this machine and rounded up.
