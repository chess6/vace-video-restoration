# Swapping the UNet, without leaving Wan

What the options actually are for replacing `model.diffusion_model` with a
community fine-tune, what each costs against this repo's pins, and which claims
here are verified rather than assumed.

Read `docs/STATE.md` first. The short version of why this is a small prize: the
plate carries the measured quality and ships, VACE regenerates ~4.4% of the
subject under a control video that pins every pixel, and in those pixels it
already measures below a plain Lanczos upscale. A better VACE backbone moves
that 4.4%. Nothing here changes the plate, which takes no prompt and no LoRA.

---

## The constraint everything else follows from

The graph feeds `WanVaceToVideo`. That node's conditioning — control video, mask,
reference image — only exists if the checkpoint carries VACE's control scopes. A
plain T2V checkpoint does not, and swapping to one does not weaken conditioning,
it **removes** it, while still producing frames and exiting 0.

So the first question about any candidate is not "is it good", it is "does it
carry the scopes". `scripts/verify_checkpoint.py` answers that from the
safetensors header in milliseconds, by diffing the candidate against the
checkpoint it would replace — no assumption about what the scopes are called.

---

## Option 1 — another ready-made VACE checkpoint

One field: `model.diffusion_model`. Genuinely zero graph work.

| | |
|---|---|
| Cost | a download; no code change |
| Verified here | VACE-14B was tested and is **worse** — in-mask 7.2 vs 1.3B's 8.0 |
| Catch | at 1.3B there is very little on offer that is not the official weights, a re-quantisation of them, or the ali-vilab preview |
| Excluded | GGUF quantisations need a loader from a custom node pack, and this repo installs none (`reports/versions.md`) |

The one notable 1.3B community VACE checkpoint,
[`lym00/Wan2.1_T2V_1.3B_SelfForcing_VACE`](https://huggingface.co/lym00/Wan2.1_T2V_1.3B_SelfForcing_VACE),
is a Self-Forcing backbone with VACE scopes injected. Its own card warns that
Self-Forcing/CausVid sampling is not natively implemented in ComfyUI, so this
project's `steps`/`cfg`/`sampler` settings would not be the right ones for it —
which makes it a sampler experiment as well as a checkpoint swap.

## Option 2 — graft the scopes onto a community fine-tune

This is the method behind the phrase "uncensored VACE variant". Such checkpoints
mostly do not exist ready-made; what exists is **T2V** fine-tunes, and the VACE
scopes get transplanted onto them. Both published examples work with the native
`WanVaceToVideo` node:

- [`QuantStack/Wan2.1_T2V_14B_FusionX_VACE`](https://huggingface.co/QuantStack/Wan2.1_T2V_14B_FusionX_VACE)
  — VACE-14B scopes injected into the FusionX T2V fine-tune
- [`lym00/Wan2.1_T2V_1.3B_SelfForcing_VACE`](https://huggingface.co/lym00/Wan2.1_T2V_1.3B_SelfForcing_VACE)
  — the same at 1.3B

`scripts/graft_vace.py` does it here, deriving the scopes by diffing VACE against
the plain T2V checkpoint it was built from rather than naming them:

```bash
scripts/graft_vace.py --vace wan2.1_vace_1.3B_fp16.safetensors \
    --base wan2.1_t2v_1.3B_bf16.safetensors \
    --target <fine-tune>.safetensors --out <name>.safetensors --dry-run
```

It refuses a target that is not a fine-tune of `--base` (missing or reshaped
backbone tensors), verifies what it wrote against the plan, and writes a
`.provenance.json` beside the output — a grafted checkpoint has no model card,
and its digest is the only thing that identifies it afterwards.

**What a graft does not promise.** The backbone and the scopes were trained
together and after a graft they were not: the scopes steer features that have
moved. The published merges work, which is evidence it holds, not evidence of
how well. Measure in-mask against the plate.

## Option 3 — Wan2.2 Fun-VACE

[`alibaba-pai/Wan2.2-VACE-Fun-A14B`](https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B)
(Apache-2.0), repackaged by Comfy-Org as a **high-noise / low-noise pair**:
34.7 GB each in bf16, 17.3 GB each fp8_scaled.

This is a port, not a swap. It needs a ComfyUI newer than the pin, two model
loaders and Wan2.2's two-stage sampling in the graph, roughly double the disk and
VRAM, and every constraint in CLAUDE.md rule 6 re-read from the new revision.

**There is no Wan2.1 VACE-Fun.** The Fun line at 2.1 is
`Wan2.1-Fun-*-Control/InP`, which are their own control models rather than VACE,
and VACE-Fun appears at 2.2.

## Option 4 — a LoRA instead

Cheapest, already wired, and the stack now holds one *alongside* the subject
LoRA. See `configs/cloud_720p_1p3b_lora_stack.yaml`. Prefer this until a
checkpoint swap has a measured reason.

---

## Before and after any swap

```bash
scripts/verify_checkpoint.py <candidate> --base wan2.1_t2v_1.3B_bf16.safetensors
scripts/verify_checkpoint.py <candidate> --digest      # which weights, not which filename
```

`run_chunks.py` records the checkpoint's digest on every result and regenerates a
chunk whose recorded digest no longer matches, so a file replaced under the same
name can no longer be mistaken for the weights that produced an earlier result.
A result recorded before this existed carries no digest and is left alone.

Rule 7 applies to anything downloaded: `scripts/record_versions.sh` writes its
size and SHA256 into `reports/versions.md`. Licences travel with the weights —
the community fine-tunes above inherit theirs from the models they were built on.
