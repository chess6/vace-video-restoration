# Cloud runbook

Moving the pilot to rented GPUs. Two stages: get the 1.3B architecture proven
correct on a 24 GB card, then swap to an 80 GB card for one 14B pass at 720p.

Rule 2a normally forbids the user's material reaching any remote. The user has
authorised a transfer to their own rented box, scoped to the list in
"What may be transferred". That authorisation covers nothing else, and it is
recorded here rather than remembered.

---

## Connecting

RunPod's SSH proxy has two quirks that make an ordinary `ssh host 'command'`
fail in confusing ways:

1. **It refuses a session without a PTY.** `ssh` only allocates one when it has
   a real terminal, which an agent or a script does not. Wrap it: `script -qec
   "ssh runpod" /dev/null`.
2. **It starts an interactive login shell and ignores a remote command passed as
   an argument.** The command must arrive on **stdin**.

Also required in `~/.ssh/config`: `IdentitiesOnly yes`. Without it `ssh` offers
every key in the agent, and the proxy closes the connection before reaching the
right one — which presents as `Permission denied (publickey)` even when the key
is correctly registered.

`.pub` is the half that goes into a web form. A private key is many lines and
begins `-----BEGIN OPENSSH PRIVATE KEY-----`; if a field takes a multi-line blob
for an SSH key, it is the wrong field. A private key pasted anywhere is burnt —
generate a new pair rather than reusing it.

## Stage 1 — 24 GB card, 1.3B

Surveyed: RTX 4090 24 GB, 32 cores, 124 GB RAM, `/workspace` on a network
volume, Python 3.12.3 (so `requirements.lock.txt` applies unchanged).

1. Clone into `/workspace`. Done at the commit recorded in the transfer receipt.
2. `scripts/bootstrap.sh` — pinned ComfyUI checkout, pinned environment.
3. `scripts/download_models.sh` — 1.3B weights **directly onto the volume**, not
   through the local machine.
4. **Benchmark one unchanged 1.3B run** before any correction, so the cloud box
   has a baseline of its own rather than inheriting a 3060's numbers.

### What may be transferred

Only these. Everything else stays local.

| | why it is needed |
|---|---|
| the exact five-second pilot interval | the only footage being restored |
| identity-verified references | conditioning; excludes any image on the run's exclusion list |
| subject masks, depth, source-derived controls | regenerating them costs GPU and must reproduce bit-exactly |
| the SeedVR2 conservative plate | reusable, expensive, and unchanged by any of the fixes |

Not transferred: the full source video, unverified references, anything derived
from the wrong-person track.

### Gates before spending a generation

All of these are enforced in code, not by memory:

- the run's exclusion list is present and non-empty (untracked, run-local);
- the tracking overlay has been reviewed by a human and approved —
  `scripts/approve_tracking.py`, bound to the mask's content hash;
- `subject_status == needs_user` blocks generation outright;
- clothing and any face covering are protected — `make_protected_mask.py` runs
  before the reference pack, because the pack's face gating depends on its
  verdict and fails closed without it.

Verify with `scripts/run_chunks.py --pilot --protected --dry-run`, which runs
every check and stops before touching the GPU. **Use it.** Confirming a guard by
starting a real generation and watching what happens wastes ~18 minutes and is
how a guard gets confirmed in the wrong direction.

### Order of generation

1. **Corrected full-frame variant only.** Inspect the tracking overlay, the face
   covering, sleeves, torso coverage and garment structure.
2. **ROI only if that passes.** `--protected` refuses to combine with `--roi`:
   the submask is derived at full-frame geometry and re-warping it is not
   something to approximate.

## Stage 2 — 80 GB card, 14B at 720p

Terminate the 24 GB pod first; keep the volume. Attach it to the 80 GB pod, then
`scripts/download_models.sh --config configs/cloud_14b.yaml`.

`configs/cloud_14b.yaml` is complete but **entirely unverified** — every value is
a local setting scaled by resolution and VRAM, and nothing in it has been run.
Two entries there are decisions rather than scaling, and are explained in the
file: SeedVR2 stays on 3B, and ROI is off.

**Re-measure the protected-regenerable fraction at 720p before deciding whether
VACE is worth running at all.** On the 240p source it was 1.57% of the tracked
figure — about a 40x40 patch, with the plate supplying everything else. That
number is resolution-dependent: the parser is only confident where it has pixels,
and at 720p the figure gains roughly 2.4x linear resolution. If it stays low, the
correct output for the shot is the restored plate alone, and no amount of GPU
changes that.

## Teardown

Terminate the GPU pod **immediately** after copying outputs back — it bills by
the second and an idle box after a run is pure waste. Keep the network volume
only if more runs are planned; it bills separately and holds the weights, so
deleting it means re-downloading them.

Copy back: generated variants, metrics, reports, review bundle. Not the weights.
