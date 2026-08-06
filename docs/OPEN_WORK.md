# Open work

Where things stand and what is waiting on a decision. Task status, not durable
memory — `docs/STATE.md` holds what is expensive to rediscover, this holds what
is merely unfinished. Delete an entry when it is done rather than marking it.

Rule 2a applies: tracked, so nothing here names or describes the material.

---

## Built this session, not yet run on a GPU

Nothing below has been committed, and no workflow has been rebuilt (that needs
the box, since `build_workflows.py` reads `/object_info` from a running ComfyUI).

**The LoRA slot became a stack.** `model.lora` takes a list; loaders chain; every
entry is bind-checked separately before any GPU time; contents are hashed into
`vace_key`. A behaviour LoRA no longer evicts the subject LoRA — that was the
real defect behind "swapping the LoRA is a two-field change", which it was not.
Third-party LoRA names live in the untracked overlay under `loras:` and are
patched in at run time, so tracked configs and `workflows/*.json` never carry
them. `configs/cloud_720p_1p3b_lora_stack.yaml`, `scripts/test_lora_stack.py`.

**Checkpoint swapping became checkable.** `scripts/verify_checkpoint.py` answers
"is this UNet a drop-in" from the safetensors header alone — milliseconds on a
34 GB file, no torch — by diffing the candidate against the checkpoint it would
replace. `scripts/graft_vace.py` transplants VACE's control scopes onto a
community fine-tune of the same base, deriving the scopes rather than naming
them. `run_chunks.py` now records the checkpoint's digest per result and
regenerates a chunk whose digest changed; records written before this exists
carry none and are left alone, so nothing already finished went stale.
`docs/MODEL_SWAP.md` has the four options and what each costs.
`scripts/test_checkpoint_swap.py`.

---

## Waiting on a decision

**A behaviour LoRA, if the base model turns out to be the limit.** Nothing is
downloaded; which file to trust is a judgement call, not a script's. Steps are in
`NEEDS_USER.md` under Optional. Before spending on it: SeedVR2 takes no prompt
and no LoRA, so the plate — which is what ships — cannot be affected by this at
all, and VACE already measures below a plain upscale inside the region it
repaints.

**The reference-pack scope.** `grey_candidate_attributes.py` defaults to keeping
the exposed regions as well as the anchor; `make_reference_pack.py` still builds
panels with the narrow `MATCH_ONLY` set, so the panel that reaches VACE is the
anchor region and grey. Widening it to "everything that is not an attribute" is
your stated rule, and it is a conditioning change: per `docs/STATE.md` it must be
reported as match gain and attribute fidelity separately, and the covered-anchor
safety (the `allowed` intersection) has to survive it.

**Whether to widen the LoRA's training crops.** `make_lora_dataset.py` crops at
`--margin 1.9`, so every training image is a tight anchor crop and the trigger
carries that framing. This is why extent-grid candidates come out tighter and
score lower than asked. A wider or mixed dataset is the only thing that makes a
generated extent the subject's proportions rather than the base model's
invention — and it only works if the references contain that extent at all,
which is the binding constraint. Cost: a retrain.

**A real full-extent reference beats all of the above** and is already ranked
first in `NEEDS_USER.md`. A generated extent is invented outside the anchor and
is greyed before anything sees it, so it can never donate attributes.

---

## Queued for me, needs the box up

1. Re-run the extent grid at reduced LoRA strength (0.6–0.7) with more seeds, and
   select on best rather than median — the single best candidate measured so far
   is already an extent one. Cheap; no code change, the generator now takes
   per-entry multipliers.
2. Emit per-image scores from `score_lora_match.py`. The scores artefact records
   only per-group median/best/worst, so the best individual candidates cannot be
   picked from it.
3. Rebuild workflows and run `verify_lora_loads.py` / `verify_checkpoint.py`
   against the real weights. Everything above was tested against fixtures.

---

## Guard gap worth closing

`check_repo_clean.sh` scans `git ls-files`, so a newly created file is not
covered by checks 3, 6, 6b or 7 until it is staged — its PASSED verdict says
nothing about untracked work in progress. `git add -N` before the scan would fix
it, either in the script or as a habit.
