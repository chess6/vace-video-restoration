#!/usr/bin/env python
"""Put VACE's control scopes onto a different Wan backbone of the same size.

WHY THIS EXISTS
The pipeline's checkpoint is VACE, and VACE is the only thing that accepts the
control video, the mask and the reference sheet. So "swap the UNet for a
community fine-tune" has a problem: community fine-tunes of Wan2.1 are almost
all T2V, and a T2V checkpoint has no control scopes. Swapping to one does not
weaken the conditioning, it removes it - and removes it silently, which is the
failure this repo keeps designing against.

The way out is the same observation the LoRA path rests on: a VACE checkpoint is
a T2V backbone plus tensors the T2V one does not have. So the backbone can come
from the fine-tune and the added tensors from official VACE, and the result is a
checkpoint the native WanVaceToVideo node drives exactly as before. Published
merges do precisely this at both sizes; docs/MODEL_SWAP.md names them, and the
other three ways to change the model, with what each costs here.

WHAT IS AND IS NOT PROMISED
This is weight surgery, not training. The backbone and the control scopes were
trained together, and after a graft they were not: the scopes are being asked to
steer features that have moved. Expect the control to hold - the published
merges work - and expect no guarantee about how well. That is a measurement,
in-mask against the plate, not an assumption.

The scopes are DERIVED, not named: they are the tensors `--vace` has that
`--base` does not (rule 6 - read it from the files, do not recall it). Nothing
here knows or states what they are called.

    scripts/graft_vace.py --vace VACE.safetensors --base T2V.safetensors \\
                          --target FINETUNE.safetensors --out NEW.safetensors
    scripts/graft_vace.py ... --dry-run      # plan and refusals only, no write

--base must be the checkpoint --vace was built from, and --target must be a
fine-tune of that same base. Everything else is refused rather than blended.
Memory: the output is assembled in RAM, so a 1.3B graft needs a few GB and a 14B
one needs tens.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, file_digest, safetensors_index, setup_logging  # noqa: E402


def plan(vace: dict, base: dict, target: dict) -> dict:
    """Decide where every tensor of the output comes from.

    Pure - indices in, plan out - so the decision that matters can be tested
    without four checkpoints on disk.

    The output's key set is exactly VACE's, because that is what makes it a
    drop-in. Two ways that cannot be satisfied are refusals, not warnings:
      missing  - the target lacks a backbone tensor VACE needs
      reshaped - the target has it at a different shape, i.e. it is a fine-tune
                 of something else, or of a different size
    """
    scopes = sorted(set(vace) - set(base))
    backbone = sorted(set(vace) & set(base))
    missing = [k for k in backbone if k not in target]
    reshaped = [k for k in backbone
                if k in target
                and list(target[k].get("shape", [])) != list(vace[k].get("shape", []))]
    # A scope the target ALSO carries is left to VACE anyway: the target is a
    # T2V fine-tune, so if it has one it is coincidence or a previous graft, and
    # in both cases the control tensors should come from the model that trained
    # them together.
    return {
        "from_target": [k for k in backbone if k in target],
        "from_vace": scopes,
        "missing": missing,
        "reshaped": reshaped,
        "target_only": sorted(set(target) - set(vace)),
    }


def dtype_summary(index: dict, keys) -> dict[str, int]:
    out: dict[str, int] = {}
    for k in keys:
        d = index[k].get("dtype", "?")
        out[d] = out.get(d, 0) + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vace", type=Path, required=True,
                    help="Official VACE checkpoint: the source of the control scopes")
    ap.add_argument("--base", type=Path, required=True,
                    help="The plain T2V checkpoint --vace was built from")
    ap.add_argument("--target", type=Path, required=True,
                    help="The community fine-tune to graft onto")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the plan and any refusal; write nothing")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("graft_vace", args.verbose)
    for p in (args.vace, args.base, args.target):
        if not p.exists():
            log.error("No such checkpoint: %s", p)
            return 1
    if not args.dry_run and args.out is None:
        log.error("--out is required unless --dry-run")
        return 1
    if args.out is not None and args.out.exists():
        log.error("%s exists. Refusing to overwrite a checkpoint.", args.out)
        return 1

    try:
        idx = {n: safetensors_index(p) for n, p in
               (("vace", args.vace), ("base", args.base), ("target", args.target))}
    except (OSError, ValueError) as e:
        log.error("%s", e)
        return 1

    p = plan(idx["vace"], idx["base"], idx["target"])
    log.info("%s: %d tensors | %s: %d | %s: %d", args.vace.name, len(idx["vace"]),
             args.base.name, len(idx["base"]), args.target.name, len(idx["target"]))
    log.info("control scopes VACE adds to its base: %d", len(p["from_vace"]))
    log.info("backbone tensors to take from the fine-tune: %d", len(p["from_target"]))
    if p["target_only"]:
        log.info("%d tensor(s) exist only in the fine-tune and are dropped: the "
                 "output's key set must be VACE's to be a drop-in.",
                 len(p["target_only"]))
    if not p["from_vace"]:
        log.error("%s adds nothing to %s. Either --vace is not the VACE variant "
                  "or --base is not the checkpoint it was built from; grafting "
                  "would just copy the fine-tune.", args.vace.name, args.base.name)
        return 1
    if p["missing"] or p["reshaped"]:
        log.error("%s is not a fine-tune of %s.", args.target.name, args.base.name)
        for k in p["missing"][:10]:
            log.error("  absent from the fine-tune: %s", k)
        for k in p["reshaped"][:10]:
            log.error("  reshaped: %s target %s vs vace %s", k,
                      idx["target"][k].get("shape"), idx["vace"][k].get("shape"))
        log.error("Grafting anyway would produce a checkpoint that loads and is "
                  "not either model. Refusing.")
        return 1

    d_scope = dtype_summary(idx["vace"], p["from_vace"])
    d_back = dtype_summary(idx["target"], p["from_target"])
    log.info("dtypes - scopes: %s | backbone: %s",
             dict(sorted(d_scope.items())), dict(sorted(d_back.items())))
    if set(d_scope) != set(d_back):
        log.warning("The two halves are not the same dtype. safetensors allows "
                    "it and ComfyUI casts, but the result is a mixed-precision "
                    "file - record which half is which before comparing runs.")

    if args.dry_run:
        log.info("DRY RUN: nothing was written. %d tensor(s) would come from the "
                 "fine-tune and %d from VACE.", len(p["from_target"]), len(p["from_vace"]))
        return 0

    # Torch and safetensors only from here: --dry-run must stay runnable on a
    # machine with neither, since deciding whether a graft is even possible is a
    # question about headers.
    from safetensors import safe_open
    from safetensors.torch import save_file

    log.info("assembling %s in memory", args.out.name)
    tensors = {}
    with safe_open(str(args.target), framework="pt") as f:
        for k in p["from_target"]:
            tensors[k] = f.get_tensor(k)
    with safe_open(str(args.vace), framework="pt") as f:
        for k in p["from_vace"]:
            tensors[k] = f.get_tensor(k)
    if set(tensors) != set(idx["vace"]):
        log.error("Assembled %d tensors against VACE's %d. Refusing to write.",
                  len(tensors), len(idx["vace"]))
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    del tensors

    # Rule 4: it wrote a file, which is not the same as having written the right
    # one. Re-read the header and check the result against what was intended,
    # then byte-compare a sample of tensors from each half against their source.
    out_idx = safetensors_index(args.out)
    intended = {k: idx["vace"][k] for k in p["from_vace"]}
    intended.update({k: idx["target"][k] for k in p["from_target"]})
    bad = [k for k in intended
           if k not in out_idx
           or list(out_idx[k].get("shape", [])) != list(intended[k].get("shape", []))]
    if bad or set(out_idx) != set(intended):
        log.error("The written file does not match the plan (%d bad, %d vs %d "
                  "tensors).", len(bad), len(out_idx), len(intended))
        return 1

    def sample(keys: list[str]) -> list[str]:
        return [keys[i] for i in {0, len(keys) // 2, len(keys) - 1}] if keys else []

    with safe_open(str(args.out), framework="pt") as fo:
        for src, keys in ((args.vace, sample(p["from_vace"])),
                          (args.target, sample(p["from_target"]))):
            with safe_open(str(src), framework="pt") as fs:
                for k in keys:
                    if not fo.get_tensor(k).equal(fs.get_tensor(k)):
                        log.error("%s does not match its source in %s.", k, src.name)
                        return 1
    log.info("verified: key set matches VACE's, sampled tensors match their sources")

    # Provenance beside the file, because a grafted checkpoint has none of its
    # own: it is not a download with a model card, and reports/versions.md can
    # only record its digest. This is what says which three files it came from.
    meta = args.out.with_suffix(".provenance.json")
    meta.write_text(json.dumps({
        "output": args.out.name,
        "output_digest": file_digest(args.out),
        "scopes_from": {"file": args.vace.name, "digest": file_digest(args.vace),
                        "tensors": len(p["from_vace"])},
        "backbone_from": {"file": args.target.name, "digest": file_digest(args.target),
                          "tensors": len(p["from_target"])},
        "scopes_derived_against": {"file": args.base.name,
                                   "digest": file_digest(args.base)},
        "dropped_target_only_tensors": len(p["target_only"]),
    }, indent=2) + "\n")
    log.info("wrote %s and %s", args.out, meta.name)
    log.info("Next: scripts/verify_checkpoint.py %s --against %s --base %s",
             args.out.name, args.vace.name, args.base.name)
    log.info("This says the file is well formed, NOT that it generates better "
             "pixels. The scopes and this backbone were not trained together.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
