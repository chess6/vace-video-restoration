#!/usr/bin/env python
"""Is this UNet actually a drop-in for the one the pipeline runs?

`model.diffusion_model` is one config field, so swapping the checkpoint for a
community fine-tune LOOKS like the cheapest change available. It is the most
dangerous one, because nothing downstream can tell that it happened:

  * ComfyUI picks a model class from the state dict. Hand it a plain T2V
    checkpoint where a VACE one is expected and the VACE scopes are simply not
    there - the control video, the mask and the reference image stop being
    conditioning and the run becomes an unconditioned generation that still
    produces frames, still writes a file, and still exits 0. Rule 4.
  * `vace_key` records the checkpoint by NAME, so a file swapped in place keys
    identically to the weights it replaced. `--digest` is what closes that.

So the question is asked as a DIFF against the checkpoint being replaced, not
against any memorised knowledge of what a VACE checkpoint contains (rule 6). If
every tensor the reference declares is present in the candidate at the same
shape, the graph that ran against one runs against the other; if any is missing
or reshaped, it does not, whatever the file is called.

Header-only: tensor names, dtypes and shapes come from the safetensors header,
so this costs milliseconds on a 34 GB file and needs neither torch nor CUDA.
`--digest` additionally reads the whole file, which is the slow part.

    scripts/verify_checkpoint.py CANDIDATE.safetensors
    scripts/verify_checkpoint.py CANDIDATE.safetensors --against wan2.1_vace_1.3B_fp16.safetensors
    scripts/verify_checkpoint.py CANDIDATE.safetensors --base wan2.1_t2v_1.3B_bf16.safetensors

`--base` names the plain T2V checkpoint the reference was built from. Given it,
the VACE scopes are DERIVED - they are the tensors VACE has and the base does
not - and the candidate is checked for carrying them. That is the check that
separates "a Wan checkpoint" from "a Wan checkpoint that can be controlled".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    P, cached_file_digest, load_config, safetensors_index, setup_logging,
)


def compare(candidate: dict, reference: dict) -> dict:
    """Set-and-shape diff of two safetensors indices.

    Pure, so it is testable without a checkpoint on disk.
    """
    c_keys, r_keys = set(candidate), set(reference)
    mismatched = sorted(
        k for k in (c_keys & r_keys)
        if list(candidate[k].get("shape", [])) != list(reference[k].get("shape", []))
    )
    return {
        "missing": sorted(r_keys - c_keys),      # reference has it, candidate does not
        "extra": sorted(c_keys - r_keys),        # candidate-only; usually harmless
        "mismatched": mismatched,                # same name, different shape: fatal
        "shared": len(c_keys & r_keys),
    }


def scopes(reference: dict, base: dict) -> list[str]:
    """The tensors the reference adds to its base - i.e. what makes it VACE.

    Derived, never named. A future checkpoint that calls them something else is
    still handled, and this file states nothing about the architecture it did not
    read from two files on disk.
    """
    return sorted(set(reference) - set(base))


def dtypes(index: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for spec in index.values():
        out[spec.get("dtype", "?")] = out.get(spec.get("dtype", "?"), 0) + 1
    return out


def verify(candidate_path: Path, reference_path: Path, base_path: Path | None,
           log) -> bool:
    """True if the candidate can stand in for the reference. Logs why not."""
    try:
        cand = safetensors_index(candidate_path)
        ref = safetensors_index(reference_path)
    except (OSError, ValueError) as e:
        log.error("%s", e)
        return False

    log.info("candidate: %s  (%d tensors, %s)", candidate_path.name, len(cand),
             " ".join(f"{n}x{d}" for d, n in sorted(dtypes(cand).items())))
    log.info("reference: %s  (%d tensors, %s)", reference_path.name, len(ref),
             " ".join(f"{n}x{d}" for d, n in sorted(dtypes(ref).items())))

    d = compare(cand, ref)
    ok = True
    if d["missing"]:
        log.error("%d tensor(s) the reference declares are ABSENT from the "
                  "candidate. The graph cannot run against it.", len(d["missing"]))
        for k in d["missing"][:10]:
            log.error("  missing: %s", k)
        ok = False
    if d["mismatched"]:
        log.error("%d tensor(s) share a name but not a shape. This is a "
                  "different architecture under the same key names.",
                  len(d["mismatched"]))
        for k in d["mismatched"][:10]:
            log.error("  %s: candidate %s vs reference %s", k,
                      cand[k].get("shape"), ref[k].get("shape"))
        ok = False
    if d["extra"]:
        # Not an error. ComfyUI ignores keys its model class does not ask for,
        # and a fine-tune that carries extra bookkeeping is still a drop-in.
        log.info("%d candidate-only tensor(s); ComfyUI will ignore them.",
                 len(d["extra"]))

    if base_path is not None:
        try:
            base = safetensors_index(base_path)
        except (OSError, ValueError) as e:
            log.error("%s", e)
            return False
        sc = scopes(ref, base)
        if not sc:
            log.error("%s adds no tensors to %s, so it is not the VACE variant "
                      "and there is nothing to check the candidate for.",
                      reference_path.name, base_path.name)
            return False
        held = [k for k in sc if k in cand]
        log.info("control scopes derived from %s vs %s: %d tensor(s); the "
                 "candidate carries %d.", reference_path.name, base_path.name,
                 len(sc), len(held))
        if len(held) != len(sc):
            log.error("The candidate is missing %d of them. A run would load it, "
                      "produce frames, and ignore the control video, the mask "
                      "and the reference image.", len(sc) - len(held))
            ok = False

    log.info("VERDICT: %s", "drop-in" if ok else "NOT a drop-in")
    if ok:
        log.info("This says the graph will RUN against it, NOT that the result "
                 "is better: measure in-mask against the plate before believing "
                 "a swap helped.")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--against", default=None,
                    help="Reference checkpoint filename; defaults to the config's")
    ap.add_argument("--base", default=None,
                    help="Plain T2V checkpoint the reference was built from. "
                         "Given it, the control scopes are derived and checked.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--digest", action="store_true",
                    help="Also hash the file. Minutes on a large checkpoint; "
                         "this is the number that identifies WHICH weights ran.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("verify_checkpoint", args.verbose)
    cfg = load_config(args.config)
    models = P.comfy / "models" / "diffusion_models"

    def resolve(name: str | Path) -> Path:
        p = Path(name)
        return p if p.exists() or p.is_absolute() else models / p

    cand = resolve(args.candidate)
    ref = resolve(args.against or cfg["model"]["diffusion_model"])
    base = resolve(args.base) if args.base else None
    for p in [cand, ref] + ([base] if base else []):
        if not p.exists():
            log.error("No such checkpoint: %s", p)
            return 1
    if cand.resolve() == ref.resolve():
        log.error("The candidate and the reference are the same file. Pass "
                  "--against the checkpoint you mean to replace.")
        return 1

    ok = verify(cand, ref, base, log)
    if args.digest:
        log.info("digest %s  %s", cached_file_digest(cand), cand.name)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
