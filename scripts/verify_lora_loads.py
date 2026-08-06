#!/usr/bin/env python
"""Does this LoRA actually bind to the checkpoint the pipeline runs?

A LoRA trained against Wan2.1 T2V 1.3B is loaded onto VACE 1.3B, and a key-name
mismatch does not raise: ComfyUI logs the unmatched keys and applies whatever
matched, which can be nothing. The result is a generation that ran, cost its
GPU minutes, recorded a LoRA in its metadata, and was not influenced by one.
Rule 4 in one sentence.

So the binding is measured through ComfyUI's own mapping code, in the installed
revision (rule 6), rather than assumed from the naming convention:

    every LoRA module -> a key ComfyUI maps -> a tensor in the checkpoint

Anything less than all of them is reported, and a partial match is a failure,
not a warning: half a LoRA is not a weaker LoRA, it is a different one.

A stack is N gambles, not one, so every member is measured separately: a
behaviour LoRA that fails to bind beside a subject LoRA that binds still yields
a run that produces frames and records both.

    scripts/verify_lora_loads.py [LORA.safetensors ...] [--model NAME]

With no arguments it checks every LoRA the config's stack names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, file_digest, load_config, lora_stack, setup_logging  # noqa: E402

# Verdicts are cached here so the pre-flight gate below costs a checkpoint load
# once per (LoRA, checkpoint) pair rather than once per run. Untracked, like
# everything else under intermediate/.
CACHE = P.intermediate / "lora_binding.json"


def header_digest(path: Path) -> str:
    """Hash a checkpoint's key names, dtypes and shapes - not its weights.

    Whether a LoRA binds depends on which keys exist and how they are shaped. It
    does not depend on the values in them, and the 14B checkpoints are ~34 GB, so
    hashing the whole file to answer a question about its key names would be
    minutes of I/O for no extra discrimination.

    A safetensors file is <u64 header length><header JSON><tensor data>, so the
    header is the first few hundred KB and reading it needs no dependency. This
    is still a CONTENT hash, not a name or a size: swap the checkpoint for a
    differently-shaped one and the digest moves.
    """
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if not 0 < n < (1 << 28):
            raise ValueError(f"{path.name}: not a safetensors file")
        head = f.read(n)
    if len(head) != n:
        raise ValueError(f"{path.name}: truncated safetensors header")
    return hashlib.sha256(head).hexdigest()[:16]


def binding_report(lora_path: Path, model_path: Path) -> tuple[list[str], list[str]]:
    """(modules the LoRA contains, modules that bind to the checkpoint).

    Measured through ComfyUI's own mapping code in the installed revision, never
    inferred from the naming convention. Loads the checkpoint, so it is not
    cheap; require_binding() caches the verdict.
    """
    sys.path.insert(0, str(P.comfy))
    import comfy.lora
    import comfy.sd
    import comfy.utils

    model = comfy.sd.load_diffusion_model(str(model_path))
    lora = comfy.utils.load_torch_file(str(lora_path), safe_load=True)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    # load_lora() consumes the dict it is given, so count first.
    modules = sorted({k.rsplit(".", 2)[0] for k in lora
                      if k.endswith(("lora_down.weight", "lora_A.weight"))})
    patches = comfy.lora.load_lora(dict(lora), key_map, log_missing=False)
    return modules, sorted({str(v) for v in patches})


def require_binding(lora_path: Path, model_path: Path, log: logging.Logger) -> bool:
    """Pre-flight gate: refuse to generate with a LoRA that will not bind.

    docs/STATE.md records the failure this exists to stop - eight checkpoints and
    the baseline scored identical to four decimals, because a merge had silently
    no-oped. ComfyUI does not raise on a key-name mismatch: it logs the unmatched
    keys and applies whatever matched, which can be nothing. So the only place
    that can catch it is before the GPU minutes are spent, and the cost of
    missing it is a result that recorded a LoRA in its key and was not
    conditioned by one.

    This matters more for a downloaded checkpoint than for the trained subject
    LoRA, whose base was known when it was made. A third-party file's key naming
    is whatever its author's trainer emitted.
    """
    try:
        key = f"{file_digest(lora_path)}:{header_digest(model_path)}"
    except (OSError, ValueError) as e:
        log.error("Cannot key the LoRA binding check: %s", e)
        return False

    cache = {}
    if CACHE.exists():
        try:
            cache = json.loads(CACHE.read_text())
        except (OSError, json.JSONDecodeError):
            cache = {}
    hit = cache.get(key)
    if hit:
        log.info("LoRA binding: %d/%d module(s) bind (cached)",
                 hit["matched"], hit["modules"])
        return hit["matched"] == hit["modules"] and hit["modules"] > 0

    log.info("Checking that %s binds to %s. Loads the checkpoint; once per pair.",
             lora_path.name, model_path.name)
    try:
        modules, matched = binding_report(lora_path, model_path)
    except Exception as e:  # noqa: BLE001 - a failed check must not read as a pass
        log.error("LoRA binding check could not run: %s", e)
        log.error("Re-run scripts/verify_lora_loads.py %s directly, or pass "
                  "--skip-lora-binding-check having done so.", lora_path.name)
        return False

    cache[key] = {"modules": len(modules), "matched": len(matched),
                  "lora": lora_path.name, "checkpoint": model_path.name}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")

    if not modules:
        log.error("%s contains no LoRA modules at all.", lora_path.name)
        return False
    if len(matched) != len(modules):
        log.error("%d of %d module(s) did not bind to %s. ComfyUI would have run "
                  "this silently and produced a generation the LoRA barely "
                  "touched.", len(modules) - len(matched), len(modules),
                  model_path.name)
        for k in modules[:10]:
            if not any(k in m for m in matched):
                log.error("  unbound: %s", k)
        return False
    log.info("LoRA binding: all %d module(s) bind.", len(modules))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lora", type=Path, nargs="*",
                    help="LoRA file(s). Default: every entry in the config's stack.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None,
                    help="Diffusion model filename; defaults to the config's")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("verify_lora_loads", args.verbose)
    cfg = load_config(args.config)
    name = args.model or cfg["model"]["diffusion_model"]
    model_path = P.comfy / "models" / "diffusion_models" / name
    if not model_path.exists():
        log.error("No such checkpoint: %s", model_path)
        return 1

    # Each member of a stack is its own key-name gamble, so each is measured
    # separately - one that fails alone still leaves a run that produces frames,
    # records both LoRAs, and was conditioned by one.
    loras = args.lora or [P.comfy / "models" / "loras" / e["name"]
                          for e in lora_stack(cfg)]
    if not loras:
        log.error("%s names no LoRA and none was given on the command line.",
                  cfg.get("_config_path"))
        return 1
    for p in loras:
        if not p.exists():
            log.error("No such LoRA: %s", p)
            return 1

    log.info("checkpoint: %s", name)
    rc = 0
    for p in loras:
        log.info("--- %s", p.name)
        modules, matched = binding_report(p, model_path)
        log.info("LoRA modules: %d", len(modules))
        log.info("bound to the checkpoint: %d", len(matched))
        unmatched = len(modules) - len(matched)
        if not modules:
            log.error("This file contains no LoRA modules at all.")
            rc = 1
            continue
        if unmatched:
            log.error("%d module(s) did not bind. ComfyUI would have run this "
                      "silently and produced a generation the LoRA barely touched.",
                      unmatched)
            for k in list(modules)[:10]:
                if not any(k in m for m in matched):
                    log.error("  unbound: %s", k)
            rc = 1
            continue
        log.info("All %d module(s) bind. The LoRA will reach %s.",
                 len(modules), name)
    if rc:
        return rc
    log.info("This says the weights land, NOT that they help: match is "
             "scored separately, against held-out references.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
