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

    scripts/verify_lora_loads.py LORA.safetensors [--model NAME]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, load_config, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lora", type=Path)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None,
                    help="Diffusion model filename; defaults to the config's")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("verify_lora_loads", args.verbose)
    cfg = load_config(args.config)
    name = args.model or cfg["model"]["diffusion_model"]
    model_path = P.comfy / "models" / "diffusion_models" / name
    if not args.lora.exists():
        log.error("No such LoRA: %s", args.lora)
        return 1
    if not model_path.exists():
        log.error("No such checkpoint: %s", model_path)
        return 1

    sys.path.insert(0, str(P.comfy))
    import comfy.lora
    import comfy.sd
    import comfy.utils

    log.info("checkpoint: %s", name)
    model = comfy.sd.load_diffusion_model(str(model_path))
    lora = comfy.utils.load_torch_file(str(args.lora), safe_load=True)

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    # load_lora() consumes the dict it is given, so count first.
    modules = sorted({k.rsplit(".", 2)[0] for k in lora
                      if k.endswith(("lora_down.weight", "lora_A.weight"))})
    patches = comfy.lora.load_lora(dict(lora), key_map, log_missing=False)
    matched = sorted({str(v) for v in patches})

    log.info("LoRA modules: %d", len(modules))
    log.info("bound to the checkpoint: %d", len(matched))
    unmatched = len(modules) - len(matched)
    if not modules:
        log.error("This file contains no LoRA modules at all.")
        return 1
    if unmatched:
        log.error("%d module(s) did not bind. ComfyUI would have run this "
                  "silently and produced a generation the LoRA barely touched.",
                  unmatched)
        for k in list(modules)[:10]:
            if not any(k in m for m in matched):
                log.error("  unbound: %s", k)
        return 1
    log.info("All %d module(s) bind. The LoRA will reach %s.",
             len(modules), name)
    log.info("This says the weights land, NOT that they help: identity is "
             "scored separately, against held-out photographs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
