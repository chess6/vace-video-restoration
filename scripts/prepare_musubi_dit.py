#!/usr/bin/env python
"""Write the DiT copy musubi-tuner can actually merge a LoRA into.

THE BUG THIS EXISTS FOR, and how it presented
Comfy-Org's repackaged Wan2.1 1.3B checkpoints prefix every tensor with
`model.diffusion_model.`. musubi-tuner v0.3.4 strips that prefix in
`wan/modules/model.py::load_wan_model` - but only AFTER
`load_safetensors_with_lora_and_fp8` has already merged the LoRA, and the merge
derives its lookup name from the raw file key. So it searches for
`lora_unet_model_diffusion_model_blocks_0_self_attn_q`, the trained weights are
called `lora_unet_blocks_0_self_attn_q`, nothing matches, and the only symptom
is one `Warning: not all LoRA keys are used` line in a log full of progress
bars. Generation then runs at full cost and produces output byte-identical to
the base model.

It was caught by comparing checksums of the probe images against the no-LoRA
baseline: all eight checkpoints scored 0.0226 match, to four decimals, which
is not a curve any real training produces. The lesson is the one in
docs/STATE.md - a stage that "ran" is not a stage that did anything.

Training is unaffected: it names LoRA modules from the loaded nn.Module tree,
which is already de-prefixed, which is why the trained file is correct.

ComfyUI is unaffected too: its own mapper binds all 300 modules to the VACE
checkpoint (scripts/verify_lora_loads.py). This is purely a musubi-side fix so
that the probe generations exercise the LoRA.

The output is byte-identical in content - only key names change - so the source
file's SHA256 plus this transform is its provenance (rule 7).

    scripts/prepare_musubi_dit.py IN.safetensors [--out OUT.safetensors]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import setup_logging  # noqa: E402

PREFIX = "model.diffusion_model."


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = setup_logging("prepare_musubi_dit", args.verbose)
    if not args.src.exists():
        log.error("No such file: %s", args.src)
        return 1
    out = args.out or args.src.with_name(args.src.stem + "_noprefix.safetensors")

    from safetensors.torch import load_file, save_file
    sd = load_file(str(args.src))
    n_pref = sum(1 for k in sd if k.startswith(PREFIX))
    if n_pref == 0:
        log.info("%s already has no `%s` keys; musubi can merge into it as it "
                 "stands. Nothing written.", args.src.name, PREFIX)
        return 0
    if n_pref != len(sd):
        log.error("%d of %d keys carry the prefix. A partially prefixed file is "
                  "not something to guess at.", n_pref, len(sd))
        return 1

    log.info("source   %s  sha256 %s", args.src.name, sha256(args.src)[:16])
    save_file({k[len(PREFIX):]: v for k, v in sd.items()}, str(out))
    log.info("wrote    %s  sha256 %s", out.name, sha256(out)[:16])
    log.info("%d tensor(s) renamed; no value changed.", len(sd))
    log.info("Use this copy for musubi generation. ComfyUI keeps the original: "
             "its loader handles the prefix correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
