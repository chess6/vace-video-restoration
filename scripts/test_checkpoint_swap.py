#!/usr/bin/env python
"""Checks on swapping the diffusion model for a community fine-tune.

Pure: builds tiny safetensors files in a temp directory and reasons about their
headers. No CUDA, no ComfyUI, no torch, no checkpoint downloads.

    venv/bin/python scripts/test_checkpoint_swap.py

What it proves:

  * `safetensors_index` reads names, dtypes and shapes from the header alone -
    the property that makes checking a 34 GB candidate cost milliseconds
  * a candidate missing any tensor the reference declares is NOT a drop-in, and
    one that merely adds tensors is
  * a same-name-different-shape tensor is caught. It is the dangerous case: the
    file loads far enough to look right
  * the control scopes are DERIVED by diffing VACE against its base, so nothing
    depends on knowing what they are called (rule 6)
  * `graft_vace.plan` takes the backbone from the fine-tune and the scopes from
    VACE, drops fine-tune-only tensors, and REFUSES a target that is a fine-tune
    of something else
  * `checkpoint_changed` invalidates a result whose recorded digest differs, and
    treats a record with no digest as no information - so adding this feature
    marks nothing already finished as stale
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import safetensors_index  # noqa: E402
from graft_vace import plan  # noqa: E402
from run_chunks import checkpoint_changed  # noqa: E402
from verify_checkpoint import compare, scopes  # noqa: E402

FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILED.append(name)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    """Minimal writer: {name: (dtype, shape)} -> a real safetensors file.

    Written by hand rather than with the library so this test runs anywhere,
    including a machine with no venv. The format is
    <u64 header length><header JSON><tensor data>, which is the same fact
    common.safetensors_index depends on - so a change to either side breaks
    this test rather than passing silently.
    """
    width = {"F16": 2, "BF16": 2, "F32": 4, "F8_E4M3": 1}
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = width[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + n]}
        offset += n
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ckpt_swap_test_"))

    # A base T2V checkpoint, the VACE variant of it (same tensors plus control
    # scopes), and a community fine-tune of the base.
    BASE = {"blocks.0.attn.weight": ("F16", [4, 4]),
            "blocks.1.attn.weight": ("F16", [4, 4]),
            "patch_embedding.weight": ("F16", [4, 2])}
    SCOPE = {"ctrl_blocks.0.weight": ("F16", [4, 4]),
             "ctrl_patch.weight": ("F16", [4, 2])}
    write_safetensors(tmp / "base.safetensors", BASE)
    write_safetensors(tmp / "vace.safetensors", {**BASE, **SCOPE})
    write_safetensors(tmp / "finetune.safetensors", {**BASE, "extra.bias": ("F16", [4])})

    base = safetensors_index(tmp / "base.safetensors")
    vace = safetensors_index(tmp / "vace.safetensors")
    fine = safetensors_index(tmp / "finetune.safetensors")

    print("header reading")
    check("every tensor is indexed", set(vace) == set(BASE) | set(SCOPE))
    check("shapes survive", vace["blocks.0.attn.weight"]["shape"] == [4, 4])
    check("dtypes survive", vace["patch_embedding.weight"]["dtype"] == "F16")
    check("__metadata__ is not a tensor", "__metadata__" not in vace)

    print("drop-in comparison")
    d = compare(vace, vace)
    check("a file is a drop-in for itself",
          not d["missing"] and not d["mismatched"])
    d = compare(fine, vace)
    check("a plain fine-tune is NOT a drop-in for VACE", len(d["missing"]) == 2)
    check("its scopes are what is missing", set(d["missing"]) == set(SCOPE))
    check("its own extra tensor is reported as extra", d["extra"] == ["extra.bias"])
    write_safetensors(tmp / "superset.safetensors",
                      {**BASE, **SCOPE, "extra.bias": ("F16", [4])})
    d = compare(safetensors_index(tmp / "superset.safetensors"), vace)
    check("extra tensors alone still leave it a drop-in", not d["missing"])
    wrong = dict(BASE, **{"blocks.0.attn.weight": ("F16", [8, 8])})
    write_safetensors(tmp / "wrong.safetensors", {**wrong, **SCOPE})
    d = compare(safetensors_index(tmp / "wrong.safetensors"), vace)
    check("a reshaped tensor is caught", d["mismatched"] == ["blocks.0.attn.weight"])

    print("scope derivation")
    check("scopes are VACE minus its base", set(scopes(vace, base)) == set(SCOPE))
    check("scopes of a model against itself are empty", scopes(vace, vace) == [])

    print("graft plan")
    p = plan(vace, base, fine)
    check("backbone comes from the fine-tune", set(p["from_target"]) == set(BASE))
    check("scopes come from VACE", set(p["from_vace"]) == set(SCOPE))
    check("nothing is missing or reshaped", not p["missing"] and not p["reshaped"])
    check("fine-tune-only tensors are dropped", p["target_only"] == ["extra.bias"])
    check("the output key set would be exactly VACE's",
          set(p["from_target"]) | set(p["from_vace"]) == set(vace))

    write_safetensors(tmp / "other.safetensors",
                      {"blocks.0.attn.weight": ("F16", [8, 8]),
                       "blocks.1.attn.weight": ("F16", [8, 8])})
    p = plan(vace, base, safetensors_index(tmp / "other.safetensors"))
    check("a fine-tune of a different size is refused",
          bool(p["missing"]) or bool(p["reshaped"]),
          f"- missing={p['missing']} reshaped={p['reshaped']}")
    p = plan(vace, vace, fine)
    check("no scopes to graft is visible as an empty scope set", p["from_vace"] == [])

    print("provenance")
    c = {"runs": {"baseline": {"model_digest": "aaaa"}}}
    check("a different checkpoint invalidates", checkpoint_changed(c, "baseline", "bbbb"))
    check("the same checkpoint does not", not checkpoint_changed(c, "baseline", "aaaa"))
    check("a record with no digest is not invalidated",
          not checkpoint_changed({"runs": {"baseline": {}}}, "baseline", "bbbb"))
    check("an unhashable checkpoint does not invalidate either",
          not checkpoint_changed(c, "baseline", None))
    check("another variant's digest is not consulted",
          not checkpoint_changed(c, "roi", "bbbb"))

    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)} check(s): {', '.join(FAILED)}")
        return 1
    print("All checkpoint-swap checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
