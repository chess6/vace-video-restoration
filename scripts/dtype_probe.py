#!/usr/bin/env python
"""What dtype did the model ACTUALLY load and compute at, measured on the server.

WHY THIS EXISTS. Bundles 17 and 18 compared "fp8" against "bf16" and concluded
precision was not the cause. The labels came from a directory name and an
argument passed to a loader — never from a measurement. A declared dtype is a
REQUEST, not a receipt: a loader can ignore it, cast around it, or fall back.

The argument that the question was closed anyway — the arms looked different, so
they cannot have been the same dtype — does not survive contact with what it
would need to prove. A visible difference shows that *something* differed. It
does not say which dtype either arm ran at, and specifically does not establish
that the arm labelled unquantised ran unquantised. That is a proxy standing in
for the thing itself, which is a documented trap in this project.

So: read it out of the running process instead of inferring it. This walks the
loaded model's parameters and reports what is actually there.

WHAT IT REPORTS, and how much each is worth:

  checkpoint_sha256   the file on disk. Settles WHICH weights, with no ambiguity.
  declared_dtype      what the graph asked for. The thing that was wrong to trust.
  param_dtypes        a census of actual tensor dtypes across the loaded model,
                      counted by parameter, not sampled from one layer.
  compute_dtype       what the model object reports it will compute in, where the
                      installed revision exposes it.
  loader_class        which implementation ran, since two loaders honour the same
                      argument differently.

RUNS ON THE POD, inside ComfyUI's own process — that is the only place the
loaded object exists. Invoked over the API by a tiny custom node would be
cleaner; this instead imports ComfyUI's loader directly with the same arguments
the graph uses, which is the same code path and needs no node installed.

Writes JSON to disk and prints the path. Displays nothing (rule 1).

    dtype_probe.py --unet Chroma1-HD.safetensors --weight-dtype default
    dtype_probe.py --unet Chroma1-HD.safetensors --weight-dtype fp8_e4m3fn
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("VACE_ROOT", "/workspace/vace-video-restoration"))
COMFY = ROOT / "ComfyUI"


def die(m: str) -> None:
    print(f"FATAL: {m}", file=sys.stderr)
    raise SystemExit(1)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def census(model) -> dict:
    """Actual dtypes of every parameter and buffer, counted by element.

    Counted by ELEMENT rather than by tensor: a model can hold thousands of tiny
    fp32 norm weights beside a handful of enormous quantised matrices, and a
    per-tensor count would report the wrong thing as dominant. What matters is
    where the parameters actually are.
    """
    import torch  # noqa: F401
    by_dtype: dict[str, int] = collections.Counter()
    tensors: dict[str, int] = collections.Counter()
    biggest = {"name": None, "dtype": None, "numel": 0}
    seen = 0
    for mod in (getattr(model, "model", None), model):
        if mod is None or not hasattr(mod, "named_parameters"):
            continue
        for name, t in list(mod.named_parameters()) + list(mod.named_buffers()):
            d = str(t.dtype).replace("torch.", "")
            by_dtype[d] += t.numel()
            tensors[d] += 1
            if t.numel() > biggest["numel"]:
                biggest = {"name": name, "dtype": d, "numel": int(t.numel())}
            seen += 1
        break                      # the inner module is the real one; stop there
    total = sum(by_dtype.values()) or 1
    return {
        "tensors_examined": seen,
        "elements_by_dtype": dict(by_dtype),
        "tensors_by_dtype": dict(tensors),
        "share_by_dtype": {k: round(v / total, 6) for k, v in by_dtype.items()},
        "dominant_dtype": max(by_dtype, key=by_dtype.get) if by_dtype else None,
        "largest_tensor": biggest,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default="Chroma1-HD.safetensors")
    ap.add_argument("--weight-dtype", default="default",
                    choices=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    help="the DECLARED dtype, i.e. exactly what the graph asks "
                         "for. The point of this tool is to find out whether it "
                         "is what happened.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--output-image", type=Path, default=None,
                    help="an image produced under this exact declared dtype. Its "
                         "sha256 is recorded, which is what ties the measurement "
                         "to a result rather than leaving them two separate "
                         "claims. The image itself is never committed.")
    args = ap.parse_args()

    if not COMFY.exists():
        die(f"{COMFY} not found; this must run on the box with ComfyUI installed")
    sys.path.insert(0, str(COMFY))

    import torch
    import comfy.sd
    import folder_paths
    import nodes

    path = folder_paths.get_full_path("diffusion_models", args.unet) or \
        folder_paths.get_full_path("unet", args.unet)
    if not path:
        die(f"{args.unet} is not in ComfyUI's diffusion_models/unet search paths")
    path = Path(path)

    # Same mapping the UNETLoader node applies, read from the installed revision
    # rather than reproduced from memory (rule 6). If this build maps the string
    # differently, this fails here instead of silently probing something else.
    loader = nodes.NODE_CLASS_MAPPINGS.get("UNETLoader")
    if loader is None:
        die("this build has no UNETLoader; the graph and this probe disagree")

    print(f"checkpoint : {path}")
    print("hashing (once per file, a minute of I/O) ...")
    digest = sha256_file(path)
    print(f"sha256     : {digest}")
    print(f"declared   : {args.weight_dtype}")

    inst = loader()
    fn = getattr(inst, loader.FUNCTION)
    out = fn(args.unet, args.weight_dtype)
    model = out[0]

    c = census(model)
    patcher = getattr(model, "patcher", model)
    rec = {
        "checkpoint_name": args.unet,
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "declared_weight_dtype": args.weight_dtype,
        "loader_class": type(inst).__name__,
        "loader_function": loader.FUNCTION,
        "model_class": type(getattr(model, "model", model)).__name__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **c,
    }
    # Where the installed revision exposes a compute dtype, take it. Absence is
    # recorded as absence rather than filled with a guess.
    for attr in ("get_dtype", "get_compute_dtype"):
        f = getattr(patcher, attr, None) or getattr(getattr(model, "model", None), attr, None)
        if callable(f):
            try:
                rec["compute_dtype"] = str(f()).replace("torch.", "")
                rec["compute_dtype_source"] = attr
                break
            except Exception as e:
                rec["compute_dtype_error"] = f"{attr}: {e}"
    rec.setdefault("compute_dtype", None)

    # Output provenance. Without it the probe says what the loader did and the
    # bundle says what the pixels look like, with nothing joining them.
    if args.output_image:
        if not args.output_image.exists():
            die(f"--output-image {args.output_image} does not exist")
        rec["output_image"] = args.output_image.name
        rec["output_sha256"] = sha256_file(args.output_image)

    # The verdict this whole tool exists to produce, stated plainly.
    dom = rec.get("dominant_dtype")
    declared_is_quantised = args.weight_dtype.startswith("fp8")
    actual_is_quantised = bool(dom and "float8" in dom)
    rec["declared_matches_actual"] = (declared_is_quantised == actual_is_quantised)
    rec["verdict"] = (
        f"declared {args.weight_dtype!r}; parameters are predominantly {dom!r} "
        f"({rec['share_by_dtype'].get(dom, 0):.1%} of elements)"
    )

    out_path = args.out or (ROOT / "intermediate" / "dtype_probe"
                            / f"{Path(args.unet).stem}_{args.weight_dtype}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=2))

    print(f"\nloader     : {rec['loader_class']}.{rec['loader_function']}")
    print(f"model      : {rec['model_class']}")
    print(f"tensors    : {rec['tensors_examined']}")
    print(f"dtypes     : {rec['tensors_by_dtype']}")
    print(f"by element : {rec['share_by_dtype']}")
    print(f"compute    : {rec['compute_dtype']}  (source: {rec.get('compute_dtype_source')})")
    print(f"VERDICT    : {rec['verdict']}")
    if not rec["declared_matches_actual"]:
        print("  *** DECLARED AND ACTUAL DISAGREE. Any arm labelled by the "
              "declared value is mislabelled. ***")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
