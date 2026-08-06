#!/usr/bin/env python
"""Checks on the LoRA stack: how it is parsed, wired and patched.

Pure: no CUDA, no ComfyUI, no model files, so it runs anywhere in a second.

    venv/bin/python scripts/test_lora_stack.py

What it proves, and why each one is here rather than left to review:

  * both spellings of `model.lora` parse, including `name: ""`, which is how
    every profile and the LoRA experiment's control arm say "no LoRA"
  * the untracked overlay's entries append AFTER the config's, so a tracked
    config can ask for a local behaviour LoRA without naming one (rule 2a)
  * a stack that names one file twice is refused, because loading it twice
    applies the patch twice and looks like a strength that was never configured
  * common_models() CHAINS the loaders - each takes the previous node's model.
    A stack that silently ran one LoRA would be the eight-identical-checkpoints
    failure again (docs/STATE.md), and the graph is where that would happen
  * an overlay LoRA reaches the graph with an EMPTY name, because workflows/ is
    tracked and its filename is not publishable
  * set_input(occurrence=i) patches the i-th loader and only it. This is the
    regression that made a stack unsafe to wire by hand before: set_input defaults
    to occurrence 0, so every entry after the first kept its build-time value
    while the manifest recorded what the config said
  * an empty stack adds NOTHING to the generation key, so turning this feature
    on does not mark finished no-LoRA chunks stale
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import generation_key, lora_stack  # noqa: E402
from comfy_client import set_input  # noqa: E402

FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILED.append(name)


def raises(name: str, fn, needle: str = "") -> None:
    try:
        fn()
    except ValueError as e:
        check(name, needle.lower() in str(e).lower(),
              f"- message did not mention {needle!r}: {e}")
    except Exception as e:  # noqa: BLE001
        check(name, False, f"- raised {type(e).__name__}, not ValueError: {e}")
    else:
        check(name, False, "- did not raise")


def cfg_with(lora) -> dict:
    return {"model": {"diffusion_model": "x.safetensors", "lora": lora}}


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lora_stack_test_"))
    # Point the overlay lookup at a scratch directory. The real
    # configs/prompt.local.yaml is the user's and is never read or written here.
    common.P.configs = tmp
    overlay = tmp / common.PROMPT_OVERLAY

    print("parsing")
    check("no lora key", lora_stack({"model": {}}) == [])
    check("empty name is no LoRA", lora_stack(cfg_with({"name": "", "strength": 1.0})) == [])
    check("mapping form",
          lora_stack(cfg_with({"name": "a.safetensors", "strength": 0.5}))
          == [{"name": "a.safetensors", "strength": 0.5, "source": "config"}])
    check("default strength is 1.0",
          lora_stack(cfg_with({"name": "a.safetensors"}))[0]["strength"] == 1.0)
    stack = lora_stack(cfg_with([{"name": "a.safetensors"},
                                 {"name": "b.safetensors", "strength": 0.6}]))
    check("list form keeps order",
          [e["name"] for e in stack] == ["a.safetensors", "b.safetensors"])
    check("list form keeps strengths", [e["strength"] for e in stack] == [1.0, 0.6])

    print("refusals")
    raises("duplicate file",
           lambda: lora_stack(cfg_with([{"name": "a.safetensors"},
                                        {"name": "a.safetensors"}])), "twice")
    raises("path escape",
           lambda: lora_stack(cfg_with({"name": "../a.safetensors"})), "bare filename")
    raises("non-numeric strength",
           lambda: lora_stack(cfg_with({"name": "a.safetensors", "strength": "loud"})),
           "number")
    raises("wrong type",
           lambda: lora_stack(cfg_with(3)), "mapping")

    print("overlay")
    overlay.write_text("loras:\n  - {name: local.safetensors, strength: 0.6}\n")
    stack = lora_stack(cfg_with({"name": "a.safetensors"}))
    check("overlay entry appends last",
          [e["name"] for e in stack] == ["a.safetensors", "local.safetensors"])
    check("overlay entry is labelled",
          [e["source"] for e in stack] == ["config", "overlay"])
    check("overlay alone still yields a stack",
          [e["name"] for e in lora_stack({"model": {}})] == ["local.safetensors"])
    overlay.write_text("trigger: sometoken\n")
    check("overlay without loras adds nothing", lora_stack(cfg_with({"name": ""})) == [])
    overlay.write_text("loras:\n  - {name: a.safetensors}\n")
    raises("overlay may not repeat a config entry",
           lambda: lora_stack(cfg_with({"name": "a.safetensors"})), "twice")
    overlay.unlink()

    print("graph")
    import build_workflows as bw
    g = bw.Graph("t", "test")
    cfg = {"model": {"diffusion_model": "vace.safetensors", "weight_dtype": "default",
                     "text_encoder": "t5.safetensors", "vae": "vae.safetensors",
                     "clip_type": "wan",
                     "lora": [{"name": "a.safetensors", "strength": 1.0},
                              {"name": "b.safetensors", "strength": 0.6}]},
           "prompt": {"positive": "p", "negative": "n"}}
    mdl = bw.common_models(g, cfg)
    loaders = [n for n in g.nodes if n.cls == "LoraLoaderModelOnly"]
    check("one loader per entry", len(loaders) == 2, f"- got {len(loaders)}")
    check("names in order",
          [n.inputs["lora_name"] for n in loaders] == ["a.safetensors", "b.safetensors"])
    check("strengths in order",
          [n.inputs["strength_model"] for n in loaders] == [1.0, 0.6])
    unet = next(n for n in g.nodes if n.cls == "UNETLoader")
    check("first loader takes the checkpoint", loaders[0].inputs["model"].node is unet)
    check("second loader takes the first", loaders[1].inputs["model"].node is loaders[0])
    check("the sampler side sees the LAST loader", mdl["unet"] is loaders[1])
    check("node ids increase with stack order", loaders[0].id < loaders[1].id)

    common.P.configs = tmp
    overlay.write_text("loras:\n  - {name: local.safetensors, strength: 0.6}\n")
    g = bw.Graph("t", "test")
    cfg["model"]["lora"] = [{"name": "a.safetensors", "strength": 1.0}]
    bw.common_models(g, cfg)
    loaders = [n for n in g.nodes if n.cls == "LoraLoaderModelOnly"]
    check("overlay LoRA gets a node", len(loaders) == 2, f"- got {len(loaders)}")
    check("overlay LoRA is not named in the graph",
          loaders[1].inputs["lora_name"] == "",
          f"- graph carries {loaders[1].inputs['lora_name']!r}")
    overlay.unlink()

    print("run-time patching")
    wf = {"1": {"class_type": "UNETLoader", "inputs": {}},
          "2": {"class_type": "LoraLoaderModelOnly",
                "inputs": {"lora_name": "a.safetensors", "strength_model": 1.0}},
          "3": {"class_type": "LoraLoaderModelOnly",
                "inputs": {"lora_name": "", "strength_model": 1.0}}}
    for i, (name, mult) in enumerate([("a.safetensors", 1.0), ("real.safetensors", 0.6)]):
        set_input(wf, "LoraLoaderModelOnly", "lora_name", name, occurrence=i)
        set_input(wf, "LoraLoaderModelOnly", "strength_model", mult, occurrence=i)
    check("each loader patched separately",
          [wf[k]["inputs"]["lora_name"] for k in ("2", "3")]
          == ["a.safetensors", "real.safetensors"])
    check("strengths patched separately",
          [wf[k]["inputs"]["strength_model"] for k in ("2", "3")] == [1.0, 0.6])

    print("generation key")
    base = {"model": {"diffusion_model": "vace.safetensors"}, "seed": 1}
    empty = dict(base, **({"loras": []} if [] else {}))
    check("an empty stack leaves the key untouched",
          generation_key(empty) == generation_key(base))
    one = dict(base, loras=[{"name": "a.safetensors", "strength": 1.0, "digest": "aa"}])
    two = dict(base, loras=[{"name": "a.safetensors", "strength": 1.0, "digest": "bb"}])
    check("a changed LoRA file changes the key", generation_key(one) != generation_key(two))
    check("a stacked LoRA changes the key", generation_key(one) != generation_key(base))

    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)} check(s): {', '.join(FAILED)}")
        return 1
    print("All LoRA-stack checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
