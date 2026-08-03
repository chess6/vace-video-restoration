#!/usr/bin/env python
"""Phase 8 - generate the ComfyUI workflows.

Emits, for each graph, BOTH formats from a single definition:
  workflows/<name>.json      - UI format, openable in the ComfyUI editor
  workflows/<name>_api.json  - API format, used by scripts/run_chunk.py

Node signatures (input names, order, widget vs link, combo options,
control_after_generate) are read from the RUNNING ComfyUI's /object_info rather
than hardcoded, so a generated workflow cannot drift from the installed revision.

Graphs produced:
  vace_masked_depth_v2v_1p3b   the baseline: depth control + reference sheet +
                               tracked subject mask
  vace_unmasked_compare        same, minus the mask and reference (ablation)
  smoke_test_modelload         minimal: loads all three models and generates a
                               few frames, to prove the stack works

The key structural point, forced by the VACE node's own maths:

    inactive = control_video * (1 - mask)      <- preserved region
    reactive = control_video * mask            <- regenerated region

so control_video must carry ORIGINAL RGB outside the mask and DEPTH inside it.
That composite is built in-graph by ImageCompositeMasked. Feeding a pure depth
video would encode the background as a depth map and destroy the scene.

    scripts/build_workflows.py [--config ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, load_config, setup_logging  # noqa: E402


# ---------------------------------------------------------------------------
# graph description
# ---------------------------------------------------------------------------

class Ref:
    """A reference to output `slot` of node `node`."""
    __slots__ = ("node", "slot")

    def __init__(self, node: "N", slot: int = 0):
        self.node, self.slot = node, slot


class N:
    """One node. `inputs` maps input-name -> literal value or Ref."""
    _seq = 0

    def __init__(self, cls: str, title: str = "", **inputs):
        N._seq += 1
        self.id = N._seq
        self.cls = cls
        self.title = title or cls
        self.inputs = inputs

    def __call__(self, slot: int = 0) -> Ref:
        return Ref(self, slot)


class Graph:
    def __init__(self, name: str, description: str):
        N._seq = 0
        self.name, self.description = name, description
        self.nodes: list[N] = []

    def add(self, node: N) -> N:
        self.nodes.append(node)
        return node


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------

def fetch_object_info(host: str, port: int) -> dict:
    with urllib.request.urlopen(f"http://{host}:{port}/object_info", timeout=30) as r:
        return json.loads(r.read())


def ordered_inputs(spec: dict) -> list[tuple[str, list]]:
    out = []
    for section in ("required", "optional"):
        for k, v in (spec["input"].get(section) or {}).items():
            out.append((k, v))
    return out


def is_link_type(t) -> bool:
    """True if this input carries a wire rather than a widget value."""
    if isinstance(t, list):
        return False                     # combo -> widget
    return t not in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO",
                     "COMFY_DYNAMICCOMBO_V3")


def to_api(g: Graph, oi: dict) -> dict:
    api: dict = {}
    for n in g.nodes:
        if n.cls not in oi:
            raise KeyError(f"Node type {n.cls!r} does not exist in this ComfyUI build")
        ins: dict = {}
        for k, v in n.inputs.items():
            ins[k] = [str(v.node.id), v.slot] if isinstance(v, Ref) else v
        api[str(n.id)] = {"class_type": n.cls, "inputs": ins,
                          "_meta": {"title": n.title}}
    return api


def to_ui(g: Graph, oi: dict) -> dict:
    """Editor format. Widget values must follow declaration order."""
    # lay out in dependency columns so the graph is readable when opened
    depth: dict[int, int] = {}

    def d(n: N) -> int:
        if n.id in depth:
            return depth[n.id]
        depth[n.id] = 0                                   # cycle guard
        vals = [d(v.node) + 1 for v in n.inputs.values() if isinstance(v, Ref)]
        depth[n.id] = max(vals) if vals else 0
        return depth[n.id]

    for n in g.nodes:
        d(n)
    col_count: dict[int, int] = {}
    pos: dict[int, list[int]] = {}
    for n in sorted(g.nodes, key=lambda n: (depth[n.id], n.id)):
        c = depth[n.id]
        r = col_count.get(c, 0)
        col_count[c] = r + 1
        pos[n.id] = [c * 400, r * 210]

    links: list[list] = []
    link_id = 0
    node_out: list[dict] = []

    # pre-compute, for each node, which inputs are wired
    wired: dict[int, dict[str, Ref]] = {
        n.id: {k: v for k, v in n.inputs.items() if isinstance(v, Ref)} for n in g.nodes
    }
    # outgoing links per (node_id, slot)
    outgoing: dict[tuple[int, int], list[int]] = {}

    for n in g.nodes:
        spec = oi[n.cls]
        oin = ordered_inputs(spec)
        in_entries = []
        widget_vals = []
        for k, v in oin:
            t = v[0]
            meta = v[1] if len(v) > 1 and isinstance(v[1], dict) else {}
            if is_link_type(t):
                ref = wired[n.id].get(k)
                in_entries.append({"name": k, "type": t,
                                   "link": None, "_ref": ref})
            else:
                if k in n.inputs and not isinstance(n.inputs[k], Ref):
                    widget_vals.append(n.inputs[k])
                elif k in n.inputs and isinstance(n.inputs[k], Ref):
                    # a widget-typed input driven by a wire (e.g. trim_amount)
                    in_entries.append({"name": k, "type": t if isinstance(t, str) else "INT",
                                       "link": None, "_ref": n.inputs[k],
                                       "widget": {"name": k}})
                    widget_vals.append(meta.get("default", 0))
                else:
                    dv = meta.get("default")
                    if dv is None and isinstance(t, list):
                        dv = t[0] if t else None
                    if dv is None and isinstance(meta.get("options"), list):
                        dv = meta["options"][0]
                    widget_vals.append(dv)
                if meta.get("control_after_generate"):
                    widget_vals.append("fixed")
        outs = [{"name": o, "type": o, "slot_index": i, "links": []}
                for i, o in enumerate(spec.get("output", []))]
        node_out.append({
            "id": n.id, "type": n.cls, "pos": pos[n.id], "size": [340, 120],
            "flags": {}, "order": depth[n.id], "mode": 0,
            "inputs": in_entries, "outputs": outs,
            "properties": {"Node name for S&R": n.cls, "cnr_id": "comfy-core"},
            "widgets_values": widget_vals,
            "title": n.title,
        })

    by_id = {n["id"]: n for n in node_out}
    for n in node_out:
        for ie in n["inputs"]:
            ref: Ref | None = ie.pop("_ref", None)
            if ref is None:
                continue
            link_id += 1
            ie["link"] = link_id
            src = by_id[ref.node.id]
            src["outputs"][ref.slot]["links"].append(link_id)
            links.append([link_id, ref.node.id, ref.slot, n["id"],
                          n["inputs"].index(ie), ie["type"]])

    return {
        "last_node_id": max(n["id"] for n in node_out),
        "last_link_id": link_id,
        "nodes": node_out,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"note": g.description},
        "version": 0.4,
    }


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------

def common_models(g: Graph, cfg: dict) -> dict:
    m = cfg["model"]
    unet = g.add(N("UNETLoader", "VACE 1.3B",
                   unet_name=m["diffusion_model"], weight_dtype=m["weight_dtype"]))
    clip = g.add(N("CLIPLoader", "UMT5-XXL FP8",
                   clip_name=m["text_encoder"], type=m["clip_type"], device="default"))
    vae = g.add(N("VAELoader", "Wan 2.1 VAE", vae_name=m["vae"]))
    pos = g.add(N("CLIPTextEncode", "Positive prompt",
                  text=cfg["prompt"]["positive"].strip(), clip=clip()))
    neg = g.add(N("CLIPTextEncode", "Negative prompt",
                  text=cfg["prompt"]["negative"].strip(), clip=clip()))
    return {"unet": unet, "clip": clip, "vae": vae, "pos": pos, "neg": neg}


def sampler_tail(g: Graph, cfg: dict, mdl: dict, vace: N, fps: int,
                 prefix: str) -> N:
    s = cfg["sampling"]
    ks = g.add(N("KSampler", "Sampler",
                 model=mdl["unet"](), seed=int(s["seed"]), steps=int(s["steps"]),
                 cfg=float(s["cfg"]), sampler_name=s["sampler"],
                 scheduler=s["scheduler"], positive=vace(0), negative=vace(1),
                 latent_image=vace(2), denoise=float(s["denoise"])))
    trim = g.add(N("TrimVideoLatent", "Drop reference frame",
                   samples=ks(), trim_amount=vace(3)))
    dec = g.add(N("VAEDecode", "Decode", samples=trim(), vae=mdl["vae"]()))
    vid = g.add(N("CreateVideo", "Assemble frames", images=dec(), fps=float(fps)))
    g.add(N("SaveVideo", "Save chunk", video=vid(), filename_prefix=prefix,
            format="auto", codec="auto"))
    return ks


def graph_main(cfg: dict, masked: bool = True) -> Graph:
    v = cfg["video"]
    name = ("vace_masked_depth_v2v_1p3b" if masked else "vace_unmasked_compare")
    g = Graph(name,
              "Depth-controlled VACE v2v with reference sheet and tracked subject mask"
              if masked else
              "Ablation: identical settings with NO mask and NO reference image")
    mdl = common_models(g, cfg)

    src_v = g.add(N("LoadVideo", "Source chunk (original 240p, upscaled+padded)",
                    file="chunk_source.mp4"))
    src = g.add(N("GetVideoComponents", "Source frames", video=src_v()))
    dep_v = g.add(N("LoadVideo", "Depth control", file="chunk_depth.mp4"))
    dep = g.add(N("GetVideoComponents", "Depth frames", video=dep_v()))

    if masked:
        msk_v = g.add(N("LoadVideo", "Tracked subject mask (white = regenerate)",
                        file="chunk_mask.mp4"))
        msk_i = g.add(N("GetVideoComponents", "Mask frames", video=msk_v()))
        mask = g.add(N("ImageToMask", "Mask -> MASK", image=msk_i(), channel="red"))
        if int(cfg["mask"]["feather"]) > 0:
            mask = g.add(N("FeatherMask", "Soften mask edge", mask=mask(),
                           left=int(cfg["mask"]["feather"]),
                           top=int(cfg["mask"]["feather"]),
                           right=int(cfg["mask"]["feather"]),
                           bottom=int(cfg["mask"]["feather"])))
        # THE critical composite: original RGB outside the mask, depth inside it.
        ctrl = g.add(N("ImageCompositeMasked",
                       "Control = original outside mask, depth inside",
                       destination=src(), source=dep(), x=0, y=0,
                       resize_source=False, mask=mask()))
        ref = g.add(N("LoadImage", "Reference sheet", image="reference_sheet.png"))
        vace = g.add(N("WanVaceToVideo", "VACE conditioning",
                       positive=mdl["pos"](), negative=mdl["neg"](), vae=mdl["vae"](),
                       width=int(v["width"]), height=int(v["height"]),
                       length=int(v["chunk_frames"]), batch_size=1,
                       strength=float(cfg["sampling"]["vace_strength"]),
                       control_video=ctrl(), control_masks=mask(),
                       reference_image=ref(0)))
        prefix = "vace/masked_ref"
    else:
        vace = g.add(N("WanVaceToVideo", "VACE conditioning (no mask, no reference)",
                       positive=mdl["pos"](), negative=mdl["neg"](), vae=mdl["vae"](),
                       width=int(v["width"]), height=int(v["height"]),
                       length=int(v["chunk_frames"]), batch_size=1,
                       strength=float(cfg["sampling"]["vace_strength"]),
                       control_video=dep()))
        prefix = "vace/unmasked_noref"

    sampler_tail(g, cfg, mdl, vace, v["model_fps"], prefix)
    return g


def graph_smoke(cfg: dict) -> Graph:
    """Smallest graph that still exercises every model and the VACE node itself."""
    v = cfg["video"]
    g = Graph("smoke_test_modelload",
              "Loads VACE 1.3B + UMT5 FP8 + Wan VAE and generates a short clip "
              "through WanVaceToVideo. Proves CUDA inference and the full path.")
    mdl = common_models(g, cfg)
    vace = g.add(N("WanVaceToVideo", "VACE conditioning (smoke)",
                   positive=mdl["pos"](), negative=mdl["neg"](), vae=mdl["vae"](),
                   width=272, height=272, length=17, batch_size=1, strength=1.0))
    s = cfg["sampling"]
    ks = g.add(N("KSampler", "Sampler (few steps)",
                 model=mdl["unet"](), seed=int(s["seed"]), steps=6, cfg=float(s["cfg"]),
                 sampler_name=s["sampler"], scheduler=s["scheduler"],
                 positive=vace(0), negative=vace(1), latent_image=vace(2), denoise=1.0))
    trim = g.add(N("TrimVideoLatent", "Trim", samples=ks(), trim_amount=vace(3)))
    dec = g.add(N("VAEDecode", "Decode", samples=trim(), vae=mdl["vae"]()))
    vid = g.add(N("CreateVideo", "Assemble", images=dec(), fps=float(v["model_fps"])))
    g.add(N("SaveVideo", "Save smoke clip", video=vid(),
            filename_prefix="vace/smoke", format="auto", codec="auto"))
    return g


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    log = setup_logging("build_workflows")
    cfg = load_config(args.config)
    host = cfg["runtime"]["comfy_host"]
    port = int(cfg["runtime"]["comfy_port"])

    try:
        oi = fetch_object_info(host, port)
    except Exception as e:
        log.error("Cannot reach ComfyUI at %s:%s (%s). Start it with "
                  "scripts/start_comfyui.sh --daemon; workflows are generated "
                  "against the live node schema on purpose.", host, port, e)
        return 1
    log.info("Read %d node schemas from the running ComfyUI", len(oi))

    P.workflows.mkdir(parents=True, exist_ok=True)
    graphs = [graph_main(cfg, masked=True),
              graph_main(cfg, masked=False),
              graph_smoke(cfg)]

    for g in graphs:
        api = to_api(g, oi)
        ui = to_ui(g, oi)
        (P.workflows / f"{g.name}_api.json").write_text(json.dumps(api, indent=2))
        (P.workflows / f"{g.name}.json").write_text(json.dumps(ui, indent=2))
        log.info("%-28s %2d nodes -> %s.json + %s_api.json",
                 g.name, len(g.nodes), g.name, g.name)

        # structural validation against the live schema
        for nid, nd in api.items():
            spec = oi[nd["class_type"]]
            valid = {k for k, _ in ordered_inputs(spec)}
            unknown = set(nd["inputs"]) - valid
            if unknown:
                raise KeyError(f"{g.name}: node {nid} ({nd['class_type']}) has "
                               f"unknown input(s) {unknown}. Valid: {sorted(valid)}")
            required = set((spec["input"].get("required") or {}).keys())
            missing = required - set(nd["inputs"])
            if missing:
                raise KeyError(f"{g.name}: node {nid} ({nd['class_type']}) is missing "
                               f"required input(s) {sorted(missing)}")
        log.info("  validated: all inputs exist and all required inputs are present")

    log.info("Workflows written to %s", P.workflows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
