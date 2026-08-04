#!/usr/bin/env python
"""Phase 8 - generate the ComfyUI workflows.

Emits, for each graph, BOTH formats from a single definition:
  workflows/<name>.json      - UI format, openable in the ComfyUI editor
  workflows/<name>_api.json  - API format, used by scripts/run_chunk.py

Node signatures (input names, order, widget vs link, combo options,
control_after_generate) are read from the RUNNING ComfyUI's /object_info rather
than hardcoded, so a generated workflow cannot drift from the installed revision.

Graphs produced. The mask and the reference sheet are independent switches, so
each ablation changes exactly one variable against the baseline:
  vace_masked_depth_v2v_1p3b   the baseline: depth control + reference sheet +
                               tracked subject mask
  vace_masked_noref            baseline minus the reference sheet only
                               (the controlled reference ablation)
  vace_unmasked_ref            baseline minus the mask only
  vace_unmasked_compare        neither mask nor reference
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


# The mask and the reference sheet are INDEPENDENT switches. They used to be one,
# which meant the "no reference" ablation silently also removed the mask and so
# compared two things at once. Each combination gets its own graph and its own
# name, so run_chunks.py can drop exactly one variable at a time.
GRAPH_NAMES = {
    (True, True): "vace_masked_depth_v2v_1p3b",   # the baseline
    (True, False): "vace_masked_noref",           # controlled reference ablation
    (False, True): "vace_unmasked_ref",           # controlled mask ablation
    (False, False): "vace_unmasked_compare",      # neither: the loosest comparison
}


def graph_name(masked: bool, reference: bool, background: bool = False) -> str:
    """`_bg` variants take their preserved pixels from the SeedVR2 plate instead
    of the original. Only the masked graphs have a preserved region to speak of,
    so only those get a background variant."""
    name = GRAPH_NAMES[(bool(masked), bool(reference))]
    return f"{name}_bg" if background and masked else name


def graph_main(cfg: dict, masked: bool = True, reference: bool = True,
               background: bool = False) -> Graph:
    v = cfg["video"]
    what = ("tracked subject mask" if masked else "no mask",
            "reference sheet" if reference else "no reference image")
    g = Graph(graph_name(masked, reference, background),
              f"Depth-controlled VACE v2v with {what[0]} and {what[1]}"
              + (", preserving a SeedVR2-restored background" if background else ""))
    mdl = common_models(g, cfg)

    src_v = g.add(N("LoadVideo", "Source chunk (original 240p, upscaled+padded)",
                    file="chunk_source.mp4"))
    src = g.add(N("GetVideoComponents", "Source frames", video=src_v()))
    dep_v = g.add(N("LoadVideo", "Depth control", file="chunk_depth.mp4"))
    dep = g.add(N("GetVideoComponents", "Depth frames", video=dep_v()))

    # The plate the preserved region is taken from. WanVaceToVideo computes
    # inactive = control_video * (1 - mask), so whatever supplies control_video
    # OUTSIDE the mask is what survives into the output untouched. Pointing that
    # at the SeedVR2 restoration is the whole background integration: the model
    # sees a restored environment as the thing it must keep, and regenerates only
    # the masked figure. The depth, the mask and the chunk timing still come from
    # the original stream, so nothing structural depends on the restoration.
    if background:
        bg_v = g.add(N("LoadVideo", "SeedVR2-restored background chunk",
                       file="chunk_background.mp4"))
        base = g.add(N("GetVideoComponents", "Background frames", video=bg_v()))
    else:
        base = src

    vace_kw = dict(positive=mdl["pos"](), negative=mdl["neg"](), vae=mdl["vae"](),
                   width=int(v["width"]), height=int(v["height"]),
                   length=int(v["chunk_frames"]), batch_size=1,
                   strength=float(cfg["sampling"]["vace_strength"]))

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
        # THE critical composite: preserved-base RGB outside the mask, depth
        # inside it. The base is the original, or the SeedVR2 plate when the
        # background stage is in use.
        ctrl = g.add(N("ImageCompositeMasked",
                       "Control = base outside mask, depth inside",
                       destination=base(), source=dep(), x=0, y=0,
                       resize_source=False, mask=mask()))
        vace_kw.update(control_video=ctrl(), control_masks=mask())
    else:
        # No mask means no preserved region, so the whole frame is regenerated and
        # the control video is depth everywhere.
        vace_kw.update(control_video=dep())

    if reference:
        ref = g.add(N("LoadImage", "Reference sheet", image="reference_sheet.png"))
        vace_kw.update(reference_image=ref(0))

    vace = g.add(N("WanVaceToVideo", f"VACE conditioning ({what[0]}, {what[1]})",
                   **vace_kw))
    prefix = f"vace/{'masked' if masked else 'unmasked'}_{'ref' if reference else 'noref'}"
    sampler_tail(g, cfg, mdl, vace, v["model_fps"], prefix)
    return g


def dynamic_options(spec: dict) -> dict[str, list]:
    """Inputs of a node schema that are dynamic combos, mapped to their options.

    Their nested inputs do not appear in the schema's own input list, so any
    check that compares a prompt against `ordered_inputs` has to be told about
    them or it will reject a perfectly valid graph.
    """
    out = {}
    for section in ("required", "optional"):
        for name, val in (spec.get("input", {}).get(section) or {}).items():
            if isinstance(val, list) and val and val[0] == "COMFY_DYNAMICCOMBO_V3":
                meta = val[1] if len(val) > 1 and isinstance(val[1], dict) else {}
                out[name] = meta.get("options", [])
    return out


def graph_seedvr2(cfg: dict, profile: str) -> Graph:
    """Full-frame background restoration with SeedVR2 3B.

    Runs before VACE and independently of it. Its output is only ever used as the
    plate the subject sits on: no scene cut, timestamp, mask, depth or tracking
    decision is derived from it, so changing profile cannot move a chunk boundary
    or shift the subject.

    Node semantics are read from the installed revision,
    ComfyUI/comfy_extras/nodes_seedvr.py:
      * SeedVR2Preprocess pads to a multiple of 16 and to 4n+1 frames
      * SeedVR2TemporalChunk emits a LIST of latents; ComfyUI then runs the
        conditioning and the sampler once per chunk, which is what keeps peak
        VRAM flat instead of scaling with clip length
      * temporal_overlap is counted in LATENT frames and is clamped to
        (frames_per_chunk-1)/4 by the node itself
      * SeedVR2PostProcessing colour-matches the result back to the pre-upscale
        frames, which is the main conservative/aggressive lever
    """
    b = cfg["background"]
    p = b["profiles"][profile]
    v = cfg["video"]
    g = Graph(f"seedvr2_{profile}",
              f"SeedVR2 3B ({b['weight_dtype']}) full-frame restoration - {profile}. "
              f"{p['description'].strip()}")

    unet = g.add(N("UNETLoader", "SeedVR2 3B",
                   unet_name=b["model"], weight_dtype=b["weight_dtype"]))
    vae = g.add(N("VAELoader", "SeedVR2 VAE", vae_name=b["vae"]))

    src_v = g.add(N("LoadVideo", "Source chunk", file="bg_source.mp4"))
    src = g.add(N("GetVideoComponents", "Source frames", video=src_v()))

    # Restore at the configured short edge, then hand the SAME resized frames to
    # post-processing as the colour reference.
    tw, th = seedvr2_target_size(int(v["width"]), int(v["height"]),
                                 int(b["target_short_edge"]))
    resized = g.add(N("ImageScale", "Resize for SeedVR2", image=src(),
                      upscale_method="lanczos", width=tw, height=th,
                      crop="disabled"))
    pre = g.add(N("SeedVR2Preprocess", "Pad to 16 / 4n+1", resized_images=resized()))

    # Tiled on BOTH sides. The encode sees the whole clip at once, before
    # SeedVR2TemporalChunk has had a chance to bound anything, so a plain
    # VAEEncode is the one place where peak VRAM still scales with clip length -
    # measured at 11.9 GiB of 12.3 GiB on only 21 frames.
    enc = g.add(N("VAEEncodeTiled", "Encode (tiled)", pixels=pre(), vae=vae(),
                  tile_size=int(b["vae_tile_size"]),
                  overlap=int(b["vae_tile_overlap"]),
                  temporal_size=int(b["frames_per_chunk"]) - 1,
                  temporal_overlap=4))
    # chunking_mode is a COMFY_DYNAMICCOMBO_V3: the prompt carries the selected
    # option key under the input's own name, and each nested input of that option
    # under "<parent>.<child>" (comfy_api/latest/_io.py::finalize_prefix). A dot
    # is not a valid Python identifier, so the nested key is set after
    # construction rather than as a keyword argument.
    chunk_node = N("SeedVR2TemporalChunk", "Split into temporal chunks",
                   latent=enc(), temporal_overlap=int(b["temporal_overlap"]),
                   chunking_mode=b.get("chunking_mode", "manual"))
    if chunk_node.inputs["chunking_mode"] == "manual":
        chunk_node.inputs["chunking_mode.frames_per_chunk"] = int(b["frames_per_chunk"])
    chunk = g.add(chunk_node)
    cond = g.add(N("SeedVR2Conditioning", "Conditioning",
                   model=unet(), vae_conditioning=chunk(0)))
    ks = g.add(N("KSampler", "Sampler",
                 model=unet(), seed=int(cfg["sampling"]["seed"]),
                 steps=int(p["steps"]), cfg=float(p["cfg"]),
                 sampler_name=cfg["sampling"]["sampler"],
                 scheduler=cfg["sampling"]["scheduler"],
                 positive=cond(0), negative=cond(1), latent_image=chunk(0),
                 denoise=float(p["denoise"])))
    merged = g.add(N("SeedVR2TemporalMerge", "Merge temporal chunks",
                     latents=ks(), temporal_overlap=chunk(1)))
    # Tiled decode: bounds the decode peak independently of frame size, which is
    # what makes 720p restoration possible at all on a 12 GB card.
    dec = g.add(N("VAEDecodeTiled", "Decode (tiled)", samples=merged(), vae=vae(),
                  tile_size=int(b["vae_tile_size"]), overlap=int(b["vae_tile_overlap"]),
                  temporal_size=int(b["frames_per_chunk"]) - 1,
                  temporal_overlap=4))
    post = g.add(N("SeedVR2PostProcessing", "Align + colour match",
                   images=dec(), original_resized_images=resized(),
                   color_correction_method=p["color_correction"]))
    # Back to the working-stream geometry, so the plate is frame- and
    # pixel-aligned with the depth, the mask and the original.
    back = g.add(N("ImageScale", "Back to working geometry", image=post(),
                   upscale_method="lanczos", width=int(v["width"]),
                   height=int(v["height"]), crop="disabled"))
    vid = g.add(N("CreateVideo", "Assemble frames", images=back(),
                  fps=float(v["model_fps"])))
    g.add(N("SaveVideo", "Save restored background", video=vid(),
            filename_prefix=f"seedvr2/{profile}", format="auto", codec="auto"))
    return g


def seedvr2_target_size(w: int, h: int, short_edge: int) -> tuple[int, int]:
    """Scale so the SHORT edge hits `short_edge`, keeping aspect, both axes even.

    Never downscales: restoring below the working resolution would throw away
    detail the rest of the pipeline still has.
    """
    scale = max(1.0, short_edge / float(min(w, h)))
    return (int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2))


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
    graphs = [graph_main(cfg, masked=m, reference=r) for m, r in GRAPH_NAMES]
    # Background-preserving variants of the masked graphs, plus one SeedVR2
    # restoration graph per background profile.
    graphs += [graph_main(cfg, masked=True, reference=r, background=True)
               for r in (True, False)]
    if cfg.get("background", {}).get("enabled"):
        graphs += [graph_seedvr2(cfg, p) for p in cfg["background"]["profiles"]]
    graphs.append(graph_smoke(cfg))

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
            # A dynamic combo declares one input whose selected option carries
            # nested inputs, and the prompt names those "<parent>.<child>"
            # (comfy_api/latest/_io.py::finalize_prefix). They are legal inputs
            # even though the schema lists only the parent, so expand the valid
            # set with the nested names belonging to the option actually chosen.
            for pname, pspec in dynamic_options(spec).items():
                chosen = nd["inputs"].get(pname)
                for opt in pspec:
                    if opt.get("key") == chosen:
                        for child in (opt.get("inputs", {}).get("required") or {}):
                            valid.add(f"{pname}.{child}")
                        for child in (opt.get("inputs", {}).get("optional") or {}):
                            valid.add(f"{pname}.{child}")
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
