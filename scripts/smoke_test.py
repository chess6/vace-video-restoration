#!/usr/bin/env python
"""Phase 9a - smoke test.

Proves, with a real generation and real file checks (never just exit codes):
  1. ComfyUI is reachable and reports a CUDA device
  2. all three model files load through their real loader nodes
  3. WanVaceToVideo executes and is NOT bypassed
  4. the sampler runs on GPU, not CPU
  5. a valid, non-empty, non-uniform video file comes out
  6. peak VRAM during generation is measured

Writes reports/smoke_test.json.

    scripts/smoke_test.py [--config ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_client import ComfyClient, load_api_workflow  # noqa: E402
from common import (  # noqa: E402
    P, human_size, load_config, probe_dims_fps, probe_frames, run, setup_logging,
)


def check_video(path: Path, log) -> dict:
    """A file existing is not proof. Decode it and look at the pixels."""
    import numpy as np
    import cv2

    info: dict = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    info["size_bytes"] = path.stat().st_size
    info["size_human"] = human_size(path.stat().st_size)
    w, h, fps = probe_dims_fps(path)
    n = probe_frames(path)
    info.update({"width": w, "height": h, "fps": round(fps, 4), "frames": n})

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    info["decoded_frames"] = len(frames)
    if frames:
        arr = np.stack(frames).astype(np.float32)
        info["mean"] = round(float(arr.mean()), 3)
        info["std"] = round(float(arr.std()), 3)
        # temporal variation proves it is not a frozen/black clip
        info["temporal_std"] = round(float(arr.mean(axis=(1, 2)).std()), 4)
        info["is_blank"] = bool(arr.std() < 1.0)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--timeout", type=float, default=1800)
    args = ap.parse_args()

    log = setup_logging("smoke_test")
    cfg = load_config(args.config)
    host, port = cfg["runtime"]["comfy_host"], int(cfg["runtime"]["comfy_port"])
    client = ComfyClient(host, port, log)

    results: dict = {"checks": [], "passed": 0, "failed": 0}

    def check(name: str, ok: bool, detail=""):
        results["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})
        if ok:
            results["passed"] += 1
            log.info("PASS  %s %s", name, f"- {detail}" if detail else "")
        else:
            results["failed"] += 1
            log.error("FAIL  %s - %s", name, detail)
        return ok

    # ---- 1. server ----------------------------------------------------------
    if not check("ComfyUI API reachable", client.is_up(), f"{host}:{port}"):
        log.error("Start it: scripts/start_comfyui.sh --daemon")
        return 1

    stats = client._get("/system_stats")
    devs = stats.get("devices", [])
    cuda_dev = next((d for d in devs if d.get("type") == "cuda"), None)
    check("CUDA device reported by ComfyUI", cuda_dev is not None,
          cuda_dev["name"] if cuda_dev else "no cuda device - refusing CPU fallback")
    if cuda_dev is None:
        return 1
    results["device"] = cuda_dev
    results["comfyui_version"] = stats["system"].get("comfyui_version")
    vram_total_mb = cuda_dev["vram_total"] / 2**20

    # ---- 2. models present in the loaders -----------------------------------
    oi = client.object_info()

    def combo(node, inp):
        return oi[node]["input"]["required"][inp][0]

    m = cfg["model"]
    check("VACE 1.3B visible to UNETLoader",
          m["diffusion_model"] in combo("UNETLoader", "unet_name"), m["diffusion_model"])
    check("UMT5 FP8 visible to CLIPLoader",
          m["text_encoder"] in combo("CLIPLoader", "clip_name"), m["text_encoder"])
    check("Wan VAE visible to VAELoader",
          m["vae"] in combo("VAELoader", "vae_name"), m["vae"])
    check("WanVaceToVideo node exists", "WanVaceToVideo" in oi)

    # ---- 3. workflow loads with no missing nodes ----------------------------
    for wf_name in ("vace_masked_depth_v2v_1p3b", "vace_unmasked_compare",
                    "smoke_test_modelload"):
        p = P.workflows / f"{wf_name}_api.json"
        if not p.exists():
            check(f"{wf_name} exists", False, f"missing {p}; run build_workflows.py")
            continue
        wf = load_api_workflow(p)
        missing = sorted({nd["class_type"] for nd in wf.values()
                          if nd["class_type"] not in oi})
        check(f"{wf_name}: all node types installed", not missing,
              f"{len(wf)} nodes" if not missing else f"missing {missing}")

    # ---- 4. real generation -------------------------------------------------
    wf = load_api_workflow(P.workflows / "smoke_test_modelload_api.json")

    vace_nodes = [nid for nid, nd in wf.items() if nd["class_type"] == "WanVaceToVideo"]
    check("smoke workflow contains WanVaceToVideo (not bypassed)",
          len(vace_nodes) == 1, f"node {vace_nodes}")
    # a bypassed/muted node in API format would carry mode 2/4; API graphs have no
    # mode field at all, so simply assert the sampler consumes VACE's outputs
    ks = next(nd for nd in wf.values() if nd["class_type"] == "KSampler")
    wired_to_vace = all(isinstance(ks["inputs"][k], list) and
                        ks["inputs"][k][0] in vace_nodes
                        for k in ("positive", "negative", "latent_image"))
    check("sampler consumes VACE conditioning + latent", wired_to_vace,
          f"positive/negative/latent_image <- node {vace_nodes[0]}")

    log.info("Running the smoke generation (this loads ~11 GB of weights; "
             "first run is the slow one)...")
    t0 = time.time()
    try:
        hist = client.run(wf, timeout=args.timeout)
    except Exception as e:
        check("smoke generation completed", False, str(e)[:1500])
        (P.reports / "smoke_test.json").write_text(json.dumps(results, indent=2))
        log.error("See logs/comfyui.log for the server-side traceback.")
        return 1
    elapsed = time.time() - t0
    check("smoke generation completed", True, f"{elapsed:.1f}s")
    results["elapsed_sec"] = round(elapsed, 2)
    results["peak_vram_mb"] = hist.get("_peak_vram_mb", 0)
    results["peak_vram_pct"] = round(100 * hist.get("_peak_vram_mb", 0) / vram_total_mb, 1)
    log.info("Peak GPU memory during smoke run: %s MiB of %.0f MiB (%.1f%%)",
             results["peak_vram_mb"], vram_total_mb, results["peak_vram_pct"])

    check("peak VRAM was non-trivial (proves GPU execution, not CPU)",
          results["peak_vram_mb"] > 1500,
          f"{results['peak_vram_mb']} MiB")

    # ---- 5. the produced file ------------------------------------------------
    outs = ComfyClient.output_files(hist, P.comfy_output)
    check("run produced an output file", bool(outs), ", ".join(o.name for o in outs))
    if not outs:
        (P.reports / "smoke_test.json").write_text(json.dumps(results, indent=2))
        return 1

    vid = outs[0]
    vinfo = check_video(vid, log)
    results["output"] = vinfo
    check("output video decodes", vinfo.get("decoded_frames", 0) > 0,
          f"{vinfo.get('decoded_frames')} frames, {vinfo.get('size_human')}")
    check("output has the requested frame count", vinfo.get("decoded_frames") == 17,
          f"got {vinfo.get('decoded_frames')}, expected 17")
    check("output is not a blank/uniform clip", not vinfo.get("is_blank", True),
          f"pixel std {vinfo.get('std')}")
    check("output varies over time (not a frozen frame)",
          vinfo.get("temporal_std", 0) > 0.01, f"temporal std {vinfo.get('temporal_std')}")

    # keep a copy where the user expects it
    P.pilots.mkdir(parents=True, exist_ok=True)
    dest = P.pilots / "smoke_test.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(vid), "-c", "copy", str(dest)], log)
    results["saved_copy"] = str(dest.relative_to(P.root))
    log.info("Smoke clip copied to %s", dest)

    # ---- 6. no errors in the server log --------------------------------------
    logf = P.logs / "comfyui.log"
    if logf.exists():
        txt = logf.read_text(errors="replace")
        tb = txt.count("Traceback (most recent call last)")
        check("no tracebacks in ComfyUI log", tb == 0, f"{tb} traceback(s)")

    P.reports.mkdir(parents=True, exist_ok=True)
    (P.reports / "smoke_test.json").write_text(json.dumps(results, indent=2))

    log.info("=" * 62)
    log.info("SMOKE TEST: %d passed, %d failed", results["passed"], results["failed"])
    log.info("Report: reports/smoke_test.json")
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
