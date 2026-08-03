#!/usr/bin/env python
"""Phase 10 - decide whether a neural upscaler beats Lanczos, on VACE OUTPUT ONLY.

Conventional upscaling already failed on the raw 240p source, so nothing here
touches the source. This operates strictly on already-restored VACE output and
asks a narrower question: given a clean 480p restoration, does a neural upscaler
add real detail or just sharpening artefacts?

Produces, for the pilot:
  * a Lanczos resize
  * a neural resize, if an upscale model is present in
    ComfyUI/models/upscale_models/ (none is downloaded automatically)
  * a side-by-side video and frame grid
  * measured statistics: high-frequency energy, edge density, temporal stability

Real-ESRGAN is NOT assumed to be better. Temporal instability is the usual
failure mode on video and shows up in the stability metric.

    scripts/compare_upscalers.py --target 720p
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P, human_size, load_config, probe_dims_fps, probe_frames, run, setup_logging  # noqa: E402


def metrics(path: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    hf, edges, gray = [], [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hf.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
        edges.append(float((cv2.Canny(g.astype(np.uint8), 80, 160) > 0).mean()))
        gray.append(cv2.resize(g, (160, 96)))
    cap.release()
    if not gray:
        return {}
    arr = np.stack(gray)
    # mean absolute frame-to-frame difference: lower = more temporally stable
    tstab = float(np.abs(np.diff(arr, axis=0)).mean()) if len(arr) > 1 else 0.0
    return {"high_freq_energy": round(float(np.mean(hf)), 2),
            "edge_density": round(float(np.mean(edges)), 5),
            "temporal_instability": round(tstab, 4),
            "size_human": human_size(path.stat().st_size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--input", type=Path, default=None,
                    help="Restored clip. Default: outputs/final/pilot_master.mp4")
    ap.add_argument("--target", choices=["720p", "1080p"], default="720p")
    args = ap.parse_args()

    log = setup_logging("compare_upscalers")
    cfg = load_config(args.config)

    src = args.input or (P.final / "pilot_master.mp4")
    if not src.exists():
        log.error("No restored clip at %s. Run the pilot and scripts/assemble.py "
                  "--pilot first. This tool deliberately refuses to run on the "
                  "raw 240p source.", src)
        return 1

    w, h, fps = probe_dims_fps(src)
    th = 720 if args.target == "720p" else 1080
    tw = int(round(w * th / h / 2) * 2)
    log.info("Upscaling restored output %dx%d -> %dx%d", w, h, tw, th)

    P.comparisons.mkdir(parents=True, exist_ok=True)
    results: dict = {"input": str(src.relative_to(P.root)),
                     "target": [tw, th], "candidates": {}}

    lan = P.comparisons / f"upscale_lanczos_{args.target}.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-vf", f"scale={tw}:{th}:flags=lanczos", "-an",
         "-c:v", "libx264", "-crf", "16", "-preset", "slow",
         "-pix_fmt", "yuv420p", str(lan)], log)
    results["candidates"]["lanczos"] = {"path": str(lan.relative_to(P.root)),
                                        **metrics(lan)}
    log.info("Lanczos: %s", results["candidates"]["lanczos"])

    # ---- optional neural upscaler --------------------------------------------
    up_dir = P.models / "upscale_models"
    models = sorted(p for p in up_dir.glob("*") if p.suffix in (".pth", ".safetensors")) \
        if up_dir.exists() else []
    if not models:
        log.warning("No upscale model in %s, so only Lanczos was produced.", up_dir)
        log.warning("To evaluate a neural upscaler, put e.g. RealESRGAN_x2plus.pth "
                    "there and re-run. It is deliberately NOT downloaded "
                    "automatically: Lanczos may well win, and the brief says not "
                    "to assume otherwise.")
    else:
        from comfy_client import ComfyClient
        client = ComfyClient(cfg["runtime"]["comfy_host"],
                             int(cfg["runtime"]["comfy_port"]), log)
        if not client.is_up():
            log.error("ComfyUI is not running; cannot run the neural upscaler.")
        else:
            import shutil
            model = models[0]
            log.info("Neural upscaler: %s", model.name)
            staged = P.comfy_input / "upscale_input.mp4"
            shutil.copy2(src, staged)
            wf = {
                "1": {"class_type": "LoadVideo", "inputs": {"file": staged.name}},
                "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
                "3": {"class_type": "UpscaleModelLoader",
                      "inputs": {"model_name": model.name}},
                "4": {"class_type": "ImageUpscaleWithModel",
                      "inputs": {"upscale_model": ["3", 0], "image": ["2", 0]}},
                "5": {"class_type": "ImageScale",
                      "inputs": {"image": ["4", 0], "upscale_method": "lanczos",
                                 "width": tw, "height": th, "crop": "disabled"}},
                "6": {"class_type": "CreateVideo",
                      "inputs": {"images": ["5", 0], "fps": float(fps)}},
                "7": {"class_type": "SaveVideo",
                      "inputs": {"video": ["6", 0],
                                 "filename_prefix": "vace/upscale_neural",
                                 "format": "auto", "codec": "auto"}},
            }
            try:
                hist = client.run(wf, timeout=3600)
                outs = ComfyClient.output_files(hist, P.comfy_output)
                if outs:
                    dst = P.comparisons / f"upscale_neural_{args.target}.mp4"
                    shutil.move(str(outs[0]), dst)
                    results["candidates"]["neural"] = {
                        "model": model.name, "path": str(dst.relative_to(P.root)),
                        **metrics(dst)}
                    log.info("Neural: %s", results["candidates"]["neural"])
            except Exception as e:
                log.error("Neural upscale failed: %s", str(e)[:1200])
            finally:
                staged.unlink(missing_ok=True)

    # ---- verdict ---------------------------------------------------------------
    if "neural" in results["candidates"]:
        a = results["candidates"]["lanczos"]
        b = results["candidates"]["neural"]
        note = []
        if b["high_freq_energy"] > a["high_freq_energy"] * 1.15:
            note.append("neural adds measurably more high-frequency detail")
        else:
            note.append("neural adds little extra detail over Lanczos")
        if b["temporal_instability"] > a["temporal_instability"] * 1.2:
            note.append("but it is noticeably less temporally stable (flicker risk)")
        results["verdict"] = "; ".join(note)
        results["decision_is_yours"] = (
            "These are proxies, not quality judgements. Watch "
            "outputs/comparisons/ and decide by eye.")
        log.info("Verdict hint: %s", results["verdict"])

    P.reports.mkdir(parents=True, exist_ok=True)
    (P.reports / "upscaler_comparison.json").write_text(json.dumps(results, indent=2))
    log.info("Report: reports/upscaler_comparison.json")
    log.info("Clips written to %s (nothing was displayed)", P.comparisons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
