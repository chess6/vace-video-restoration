#!/usr/bin/env python
"""Phase 11 - runtime guard. Benchmark one 81-frame generation and extrapolate.

Reports:
  * seconds per generated frame
  * estimated wall-clock for the whole 30-minute video
  * estimated intermediate and output disk usage
  * peak VRAM

Writes reports/benchmark.json, which scripts/run_full.sh reads and prints before
it will start.

    scripts/benchmark.py              # uses a real chunk if one exists
    scripts/benchmark.py --synthetic  # no source media needed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_client import ComfyClient, load_api_workflow, set_input  # noqa: E402
from common import (  # noqa: E402
    P, human_size, human_time, load_config, load_manifest, probe_frames, run,
    setup_logging,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="Benchmark on generated media instead of the real source")
    ap.add_argument("--target-minutes", type=float, default=30.0)
    ap.add_argument("--timeout", type=float, default=7200)
    args = ap.parse_args()

    log = setup_logging("benchmark")
    cfg = load_config(args.config)
    v = cfg["video"]
    W, H = int(v["width"]), int(v["height"])
    LEN = int(v["chunk_frames"])
    OVERLAP = int(v["chunk_overlap"])
    fps = int(v["model_fps"])
    steps = int(cfg["sampling"]["steps"])

    client = ComfyClient(cfg["runtime"]["comfy_host"],
                         int(cfg["runtime"]["comfy_port"]), log)
    if not client.is_up():
        log.error("ComfyUI is not running. scripts/start_comfyui.sh --daemon")
        return 1

    # ---- build the benchmark job ---------------------------------------------
    wf = load_api_workflow(P.workflows / "smoke_test_modelload_api.json")
    set_input(wf, "WanVaceToVideo", "width", W)
    set_input(wf, "WanVaceToVideo", "height", H)
    set_input(wf, "WanVaceToVideo", "length", LEN)
    set_input(wf, "KSampler", "steps", steps)
    set_input(wf, "CreateVideo", "fps", float(fps))
    set_input(wf, "SaveVideo", "filename_prefix", "vace/benchmark")

    log.info("Benchmarking %dx%d x %d frames, %d steps (%s)...",
             W, H, LEN, steps, cfg["sampling"]["sampler"])
    log.info("Weights are already resident if ComfyUI has run before; the first "
             "ever run includes a one-off model load.")

    t0 = time.time()
    hist = client.run(wf, timeout=args.timeout)
    elapsed = time.time() - t0
    peak = hist.get("_peak_vram_mb", 0)

    outs = ComfyClient.output_files(hist, P.comfy_output)
    n = probe_frames(outs[0]) if outs else 0
    if n != LEN:
        log.error("Benchmark produced %d frames, expected %d", n, LEN)
        return 1

    sec_per_frame = elapsed / LEN
    log.info("Chunk of %d frames took %s -> %.3f s per generated frame",
             LEN, human_time(elapsed), sec_per_frame)
    log.info("Peak VRAM: %s MiB", peak)

    # ---- extrapolate ----------------------------------------------------------
    man = None
    try:
        man = load_manifest()
    except Exception:
        pass

    if man and man.get("chunks"):
        n_chunks = len(man["chunks"])
        gen_frames = sum(c["n_frames"] for c in man["chunks"])
        total_src_frames = man["normalized"]["total_frames"]
        source = "the real chunk manifest"
    else:
        total_src_frames = int(args.target_minutes * 60 * fps)
        stride = LEN - OVERLAP
        n_chunks = max(1, -(-total_src_frames // stride))
        gen_frames = n_chunks * LEN
        source = f"an estimate for {args.target_minutes:.0f} minutes at {fps} fps"

    total_sec = gen_frames * sec_per_frame
    log.info("Extrapolating from %s", source)

    # ---- disk -----------------------------------------------------------------
    # Grounded in measurements on this machine at 832x480 (see reports/versions.md
    # for the encoder settings), then rounded UP, because synthetic test patterns
    # compress differently from real footage and an under-estimate is the
    # dangerous direction. Measured references:
    #   h264 from SaveVideo   ~5.5-8.2 kB/frame on generated content
    #   ffv1 gray (depth)     ~15 kB/frame
    #   ffv1 gray (mask)      ~0.4 kB/frame (near-binary + feather)
    #   ffv1 colour           ~26 kB/frame
    # Detailed real footage can run several times higher, hence the headroom.
    bytes_per_frame = {"depth_ffv1_gray": 25_000, "mask_ffv1_gray": 8_000,
                       "chunk_src_h264": 25_000, "restored_h264": 40_000,
                       "assembly_ffv1": 120_000}
    inter = (total_src_frames * (bytes_per_frame["depth_ffv1_gray"] +
                                 bytes_per_frame["mask_ffv1_gray"]) +
             gen_frames * bytes_per_frame["chunk_src_h264"])
    outp = (gen_frames * bytes_per_frame["restored_h264"] +
            total_src_frames * bytes_per_frame["assembly_ffv1"])

    result = {
        "resolution": [W, H], "chunk_frames": LEN, "chunk_overlap": OVERLAP,
        "steps": steps, "sampler": cfg["sampling"]["sampler"], "model_fps": fps,
        "measured": {
            "chunk_seconds": round(elapsed, 2),
            "seconds_per_generated_frame": round(sec_per_frame, 4),
            "peak_vram_mb": peak,
        },
        "projection_basis": source,
        "projection": {
            "source_frames": total_src_frames,
            "chunks": n_chunks,
            "generated_frames": gen_frames,
            "overlap_overhead_pct": round(100 * (gen_frames / max(total_src_frames, 1) - 1), 1),
            "total_seconds": round(total_sec, 1),
            "total_human": human_time(total_sec),
            "hours": round(total_sec / 3600, 2),
        },
        "disk_estimate": {
            "intermediate_bytes": int(inter), "intermediate_human": human_size(inter),
            "outputs_bytes": int(outp), "outputs_human": human_size(outp),
            "total_human": human_size(inter + outp),
        },
    }
    P.reports.mkdir(parents=True, exist_ok=True)
    (P.reports / "benchmark.json").write_text(json.dumps(result, indent=2))

    log.info("=" * 62)
    log.info("PROJECTION for the full job")
    log.info("  chunks              : %d", n_chunks)
    log.info("  frames to generate  : %d (%.0f%% overlap overhead)",
             gen_frames, result["projection"]["overlap_overhead_pct"])
    log.info("  estimated wall clock: %s (%.1f h)",
             human_time(total_sec), total_sec / 3600)
    log.info("  intermediate disk   : %s", human_size(inter))
    log.info("  output disk         : %s", human_size(outp))
    log.info("  TOTAL disk          : %s", human_size(inter + outp))
    log.info("  peak VRAM           : %s MiB", peak)
    log.info("Report: reports/benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
