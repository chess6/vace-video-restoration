#!/usr/bin/env python
"""Phase 6 diagnostic - prove which mask colour marks the regenerated region.

Reading the node source says WHITE regenerates:

    ComfyUI/comfy_extras/nodes_wan.py :: WanVaceToVideo
        inactive = control_video * (1 - mask)     # preserved
        reactive = control_video * mask           # regenerated

This script proves it end to end instead of trusting the reading. It runs a real
generation with a control video and a mask that is WHITE on the LEFT half and
BLACK on the RIGHT half, then measures how much each half changed relative to the
control input.

Expected if white = regenerate:
    left  (white, reactive) changes a lot
    right (black, inactive) changes very little
and the script asserts the ratio, so a future ComfyUI update that flips the
convention will fail this test loudly rather than silently inverting your masks.

Uses a synthetic control video, so it needs none of your media.

    scripts/verify_mask_polarity.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_client import ComfyClient, load_api_workflow, set_input  # noqa: E402
from common import P, load_config, probe_frames, run, setup_logging  # noqa: E402

W, H, LEN, STEPS = 448, 256, 17, 6


def make_control(path: Path, log) -> None:
    """A structured, high-contrast moving pattern so change is easy to measure."""
    run(["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate=16:duration=2",
         "-frames:v", str(LEN), "-c:v", "libx264", "-crf", "10",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)], log)


def make_half_mask(path: Path, log) -> None:
    """Left half white (should regenerate), right half black (should be preserved)."""
    run(["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c=white:size={W//2}x{H}:rate=16:duration=2",
         "-f", "lavfi", "-i", f"color=c=black:size={W//2}x{H}:rate=16:duration=2",
         "-filter_complex", "[0:v][1:v]hstack=inputs=2,format=yuv420p",
         "-frames:v", str(LEN), "-c:v", "libx264", "-crf", "8",
         "-preset", "veryfast", str(path)], log)


def read_gray(path: Path, n: int) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fr = []
    while len(fr) < n:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    return np.stack(fr) if fr else np.zeros((0, H, W), np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ratio", type=float, default=1.8,
                    help="Minimum (white-half change)/(black-half change) to pass")
    args = ap.parse_args()

    log = setup_logging("verify_mask_polarity")
    cfg = load_config(args.config)
    client = ComfyClient(cfg["runtime"]["comfy_host"],
                         int(cfg["runtime"]["comfy_port"]), log)
    if not client.is_up():
        log.error("ComfyUI is not running. scripts/start_comfyui.sh --daemon")
        return 1

    wf_path = P.workflows / "vace_masked_depth_v2v_1p3b_api.json"
    if not wf_path.exists():
        log.error("Missing %s. Run scripts/build_workflows.py", wf_path)
        return 1

    P.comfy_input.mkdir(parents=True, exist_ok=True)
    ctrl = P.comfy_input / "_polarity_control.mp4"
    mask = P.comfy_input / "_polarity_mask.mp4"
    make_control(ctrl, log)
    make_half_mask(mask, log)
    log.info("Built synthetic control (%d frames) and a left-white/right-black mask",
             probe_frames(ctrl))

    # Reference sheet: reuse the real one if present, else a neutral grey frame so
    # the graph's LoadImage has something valid to read.
    ref_name = "reference_sheet.png"
    if not (P.comfy_input / ref_name).exists():
        ref_name = "_polarity_ref.png"
        run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", f"color=c=gray:size={W}x{H}", "-frames:v", "1",
             str(P.comfy_input / ref_name)], log)

    wf = load_api_workflow(wf_path)
    set_input(wf, "LoadVideo", "file", ctrl.name, title_contains="source")
    # depth input also gets the control clip: this test is about mask polarity,
    # so the two control streams are deliberately identical.
    set_input(wf, "LoadVideo", "file", ctrl.name, title_contains="depth")
    set_input(wf, "LoadVideo", "file", mask.name, title_contains="mask")
    set_input(wf, "LoadImage", "image", ref_name)
    set_input(wf, "WanVaceToVideo", "width", W)
    set_input(wf, "WanVaceToVideo", "height", H)
    set_input(wf, "WanVaceToVideo", "length", LEN)
    set_input(wf, "KSampler", "steps", STEPS)
    set_input(wf, "CreateVideo", "fps", 16.0)
    set_input(wf, "SaveVideo", "filename_prefix", "vace/polarity")
    # Feather would blur the boundary and muddy the measurement.
    for nid, nd in list(wf.items()):
        if nd["class_type"] == "FeatherMask":
            for other in wf.values():
                for k, vv in other["inputs"].items():
                    if isinstance(vv, list) and vv[0] == nid:
                        other["inputs"][k] = nd["inputs"]["mask"]
            wf.pop(nid)
            log.info("Removed FeatherMask so the half-boundary stays sharp")

    log.info("Running the polarity generation...")
    hist = client.run(wf, timeout=1800)
    outs = ComfyClient.output_files(hist, P.comfy_output)
    if not outs:
        log.error("No output produced")
        return 1
    out = outs[0]

    a = read_gray(ctrl, LEN)
    b = read_gray(out, LEN)
    n = min(len(a), len(b))
    if n == 0:
        log.error("Could not decode frames for comparison")
        return 1
    a, b = a[:n], b[:n]
    if a.shape[1:] != b.shape[1:]:
        import cv2
        b = np.stack([cv2.resize(f, (a.shape[2], a.shape[1])) for f in b])

    half = a.shape[2] // 2
    left = float(np.abs(a[:, :, :half] - b[:, :, :half]).mean())     # white half
    right = float(np.abs(a[:, :, half:] - b[:, :, half:]).mean())    # black half
    ratio = left / max(right, 1e-6)

    log.info("=" * 62)
    log.info("mean |control - output|")
    log.info("  LEFT  half, mask WHITE : %7.3f", left)
    log.info("  RIGHT half, mask BLACK : %7.3f", right)
    log.info("  ratio white/black      : %7.2f  (need >= %.2f)", ratio, args.ratio)

    verdict = "white_is_regenerate" if ratio >= args.ratio else (
        "black_is_regenerate" if ratio <= 1.0 / args.ratio else "inconclusive")
    ok = verdict == "white_is_regenerate"

    result = {
        "width": W, "height": H, "length": LEN, "steps": STEPS,
        "left_white_mean_abs_change": round(left, 4),
        "right_black_mean_abs_change": round(right, 4),
        "ratio": round(ratio, 3), "required_ratio": args.ratio,
        "verdict": verdict, "passed": ok,
        "config_expects": cfg["mask"]["polarity"],
        "source_of_truth": ("ComfyUI/comfy_extras/nodes_wan.py::WanVaceToVideo, "
                            "reactive = control_video * mask"),
        "output_clip": str(out),
    }
    P.reports.mkdir(parents=True, exist_ok=True)
    (P.reports / "mask_polarity.json").write_text(json.dumps(result, indent=2))

    if ok:
        log.info("PASS: WHITE marks the regenerated region; BLACK is preserved.")
        log.info("The tracker therefore writes the subject as WHITE.")
    else:
        log.error("FAIL: verdict %s. Do not trust the masks until this is resolved; "
                  "the config assumes %s.", verdict, cfg["mask"]["polarity"])
    log.info("Report: reports/mask_polarity.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
