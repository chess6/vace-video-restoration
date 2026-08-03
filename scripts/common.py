"""Shared helpers for the VACE restoration pipeline.

Design rules enforced here:
  * Nothing in this project ever opens a viewer, window or player. All visual
    artefacts are written to disk as files. `ffplay`, `cv2.imshow`, `PIL.Image.show`
    and `xdg-open` are never called anywhere in this codebase.
  * The original source video and reference images are opened read-only and are
    never written to, moved or re-encoded in place.
  * Long operations are resumable: they record per-item status in a JSON manifest
    and skip items already marked done.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable

# Enforce the no-display rule at import time, before any plotting library can
# pick an interactive backend from the environment. Set rather than defaulted:
# a DISPLAY-derived backend must never win here.
os.environ["MPLBACKEND"] = "Agg"

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

class P:
    root = PROJECT_ROOT
    comfy = PROJECT_ROOT / "ComfyUI"
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    configs = PROJECT_ROOT / "configs"
    inputs = PROJECT_ROOT / "inputs"
    source = PROJECT_ROOT / "inputs" / "source"
    references = PROJECT_ROOT / "inputs" / "references"
    subject_seeds = PROJECT_ROOT / "inputs" / "subject_seeds"
    intermediate = PROJECT_ROOT / "intermediate"
    normalized = PROJECT_ROOT / "intermediate" / "normalized"
    shots = PROJECT_ROOT / "intermediate" / "shots"
    chunks = PROJECT_ROOT / "intermediate" / "chunks"
    depth = PROJECT_ROOT / "intermediate" / "depth"
    masks = PROJECT_ROOT / "intermediate" / "masks"
    reference_sheets = PROJECT_ROOT / "intermediate" / "reference_sheets"
    workflows = PROJECT_ROOT / "workflows"
    scripts = PROJECT_ROOT / "scripts"
    outputs = PROJECT_ROOT / "outputs"
    pilots = PROJECT_ROOT / "outputs" / "pilots"
    comparisons = PROJECT_ROOT / "outputs" / "comparisons"
    restored_480p = PROJECT_ROOT / "outputs" / "restored_480p"
    final = PROJECT_ROOT / "outputs" / "final"
    logs = PROJECT_ROOT / "logs"
    reports = PROJECT_ROOT / "reports"
    manifest = PROJECT_ROOT / "intermediate" / "chunk_manifest.json"
    models = PROJECT_ROOT / "ComfyUI" / "models"
    comfy_input = PROJECT_ROOT / "ComfyUI" / "input"
    comfy_output = PROJECT_ROOT / "ComfyUI" / "output"


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(name: str, verbose: bool = False) -> logging.Logger:
    """Log to both stdout and logs/<name>.log so long runs leave a trail."""
    P.logs.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(sh)

    fh = logging.FileHandler(P.logs / f"{name}.log")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else P.configs / "local_1p3b.yaml"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Fail loudly on values the installed VACE node cannot accept.

    These constraints come from ComfyUI/comfy_extras/nodes_wan.py::WanVaceToVideo.
    """
    v = cfg["video"]
    n = v["chunk_frames"]
    if (n - 1) % 4 != 0:
        raise ValueError(
            f"chunk_frames={n} is invalid. WanVaceToVideo declares "
            f"length step=4 from min=1, so only 4n+1 values are accepted "
            f"(…, 73, 77, 81, 85, …). Nearest valid: {((n - 1) // 4) * 4 + 1}."
        )
    for key in ("width", "height"):
        if v[key] % 16 != 0:
            raise ValueError(
                f"video.{key}={v[key]} is invalid. WanVaceToVideo declares "
                f"{key} step=16, so it must be a multiple of 16."
            )
    if v["chunk_overlap"] >= n:
        raise ValueError(f"chunk_overlap={v['chunk_overlap']} must be < chunk_frames={n}")
    if cfg["mask"]["polarity"] != "white_is_regenerate":
        raise ValueError(
            "mask.polarity must be 'white_is_regenerate'. Verified from source: "
            "WanVaceToVideo computes reactive = control_video * mask, so a white "
            "(1.0) mask marks the region the model regenerates."
        )


def nearest_valid_length(n: int) -> int:
    """Snap a frame count down to the nearest 4n+1 accepted by WanVaceToVideo."""
    n = max(1, n)
    return ((n - 1) // 4) * 4 + 1


def round_to_16(x: int) -> int:
    return max(16, int(round(x / 16)) * 16)


# ---------------------------------------------------------------------------
# Subprocess / ffmpeg
# ---------------------------------------------------------------------------

def run(cmd: list[str], logger: logging.Logger | None = None, check: bool = True,
        capture: bool = True) -> subprocess.CompletedProcess:
    if logger:
        logger.debug("$ %s", " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], check=False,
                          capture_output=capture, text=True)
    if check and proc.returncode != 0:
        msg = f"Command failed ({proc.returncode}): {' '.join(str(c) for c in cmd)}"
        if capture:
            msg += f"\nstderr:\n{proc.stderr[-4000:]}"
        raise RuntimeError(msg)
    return proc


def ffprobe_json(path: Path) -> dict:
    proc = run(["ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", "-show_chapters", str(path)])
    return json.loads(proc.stdout)


def require_tools(*tools: str) -> None:
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"Missing required tool(s): {', '.join(missing)}. "
            f"Install with: sudo apt-get install -y {' '.join(missing)}"
        )


def parse_fraction(s: str) -> float:
    if s is None:
        return 0.0
    s = str(s)
    if "/" in s:
        a, b = s.split("/", 1)
        b = float(b)
        return float(a) / b if b else 0.0
    return float(s)


# ---------------------------------------------------------------------------
# CUDA guard - never silently fall back to CPU
# ---------------------------------------------------------------------------

def require_cuda(logger: logging.Logger | None = None) -> "torch.device":  # noqa: F821
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment and this pipeline refuses "
            "to fall back to CPU inference. Check `nvidia-smi` and that the venv's "
            "torch is a +cu build (scripts/verify_env.sh)."
        )
    dev = torch.device("cuda")
    if logger:
        props = torch.cuda.get_device_properties(0)
        logger.info("CUDA device: %s (%.1f GiB, sm_%d%d)",
                    props.name, props.total_memory / 2**30, props.major, props.minor)
    return dev


def vram_snapshot() -> dict:
    """Peak/current VRAM from both torch and nvidia-smi (nvidia-smi sees the whole GPU)."""
    out: dict[str, Any] = {}
    try:
        import torch
        if torch.cuda.is_available():
            out["torch_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
            out["torch_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20, 1)
            out["torch_max_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    except Exception:
        pass
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True)
        used, total = p.stdout.strip().split("\n")[0].split(", ")
        out["gpu_used_mb"] = int(used)
        out["gpu_total_mb"] = int(total)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """One VACE inference unit. Never spans a hard scene cut."""
    chunk_id: str
    shot_id: str
    # frame indices into the normalized, model-fps working stream
    start_frame: int
    end_frame: int            # exclusive
    n_frames: int             # == end_frame - start_frame, always 4n+1
    # overlap with the PREVIOUS chunk of the same shot, in frames (0 for first)
    overlap_prev: int
    # times in the ORIGINAL source timebase, for audio sync and re-extraction
    src_start_sec: float
    src_end_sec: float
    # generation parameters
    width: int
    height: int
    fps: int
    seed: int
    prompt: str
    negative_prompt: str
    # asset paths, relative to project root
    reference_sheet: str = ""
    depth_path: str = ""
    mask_path: str = ""
    control_path: str = ""
    output_path: str = ""
    # bookkeeping
    status: str = "pending"   # pending | running | done | failed | skipped
    attempts: int = 0
    error: str = ""
    duration_sec: float = 0.0
    peak_vram_mb: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Shot:
    shot_id: str
    start_frame: int
    end_frame: int            # exclusive
    src_start_sec: float
    src_end_sec: float
    n_frames: int
    # set by track_subject.py
    subject_confidence: float = 0.0
    subject_status: str = "pending"   # pending | auto | needs_user | failed
    subject_note: str = ""


def load_manifest(path: Path | None = None) -> dict:
    path = path or P.manifest
    if not path.exists():
        raise FileNotFoundError(
            f"No chunk manifest at {path}. Run scripts/preprocess_source.py first."
        )
    with open(path) as f:
        return json.load(f)


def save_manifest(data: dict, path: Path | None = None) -> None:
    """Atomic write so an interrupted run cannot corrupt the manifest."""
    path = path or P.manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def update_chunk(manifest: dict, chunk_id: str, **fields) -> dict:
    for c in manifest["chunks"]:
        if c["chunk_id"] == chunk_id:
            c.update(fields)
            return c
    raise KeyError(f"chunk {chunk_id} not in manifest")


def find_single(directory: Path, exts: set[str], what: str) -> Path:
    """Find exactly one media file in a directory, with a clear error otherwise."""
    files = sorted(p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(
            f"No {what} found in {directory}. "
            f"Expected one file with an extension in {sorted(exts)}."
        )
    if len(files) > 1:
        raise RuntimeError(
            f"Expected exactly one {what} in {directory} but found {len(files)}: "
            f"{[f.name for f in files]}. Pass an explicit path instead."
        )
    return files[0]


def slice_frames(src: Path, dst: Path, start_frame: int, end_frame: int, fps: int,
                 logger: logging.Logger | None = None, lossless: bool = True,
                 gray: bool = False) -> None:
    """Cut [start_frame, end_frame) out of `src` into `dst`, frame-exact.

    Uses the `trim` filter with frame indices rather than `-ss` seconds, because
    time-based seeking is not frame-accurate and would desynchronise the depth
    stream, the mask stream and the source stream from each other.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS"
    if gray:
        vf += ",format=gray,format=yuv420p"
    else:
        vf += ",format=yuv420p"
    codec = (["-c:v", "ffv1", "-level", "3"] if lossless
             else ["-c:v", "libx264", "-crf", "12", "-preset", "veryfast"])
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf,
         "-vsync", "cfr", "-r", str(fps), *codec, "-an", str(dst)], logger)


def probe_frames(path: Path) -> int:
    """Exact frame count via packet counting."""
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)])
    return int((p.stdout or "0").strip() or 0)


def probe_dims_fps(path: Path) -> tuple[int, int, float]:
    d = ffprobe_json(path)
    v = next(s for s in d["streams"] if s.get("codec_type") == "video")
    return int(v["width"]), int(v["height"]), parse_fraction(
        v.get("avg_frame_rate") or v.get("r_frame_rate"))


def assert_aligned(paths: dict[str, Path], expect_frames: int, expect_w: int,
                   expect_h: int, expect_fps: float, logger: logging.Logger,
                   tol_fps: float = 0.02) -> None:
    """Hard check that control streams agree on dims, count, fps and ordering.

    A silent one-frame mismatch between depth and mask is one of the easiest ways
    to get subtly wrong output, so this raises rather than warns.
    """
    problems: list[str] = []
    for name, p in paths.items():
        if not p.exists():
            problems.append(f"{name}: missing file {p}")
            continue
        w, h, fps = probe_dims_fps(p)
        n = probe_frames(p)
        if (w, h) != (expect_w, expect_h):
            problems.append(f"{name}: {w}x{h} != expected {expect_w}x{expect_h}")
        if n != expect_frames:
            problems.append(f"{name}: {n} frames != expected {expect_frames}")
        if abs(fps - expect_fps) > tol_fps:
            problems.append(f"{name}: {fps:.4f} fps != expected {expect_fps:.4f}")
    if problems:
        raise RuntimeError("Control stream alignment check FAILED:\n  " +
                           "\n  ".join(problems))
    logger.debug("Alignment OK: %d frames, %dx%d, %.3f fps for %s",
                 expect_frames, expect_w, expect_h, expect_fps,
                 ", ".join(paths.keys()))


def human_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def human_size(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PiB"
