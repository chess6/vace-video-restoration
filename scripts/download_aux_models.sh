#!/usr/bin/env bash
# Install and prefetch the auxiliary models used by the preprocessing stages.
# These run OUTSIDE ComfyUI, one stage at a time, so they never compete with VACE
# for VRAM.
#
#   SAM 2.1 (hiera-large)          -> full-subject mask tracking          (Phase 6)
#   Grounding DINO (base)          -> open-vocabulary candidate detection   (Phase 6)
#   CLIP ViT-L/14                  -> appearance / attributes ReID embedding (Phase 6)
#   InsightAnchor buffalo_l          -> anchor match embedding            (Phase 6)
#   Depth Anything V2 (Large)      -> structural depth control           (Phase 7)
#
# Pinned versions are recorded in reports/versions.md by scripts/record_versions.sh
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$PROJ/venv/bin/python"

# Pinned commit of the official Meta SAM 2 repository.
SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"

export HF_HOME="${HF_HOME:-$PROJ/hf_cache}"
mkdir -p "$HF_HOME"

echo "=== 1/3 Installing SAM 2 (pinned $SAM2_COMMIT) ==="
if "$PY" -c "import sam2" 2>/dev/null; then
  echo "sam2 already installed."
else
  # --no-build-isolation keeps it using the venv's already-installed torch
  "$PY" -m pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/sam2.git@${SAM2_COMMIT}" || {
      echo "ERROR: sam2 install failed"; exit 1; }
fi
"$PY" -c "import sam2; print('sam2 OK:', sam2.__file__)" || exit 1

echo
echo "=== 2/3 Prefetching Hugging Anchor models into $HF_HOME ==="
"$PY" - <<'PYEOF'
import os, sys
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from huggingface_hub import snapshot_download

MODELS = [
    ("facebook/sam2.1-hiera-large",            "SAM 2.1 hiera-large (tracking)"),
    ("IDEA-Research/grounding-dino-base",      "Grounding DINO base (open-vocab detection)"),
    ("openai/clip-vit-large-patch14",          "CLIP ViT-L/14 (appearance ReID)"),
    ("depth-anything/Depth-Anything-V2-Large-hf", "Depth Anything V2 Large (depth)"),
]
fail = 0
for repo, desc in MODELS:
    try:
        print(f"[get] {repo}  -- {desc}", flush=True)
        p = snapshot_download(repo, allow_patterns=[
            "*.json", "*.txt", "*.safetensors", "*.model", "*.yaml", "*.pt"])
        total = sum(f.stat().st_size for f in __import__("pathlib").Path(p).rglob("*") if f.is_file())
        print(f"[ok]  {repo}  ({total/2**30:.2f} GiB) -> {p}", flush=True)
    except Exception as e:
        print(f"[FAIL] {repo}: {e}", file=sys.stderr, flush=True)
        fail += 1
sys.exit(1 if fail else 0)
PYEOF
[ $? -ne 0 ] && { echo "ERROR: HF prefetch failed"; exit 1; }

echo
echo "=== 3/3 Prefetching InsightAnchor buffalo_l (anchor match) ==="
"$PY" - <<'PYEOF'
import os, sys
os.environ.setdefault("INSIGHTFACE_HOME", os.environ.get("INSIGHTFACE_HOME", os.path.expanduser("~/.insightface")))
try:
    from insightface.app import FaceAnalysis
    # Downloads buffalo_l on first construction. CPU providers are fine for the
    # prefetch; the real run selects CUDA.
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("[ok]  insightface buffalo_l ready")
except Exception as e:
    print(f"[WARN] insightface prefetch failed: {e}", file=sys.stderr)
    print("       Anchor matching will degrade to appearance-only ReID, which still works.", file=sys.stderr)
    sys.exit(0)   # non-fatal: appearance ReID alone is a valid fallback
PYEOF

echo
echo "All auxiliary models are ready. HF_HOME=$HF_HOME"
