#!/usr/bin/env bash
# Recreate this project from a fresh clone: pinned ComfyUI checkout, pinned
# Python environment, pinned model weights.
#
#   scripts/bootstrap.sh              full setup
#   scripts/bootstrap.sh --no-models  environment only, skip the weight downloads
#   scripts/bootstrap.sh --check      report what is missing, change nothing
#
# No sudo. Nothing is installed system-wide, no system CUDA toolkit is needed
# (PyTorch ships its own), and an existing conda environment is not touched.
#
# Everything version-critical is pinned here rather than resolved at install
# time, because "latest" changes the output of a run that is supposed to be
# reproducible: the ComfyUI commit fixes the WanVaceToVideo semantics the whole
# design is built on, requirements.lock.txt fixes every Python package, and the
# Hugging Anchor revisions are pinned in the scripts that load them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- pins -------------------------------------------------------------------
COMFY_REPO="https://github.com/comfyanonymous/ComfyUI.git"
COMFY_COMMIT="e377e263049f9338b4d12a3dd417b36ae62948ff"
SAM2_REPO="https://github.com/facebookresearch/sam2"
SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
# Must match the +cuXXX local version pinned in requirements.lock.txt: that build
# exists only on its own index. Pointing here at cu124 while the lock pins
# 2.9.1+cu128 makes the install unresolvable, which is how this was found.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

CHECK_ONLY=0
WITH_MODELS=1
for a in "$@"; do
  case "$a" in
    --check) CHECK_ONLY=1 ;;
    --no-models) WITH_MODELS=0 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "Unknown option: $a" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '  !! %s\n' "$*" >&2; }
ok()   { printf '  ok %s\n' "$*"; }

# ---- 0. host prerequisites ---------------------------------------------------
say "Checking host prerequisites"
MISSING=""
for t in git curl ffmpeg ffprobe; do
  command -v "$t" >/dev/null 2>&1 || MISSING="$MISSING $t"
done
command -v "$PYTHON_BIN" >/dev/null 2>&1 || MISSING="$MISSING $PYTHON_BIN"
if [ -n "$MISSING" ]; then
  warn "missing:$MISSING"
  warn "install them first, e.g.: sudo apt-get install -y$MISSING"
  [ "$CHECK_ONLY" -eq 1 ] || exit 1
else
  ok "git, curl, ffmpeg, ffprobe, $PYTHON_BIN"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  ok "NVIDIA driver: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  warn "no nvidia-smi. This pipeline refuses to run on CPU (see CLAUDE.md rule 3)."
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  say "Check only; reporting state"
  [ -d ComfyUI/.git ] && ok "ComfyUI present at $(git -C ComfyUI rev-parse --short HEAD)" \
                      || warn "ComfyUI missing"
  [ -x venv/bin/python ] && ok "venv present ($(venv/bin/python -V 2>&1))" \
                         || warn "venv missing"
  [ -f ComfyUI/models/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors ] \
      && ok "VACE 1.3B weights present" || warn "VACE weights missing"
  exit 0
fi

# ---- 1. ComfyUI at the pinned commit ----------------------------------------
say "ComfyUI @ ${COMFY_COMMIT:0:12}"
if [ ! -d ComfyUI/.git ]; then
  git clone --filter=blob:none "$COMFY_REPO" ComfyUI
fi
git -C ComfyUI fetch --depth 1 origin "$COMFY_COMMIT" 2>/dev/null || git -C ComfyUI fetch origin
git -C ComfyUI checkout -q "$COMFY_COMMIT"
ok "ComfyUI at $(git -C ComfyUI rev-parse --short HEAD)"

# ---- 2. the virtual environment ---------------------------------------------
say "Python environment"
if [ ! -x venv/bin/python ]; then
  "$PYTHON_BIN" -m venv venv
fi
venv/bin/python -m pip install --quiet --upgrade pip wheel
ok "$(venv/bin/python -V 2>&1)"

if [ -f requirements.lock.txt ]; then
  # The lock file is the whole environment, torch included, as it was resolved on
  # the machine the pipeline was validated on.
  say "Installing from requirements.lock.txt (exact versions)"
  venv/bin/python -m pip install --quiet \
      --extra-index-url "$TORCH_INDEX" -r requirements.lock.txt
  ok "locked dependencies installed"
else
  warn "requirements.lock.txt not found; installing ComfyUI's own requirements"
  venv/bin/python -m pip install --quiet --extra-index-url "$TORCH_INDEX" torch torchvision
  venv/bin/python -m pip install --quiet -r ComfyUI/requirements.txt
  venv/bin/python -m pip install --quiet \
      "git+${SAM2_REPO}@${SAM2_COMMIT}" transformers insightface onnxruntime \
      opencv-python-headless scenedetect pyyaml
fi

# opencv MUST stay headless: the GUI build would make the no-display rule
# unenforceable at runtime (CLAUDE.md rule 1, scripts/check_no_display.sh).
if venv/bin/python -c "import cv2, sys; sys.exit(0 if hasattr(cv2,'imshow') and 'headless' not in getattr(cv2,'__file__','') else 1)" 2>/dev/null; then
  warn "the GUI build of OpenCV may be installed; reinstalling the headless one"
  venv/bin/python -m pip install --quiet --force-reinstall opencv-python-headless
fi

# ---- 3. model weights --------------------------------------------------------
if [ "$WITH_MODELS" -eq 1 ]; then
  say "Model weights (checksummed)"
  scripts/download_models.sh
  scripts/download_aux_models.sh
else
  say "Skipping model downloads (--no-models)"
fi

# ---- 4. verify ---------------------------------------------------------------
say "Verifying"
scripts/verify_env.sh || warn "verify_env.sh reported problems (see above)"
# The whole synthetic suite, hermetically: no GPU, and no binding file reachable.
# Bootstrap is the FRESH-CHECKOUT path, which is exactly where the claim matters -
# the stages must import with no backend binding present, and must still refuse
# to run rather than degrade once they need one.
scripts/run_unit_tests.sh || warn "unit tests FAILED"
scripts/check_no_display.sh >/dev/null && ok "no-display rule holds"

say "Done"
cat <<'EOF'
  Next:
    scripts/start_comfyui.sh --daemon
    venv/bin/python scripts/build_workflows.py
    cp /path/to/video.mp4 inputs/source/ && cp /path/to/stills/* inputs/references/
    see README.md "Quick start"
EOF
