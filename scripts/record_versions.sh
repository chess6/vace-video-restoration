#!/usr/bin/env bash
# Record exact commits, model URLs, sizes and checksums for reproducibility.
# Writes reports/versions.md.
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$PROJ/venv/bin/python"
OUT="$PROJ/reports/versions.md"
mkdir -p "$PROJ/reports"

{
echo "# Pinned versions and model provenance"
echo
echo "Generated: $(date -Iseconds)"
echo
echo "## Repositories"
echo
echo "| Component | Source | Commit / version |"
echo "|---|---|---|"
if [ -d "$PROJ/ComfyUI/.git" ]; then
  echo "| ComfyUI | https://github.com/comfyanonymous/ComfyUI | \`$(git -C "$PROJ/ComfyUI" rev-parse HEAD)\` ($(git -C "$PROJ/ComfyUI" describe --tags --abbrev=0 2>/dev/null || echo untagged)) |"
fi
SAM2V=$("$PY" -c "import sam2,os;print(getattr(sam2,'__version__','1.0'))" 2>/dev/null || echo "n/a")
echo "| SAM 2 | https://github.com/facebookresearch/sam2 | pinned \`2b90b9f5ceec907a1c18123530e92e794ad901a4\` (v$SAM2V) |"
echo
echo "**Custom ComfyUI nodes installed: none.**"
echo
echo "The baseline workflow uses only native nodes (\`WanVaceToVideo\`,"
echo "\`LoadVideo\`, \`GetVideoComponents\`, \`ImageCompositeMasked\`, \`ImageToMask\`,"
echo "\`FeatherMask\`, \`CreateVideo\`, \`SaveVideo\`, …), so there are no third-party"
echo "nodes to pin or to break on a ComfyUI update. Depth estimation, detection"
echo "and SAM 2 tracking run as separate processes outside ComfyUI, which also"
echo "keeps them from competing with VACE for VRAM."
echo
echo "## Python environment"
echo
echo '```'
"$PY" - <<'PYEOF'
import importlib.metadata as md
for p in ["torch","torchvision","torchaudio","transformers","huggingface-hub",
          "opencv-python-headless","numpy","scenedetect","insightface",
          "onnxruntime-gpu","timm","SAM-2","safetensors","Pillow","scipy"]:
    try: print(f"{p:26s} {md.version(p)}")
    except Exception: print(f"{p:26s} (not installed)")
PYEOF
echo '```'
echo
echo "## Model files"
echo
echo "| File | Bytes | SHA256 | Source |"
echo "|---|---|---|---|"
REPO="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main"
for spec in \
  "diffusion_models/wan2.1_vace_1.3B_fp16.safetensors|split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors" \
  "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors|split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  "vae/wan_2.1_vae.safetensors|split_files/vae/wan_2.1_vae.safetensors" ; do
  rel="${spec%%|*}"; rpath="${spec##*|}"
  f="$PROJ/ComfyUI/models/$rel"
  if [ -f "$f" ]; then
    echo "| \`$(basename "$f")\` | $(stat -c%s "$f") | \`$(sha256sum "$f" | cut -c1-16)…\` | $REPO/$rpath |"
  else
    echo "| \`$(basename "$f")\` | MISSING | - | $REPO/$rpath |"
  fi
done
echo
echo "Full checksums are re-verified on every run of \`scripts/download_models.sh\`."
echo
echo "## Auxiliary models (Hugging Anchor, cached in \`hf_cache/\`)"
echo
echo "| Model | Purpose |"
echo "|---|---|"
echo "| \`facebook/sam2.1-hiera-large\` | full-subject mask tracking |"
echo "| \`IDEA-Research/grounding-dino-base\` | open-vocabulary candidate detection |"
echo "| \`openai/clip-vit-large-patch14\` | appearance / attributes ReID embedding |"
echo "| \`depth-anything/Depth-Anything-V2-Large-hf\` | depth structural control |"
echo "| \`insightface buffalo_l\` (ArcFace) | anchor match embedding |"
echo
echo "## LoRAs installed in \`ComfyUI/models/loras\`"
echo
# Rule 7 pins every model dependency by size and SHA256, and a LoRA downloaded
# from a third party is exactly the dependency that needs it: the same filename
# is reused across revisions, and a swapped file changes the pixels while the
# config still reads the same. The digest here is what identifies which weights
# a result came from, and vace_key now hashes the same contents per chunk.
#
# Rule 2a decides whether the NAME can be printed. A LoRA named in a tracked
# config is already public and is named. Anything else is local to this machine,
# and its author named it after what it makes the model produce - so it is
# recorded by digest alone. That is not a gap: the digest is the identifier, and
# the name lives in configs/prompt.local.yaml, which travels in the state bundle.
LORA_DIR="$PROJ/ComfyUI/models/loras"
if [ -d "$LORA_DIR" ] && [ -n "$(ls -A "$LORA_DIR" 2>/dev/null)" ]; then
  echo "| File | Bytes | SHA256 |"
  echo "|---|---|---|"
  for f in "$LORA_DIR"/*; do
    [ -f "$f" ] || continue
    b="$(basename "$f")"
    if grep -qF -- "$b" "$PROJ"/configs/*.yaml 2>/dev/null \
       && ! grep -qF -- "$b" "$PROJ/configs/prompt.local.yaml" 2>/dev/null; then
      label="\`$b\`"
    else
      label="(local, name withheld — rule 2a)"
    fi
    echo "| $label | $(stat -c%s "$f") | \`$(sha256sum "$f" | cut -c1-16)…\` |"
  done
else
  echo "No LoRAs are installed on this machine."
fi
echo
echo "## Subject-LoRA training (musubi-tuner)"
echo
MUSUBI_DIR="${MUSUBI_DIR:-/workspace/musubi-tuner}"
WAN_TRAIN_MODELS="${WAN_TRAIN_MODELS:-/workspace/wan_train_models}"
if [ -d "$MUSUBI_DIR/.git" ]; then
  echo "| Component | Source | Commit / version |"
  echo "|---|---|---|"
  echo "| musubi-tuner | https://github.com/kohya-ss/musubi-tuner | \`$(git -C "$MUSUBI_DIR" rev-parse HEAD)\` ($(git -C "$MUSUBI_DIR" describe --tags --always 2>/dev/null)) |"
  echo
  echo "Trainer weights are a SEPARATE set from the pipeline's, and deliberately so:"
  echo "musubi has no VACE task, and rejects the fp8_scaled text encoder ComfyUI uses."
  echo "The VAE is shared, not duplicated."
  echo
  echo "| File | Bytes | SHA256 | Source |"
  echo "|---|---|---|---|"
  for spec in \
    "$WAN_TRAIN_MODELS/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors|https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors" \
    "$WAN_TRAIN_MODELS/models_t5_umt5-xxl-enc-bf16.pth|https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth" ; do
    f="${spec%%|*}"; url="${spec##*|}"
    if [ -f "$f" ]; then
      echo "| \`$(basename "$f")\` | $(stat -c%s "$f") | \`$(sha256sum "$f" | cut -c1-16)…\` | $url |"
    else
      echo "| \`$(basename "$f")\` | MISSING | - | $url |"
    fi
  done
else
  echo "musubi-tuner is not installed on this machine (\`$MUSUBI_DIR\`), so no"
  echo "training versions are recorded here. It lives on the rented GPU box."
fi
echo
echo "## Deliberately NOT installed"
echo
echo "- Wan2.1-VACE-**14B** weights — will not fit 12 GB; see \`configs/cloud_14b.yaml\`"
echo "- extra T2V / I2V checkpoints"
echo "- acceleration LoRAs (CausVid etc.) — excluded from the baseline by design"
echo "- GFPGAN / CodeFormer — anchor-only restoration was explicitly ruled out"
echo "- Real-ESRGAN — optional *post*-restoration resize only, and only if it"
echo "  beats Lanczos in \`scripts/compare_upscalers.py\`"
} > "$OUT"

echo "Wrote $OUT"
