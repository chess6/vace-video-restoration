#!/usr/bin/env bash
# Download the local (1.3B) model profile into ComfyUI's model directories.
#
# Every file is verified by SHA256 against the value published by the Hugging Face
# API at the time this script was written. A size-only check is NOT sufficient, so
# a checksum mismatch is a hard failure.
#
# Resumable: already-present files with a correct checksum are skipped.
# Deliberately does NOT download the 14B model (see configs/cloud_14b.yaml).
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$PROJ/ComfyUI/models"
REPO="Comfy-Org/Wan_2.1_ComfyUI_repackaged"
BASE="https://huggingface.co/$REPO/resolve/main"

mkdir -p "$MODELS/diffusion_models" "$MODELS/text_encoders" "$MODELS/vae"

# dest_subdir | filename | remote_path | sha256 | size_bytes
FILES=(
"diffusion_models|wan2.1_vace_1.3B_fp16.safetensors|split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors|640ccc0577e6a5d4bb15cd91b11b699ef914fc55f126c5a1c544e152130784f2|4309519800"
"text_encoders|umt5_xxl_fp8_e4m3fn_scaled.safetensors|split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors|c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68|6735906897"
"vae|wan_2.1_vae.safetensors|split_files/vae/wan_2.1_vae.safetensors|2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b|253815318"
)

rc=0
for entry in "${FILES[@]}"; do
  IFS='|' read -r sub name rpath sha size <<< "$entry"
  dest="$MODELS/$sub/$name"

  if [ -f "$dest" ]; then
    actual_size=$(stat -c%s "$dest")
    if [ "$actual_size" = "$size" ]; then
      echo "[check] $name present, verifying SHA256..."
      actual_sha=$(sha256sum "$dest" | cut -d' ' -f1)
      if [ "$actual_sha" = "$sha" ]; then
        echo "[ok]    $name verified, skipping."
        continue
      fi
      echo "[warn]  $name checksum mismatch, re-downloading."
    else
      echo "[warn]  $name size $actual_size != $size, resuming/re-downloading."
    fi
  fi

  echo "[get]   $name  ($(numfmt --to=iec "$size"))"
  echo "        $BASE/$rpath"
  # -C - resumes a partial file; --retry survives transient HF 5xx
  if ! curl -L --fail --retry 5 --retry-delay 5 --no-progress-meter -C - -o "$dest" "$BASE/$rpath"; then
    # -C - fails if the server already sent the whole file; retry without resume
    curl -L --fail --retry 5 --retry-delay 5 --no-progress-meter -o "$dest" "$BASE/$rpath" || { echo "[FAIL]  download $name"; rc=1; continue; }
  fi

  actual_size=$(stat -c%s "$dest")
  actual_sha=$(sha256sum "$dest" | cut -d' ' -f1)
  if [ "$actual_sha" != "$sha" ]; then
    echo "[FAIL]  $name SHA256 MISMATCH"
    echo "        expected $sha"
    echo "        actual   $actual_sha"
    rc=1
  elif [ "$actual_size" != "$size" ]; then
    echo "[FAIL]  $name size mismatch: $actual_size != $size"; rc=1
  else
    echo "[ok]    $name verified ($actual_size bytes, sha256 $actual_sha)"
  fi
done

echo
echo "=== Local model profile ==="
ls -lh "$MODELS/diffusion_models" "$MODELS/text_encoders" "$MODELS/vae" 2>/dev/null | grep -v '^total'
echo
if [ $rc -eq 0 ]; then echo "All models present and checksum-verified."
else echo "ERROR: one or more models failed verification."; fi
exit $rc
