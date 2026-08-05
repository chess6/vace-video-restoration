#!/usr/bin/env bash
# Download the SeedVR2 background-restoration model into ComfyUI's model dirs.
#
# 3B in fp8 ONLY. The 7B model is deliberately not downloaded: it does not fit
# this 12 GB card at 480p video length, and a run that silently swaps to CPU or
# thrashes is worse than one that never starts (CLAUDE.md rule 3).
#
# Same contract as download_models.sh: SHA256 from the Hugging Face API, hard
# failure on mismatch, resumable, skips files already verified.
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$PROJ/ComfyUI/models"
REPO="Comfy-Org/SeedVR2"
BASE="https://huggingface.co/$REPO/resolve/main"

mkdir -p "$MODELS/diffusion_models" "$MODELS/vae"

# dest_subdir | filename | remote_path | sha256 | size_bytes
#
# The sizes below are the sizes of the files whose SHA256 are recorded here, as
# measured after a verified download. They were previously 3388459832 and
# 503341328, which no file with these checksums has ever had: the size test runs
# first, so it failed on a correct file and forced a 3.9 GB re-download on every
# single invocation, after which the SHA256 passed and declared it fine. A size
# that disagrees with a matching checksum is a bug in the size, not the file.
FILES=(
"diffusion_models|seedvr2_3b_fp8_e4m3fn.safetensors|diffusion_models/seedvr2_3b_fp8_e4m3fn.safetensors|a0226eaa2c3e6f47ae5ce83225120f16479da890ced1a3bc32b1a14619787914|3392794232"
"vae|seedvr2_ema_vae_fp16.safetensors|vae/seedvr2_ema_vae_fp16.safetensors|20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1|501324814"
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
  if ! curl -L --fail --retry 5 --retry-delay 5 --no-progress-meter -C - -o "$dest" "$BASE/$rpath"; then
    curl -L --fail --retry 5 --retry-delay 5 --no-progress-meter -o "$dest" "$BASE/$rpath" \
      || { echo "[FAIL]  download $name"; rc=1; continue; }
  fi

  actual_size=$(stat -c%s "$dest")
  actual_sha=$(sha256sum "$dest" | cut -d' ' -f1)
  if [ "$actual_sha" != "$sha" ]; then
    echo "[FAIL]  $name SHA256 MISMATCH"
    echo "        expected $sha"
    echo "        actual   $actual_sha"
    rc=1
  else
    echo "[ok]    $name verified ($actual_size bytes)"
  fi
done

echo
if [ $rc -eq 0 ]; then echo "SeedVR2 3B fp8 present and checksum-verified."
else echo "ERROR: SeedVR2 model verification failed."; fi
exit $rc
