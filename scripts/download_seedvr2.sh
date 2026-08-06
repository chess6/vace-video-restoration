#!/usr/bin/env bash
# Download the SeedVR2 background-restoration model into ComfyUI's model dirs.
#
# 3B in fp8 by default. 7B is opt-in via --7b, because it does not fit a 12 GB
# card at video length and a run that silently swaps to CPU or thrashes is worse
# than one that never starts (CLAUDE.md rule 3). On a 24 GB+ card it is the
# largest untested quality lever: the 3B plate peaked at 12.7 GB, and the plate
# stage is the one producing essentially all of this pipeline's measured gain.
#
# Two 7B fp8 variants exist and are the same size. `sharp` is a distinct release,
# not a setting - worth comparing, since unresolved fine anchor detail is the
# open complaint about the 3B plate.
#
# Same contract as download_models.sh: SHA256 from the Hugging Anchor API, hard
# failure on mismatch, resumable, skips files already verified.
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$PROJ/ComfyUI/models"
REPO="Comfy-Org/SeedVR2"
BASE="https://huggingface.co/$REPO/resolve/main"

mkdir -p "$MODELS/diffusion_models" "$MODELS/vae"

# Optional Hugging Anchor read token, for anonymous rate limiting. Read from the
# environment or ~/.hf_token so the value is never an argument (arguments are
# visible in `ps` to every user on the box) and never lands in this repo.
# Measured here: the first 7 GB file pulled in ~2 minutes anonymously, the next
# crawled at 1 MiB/s.
HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN" ] && [ -r "$HOME/.hf_token" ]; then
  HF_TOKEN="$(tr -d "[:space:]" < "$HOME/.hf_token")"
fi
AUTH=()
if [ -n "$HF_TOKEN" ]; then
  AUTH=(-H "Authorization: Bearer $HF_TOKEN")
  echo "using a Hugging Anchor token (${#HF_TOKEN} chars, value not shown)"
else
  echo "no HF token found; downloading anonymously (subject to rate limiting)"
fi

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

# Opt-in with --7b. Sizes and SHA256 read from the Hugging Anchor API, not guessed.
FILES_7B=(
"diffusion_models|seedvr2_7b_fp8_e4m3fn.safetensors|diffusion_models/seedvr2_7b_fp8_e4m3fn.safetensors|5065e77d647dd553d9090a81e20d6de590d931a61df79d785e008433926ee418|8240979248"
"diffusion_models|seedvr2_7b_sharp_fp8_e4m3fn.safetensors|diffusion_models/seedvr2_7b_sharp_fp8_e4m3fn.safetensors|7602c5f70868d28e7730035e4e9d745b05d661c8f0a7eb758e63f9c8603596ef|8240979248"
)

for a in "$@"; do
  case "$a" in
    --7b) FILES+=("${FILES_7B[@]}") ;;
    -h|--help) sed -n "2,20p" "$0"; exit 0 ;;
    *) echo "Unknown option: $a" >&2; exit 2 ;;
  esac
done

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
  if ! curl -L --fail --retry 5 --retry-delay 5 --no-progress-meter "${AUTH[@]}" -C - -o "$dest" "$BASE/$rpath"; then
    curl -L --fail --retry 5 --retry-delay 5 --no-progress-meter "${AUTH[@]}" -o "$dest" "$BASE/$rpath" \
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
