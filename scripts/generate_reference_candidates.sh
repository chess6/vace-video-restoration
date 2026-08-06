#!/usr/bin/env bash
# Phase 11b - widen the reference set with generated candidates, then judge them
# by the same door a photograph goes through.
#
# WHAT THIS IS FOR
# The seventeen photographs were selected for identity evidence, not for pose or
# lighting coverage, which is why reference-based super-resolution has so little
# to match against. This generates candidates across a pose and lighting grid
# using the subject LoRA, and hands the whole set to score_lora_identity.py.
#
# WHAT DECIDES WHETHER A CANDIDATE IS USABLE - two numbers, not one:
#
#   identity   scored against the HELD-OUT photographs only, the three no
#              training ever saw. Real photographs score ~0.745 on that bank and
#              the base model with no LoRA scores ~0.023, so both ends of the
#              scale are known and a candidate can be placed on it.
#   eye px     inter-ocular distance. A reference can only donate detail it
#              actually has, and the target face is ~70px. A candidate that
#              matches the identity but carries no more detail per feature than
#              the frame is worth nothing as a RefSR reference, however good it
#              looks.
#
# A candidate that fails either is not a reference, and nothing here promotes
# anything into inputs/ automatically: generated images are not photographs of
# the user, and the decision to treat them as evidence is theirs.
#
# Rule 1: everything lands on disk. Nothing is displayed.
#
#   scripts/generate_reference_candidates.sh [--seeds "1 2 3"] [--steps 25]
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

MUSUBI_DIR="${MUSUBI_DIR:-/workspace/musubi-tuner}"
WAN_TRAIN_MODELS="${WAN_TRAIN_MODELS:-/workspace/wan_train_models}"
DIT="${DIT:-$WAN_TRAIN_MODELS/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16_noprefix.safetensors}"
T5="${T5:-$WAN_TRAIN_MODELS/models_t5_umt5-xxl-enc-bf16.pth}"
VAE="${VAE:-$ROOT/ComfyUI/models/vae/wan_2.1_vae.safetensors}"
LORA="${LORA:-$ROOT/intermediate/lora_out/subject_1p3b_v2.safetensors}"
DATASET="${DATASET:-$ROOT/intermediate/lora_dataset}"
OUT="${OUT:-$ROOT/intermediate/ref_candidates}"

SEEDS="101 202 303 404 505"
STEPS=25
# 480x832 portrait: t2v-1.3B declares ('480*832','832*480') as its supported
# sizes, and the probe runs at 384 warned about exactly that. Portrait also puts
# the pixels where a head shot needs them.
W=480
H=832

while [ $# -gt 0 ]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

LOG="$ROOT/logs/ref_candidates.log"
mkdir -p "$ROOT/logs" "$OUT"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 || {
  echo "FATAL: no CUDA device (rule 3)." >&2; exit 1; }
[ -f "$LORA" ] || { echo "FATAL: no LoRA at $LORA" >&2; exit 1; }
[ -f "$DIT" ] || { echo "FATAL: no de-prefixed DiT at $DIT (see prepare_musubi_dit.py)" >&2; exit 1; }

TRIGGER="$(cat "$(find "$DATASET/train" -name '*.txt' | sort | head -1)")"
say "trigger '$TRIGGER' | seeds: $SEEDS | ${W}x${H} | $STEPS steps"

# The grid is pose first, lighting second: pose is what a RefSR matcher aligns
# on, lighting is what makes a transferred texture look wrong if it disagrees.
VARIANTS=(
  "head_on:looking straight at the camera, head and shoulders"
  "turned_left:head turned to the left, head and shoulders"
  "turned_right:head turned to the right, head and shoulders"
  "side_on:side view of the head, profile, head and shoulders"
  "tilted_down:head tilted slightly down, head and shoulders"
  "soft_daylight:looking at the camera, soft even daylight, head and shoulders"
  "warm_lamp:looking at the camera, warm indoor lamp light, head and shoulders"
  "overhead:looking at the camera, overhead room light, slight shadow under the brow"
)

for v in "${VARIANTS[@]}"; do
  name=${v%%:*}; desc=${v#*:}
  dst="$OUT/$name"
  mkdir -p "$dst"
  say "variant $name"
  for s in $SEEDS; do
    # Resume-friendly (rule 8): a seed already generated is not generated again.
    if find "$dst" -name "*_${s}_*" 2>/dev/null | grep -q .; then continue; fi
    "$MUSUBI_DIR/.venv/bin/python" "$MUSUBI_DIR/src/musubi_tuner/wan_generate_video.py" \
        --task t2v-1.3B --dit "$DIT" --vae "$VAE" --t5 "$T5" \
        --prompt "a photo of $TRIGGER, $desc" \
        --video_size "$H" "$W" --video_length 1 --infer_steps "$STEPS" \
        --seed "$s" --attn_mode sdpa --output_type images --save_path "$dst" \
        --lora_weight "$LORA" --lora_multiplier 1.0 >>"$LOG" 2>&1 || \
        say "  seed $s FAILED (continuing)"
  done
  n=$(find "$dst" -name '*.png' | wc -l | tr -d ' ')
  say "  $name: $n image(s)"
done

say "scoring every candidate against the held-out bank"
"$ROOT/venv/bin/python" "$ROOT/scripts/score_lora_identity.py" \
    --media "$OUT"/*/ --dataset "$DATASET" \
    --out "$OUT/candidate_scores.json" 2>&1 | tee -a "$LOG"

say "Nothing was promoted into inputs/. A generated image is not a photograph of"
say "the user; whether it counts as a reference is their call, on these numbers."
