#!/usr/bin/env bash
# Phase 5e - generate a probe set per LoRA checkpoint, then score identity.
#
# WHY GENERATE AT ALL, RATHER THAN INSPECT THE WEIGHTS
# A LoRA that loads, is finite and is non-zero has proven nothing about whose
# face it learned. The only question worth asking is whether the model, prompted
# with the trigger token, produces a face that agrees with photographs the
# training never saw. So each checkpoint generates a small probe set from the
# base T2V model - no VACE, no plate, no mask - which isolates the LoRA from
# every other stage of the pipeline.
#
# WHY EVERY CHECKPOINT AND NOT THE LAST ONE
# Nine training images overfit quickly, and the failure is not visible in the
# loss: it keeps falling while the model memorises framing and lighting. The
# checkpoint to ship is the one that scores best against the held-out bank, and
# that is a measurement, not a guess. The last checkpoint is included but has no
# privileged status.
#
# The baseline row is the same prompt and seeds with NO LoRA. Without it a
# similarity of 0.3 is unreadable: the base model produces a person too, and
# some faces resemble each other.
#
# Rule 1: everything is written to disk. Nothing is displayed.
#
#   scripts/eval_lora_checkpoints.sh [--seeds "1 2 3 4"] [--size 384]
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

MUSUBI_DIR="${MUSUBI_DIR:-/workspace/musubi-tuner}"
WAN_TRAIN_MODELS="${WAN_TRAIN_MODELS:-/workspace/wan_train_models}"
# The de-prefixed copy, NOT the repackaged file the trainer reads: musubi's
# generation path merges the LoRA before it strips `model.diffusion_model.`
# from the checkpoint's keys, so against the original file every LoRA silently
# fails to bind and every checkpoint renders the base model. See
# scripts/prepare_musubi_dit.py, which writes the copy and explains the bug.
DIT="${DIT:-$WAN_TRAIN_MODELS/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16_noprefix.safetensors}"
T5="${T5:-$WAN_TRAIN_MODELS/models_t5_umt5-xxl-enc-bf16.pth}"
VAE="${VAE:-$ROOT/ComfyUI/models/vae/wan_2.1_vae.safetensors}"
DATASET="${DATASET:-$ROOT/intermediate/lora_dataset}"
LORA_DIR="${LORA_DIR:-$ROOT/intermediate/lora_out}"
EVAL_DIR="${EVAL_DIR:-$ROOT/intermediate/lora_eval}"

SEEDS="11 22 33 44"
SIZE=384
STEPS=20

while [ $# -gt 0 ]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --size)  SIZE="$2";  shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

LOG="$ROOT/logs/eval_lora.log"
mkdir -p "$ROOT/logs" "$EVAL_DIR"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 || {
  echo "FATAL: no CUDA device (rule 3)." >&2; exit 1; }

if [ ! -f "$DIT" ]; then
  say "de-prefixed DiT missing; writing it"
  "$ROOT/venv/bin/python" "$ROOT/scripts/prepare_musubi_dit.py" \
      "${DIT%_noprefix.safetensors}.safetensors" --out "$DIT" 2>&1 | tee -a "$LOG"
fi

TRIGGER="$(cat "$(find "$DATASET/train" -name '*.txt' | sort | head -1)")"
PROMPT="a photo of $TRIGGER, head and shoulders, looking at the camera, plain background"
say "trigger '$TRIGGER' | seeds: $SEEDS | ${SIZE}px | $STEPS steps"
say "prompt: $PROMPT"

PY="$MUSUBI_DIR/.venv/bin/python"
gen() {  # gen <out_subdir> [extra args...]
  local out="$EVAL_DIR/$1"; shift
  mkdir -p "$out"
  for s in $SEEDS; do
    # Resume-friendly (rule 8): a seed already generated is not generated again.
    if find "$out" -name "*_${s}_*" -o -name "*seed${s}*" | grep -q .; then
      continue
    fi
    "$PY" "$MUSUBI_DIR/src/musubi_tuner/wan_generate_video.py" \
        --task t2v-1.3B --dit "$DIT" --vae "$VAE" --t5 "$T5" \
        --prompt "$PROMPT" --video_size "$SIZE" "$SIZE" --video_length 1 \
        --infer_steps "$STEPS" --seed "$s" --attn_mode sdpa \
        --output_type images --save_path "$out" "$@" >>"$LOG" 2>&1
  done
}

say "baseline: no LoRA"
gen "baseline_no_lora"

shopt -s nullglob
CKPTS=("$LORA_DIR"/*.safetensors)
[ ${#CKPTS[@]} -gt 0 ] || { echo "FATAL: no checkpoints in $LORA_DIR" >&2; exit 1; }
for ckpt in "${CKPTS[@]}"; do
  name="$(basename "$ckpt" .safetensors)"
  say "checkpoint $name"
  gen "$name" --lora_weight "$ckpt" --lora_multiplier 1.0
done

# Rule 4: the generations must exist before anything is scored.
missing=0
for d in "$EVAL_DIR"/*/; do
  n=$(find "$d" -name '*.png' -o -name '*.jpg' | wc -l | tr -d ' ')
  say "$(basename "$d"): $n image(s)"
  [ "$n" -gt 0 ] || missing=1
done
[ "$missing" -eq 0 ] || { echo "FATAL: a probe set is empty; generation failed silently." >&2; exit 1; }

# And they must DIFFER from the baseline. Same prompt, same seeds, same steps:
# if a LoRA reached the sampler the pixels cannot be identical, and if it did
# not, the whole comparison scores the base model against itself. That is not a
# hypothetical - it is what the first run of this script did, for eight
# checkpoints, and the scores agreed to four decimal places before anyone
# noticed. A warning line in a log is not a guard; this is.
base_sum=$(find "$EVAL_DIR/baseline_no_lora" -name '*.png' | sort | xargs md5sum | awk '{print $1}' | sort | md5sum)
identical=0
for d in "$EVAL_DIR"/*/; do
  name="$(basename "$d")"
  [ "$name" = "baseline_no_lora" ] && continue
  s=$(find "$d" -name '*.png' | sort | xargs md5sum | awk '{print $1}' | sort | md5sum)
  if [ "$s" = "$base_sum" ]; then
    echo "FATAL: $name generated pixel-identical output to the no-LoRA baseline." >&2
    identical=1
  fi
done
[ "$identical" -eq 0 ] || {
  echo "The LoRA did not reach the sampler. Check for 'not all LoRA keys are used'" >&2
  echo "in $LOG, and see scripts/prepare_musubi_dit.py." >&2; exit 1; }

say "scoring against the held-out bank"
"$ROOT/venv/bin/python" "$ROOT/scripts/score_lora_identity.py" \
    --media "$EVAL_DIR"/*/ --dataset "$DATASET" 2>&1 | tee -a "$LOG"
