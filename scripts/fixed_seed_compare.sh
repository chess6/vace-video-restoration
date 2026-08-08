#!/usr/bin/env bash
# One variable, everything else pinned, with a receipt for what actually ran.
#
# WHY IT EXISTS. Bundles 17 and 18 compared two precisions and recorded neither
# the runtime dtype nor which prompt key produced the text. Reproducing one arm
# months later took three attempts: the first used the wrong seed, because the
# grid index offsets --base-seed; the second used the wrong prompt key and landed
# 138.5 mean levels away from the target; only the third was bit-identical. None
# of that was visible from the bundle. This script makes an arm reproducible by
# construction, and prints enough to tell when it is not.
#
# WHAT IT PINS. Everything except the one axis named on the command line:
# resolution, steps, cfg, sampler, scheduler, flow shift, adapter, adapter
# strength, seed, prompt key. Comparability is not a nicety here - resolution,
# cfg, flow shift and sampler all move absolute values, so two arms that differ
# in any of them are not an A/B.
#
# WHAT IT RECORDS, per arm, without ever writing conditioning text:
#   * positive_digest / negative_digest, printed by the generator
#   * a dtype probe against the loaded model, tied to the produced image's sha256
#
# NO PROMPT TEXT, NO MEDIA, NO IMAGES are written by this script into anything
# tracked. Conditioning comes from the untracked overlay by KEY; results land
# under intermediate/ and are collected into a review bundle separately.
#
#   scripts/fixed_seed_compare.sh --axis weight-dtype --values default,fp8_e4m3fn
#   scripts/fixed_seed_compare.sh --axis lora --values none,chroma_subject_v2_combined.safetensors
set -uo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
PY="${PY:-$PROJ/venv/bin/python}"
GEN="${GEN:-$PROJ/intermediate/tools/gen_chroma.py}"

# The pinned baseline. Every value is an argument so a different baseline is a
# flag rather than an edit, and so the line this prints IS the record.
GRID_KEY="${GRID_KEY:-review4}"
VARIANT="${VARIANT:-view_front}"
POSITIVE_KEY="${POSITIVE_KEY:-positive}"
BASE_SEED="${BASE_SEED:-929}"
WIDTH="${WIDTH:-896}"; HEIGHT="${HEIGHT:-1152}"
STEPS="${STEPS:-26}"; CFG="${CFG:-3.8}"
SAMPLER="${SAMPLER:-euler}"; SCHEDULER="${SCHEDULER:-beta}"
FLOW_SHIFT="${FLOW_SHIFT:-1}"
LORA="${LORA:-}"
LORA_STRENGTH="${LORA_STRENGTH:-1.0}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-default}"

AXIS=""; VALUES=""; OUT="$PROJ/intermediate/fixed_seed_compare"
while [ $# -gt 0 ]; do
  case "$1" in
    --axis)   AXIS="$2"; shift 2 ;;
    --values) VALUES="$2"; shift 2 ;;
    --out)    OUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$AXIS" ] && [ -n "$VALUES" ] || { sed -n '2,28p' "$0"; exit 2; }

curl -s --max-time 10 http://127.0.0.1:8188/system_stats >/dev/null || {
  echo "FATAL: no ComfyUI on 8188. scripts/start_comfyui.sh (which passes" >&2
  echo "       --disable-auto-launch, CLAUDE.md rule 1)." >&2; exit 1; }

mkdir -p "$OUT"
echo "axis            : $AXIS = $VALUES"
echo "pinned          : ${WIDTH}x${HEIGHT} steps=$STEPS cfg=$CFG $SAMPLER/$SCHEDULER"
echo "                  flow_shift=$FLOW_SHIFT positive_key=$POSITIVE_KEY"
echo "                  grid=$GRID_KEY variant=$VARIANT base_seed=$BASE_SEED"
echo "                  lora=${LORA:-none} strength=$LORA_STRENGTH dtype=$WEIGHT_DTYPE"
echo "NOTE: the grid index offsets --base-seed, so the seed the sampler sees is"
echo "      NOT base_seed. The generator prints it; check it against the arm you"
echo "      are reproducing."
echo

IFS=',' read -ra VALS <<< "$VALUES"
for v in "${VALS[@]}"; do
  wd="$WEIGHT_DTYPE"; lora="$LORA"
  case "$AXIS" in
    weight-dtype) wd="$v" ;;
    lora)         lora="$v"; [ "$v" = "none" ] && lora="" ;;
    *) echo "FATAL: unsupported axis '$AXIS' (weight-dtype|lora)" >&2; exit 2 ;;
  esac

  tag="${AXIS}_${v//[^A-Za-z0-9._-]/_}"
  echo "=== $tag ==="
  args=(--grid-key "$GRID_KEY" --only "$VARIANT" --base-seed "$BASE_SEED"
        --positive-key "$POSITIVE_KEY" --width "$WIDTH" --height "$HEIGHT"
        --steps "$STEPS" --cfg "$CFG" --sampler "$SAMPLER"
        --scheduler "$SCHEDULER" --flow-shift "$FLOW_SHIFT"
        --weight-dtype "$wd" --lora-strength "$LORA_STRENGTH")
  [ -n "$lora" ] && args+=(--lora "$lora")

  log="$OUT/$tag.txt"
  "$PY" "$GEN" "${args[@]}" 2>&1 | tee "$log" | grep -E "digest|seed |->"
  img="$(grep -oE '/[^ ]+\.png' "$log" | tail -1)"
  if [ -z "$img" ] || [ ! -f "$img" ]; then
    echo "  FATAL: no image produced for $tag" >&2; exit 1
  fi
  cp -p "$img" "$OUT/$tag.png"

  # The receipt. Declared dtype is a request; this reads the loaded model.
  "$PY" "$PROJ/scripts/dtype_probe.py" --weight-dtype "$wd" \
      --output-image "$OUT/$tag.png" --out "$OUT/$tag.dtype.json" \
      2>&1 | grep -E "VERDICT|DECLARED AND ACTUAL"
  echo
done

echo "arms under $OUT"
echo "Compare them with a pixel diff, not by eye alone: two arms that differ by"
echo "0.000 are the same run, and two that differ by ~16 are a real change."
