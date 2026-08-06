#!/usr/bin/env bash
# Phase 11b - widen the reference set with generated candidates, then judge them
# by the same door a real reference goes through.
#
# WHAT THIS IS FOR
# The references were selected for match evidence, not for angle or lighting
# coverage, which is why reference-based super-resolution has so little to match
# against. This generates candidates across an angle and lighting grid using the
# subject LoRA - plus any behaviour LoRA the overlay names, stacked with it - and
# hands the whole set to score_lora_match.py.
#
# WHAT DECIDES WHETHER A CANDIDATE IS USABLE - two numbers, not one:
#
#   match      scored against the HELD-OUT references only, the three no training
#              ever saw. Real references score ~0.745 on that bank and the base
#              model with no LoRA scores ~0.023, so both ends of the scale are
#              known and a candidate can be placed on it.
#   span px    the anchor keypoint span, i.e. how much detail the anchor is drawn
#              with. A reference can only donate detail it actually has, and the
#              target's span is ~70px. A candidate that matches on match but
#              carries no more detail per feature than the frame is worth nothing
#              as a RefSR reference, however good it looks.
#
# A candidate that fails either is not a reference, and nothing here promotes
# anything into inputs/ automatically: a generated image is not a record of the
# user's subject, and the decision to treat one as evidence is theirs.
#
# WHERE THE PROMPT TEXT LIVES
# The grid's descriptions are conditioning text, and conditioning text describes
# the subject in order to work - so, like the profile prompts, it lives in the
# untracked configs/prompt.local.yaml and not here. What stays tracked is the
# structure: two grids, how they are selected, how they are scored, and that a
# seed already generated is never generated twice. See common.py, which explains
# why a functional input is withheld anyway.
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
GRID=head
# 480x832 portrait: t2v-1.3B declares ('480*832','832*480') as its supported
# sizes, and the probe runs at 384 warned about exactly that. Portrait also puts
# the pixels where a tightly framed anchor needs them.
W=480
H=832

while [ $# -gt 0 ]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --grid)  GRID="$2";  shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$GRID" in
  head|subject|both) ;;
  *) echo "FATAL: --grid must be head, subject or both" >&2; exit 2 ;;
esac

LOG="$ROOT/logs/ref_candidates.log"
mkdir -p "$ROOT/logs" "$OUT"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 || {
  echo "FATAL: no CUDA device (rule 3)." >&2; exit 1; }
[ -f "$LORA" ] || { echo "FATAL: no LoRA at $LORA" >&2; exit 1; }
[ -f "$DIT" ] || { echo "FATAL: no de-prefixed DiT at $DIT (see prepare_musubi_dit.py)" >&2; exit 1; }

# ---------------------------------------------------------------------------
# The LoRA stack
# ---------------------------------------------------------------------------
# This stage, not VACE, is where a limitation in the base model would actually
# show. VACE repaints ~4.4% of the subject under a control video that pins every
# pixel; here the base model draws a whole image from a prompt, unconstrained, so
# whatever it will not or cannot render it simply does not render. If a behaviour
# LoRA is ever going to be worth anything to this project, this is the stage that
# demonstrates it, and the candidates it produces are scored on the same two
# numbers as everything else - match against the held-out bank, and span px.
#
# The subject LoRA stays loaded alongside it. Without it a candidate is not a
# candidate: the base model with no LoRA scores ~0.023 on the held-out bank.
#
# Extra entries come from the untracked overlay, for the reason
# scripts/common.py::lora_stack gives - a third-party LoRA's filename says what
# it makes the model produce, which is a category this repo does not state in a
# tracked file. Names resolve inside ComfyUI/models/loras. EXTRA_LORAS overrides,
# space-separated, as `path` or `path:multiplier`.
EXTRA_LORAS="${EXTRA_LORAS:-$(./venv/bin/python -c "
import yaml
from pathlib import Path
try:
    d = yaml.safe_load(open('configs/prompt.local.yaml')) or {}
except FileNotFoundError:
    d = {}
for e in (d.get('loras') or []):
    e = {'name': e} if isinstance(e, str) else e
    if e.get('name'):
        print(f\"{Path('ComfyUI/models/loras') / e['name']}:{e.get('strength', 1.0)}\")
" | tr '\n' ' ')}"

LORA_FILES=("$LORA"); LORA_MULTS=("${LORA_STRENGTH:-1.0}")
for spec in $EXTRA_LORAS; do
  f="${spec%:*}"; mult="${spec##*:}"
  [ "$f" = "$mult" ] && mult=1.0
  [ -f "$f" ] || { echo "FATAL: no LoRA at $f" >&2; exit 1; }
  LORA_FILES+=("$f"); LORA_MULTS+=("$mult")
done

# Rule 6: whether this trainer's generator takes more than one --lora_weight is
# read from the installed source, not assumed from the flag's name. Passing two
# values to a single-value flag would be a full grid of GPU time spent on either
# an argparse error or - worse - one silently ignored LoRA.
if [ "${#LORA_FILES[@]}" -gt 1 ]; then
  GEN="$MUSUBI_DIR/src/musubi_tuner/wan_generate_video.py"
  for flag in lora_weight lora_multiplier; do
    grep -A4 -- "--$flag" "$GEN" | grep -q 'nargs' || {
      echo "FATAL: $(basename "$GEN") does not declare --$flag as multi-valued in" >&2
      echo "       the installed revision, so a stack cannot be passed to it." >&2
      exit 1; }
  done
fi
say "LoRAs: ${#LORA_FILES[@]} ($(printf '%s ' "${LORA_MULTS[@]}"))"

# The trigger comes from dataset.json, which records it, rather than from a
# caption file beside the crops: the crops are large and regenerable, the split
# is not, and a machine that has been given only dataset.json still knows every
# token and filename the run needs.
TRIGGER="$(./venv/bin/python -c "
import json,sys
d=json.load(open('$DATASET/dataset.json'))
t=d.get('trigger')
sys.exit('dataset.json records no trigger') if not t else print(t)
")"
[ -n "$TRIGGER" ] || { echo "FATAL: no trigger token in $DATASET/dataset.json" >&2; exit 1; }
say "trigger '$TRIGGER' | grid $GRID | seeds: $SEEDS | ${W}x${H} | $STEPS steps"

# The grid varies angle first, lighting second: angle is what a RefSR matcher
# aligns on, lighting is what makes a transferred texture look wrong if it
# disagrees. Two grids exist - `anchor` frames the anchor tightly, `extent` frames
# the whole subject.
#
# THE ATTRIBUTE WARNING, which is why the `extent` grid is not a free win. This
# LoRA was trained on tightly framed anchor crops, deliberately, so it knows the
# anchor region and nothing else. Everything outside that region in these images
# is the base model inventing attributes, and they are not the attributes in the
# source interval. docs/STATE.md is explicit that the attribute in the source is
# the sole ground truth and that references may condition match only. A RefSR
# model handed one of these transfers texture wholesale and cannot be told to take
# the anchor and leave the attribute. Treat them as evidence about proportion,
# never as an attribute donor. The `extent` grid also trades anchor detail for
# coverage: the same canvas now holds the whole subject, so detail per feature
# falls sharply - measured, not assumed, in the span px column.
#
# The descriptions themselves are conditioning text and live in the untracked
# overlay, keyed by grid. No default: generating a full grid against wording that
# is not the wording that produced the recorded numbers spends GPU on results that
# cannot be compared with them.
readarray -t VARIANTS < <(./venv/bin/python -c "
import sys, yaml
try:
    d = yaml.safe_load(open('configs/prompt.local.yaml')) or {}
except FileNotFoundError:
    sys.exit('configs/prompt.local.yaml is missing; it holds the grid descriptions')
grid = d.get('candidate_grid') or {}
want = {'head': ['anchor'], 'subject': ['extent'], 'both': ['anchor', 'extent']}['$GRID']
out = []
for k in want:
    if not grid.get(k):
        sys.exit(f'configs/prompt.local.yaml: candidate_grid.{k} is empty')
    out += [f'{name}:{desc}' for name, desc in grid[k].items()]
print('\\n'.join(out))
")
[ "${#VARIANTS[@]}" -gt 0 ] || { echo "FATAL: no grid variants" >&2; exit 1; }

# The prompt template, also from the overlay: '{trigger}' and '{desc}' are
# substituted per variant.
TEMPLATE="$(./venv/bin/python -c "
import sys, yaml
d = yaml.safe_load(open('configs/prompt.local.yaml')) or {}
t = d.get('candidate_prompt')
sys.exit('configs/prompt.local.yaml records no candidate_prompt') if not t else print(t)
")"

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
        --prompt "$(P="${TEMPLATE//\{trigger\}/$TRIGGER}"; echo "${P//\{desc\}/$desc}")" \
        --video_size "$H" "$W" --video_length 1 --infer_steps "$STEPS" \
        --seed "$s" --attn_mode sdpa --output_type images --save_path "$dst" \
        --lora_weight "${LORA_FILES[@]}" --lora_multiplier "${LORA_MULTS[@]}" \
        >>"$LOG" 2>&1 || \
        say "  seed $s FAILED (continuing)"
  done
  n=$(find "$dst" -name '*.png' | wc -l | tr -d ' ')
  say "  $name: $n image(s)"
done

say "scoring every candidate against the held-out bank"
"$ROOT/venv/bin/python" "$ROOT/scripts/score_lora_match.py" \
    --media "$OUT"/*/ --dataset "$DATASET" \
    --out "$OUT/candidate_scores.json" 2>&1 | tee -a "$LOG"

say "Nothing was promoted into inputs/. A generated image is not a record of the"
say "user's subject; whether it counts as a reference is their call, on these numbers."
