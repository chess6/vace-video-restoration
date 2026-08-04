#!/usr/bin/env bash
# Build the whole controlled background/subject pilot comparison, end to end.
#
# Every variant covers the same interval with the same mask, depth, reference
# sheet, prompt and VACE seed. Only the background treatment and the integration
# path change, which is what makes the comparison worth anything.
#
# Runs strictly sequentially: SeedVR2 and VACE each want most of a 12 GB card, so
# overlapping them would just OOM.
#
#   VACE_RUN=<run> scripts/run_pilot_comparison.sh
#   VACE_RUN=<run> scripts/run_pilot_comparison.sh --skip-background
#
# Stops after the comparison. It never touches the rest of the video: that still
# needs scripts/run_full.sh --confirm-full-run.
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
PY="venv/bin/python"

SKIP_BG=0
for a in "$@"; do
  case "$a" in
    --skip-background) SKIP_BG=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "Unknown option: $a" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m  (%s)\n' "$1" "$(date +%H:%M:%S)"; }
rc=0
step() {                      # step <label> <command...>
  local label="$1"; shift
  say "$label"
  if "$@"; then
    echo "    ok"
  else
    echo "    FAILED: $label" >&2
    rc=1
    return 1
  fi
}

# 1. Background plates. Cached by interval + config hash, so re-running is cheap.
if [ "$SKIP_BG" -eq 0 ]; then
  step "SeedVR2 plates (both profiles)" \
    $PY scripts/restore_background.py --all-profiles --pilot || exit 1
fi

# 2. VACE. Three generations: the baseline plus one per background profile.
#    The baseline's subject is also what the composite path reuses, so the two
#    integration paths are compared on identical generated pixels.
step "VACE baseline (original preserved outside the mask)" \
  $PY scripts/run_chunks.py --pilot || exit 1

step "VACE preserving the conservative plate" \
  $PY scripts/run_chunks.py --pilot --background background_conservative || exit 1

step "VACE preserving the aggressive plate" \
  $PY scripts/run_chunks.py --pilot --background background_aggressive || exit 1

# 3. Assemble every variant, including the composite path (no GPU).
step "Build the variant set" $PY scripts/pilot_compare.py --skip-missing

# 4. Measure them, and cost the stages separately.
step "Measure the variants"  $PY scripts/evaluate_pilot.py
step "Runtime / VRAM / disk"  $PY scripts/estimate_pipeline.py

# 5. Everything that needs a human eye, in one archive. Always the last step:
#    nothing in this project displays anything, so the review bundle is how the
#    work actually reaches the person who has to judge it.
step "Review bundle" $PY scripts/make_review_bundle.py

say "Done"
echo "  REVIEW   : outputs/review_bundle.zip   <- open this"
echo "  Variants : outputs/comparisons/"
echo "  Metrics  : reports/pilot_metrics.json"
echo "  Costs    : reports/pipeline_estimates.md"
echo
echo "  Nothing was displayed. Open the comparison files yourself and record your"
echo "  judgement in reports/pilot_results.md - the metrics are evidence, not a"
echo "  verdict. The full video still requires run_full.sh --confirm-full-run."
exit $rc
