#!/usr/bin/env bash
# Positive control for check_repo_clean.sh checks 6 and 6b.
#
# WHY THIS EXISTS
# Moving the vocabulary out of the tracked guard and into an untracked file
# turned checks 6/6b from "obviously present" into "loaded at runtime from
# somewhere else", and the first attempt at that load silently dropped an entire
# section, built a malformed pattern, and reported OK. Nothing noticed, because
# every existing test only ever asked whether the guard PASSES on a clean repo -
# which a guard that scans nothing also does, perfectly, forever.
#
# So this asks the opposite question: plant a violation and confirm the guard
# fails. The needle is lifted from the vocab file at runtime rather than written
# here, because writing one here would put the withheld word into a tracked file
# and hand the leak straight back.
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ" || exit 1
VOCABFILE=configs/vocab.local.txt
GUARD=scripts/check_repo_clean.sh
pass=0; fail=0

if [ ! -f "$VOCABFILE" ]; then
  echo "SKIPPED: $VOCABFILE is absent on this machine (untracked by design)."
  echo "Restore it from a state bundle to run this test."
  exit 0
fi

# A literal, greppable word from a section - no regex metacharacters, so it can
# be planted verbatim and will match itself.
needle_from() {
  awk -v want="[$1]" '
    /^\[[a-z][a-z-]*\]$/ { inside = ($0 == want); next }
    /^[[:space:]]*#/     { next }
    inside && NF         { print }
  ' "$VOCABFILE" \
  | tr '|' '\n' \
  | grep -oE '^[a-z]{5,}$' \
  | head -1
}

TMP=intermediate/.vocab_guard_probe.md
cleanup() { git rm -q --cached "$TMP" >/dev/null 2>&1; rm -f "$TMP"; }
trap cleanup EXIT

plant_and_check() {  # $1 = needle, $2 = label of the check expected to fire
  local needle="$1" label="$2" out
  [ -z "$needle" ] && { echo "  FAIL  could not lift a needle for $label"; fail=$((fail+1)); return; }
  mkdir -p "$(dirname "$TMP")"
  printf 'probe %s probe\n' "$needle" > "$TMP"
  git add -f "$TMP" 2>/dev/null
  out=$(bash "$GUARD" 2>&1)
  cleanup
  if printf '%s' "$out" | grep -q "VIOLATION ($label"; then
    pass=$((pass+1)); echo "  ok    planted needle caught by check $label"
  else
    fail=$((fail+1)); echo "  FAIL  planted needle NOT caught by check $label"
  fi
}

echo "=== a planted violation is detected ==="
plant_and_check "$(needle_from category)" "6b"
plant_and_check "$(needle_from describe)" "6"

echo
echo "=== a missing vocab file FAILS rather than passing ==="
mv "$VOCABFILE" "$VOCABFILE.testbak"
out=$(bash "$GUARD" 2>&1); rc_guard=$?
mv "$VOCABFILE.testbak" "$VOCABFILE"
if [ $rc_guard -ne 0 ] && printf '%s' "$out" | grep -q "cannot run"; then
  pass=$((pass+1)); echo "  ok    guard fails loudly with no vocabulary"
else
  fail=$((fail+1)); echo "  FAIL  guard passed (or was silent) with no vocabulary"
fi

echo
echo "=== the clean tree still passes ==="
if bash "$GUARD" >/dev/null 2>&1; then
  pass=$((pass+1)); echo "  ok    clean worktree passes"
else
  echo "  note  clean worktree does not currently pass; see the guard's output"
fi

echo
echo "passed=$pass failed=$fail"
[ $fail -eq 0 ]
