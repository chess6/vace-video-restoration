#!/usr/bin/env bash
# Every synthetic test, in one place, with no GPU and no private configuration.
#
# WHY IT EXISTS. The tests were standalone scripts nobody ran together, and only
# one of them was wired into bootstrap. Two of them could not run at all on a
# fresh checkout, because importing the stage they test resolved a backend role
# at import time and the binding is untracked by design.
#
# HERMETIC BY DEFAULT. This runs with VACE_BACKENDS_CONFIG pointed at a path that
# does not exist, so every role behaves exactly as it would on a clone with no
# binding. Tests that need one inject the synthetic fixture in scripts/
# test_fixtures.py - invented, so it describes nothing and can be tracked. If a
# test only passes because this machine happens to hold private configuration,
# it fails here, which is the point.
#
# Pass --with-local-bindings to run against the real binding instead, which is
# worth doing on a box that has one: the fixture proves the code is correct for
# any label map, the real map proves this map satisfies it.
#
#   scripts/run_unit_tests.sh
#   scripts/run_unit_tests.sh --with-local-bindings
set -uo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"

PY="${PY:-}"
if [ -z "$PY" ]; then
  if [ -x "$PROJ/venv/bin/python" ]; then PY="$PROJ/venv/bin/python"; else PY=python3; fi
fi

if [ "${1:-}" = "--with-local-bindings" ]; then
  echo "== unit tests, against this machine's REAL binding =="
  unset VACE_BACKENDS_CONFIG
else
  echo "== unit tests, hermetic: no binding file reachable =="
  export VACE_BACKENDS_CONFIG=/nonexistent/vace-hermetic-check.yaml
fi
echo "   python: $PY"
echo

TESTS=(
  test_fixtures            # the helper itself, before anything relies on it
  test_backend_fail_loud   # missing bindings terminate the stage
  test_reference_pack      # authority split + staleness
  test_chunking
  test_lora_stack
  test_metrics
  test_composite
  test_evaluate_report
)

pass=0; fail=0; failed=()
for t in "${TESTS[@]}"; do
  out="$("$PY" "$PROJ/scripts/$t.py" 2>&1)"
  rc=$?
  last="$(printf '%s' "$out" | tail -1)"
  if [ $rc -eq 0 ]; then
    printf '  ok    %-24s %s\n' "$t" "${last:0:74}"
    pass=$((pass+1))
  else
    printf '  FAIL  %-24s %s\n' "$t" "${last:0:74}"
    fail=$((fail+1)); failed+=("$t")
  fi
done

echo
if [ $fail -gt 0 ]; then
  echo "UNIT TESTS: $pass passed, $fail FAILED — ${failed[*]}"
  exit 1
fi
echo "UNIT TESTS: $pass passed"
