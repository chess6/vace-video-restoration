#!/usr/bin/env bash
# Tests scripts/agent_guard.sh, the PreToolUse hook enforcing CLAUDE.md rule 2c.
#
# The cases are held in this file rather than passed on a command line, because
# a command line naming a denied path is blocked by the very hook under test.
# That is not a testing inconvenience to work around - it is the guard working -
# so the harness is shaped to avoid tripping it instead of being exempted from it.
#
# Two regressions are pinned here, both found by this file rather than by
# reading the code, and both from the same mistake of reducing a glob to its
# literal prefix:
#   * `intermediate/**/dataset.json` denied the whole intermediate tree, so an
#     ffprobe of a normalized clip was refused
#   * the same reduction missed `find intermediate -name dataset.json`, because
#     the command never writes the trailing slash the needle carried
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GUARD="$PROJ/scripts/agent_guard.sh"
DENYLIST="$PROJ/configs/agent_denylist.txt"
pass=0; fail=0

check() {  # want cmd desc
  local want="$1" cmd="$2" desc="$3" rc got
  printf '%s' "$(CMD="$cmd" python3 -c 'import json,os;print(json.dumps({"tool_input":{"command":os.environ["CMD"]}}))')" \
    | bash "$GUARD" >/dev/null 2>&1
  rc=$?
  got=PERMIT; [ $rc -eq 2 ] && got=BLOCK
  if [ "$got" = "$want" ]; then pass=$((pass+1)); printf '  ok    %-6s %s\n' "$got" "$desc"
  else fail=$((fail+1)); printf '  FAIL  want=%s got=%s  %s\n' "$want" "$got" "$desc"; fi
}

echo "=== denied paths are refused, including via routes around the Read tool ==="
check BLOCK 'cat configs/prompt.local.yaml'                                  'read the conditioning overlay'
check BLOCK 'cat configs/backends.local.yaml'                                'read the backend bindings'
check BLOCK 'ls -la inputs/references/'                                      'list reference filenames'
check BLOCK 'ls inputs'                                                      'list inputs with no trailing slash'
check BLOCK 'grep -ri trigger logs/'                                         'grep the run logs'
check BLOCK 'python3 -c "print(open(chr(99)+\"onfigs/prompt.local.yaml\").read())"' 'python instead of Read'
check BLOCK 'find intermediate -name dataset.json -exec cat {} +'            'find + cat the training manifest'
check BLOCK 'tail -5 intermediate/reference_exclusions.txt'                  'tail the exclusion notes'
check BLOCK 'head -1 intermediate/inspection_allowlist.txt'                  'read the rule-2b grant ledger'
check BLOCK 'cat .claude/settings.local.json'                                'read the permission allow-list'
check BLOCK 'cp inputs/references/x.jpg /tmp/'                               'copy material out of inputs'
check BLOCK 'tar -tf intermediate/state_bundles/b.tar'                       'list a state bundle'

echo
echo "=== ordinary work is not blocked ==="
check PERMIT 'git status --porcelain'                                        'ordinary git command'
check PERMIT 'python3 scripts/smoke_test.py'                                 'run a pipeline script'
check PERMIT 'ls scripts/'                                                   'list the scripts directory'
check PERMIT 'bash scripts/check_repo_clean.sh'                              'run the push guard'
check PERMIT 'ffprobe -v error -show_streams intermediate/normalized/x.mp4'  'probe a derived clip'
check PERMIT 'ls outputs/'                                                   'inputs must not fire inside outputs'
check PERMIT 'python3 scripts/assemble.py --config configs/local_1p3b.yaml'  'read a tracked config'

echo
echo "=== the guard fails closed ==="
mv "$DENYLIST" "$DENYLIST.testbak"
printf '{"tool_input":{"command":"echo hi"}}' | bash "$GUARD" >/dev/null 2>&1
rc=$?
mv "$DENYLIST.testbak" "$DENYLIST"
if [ $rc -eq 2 ]; then pass=$((pass+1)); echo "  ok    BLOCK  harmless command refused while the denylist is missing"
else fail=$((fail+1)); echo "  FAIL  guard permitted with no denylist (rc=$rc)"; fi

# A malformed payload must not be a way through: if the JSON cannot be parsed,
# the raw text is matched instead of the command being waved past.
printf 'not json at all configs/prompt.local.yaml' | bash "$GUARD" >/dev/null 2>&1
if [ $? -eq 2 ]; then pass=$((pass+1)); echo "  ok    BLOCK  unparseable payload naming a denied path still refused"
else fail=$((fail+1)); echo "  FAIL  unparseable payload permitted"; fi

echo
echo "passed=$pass failed=$fail"
[ $fail -eq 0 ]
