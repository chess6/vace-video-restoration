#!/usr/bin/env bash
# PreToolUse hook for Bash. Enforces CLAUDE.md rule 2c at the tool boundary.
#
# WHY A HOOK AND NOT JUST permissions.deny
# A `deny` entry on Read/Glob/Grep stops those tools and nothing else. Bash is a
# hole straight through it: `cat`, `head`, `grep -r`, `ls`, `find`, `python3 -c
# "print(open(...).read())"` all reach the same bytes without going near the
# denied tool. So the deny rules cover the file-reading tools and this hook
# covers the shell, and check 8 in check_repo_clean.sh verifies both still
# reference every line of configs/agent_denylist.txt.
#
# WHAT IT MATCHES, AND WHY IT IS DELIBERATELY BLUNT
# It matches the denylisted PATH appearing anywhere in the command string, not a
# parsed argument list. That over-blocks - `echo "see configs/prompt.local.yaml"`
# is refused, and so is a command that merely mentions the path in a comment.
# That is the correct direction to err. Parsing a shell command precisely enough
# to know which words are file arguments means reimplementing the shell, and
# every gap in such a parser is a silent read of the thing this exists to
# protect. A false refusal costs one round trip; a false permit cannot be undone
# once the bytes are in a context window.
#
# WHAT IT DOES NOT AND CANNOT DO
# This binds an agent's Bash tool. It does not bind the pipeline: scripts/*.py
# read these files directly and must, or nothing runs. It also cannot stop an
# agent that reaches the bytes without naming the path - a wildcard that happens
# to expand over one, a script written to a scratch file and then executed, a
# path assembled from variables. Rule 2c is what covers those; this hook is the
# part that does not depend on the agent cooperating.
#
# Contract: stdin is the PreToolUse JSON payload. Exit 0 permits, exit 2 blocks
# and returns stderr to the agent as the reason.
set -uo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DENYLIST="$PROJ/configs/agent_denylist.txt"

payload="$(cat)"

# The command text, however the payload is shaped. python3 is the parser when it
# is available because a Bash command can contain anything at all, including
# quotes and newlines that defeat a sed-based extraction.
if command -v python3 >/dev/null 2>&1; then
  cmd="$(printf '%s' "$payload" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    # Unparseable payload: emit the raw text so matching still happens. Failing
    # open here would make a malformed payload a way through the guard.
    sys.stdout.write(sys.stdin.read() if sys.stdin.readable() else "")
    sys.exit(0)
ti = d.get("tool_input") or {}
sys.stdout.write(" ".join(str(ti.get(k, "")) for k in ("command", "file_path", "path", "pattern")))
' 2>/dev/null)"
  # Parser produced nothing at all: fall back to the whole payload rather than
  # permitting on an empty string.
  [ -z "${cmd//[[:space:]]/}" ] && cmd="$payload"
else
  cmd="$payload"
fi

if [ ! -f "$DENYLIST" ]; then
  echo "BLOCKED (rule 2c): $DENYLIST is missing, so no command can be checked" >&2
  echo "against it. Restore it before running shell commands in this project." >&2
  exit 2
fi

# NEEDLE DERIVATION
# Reducing every glob to its literal prefix was the obvious approach and it is
# wrong in both directions at once. `intermediate/**/dataset.json` reduces to
# `intermediate/`, which denies a tree that is almost entirely innocent - depth
# maps, masks, chunks - while `find intermediate -name dataset.json` slips past
# because the command never writes the trailing slash. Both were caught by the
# case file; a guard tuned only against the paths it expects to see is a guard
# tuned against the wrong thing.
#
# So the reduction depends on where the wildcard sits:
#   no wildcard          -> the whole path, AND its basename, so that a command
#                           run from inside the directory is caught too
#   wildcard, literal    -> the basename ONLY. `dataset.json` is distinctive;
#   basename                the directory above it is not, and denying the
#                           directory is what over-blocked.
#   wildcard in basename -> the literal prefix, trailing slash stripped, since
#                           the specific filename is unknowable in advance
emit_needles() {
  local pat="$1" base="${1##*/}" prefix
  if [ "${pat#*\*}" = "$pat" ]; then
    printf '%s\n' "$pat"
    [ "$base" != "$pat" ] && printf '%s\n' "$base"
  elif [ "${base#*\*}" = "$base" ]; then
    printf '%s\n' "$base"
  else
    prefix="${pat%%\**}"
    prefix="${prefix%/}"
    [ -n "$prefix" ] && printf '%s\n' "$prefix"
  fi
}

# Matched on a word boundary rather than as a bare substring, so `inputs` does
# not fire inside `outputs`. The boundary is still loose enough that a denied
# directory name appearing as an ordinary English word in quoted prose - `echo
# "the pipeline runs"` - is refused. That is over-blocking, it is known, and it
# is left in place: the message says why, rephrasing costs one round trip, and
# the alternative is parsing the shell to decide which words are paths.
NEEDLES="$(while IFS= read -r pat; do
  pat="${pat%%#*}"
  pat="$(printf '%s' "$pat" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
  [ -z "$pat" ] && continue
  emit_needles "$pat"
done < "$DENYLIST" | sort -u)"

while IFS= read -r needle; do
  [ -z "$needle" ] && continue
  esc="$(printf '%s' "$needle" | sed 's/[][\.^$*+?(){}|/-]/\\&/g')"
  if printf '%s' "$cmd" | grep -qE -- "(^|[^[:alnum:]_.-])${esc}([^[:alnum:]_-]|$)"; then
    echo "BLOCKED by CLAUDE.md rule 2c: this command references '$needle'," >&2
    echo "which is on configs/agent_denylist.txt - an agent may not read it." >&2
    echo >&2
    echo "These paths carry the conditioning text, the backend bindings, or" >&2
    echo "the names and content of the user's media. The pipeline reads them;" >&2
    echo "you do not. If you need a fact from one, ask the user to run the" >&2
    echo "command and report only the structural result (a count, a shape, a" >&2
    echo "pass/fail) - never the content itself." >&2
    exit 2
  fi
done <<< "$NEEDLES"

exit 0
