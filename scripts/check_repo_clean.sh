#!/usr/bin/env bash
# Pre-push guard for CLAUDE.md rule 2a.
#
# Verifies that nothing about the user's media can reach a remote:
#   1. no media, archive or compiled-Python files are tracked
#   2. no tracked file is under inputs/ except the two generic docs
#   3. no tracked file's CONTENT mentions any filename present in inputs/
#   4. no media-derived report is tracked
#
# Check 3 is the important one: it derives the forbidden words from whatever is
# actually in inputs/ right now, so it keeps working for any future media
# without this script needing to know anything about it.
#
# Exit non-zero on any violation. Run before every push.
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ" || exit 1
rc=0

tracked() { git ls-files 2>/dev/null; }

echo "=== 1. No media / archive / pycache tracked ==="
hits=$(tracked | grep -iE '\.(zip|tar|gz|tgz|7z|rar|mp4|mkv|mov|avi|webm|m4v|mpg|mpeg|ts|wmv|flv|png|jpg|jpeg|webp|bmp|tif|tiff|wav|mp3|aac|m4a|flac|safetensors|ckpt|pth|pt|bin|onnx)$|__pycache__|\.pyc$' || true)
if [ -n "$hits" ]; then echo "$hits" | sed 's/^/  VIOLATION: /'; rc=1; else echo "  OK"; fi

echo
echo "=== 2. Nothing under inputs/ except generic docs ==="
hits=$(tracked | grep '^inputs/' | grep -vE '(\.gitkeep|README\.md)$' || true)
if [ -n "$hits" ]; then echo "$hits" | sed 's/^/  VIOLATION: /'; rc=1; else echo "  OK"; fi

echo
echo "=== 3. No tracked file mentions any real input filename ==="
# Forbidden words are derived from the live contents of inputs/: the basenames of
# actual media FILES, plus the entry names inside any archive (which is where the
# revealing names usually are, and which are otherwise invisible because the
# archive is never extracted).
#
# Project-structural and generic words are excluded, otherwise the pipeline's own
# vocabulary ("source", "references", "subject", "video") would trip every file.
#
# The second group is ORDINARY ENGLISH that shows up both in filenames and in
# prose. A filename built from common words ("..._with_...", "..._plus_...")
# turns those words into needles, and they then match innocent sentences in every
# tracked file - including this script, whose own comments discuss filenames.
# That produced a hard FAILED verdict with no leak behind it, which is worse than
# a miss: a guard that cries wolf gets ignored or worked around.
#
# The cost is real and accepted: a user file named exactly "back.jpg" would no
# longer be caught by check 3. Distinctive names still are, and checks 6/6b/7
# cover the description side independently. Extend this list when a new filename
# collides rather than muting the check.
STOP='^(source|sources|references|reference|subject|subjects|seed|seeds|video|videos|image|images|photo|photos|media|clip|clips|input|inputs|file|files|frame|frames|mask|masks|depth|output|outputs|readme|gitkeep|macosx|ds_store|store|copy|final|test|temp|tmp|data|sample|samples|record|records|folder|archive|export|render|master|origin|second|middle|centre|center|normal|single|double|upper|lower|middle|before|after|number|version|original'
STOP="$STOP"'|with|without|back|side|sides|plus|minus|front|left|right|down|over|under|near|full|half|wide|narrow|long|short|open|close|first|last|next|prev|main|part|parts|view|views|angle|angles|shot|shots|size|line|edge|edges|high|low|more|less|same|other|both|each|only|some|more)$'

# WHERE THE NAMES COME FROM
# `inputs/` is where the user's material is supposed to live, and for a long time
# it was the only place looked at. That was not enough: unzipping an archive into
# the project root leaves a sibling directory whose NAME is itself the leak, and
# nothing derived anything from it. Such a directory is typically ignored only
# because everything inside it happens to match a media extension - one stray
# `.txt` and `git add -A` publishes the name.
#
# So two things are collected now, and they are collected differently:
#
#   inputs/, in full        - every name, recursively, plus archive entries.
#   every OTHER untracked   - the DIRECTORY NAME always, and its contents only if
#   top-level directory       the directory holds media, which is what marks it as
#                             somewhere the user's material actually landed.
#
# The name-only half is the cheap, important half: it is the case that was being
# missed entirely. The contents half is gated on media because the alternative -
# sweeping every untracked tree - drags in the pipeline's own output names and
# the tooling's config filenames, thousands of them, none of them about the user.
MEDIA_RE='.*\.\(jpg\|jpeg\|png\|webp\|bmp\|tif\|tiff\|heic\|mp4\|mkv\|mov\|avi\|webm\|m4v\|mpg\|mpeg\|ts\|wmv\|flv\)'

collect_names() {
  local others d
  # Top-level untracked directories, by name. A folder can be named after a
  # candidate, and `__MACOSX/<name>/` is what unzipping an archive here leaves.
  others=$(git ls-files --others --directory --no-empty-directory 2>/dev/null \
           | grep '/$' | sed 's|/$||' | grep -v '/' | sort -u)
  echo "$others"

  for d in inputs $others; do
    [ -d "$d" ] || continue
    if [ "$d" != inputs ] && \
       ! find "$d" -maxdepth 3 -type f -iregex "$MEDIA_RE" -print -quit 2>/dev/null | grep -q .; then
      continue          # no media in it: the name alone, already emitted above
    fi
    # file and directory names (a folder can be named after a candidate)
    find "$d" -mindepth 1 \( -type f -o -type d \) 2>/dev/null | sed 's|.*/||'
    # entries inside archives, listed without extracting
    for a in $(find "$d" -type f -iname '*.zip' 2>/dev/null); do
      unzip -Z1 "$a" 2>/dev/null | sed 's|/$||' | sed 's|.*/||'
    done
    for a in $(find "$d" -type f \( -iname '*.tar' -o -iname '*.tar.gz' -o -iname '*.tgz' \) 2>/dev/null); do
      tar -tf "$a" 2>/dev/null | sed 's|/$||' | sed 's|.*/||'
    done
  done
}

# Two tiers, because a bare 4-letter token is worthless as a signal:
#   PHRASES - whole filename stems, normalised to underscores. Compound names
#             are distinctive and are the primary signal.
#   TOKENS  - single words, but only >= 6 characters and not obviously generic.
#             Short words collide with ordinary English in source comments and
#             are pure noise.
#
# NOTE: this file deliberately contains NO example derived from real input
# names. Writing one here would itself be the leak this script exists to catch.
#
# Both tiers drop the pipeline's own directory names and the tooling's, which are
# project structure rather than anything about the user. Keep this list to things
# the project itself creates - never add a word because it happened to collide.
OURS='^(source|references|subject_seeds|ds_store|macosx|intermediate|outputs|runs|logs|reports|configs|scripts|workflows|models_staging|comfyui|claude|venv|hf_cache|cache)$'

PHRASES=$(collect_names \
        | grep -vE '^(\.gitkeep|README\.md)$' \
        | sed 's/\.[A-Za-z0-9]\+$//' \
        | tr 'A-Z' 'a-z' \
        | sed 's/[^a-z0-9]\+/_/g; s/^_//; s/_$//' \
        | awk 'length($0) >= 6 && /_/' \
        | grep -vE "$OURS" | grep -vE "$STOP" \
        | sort -u)

TOKENS=$(collect_names \
        | grep -vE '^(\.gitkeep|README\.md)$' \
        | sed 's/\.[A-Za-z0-9]\+$//' \
        | tr 'A-Z' 'a-z' \
        | tr -cs 'a-z0-9' '\n' \
        | awk 'length($0) >= 6' \
        | grep -vE "$OURS" | grep -vE "$STOP" \
        | sort -u)

WORDS=$(printf '%s\n%s\n' "$PHRASES" "$TOKENS" | grep -v '^$' | sort -u)
if [ -z "$WORDS" ]; then
  echo "  (no user media in the worktree right now; nothing to check against)"
else
  n=$(echo "$WORDS" | wc -l | tr -d ' ')
  echo "  derived $n forbidden token(s) from the worktree's untracked material"
  # One pass over the tracked files with every token in a single alternation.
  # The old shape was a grep per token per file, which is fine for a dozen
  # tokens and takes minutes once the derivation widens past inputs/.
  PAT="\\b($(echo "$WORDS" | paste -sd'|' -))\\b"
  # shellcheck disable=SC2046
  # This script is exempt from its OWN check 3, and only that check.
  #
  # Its source necessarily contains the STOP vocabulary and prose about how
  # filenames are handled, so it matches tokens derived from filenames built out
  # of ordinary words. That is a structural self-match, not a leak, and it
  # produced a standing FAILED verdict - which is the worst outcome available,
  # because a guard that always fails stops being read.
  #
  # Narrow and auditable, in the same spirit as the dependency-floor exemptions:
  # ONE named file, ONE check. Checks 6, 6b and 7 still scan this file, so a real
  # description or a real prompt phrase appearing here is still caught.
  hits=$(tracked | grep -vxF 'scripts/check_repo_clean.sh' \
         | tr '\n' '\0' | xargs -0 grep -IlniE -- "$PAT" 2>/dev/null | sort -u)
  if [ -n "$hits" ]; then
    # The token is deliberately not echoed: printing it here would write the
    # user's filename into this script's output and into any CI log.
    echo "  VIOLATION: a name derived from the user's material appears in:"
    echo "$hits" | sed 's/^/      /'
    rc=1
  else
    echo "  OK: no tracked file mentions any of them"
  fi
fi

echo
echo "=== 3b. No COMMIT MESSAGE mentions an input filename ==="
# Tracked file contents are not the only thing that gets published. Commit
# messages, author names and branch names travel with the repository too, and
# are far harder to redact after the fact.
if [ -z "${WORDS:-}" ] || ! git rev-parse HEAD >/dev/null 2>&1; then
  echo "  (nothing to check)"
else
  found=0
  meta=$(git log --all --format='%B%n%an%n%ae%n%s' 2>/dev/null;
         git for-each-ref --format='%(refname)' 2>/dev/null)
  while read -r w; do
    [ -z "$w" ] && continue
    if printf '%s' "$meta" | grep -qiE "\b${w}\b"; then
      echo "  VIOLATION: token '${w}' appears in a commit message, author field or ref name"
      found=1
    fi
  done <<< "$WORDS"
  [ $found -eq 1 ] && rc=1 || echo "  OK: commit messages, authors and refs are clean"
fi

echo
echo "=== 4. No media-derived reports tracked ==="
hits=$(tracked | grep -E '^reports/(source_info|tracking_report|assembly|upscaler_comparison)' || true)
if [ -n "$hits" ]; then echo "$hits" | sed 's/^/  VIOLATION: /'; rc=1; else echo "  OK"; fi

echo
echo "=== 5. docs/STATE.md is present and within its size limit ==="
# A working memory that grows without bound stops being read, which defeats its
# purpose. The cap is deliberately small: anything now enforced by a test or a
# guard belongs in that test, and history belongs in git.
STATE=docs/STATE.md
MAX_LINES=200
MAX_BYTES=12288
if [ ! -f "$STATE" ]; then
  echo "  VIOLATION: $STATE is missing (CLAUDE.md rule 0 points every session at it)"
  rc=1
else
  L=$(wc -l < "$STATE"); B=$(wc -c < "$STATE")
  if [ "$L" -gt "$MAX_LINES" ] || [ "$B" -gt "$MAX_BYTES" ]; then
    echo "  VIOLATION: $STATE is ${L} lines / ${B} bytes (limit ${MAX_LINES} / ${MAX_BYTES})."
    echo "             Delete the oldest resolved entries; git history keeps them."
    rc=1
  else
    echo "  OK: ${L}/${MAX_LINES} lines, ${B}/${MAX_BYTES} bytes"
  fi
fi

echo
echo "=== 6/6b. No tracked file describes the footage or names the category ==="
# Checks 3 and 3b derive their words from the worktree and so catch FILENAMES.
# These two catch the other half, which no amount of deriving will find: prose
# that explains a real fix by describing what is in the footage (check 6), and
# the weaker but far more pervasive leak of words revealing what KIND of thing
# the subject is, even when nothing about the footage is stated (check 6b).
#
# WHY THE WORDS ARE NO LONGER IN THIS FILE
# They used to be, inline, and that was itself the largest remaining breach of
# rule 2a. A list of every word the project refuses to say is a negative image of
# the subject: read the guard and you know exactly what it protects, with no
# other file needed - and the guard is the one file guaranteed to be in every
# clone. The machinery is what is worth tracking. The vocabulary is not.
#
# So the words live in an untracked vocab file, on the rule-2c denylist, carried
# between machines by state_bundle.sh. Its sections:
#   [describe]       check 6  - describes THIS footage
#   [category]       check 6b - names the subject's category
#   [exempt-vendor]  third-party package/model/class names; a rename breaks the call
#   [exempt-labels]  parser label literals, quoted; indexed by string
#   [exempt-idiom]   ordinary English colliding with the category list
#
# THE DEPENDENCY FLOOR is what the exempt sections encode: functional inputs
# where rewording changes behaviour rather than wording. Prompt text is NOT on
# it (check 7), and backend identities no longer are either - they moved to
# their own untracked config, so the vendor section should keep shrinking. What
# cannot leave is the lockfile and bootstrap, which must name a package to
# install it. That residue is known and accepted; see CLAUDE.md rule 2a.
#
# A line whose word is functional may opt out with a trailing  rule2a-ok  marker.
VOCABFILE=configs/vocab.local.txt
if [ ! -f "$VOCABFILE" ]; then
  # NOT a skip. A guard that cannot check anything must not print OK: the entire
  # value of these two checks is that a FAILED verdict blocks a push, and
  # "vocabulary unavailable, carrying on" is indistinguishable from "clean" to
  # everyone downstream. Restore it from a state bundle.
  echo "  VIOLATION: $VOCABFILE is missing, so checks 6 and 6b cannot run."
  echo "             Restore it from a state bundle (scripts/state_bundle.sh);"
  echo "             it is untracked by design - see CLAUDE.md rules 2a and 2c."
  rc=1
else
  vocab_section() {  # $1 = section name -> fragments joined into one alternation
    # The header pattern is anchored to a COMPLETE [name] line, not merely a
    # leading '['. Fragments in the exempt-labels section are character classes
    # and so begin with '[' themselves; a looser rule read every one of them as
    # a section header and silently dropped the whole section. That produced an
    # empty alternation, which made the combined pattern contain '||', which
    # made grep exit 2 - and the caller's `|| true` turned that into a green OK.
    # A guard reporting OK because its pattern was too broken to run is the
    # worst outcome available, so both halves are now checked below.
    awk -v want="[$1]" '
      /^\[[a-z][a-z-]*\]$/ { inside = ($0 == want); next }
      /^[[:space:]]*#/     { next }
      inside && NF         { printf "%s|", $0 }
    ' "$VOCABFILE" | sed 's/|$//'
  }

  # Every section must be present and non-empty, checked INDIVIDUALLY. Testing
  # only the joined EXEMPT string would pass: with one section empty it is still
  # a long non-empty string, just a malformed one.
  vocab_missing=0
  for sec in describe category exempt-vendor exempt-labels exempt-idiom; do
    if [ -z "$(vocab_section "$sec")" ]; then
      echo "  VIOLATION: section [$sec] of $VOCABFILE is missing or empty."
      vocab_missing=1
    fi
  done
  DESCRIBE=$(vocab_section describe)
  VOCAB=$(vocab_section category)
  EXEMPT="$(vocab_section exempt-vendor)|$(vocab_section exempt-labels)|$(vocab_section exempt-idiom)"

  # And every pattern must actually compile. grep exits 2 on a bad regex, which
  # is indistinguishable from "no matches" once a `|| true` has been applied.
  for pat in "$DESCRIBE" "$VOCAB" "$EXEMPT"; do
    printf '' | grep -qE "$pat" 2>/dev/null
    if [ $? -gt 1 ]; then
      echo "  VIOLATION: a pattern built from $VOCABFILE is not a valid regex,"
      echo "             so checks 6 and 6b would scan nothing and report OK."
      vocab_missing=1
    fi
  done

  if [ $vocab_missing -eq 1 ]; then
    rc=1
  else
    # Both checks still skip this script, which necessarily discusses how they
    # work. They no longer need to skip it for containing the words themselves.
    scan() {  # $1 = pattern, $2 = post-filter
      tracked | while read -r f; do
        [ -f "$f" ] && [ "$f" != "scripts/check_repo_clean.sh" ] \
          && grep -IHniE "$1" "$f" 2>/dev/null
      done | grep -viE "$2" || true
    }

    hits=$(scan "$DESCRIBE" 'rule2a-ok')
    if [ -n "$hits" ]; then
      echo "$hits" | sed 's/^/  VIOLATION (6, describes the footage): /'
      echo "  Say it in role terms (a subject, a non-target, scenery, an attribute),"
      echo "  or add a trailing 'rule2a-ok' marker if the word is functional."
      rc=1
    else
      echo "  OK (6): no tracked file describes the footage"
    fi

    hits=$(scan "$VOCAB" "$EXEMPT")
    if [ -n "$hits" ]; then
      echo "$hits" | sed 's/^/  VIOLATION (6b, names the category): /'
      echo "  Use a role word (subject, candidate, non-target, reference, match,"
      echo "  attribute, appearance, anchor, extent, exposed region, covering,"
      echo "  extremity, scenery), or add 'rule2a-ok' if it is a functional input."
      rc=1
    else
      echo "  OK (6b): no tracked file names the subject's category"
    fi
  fi
fi
echo
echo "=== 7. No tracked file repeats the untracked conditioning text ==="
# Prompt text is withheld (CLAUDE.md rule 2a, docs/STATE.md) and lives only in
# configs/prompt.local.yaml. Check 6b catches the vocabulary; this catches the
# text itself, however it is worded - including wording that uses no listed word
# at all. It derives its needles from the live overlay, exactly as check 3
# derives its needles from the live contents of inputs/, so it keeps working for
# any future wording without this script needing to know any of it.
#
# The commonest way this regresses is not an edit to a config: it is running
# build_workflows.py, which bakes the loaded prompt into every workflows/*.json
# and leaves them dirty in the working tree, ready to be committed by a `git add
# -A`. That is why workflows/ is scanned here and no longer skipped in 6b.
OVERLAY=configs/prompt.local.yaml
if [ ! -f "$OVERLAY" ]; then
  echo "  (no $OVERLAY on this machine; nothing to check against)"
elif ! command -v python3 >/dev/null 2>&1; then
  echo "  SKIPPED: python3 is unavailable, so the overlay could not be parsed"
else
  # Every leaf string in the overlay, split into clauses on commas and newlines,
  # MINUS every clause the tracked configs already publish as the category-free
  # default. What is left is the wording that exists only in the overlay, which
  # is the only wording worth hunting for.
  #
  # Clauses >= 12 characters only: a lone word from a prompt ("the same") is
  # ordinary English and would fire on everything. And the subtraction is what
  # makes the rest usable - the default deliberately keeps the generic clauses
  # ("new background", "jpeg artifacts"), so without it every config would
  # report itself. A clause the default already publishes is by construction not
  # distinctive; anything distinctive that slips into a tracked file with a
  # category word in it is check 6b's to catch, and the two cover each other.
  PHRASES=$(PY_OVERLAY="$OVERLAY" python3 - <<'PYEOF' 2>/dev/null
import glob, os, re, sys
try:
    import yaml
except ImportError:
    sys.exit(0)

def leaves(v):
    if isinstance(v, dict):
        for x in v.values(): yield from leaves(x)
    elif isinstance(v, list):
        for x in v: yield from leaves(x)
    elif isinstance(v, str):
        yield v

def clauses(doc):
    out = set()
    for s in leaves(doc):
        for part in re.split(r"[,\n]", s):
            part = re.sub(r"\{\w+\}", " ", part).strip().lower()
            part = re.sub(r"\s+", " ", part)
            if len(part) >= 12:
                out.add(part)
    return out

overlay = os.environ["PY_OVERLAY"]
with open(overlay) as fh:
    needles = clauses(yaml.safe_load(fh) or {})

for cfg in glob.glob("configs/*.yaml"):
    if os.path.abspath(cfg) == os.path.abspath(overlay):
        continue
    try:
        with open(cfg) as fh:
            block = (yaml.safe_load(fh) or {}).get("prompt") or {}
    except Exception:
        continue
    needles -= clauses(block)

print("\n".join(sorted(needles)))
PYEOF
)
  if [ -z "$PHRASES" ]; then
    echo "  SKIPPED: could not read phrases from $OVERLAY (is PyYAML installed?)"
  else
    n=$(echo "$PHRASES" | wc -l | tr -d ' ')
    echo "  derived $n phrase(s) from $OVERLAY"
    found=0
    while IFS= read -r ph; do
      [ -z "$ph" ] && continue
      hits=$(tracked | while read -r f; do
               [ -f "$f" ] && grep -IlniF -- "$ph" "$f" 2>/dev/null
             done | sort -u)
      if [ -n "$hits" ]; then
        # Do not echo the phrase: printing it here would put it back in a
        # tracked file's output and, worse, into a CI log.
        echo "  VIOLATION: a phrase from $OVERLAY appears in:"; echo "$hits" | sed 's/^/      /'
        found=1
      fi
    done <<< "$PHRASES"
    if [ $found -eq 1 ]; then
      echo "  Regenerate workflows from a checkout without the overlay, or revert"
      echo "  the config. The tracked default is the category-free one."
      rc=1
    else
      echo "  OK: the withheld wording appears in no tracked file"
    fi
  fi
fi

echo
echo "=== 8. The agent denylist is wired up and complete (rule 2c) ==="
# Checks 1-7 all guard the same exit: what reaches a remote. Rule 2c guards a
# different one - what an agent working in the worktree can read - and nothing
# above would notice if that enforcement quietly stopped existing. It is three
# files that have to agree, which is exactly the shape that drifts: someone adds
# a path to the manifest and not to the settings, or edits the settings and
# leaves the manifest behind, and the half nobody re-reads is the half that was
# doing the protecting.
DENYLIST=configs/agent_denylist.txt
SETTINGS=.claude/settings.json
HOOK=scripts/agent_guard.sh
if [ ! -f "$DENYLIST" ]; then
  echo "  VIOLATION: $DENYLIST is missing - rule 2c has no list to enforce"; rc=1
elif [ ! -f "$SETTINGS" ]; then
  echo "  VIOLATION: $SETTINGS is missing - nothing denies Read/Glob/Grep"; rc=1
elif [ ! -f "$HOOK" ]; then
  echo "  VIOLATION: $HOOK is missing - Bash walks around the deny rules"; rc=1
else
  miss=0
  # Every manifest entry must appear in the deny rules. Compared as literal
  # text: the settings file spells each path out, so a missing one is a real
  # gap rather than a formatting difference.
  while IFS= read -r pat; do
    pat="${pat%%#*}"
    pat="$(printf '%s' "$pat" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    [ -z "$pat" ] && continue
    if ! grep -qF -- "$pat" "$SETTINGS"; then
      echo "  VIOLATION: '$pat' is on $DENYLIST but no deny rule in $SETTINGS covers it"
      miss=1
    fi
  done < "$DENYLIST"

  if ! grep -q 'PreToolUse' "$SETTINGS" || ! grep -qF 'agent_guard.sh' "$SETTINGS"; then
    echo "  VIOLATION: $SETTINGS does not register $HOOK as a PreToolUse hook."
    echo "             Read/Glob/Grep denials alone leave Bash wide open."
    miss=1
  fi
  if [ ! -x "$HOOK" ]; then
    echo "  VIOLATION: $HOOK is not executable, so the hook silently never runs"; miss=1
  fi
  # The three tracked pieces must actually be tracked, or a fresh clone lands
  # with no enforcement at all and nothing says so.
  for f in "$DENYLIST" "$SETTINGS" "$HOOK" scripts/test_agent_guard.sh; do
    tracked | grep -qxF "$f" || { echo "  VIOLATION: $f is not tracked; a clone would arrive unprotected"; miss=1; }
  done
  # And the local-only files the denylist names must be ignored, since a path
  # worth hiding from an agent is a path worth never pushing.
  for f in configs/prompt.local.yaml configs/backends.local.yaml configs/vocab.local.txt; do
    if ! git check-ignore -q "$f" 2>/dev/null; then
      echo "  VIOLATION: $f is not ignored by .gitignore"; miss=1
    fi
  done
  [ $miss -eq 1 ] && rc=1 || echo "  OK: manifest, deny rules and Bash hook agree"
fi

echo
echo "=== tracked file count: $(tracked | wc -l) ==="
if [ $rc -eq 0 ]; then echo "REPO CLEAN CHECK: PASSED — safe to push"
else echo "REPO CLEAN CHECK: FAILED — do NOT push"; fi
exit $rc
