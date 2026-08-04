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
STOP='^(source|sources|references|reference|subject|subjects|seed|seeds|video|videos|image|images|photo|photos|media|clip|clips|input|inputs|file|files|frame|frames|mask|masks|depth|output|outputs|readme|gitkeep|macosx|ds_store|store|copy|final|test|temp|tmp|data|sample|samples|record|records|folder|archive|export|render|master|origin|second|middle|centre|center|normal|single|double|upper|lower|middle|before|after|number|version|original)$'

collect_names() {
  # media files directly in inputs/
  find inputs -mindepth 1 -type f 2>/dev/null | sed 's|.*/||'
  # directory names under inputs/ (a folder can be named after a person)
  find inputs -mindepth 1 -type d 2>/dev/null | sed 's|.*/||'
  # entries inside archives, listed without extracting
  for a in $(find inputs -type f \( -iname '*.zip' \) 2>/dev/null); do
    unzip -Z1 "$a" 2>/dev/null | sed 's|/$||' | sed 's|.*/||'
  done
  for a in $(find inputs -type f \( -iname '*.tar' -o -iname '*.tar.gz' -o -iname '*.tgz' \) 2>/dev/null); do
    tar -tf "$a" 2>/dev/null | sed 's|/$||' | sed 's|.*/||'
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
# Both tiers drop the pipeline's own directory names, which are project
# structure rather than anything about the user.
OURS='^(source|references|subject_seeds|ds_store)$'

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
  echo "  (inputs/ holds no user media right now; nothing to check against)"
else
  n=$(echo "$WORDS" | wc -l)
  echo "  derived $n forbidden token(s) from the current contents of inputs/"
  found=0
  while read -r w; do
    [ -z "$w" ] && continue
    hits=$(tracked | while read -r f; do
             [ -f "$f" ] && grep -IlniE "\b${w}\b" "$f" 2>/dev/null
           done | sort -u)
    if [ -n "$hits" ]; then
      echo "  VIOLATION: token '${w}' appears in:"; echo "$hits" | sed 's/^/      /'
      found=1
    fi
  done <<< "$WORDS"
  [ $found -eq 1 ] && rc=1 || echo "  OK: no tracked file mentions any of them"
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
echo "=== tracked file count: $(tracked | wc -l) ==="
if [ $rc -eq 0 ]; then echo "REPO CLEAN CHECK: PASSED — safe to push"
else echo "REPO CLEAN CHECK: FAILED — do NOT push"; fi
exit $rc
