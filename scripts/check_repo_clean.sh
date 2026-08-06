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
  hits=$(tracked | tr '\n' '\0' | xargs -0 grep -IlniE -- "$PAT" 2>/dev/null | sort -u)
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
echo "=== 6. No tracked file DESCRIBES the footage ==="
# Rule 2a's prose clause. Checks 3 and 3b derive their words from inputs/ and so
# catch filenames; this one catches the other half, which no amount of deriving
# will find: a comment or doc that explains a real fix by describing what is
# actually in the user's footage. Every word below leaked in exactly that way -
# in a docstring, next to the code it justified, sounding like good technical
# writing. Role words (subject, non-target, scenery, attribute, extent) are the
# fix, and they are not in this list.
#
# Deliberately NOT listed, because in this repo they are vocabulary rather than
# description:
#   * dress, skirt, pants, hat, belt - the ATR segmentation model's own label
#     set (make_reference_pack.py). Model taxonomy, not this footage.
#   * standing, seated, walking, side view - the GENERATED candidate pose grid
#     in generate_reference_candidates.sh spans poses on purpose.
#
# A line whose word is functional rather than descriptive may opt out with a
# trailing  rule2a-ok  marker and a reason. One exists today: the Grounding DINO
# detection prompt, which must name poses to detect them.
DESCRIBE='reclin|bystander|doorway|jacket|shirt|blouse|hoodie|sweater|coat|handbag'
hits=$(tracked | while read -r f; do
         [ -f "$f" ] && [ "$f" != "scripts/check_repo_clean.sh" ] \
           && grep -IHniE "$DESCRIBE" "$f" 2>/dev/null
       done | grep -v 'rule2a-ok' || true)
if [ -n "$hits" ]; then
  echo "$hits" | sed 's/^/  VIOLATION: /'
  echo "  Say it in role terms (a subject, a non-target, scenery, an attribute),"
  echo "  or add a trailing 'rule2a-ok' marker if the word is functional."
  rc=1
else
  echo "  OK: no tracked file describes the footage"
fi

echo
echo "=== 6b. No tracked file uses the RETIRED category vocabulary ==="
# Check 6 catches words that describe THIS footage. This one catches the weaker
# but more pervasive leak: words that reveal what KIND of thing the subject is,
# even when nothing about the actual footage is stated. A tracked file may say
# a stage matches a subject against a reference; it may not say what category
# of thing is being matched.
#
# Everything below has a role word that says the same thing:
#   subject, candidate, non-target, reference, match, attribute, appearance,
#   anchor (and anchor region / orientation / keypoints), extent, peripheral
#   extent, exposed region, covering, extremity, scenery.
#
# Three groups, all one alternation:
#   category    what the subject IS
#   anatomy     the parts it is made of - the commonest way the category leaks
#               back in after a scrub, because each word sounds merely technical
#   capture     "photograph" and friends, which say the references are pictures
#               of a real subject, and "landmark"/"yaw", which are only ever
#               named that for one kind of subject
VOCAB='\b(garments?|outfits?|clothing|clothes|apparel|wardrobe|dressed|undress(ed)?|wearing|worn'
VOCAB="$VOCAB"'|identity|identities|persons?|peoples?|humans?|strangers?|men|women|figures?'
VOCAB="$VOCAB"'|faces?|facial|hairs?|hairstyles?|heads?|bodies|torsos?|limbs?|skin|necks?|shoulders?|waists?|chests?'
VOCAB="$VOCAB"'|legs?|hands?|feet|foot|eyes?|eyebrows?|brow|nose|mouth|chin|ocular|sleeves?|hemlines?|collars?'
VOCAB="$VOCAB"'|photographs?|photos?|selfies?|headshots?|landmarks?|yaw|anatomy|gender|posture)\b'
#
# Deliberately NOT listed, because in this repo they carry an unrelated sense far
# more often than the anatomical one, and a check that fires constantly is a
# check that gets ignored:
#   arm         an experiment arm - the unit every result table is built on
#   child       a ComfyUI dynamic-combo child input, `<parent>.<child>`
#   individual  an adjective ("the individual variants")
#   portrait    an image ORIENTATION (480x832), never a framing
# If one of them ever needs to mean a part of a subject, use the role word
# (extremity, peripheral extent) - these four are unmonitored, not permitted.
#
# THE DEPENDENCY FLOOR - exempt because these are functional inputs, not prose.
# Renaming any of them changes behaviour rather than wording:
#   * third-party package, model, class and vendor names (insightface, ArcFace,
#     Hugging Face, sdpose_wholebody, ...)
#   * ComfyUI node types and widget keys (draw_body, draw_face) - the node
#     input contract; a renamed key is an unrecognised key
#   * the ATR parser's own label literals, which are indexed by string
#
# PROMPT TEXT IS NO LONGER ON THIS FLOOR. It used to be, and that exemption was
# the single largest remaining leak: the profile prompts named the subject's
# category outright in configs/*.yaml and in every generated workflows/*.json,
# in the clear, exempted by this very script. Prompts now live in untracked
# configs/prompt.local.yaml (check 7 below), the tracked configs carry a
# category-free default, and neither the YAML prompt blocks nor workflows/ is
# skipped here any more.
#
# Three exemption groups, kept separate so each stays auditable.
#
# 1. VENDOR - third-party package, model, class and vendor names.
EXEMPT='insightface|arcface|faceanalysis|facexlib|facebook|huggingface|hugging face'
EXEMPT="$EXEMPT"'|sdpose|dwpose|openpose|segformer|wholebody|draw_body|draw_face|draw_hand|draw_feet|draw_head'
#
# 2. ATR LABELS - the SegFormer model card's id2label, verbatim and QUOTED. They
#    are indexed by string, so a rename is a KeyError. Only the quoted forms are
#    exempt; the same word in prose is not.
EXEMPT="$EXEMPT"'|["'"'"'](background|hat|hair|sunglasses|upper|skirt|pants|dress|belt|left_shoe|right_shoe|face|left_leg|right_leg|left_arm|right_arm|bag|scarf)["'"'"']'
#
# 3. IDIOM - English that collides with the anatomy group while saying nothing
#    about any subject. Listed here rather than dropped from VOCAB, so each word
#    stays caught everywhere it is NOT one of these.
#      head    `git rev-parse HEAD`, `head -1`, a header, "the head and tail"
#      hand    "hand over", "by hand", "hands off to"
#      human   the reviewer - this project's gates are explicitly human-gated
#      eye     human judgement: "decide by eye", "contradicts the eye", np.eye
EXEMPT="$EXEMPT"'|rev-parse|head -[0-9]|\| *head|HEAD\b|ahead|overhead|header|interface|head and tail'
EXEMPT="$EXEMPT"'|hands? (over|back|off|it|the)|by hand|hand-paint|hand-author|second hand|handed'
EXEMPT="$EXEMPT"'|human|by eye|the eye|your eye|user.s eyes|np\.eye|eyes —|eyes -'
EXEMPT="$EXEMPT"'|rule2a-ok'
hits=$(tracked | while read -r f; do
         [ -f "$f" ] && [ "$f" != "scripts/check_repo_clean.sh" ] \
           && grep -IHniE "$VOCAB" "$f" 2>/dev/null
       done | grep -viE "$EXEMPT" || true)
if [ -n "$hits" ]; then
  echo "$hits" | sed 's/^/  VIOLATION: /'
  echo "  Use a role word (subject, candidate, non-target, reference, match,"
  echo "  attribute, appearance, anchor, extent, exposed region, covering,"
  echo "  extremity, scenery), or add 'rule2a-ok' if it is a functional input."
  rc=1
else
  echo "  OK: no tracked file names the subject's category"
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
echo "=== tracked file count: $(tracked | wc -l) ==="
if [ $rc -eq 0 ]; then echo "REPO CLEAN CHECK: PASSED — safe to push"
else echo "REPO CLEAN CHECK: FAILED — do NOT push"; fi
exit $rc
