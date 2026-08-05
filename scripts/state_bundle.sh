#!/usr/bin/env bash
# Move the irreplaceable half of a run between a rented box and this laptop.
#
# WHY THIS EXISTS
# A rented box is not storage. Vast volumes are HOST volumes - the instance
# guide is explicit that one persists through recycle and destroy and may be
# shared with other instances *on the same machine* - so there is no
# network-shared disk to rent, and a box whose GPU another customer is using
# cannot be started to fetch anything off it. This session hit exactly that: the
# 960x720 masks, protected submask, depth, control stream and plates existed
# only on a box that would not start, while the laptop held the previous
# geometry's copies and nothing else. Everything else was recoverable; those
# were not, because tracking is the one stage a human has to approve.
#
# WHAT IS WORTH CARRYING, AND WHAT IS NOT
# Not the models: 24 GB that re-downloads from Hugging Face in minutes for free,
# already scripted in download_models.sh. Not the generated variants: they are
# outputs, and re-running them is the cheap part. What matters is the small,
# expensive, human-gated middle - masks, the protected submask, the manifest
# that binds them - plus the streams that cost GPU minutes to rebuild.
#
# Rule 2a: everything this moves is the user's material. It goes between the
# laptop and their own rented box and nowhere else, it is written under paths
# .gitignore denies wholesale, and the archive must never be committed.
#
#   scripts/state_bundle.sh export [ssh-host] [dest-dir]
#   scripts/state_bundle.sh restore [ssh-host] [src-dir]
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

MODE="${1:-}"
HOST="${2:-vast}"
DIR="${3:-$ROOT/intermediate/state_bundles}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/vace-video-restoration}"

# Relative to the project root, on both ends. Anything absent is skipped rather
# than failing the run: a bundle taken mid-pipeline is still worth having.
PATHS=(
  intermediate/chunk_manifest.json
  intermediate/masks
  intermediate/depth
  intermediate/chunks
  intermediate/normalized
  intermediate/background
  intermediate/reference_packs
  intermediate/reference_sheets
  intermediate/shots
  intermediate/reference_exclusions.txt
  intermediate/inspection_allowlist.txt
  intermediate/lora_dataset
)

usage() { sed -n '2,30p' "$0"; exit 2; }
[ -n "$MODE" ] || usage

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

case "$MODE" in
export)
  mkdir -p "$DIR"
  ARCHIVE="$DIR/state_${STAMP}.tar.gz"
  echo "Reading state from $HOST:$REMOTE_ROOT"
  # `tar` chooses what exists remotely; missing members are reported, not fatal.
  # shellcheck disable=SC2029
  ssh -n "$HOST" "cd '$REMOTE_ROOT' && tar czf - \$(for p in ${PATHS[*]}; do [ -e \"\$p\" ] && printf '%s ' \"\$p\"; done)" > "$ARCHIVE"
  [ -s "$ARCHIVE" ] || { echo "FATAL: the archive is empty." >&2; exit 1; }
  # Rule 4: a tar that transferred is not a tar that reads back.
  tar tzf "$ARCHIVE" > "$ARCHIVE.list" || { echo "FATAL: $ARCHIVE does not read back." >&2; exit 1; }
  shasum -a 256 "$ARCHIVE" > "$ARCHIVE.sha256" 2>/dev/null || sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
  echo "wrote $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1), $(wc -l < "$ARCHIVE.list") entries)"
  # Print the geometry it carries. A bundle is only useful against a run of the
  # same geometry, and this session lost time to laptop copies that turned out
  # to be a previous one.
  echo "Geometry in this bundle:"
  PY_BIN="./venv/bin/python"; [ -x "$PY_BIN" ] || PY_BIN=python3
  "$PY_BIN" - "$ARCHIVE" <<'PY'
import json, sys, tarfile
with tarfile.open(sys.argv[1]) as t:
    try:
        m = t.extractfile("intermediate/chunk_manifest.json")
    except KeyError:
        m = None
    if m is None:
        print("  no manifest in the bundle")
    else:
        n = json.load(m).get("normalized", {})
        print(f"  {n.get('width')}x{n.get('height')} @ {n.get('fps')} fps, "
              f"{n.get('total_frames')} frames")
PY
  echo "Keep it. It is the half that cannot be regenerated without a human."
  ;;
restore)
  ARCHIVE="${4:-$(ls -t "$DIR"/state_*.tar.gz 2>/dev/null | head -1)}"
  [ -n "$ARCHIVE" ] && [ -f "$ARCHIVE" ] || { echo "FATAL: no bundle found in $DIR" >&2; exit 1; }
  echo "Restoring $(basename "$ARCHIVE") to $HOST:$REMOTE_ROOT"
  if [ -f "$ARCHIVE.sha256" ]; then
    (shasum -a 256 -c "$ARCHIVE.sha256" >/dev/null 2>&1 || sha256sum -c "$ARCHIVE.sha256" >/dev/null 2>&1) \
      || { echo "FATAL: $ARCHIVE fails its own checksum; do not push it anywhere." >&2; exit 1; }
  fi
  # shellcheck disable=SC2029
  ssh -n "$HOST" "mkdir -p '$REMOTE_ROOT'"
  ssh "$HOST" "cd '$REMOTE_ROOT' && tar xzf -" < "$ARCHIVE"
  echo "Verifying what landed:"
  # shellcheck disable=SC2029
  ssh -n "$HOST" "cd '$REMOTE_ROOT' && for p in ${PATHS[*]}; do [ -e \"\$p\" ] && printf '  %-42s %s\n' \"\$p\" \"\$(du -sh \"\$p\" | cut -f1)\"; done"
  echo "Restored. Rebuild the workflows before running: they are build artefacts"
  echo "of whichever config was last used, and a stale one runs the old geometry."
  ;;
*)
  usage
  ;;
esac
