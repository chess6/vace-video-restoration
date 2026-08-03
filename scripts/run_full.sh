#!/usr/bin/env bash
# Phase 11 - the ONLY entry point that processes the entire video.
#
# It refuses to start without --confirm-full-run, and prints the measured time
# and disk estimates first. The default pipeline stops after the pilot.
#
#   scripts/run_full.sh                     # prints estimates, then refuses
#   scripts/run_full.sh --confirm-full-run  # actually runs
#
# Resumable: it drives scripts/run_chunks.py, which records per-chunk status in
# the manifest. Re-running continues where it stopped. Failed chunks alone can be
# retried with scripts/run_chunks.py --resume-failed.
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$PROJ/venv/bin/python"
CONFIRM=0
DELIVER=""
for a in "$@"; do
  case "$a" in
    --confirm-full-run) CONFIRM=1 ;;
    --deliver=*) DELIVER="${a#*=}" ;;
    *) echo "Unknown argument: $a"; exit 2 ;;
  esac
done

BENCH="$PROJ/reports/benchmark.json"
MANIFEST="$PROJ/intermediate/chunk_manifest.json"

echo "==============================================================="
echo " FULL RUN - 30 minute source"
echo "==============================================================="

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: no chunk manifest at $MANIFEST"
  echo "       Run scripts/preprocess_source.py first."
  exit 1
fi

"$PY" - "$MANIFEST" "$BENCH" <<'PYEOF'
import json, sys, os
man = json.load(open(sys.argv[1]))
chunks = man["chunks"]
done = sum(1 for c in chunks if c["status"] == "done")
failed = sum(1 for c in chunks if c["status"] == "failed")
pending = len(chunks) - done - failed

print(f"  source        : {man['source']['filename']}")
print(f"  duration      : {man['normalized']['duration_sec']/60:.1f} min "
      f"({man['normalized']['total_frames']} frames @ {man['normalized']['fps']} fps)")
print(f"  resolution    : {man['normalized']['width']}x{man['normalized']['height']}")
print(f"  shots         : {len(man['shots'])}")
print(f"  chunks        : {len(chunks)}  (done {done}, failed {failed}, pending {pending})")

needs = [s['shot_id'] for s in man['shots'] if s.get('subject_status') == 'needs_user']
if needs:
    print(f"  !! {len(needs)} shot(s) still flagged needs_user: {', '.join(needs[:8])}"
          + (" ..." if len(needs) > 8 else ""))

if os.path.exists(sys.argv[2]):
    b = json.load(open(sys.argv[2]))
    spf = b["measured"]["seconds_per_generated_frame"]
    gen = sum(c["n_frames"] for c in chunks if c["status"] != "done")
    secs = gen * spf
    print()
    print("  MEASURED on this machine:")
    print(f"    {spf:.3f} s per generated frame, peak VRAM {b['measured']['peak_vram_mb']} MiB")
    print(f"    {gen} frames still to generate")
    print(f"    ESTIMATED REMAINING WALL CLOCK: {secs/3600:.1f} hours")
    print(f"    disk (intermediate): {b['disk_estimate']['intermediate_human']}")
    print(f"    disk (outputs)     : {b['disk_estimate']['outputs_human']}")
    print(f"    disk (total)       : {b['disk_estimate']['total_human']}")
else:
    print()
    print("  NO BENCHMARK FOUND. Run scripts/benchmark.py for a real estimate.")
PYEOF

echo
df -h "$PROJ" | tail -1 | awk '{print "  free disk on this volume: " $4}'
echo

if [ "$CONFIRM" -ne 1 ]; then
  echo "---------------------------------------------------------------"
  echo "STOPPING. This would process the entire video."
  echo "Re-run with the explicit flag if you accept the estimates above:"
  echo
  echo "    scripts/run_full.sh --confirm-full-run"
  echo "---------------------------------------------------------------"
  exit 3
fi

echo "Confirmed. Starting the full run at $(date -Iseconds)."
echo "Progress is logged to logs/run_chunks.log; this is resumable."
echo

"$PY" "$PROJ/scripts/run_chunks.py" --all
rc=$?
if [ $rc -ne 0 ]; then
  echo
  echo "Some chunks failed. Retry only those with:"
  echo "    $PY $PROJ/scripts/run_chunks.py --resume-failed"
  exit $rc
fi

echo
echo "All chunks done. Assembling..."
ASM=("$PY" "$PROJ/scripts/assemble.py")
[ -n "$DELIVER" ] && ASM+=(--deliver "$DELIVER")
"${ASM[@]}"
rc=$?
echo
[ $rc -eq 0 ] && echo "FULL RUN COMPLETE at $(date -Iseconds)." \
              || echo "Assembly reported a problem; see logs/assemble.log"
exit $rc
