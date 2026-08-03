#!/usr/bin/env bash
# End-to-end validation of the whole pipeline on SYNTHETIC media.
#
# Purpose: prove every script actually runs and produces correct artefacts before
# any real footage exists. It exercises inspection, CFR normalization, scene-cut
# detection, 4n+1 chunking, depth, SAM 2 tracking, mask export, VACE generation,
# chunk assembly, audio remux and A/V sync verification.
#
# What it does NOT prove: identity matching. A synthetic humanoid is not a person,
# so Grounding DINO / ArcFace matching cannot be judged on it. That stage is
# exercised here with a manual seed, and is validated for real when you supply
# actual references. This limitation is stated in the report rather than hidden.
#
# SAFETY:
#   * refuses to run if inputs/source or inputs/references contain YOUR files
#   * restores any pre-existing chunk manifest afterwards
#   * removes its own synthetic media on completion unless --keep is given
#   * never displays any image or video
#
#   scripts/selftest.sh [--keep] [--steps 6]
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$PROJ/venv/bin/python"
KEEP=0
STEPS=6
for a in "$@"; do
  case "$a" in
    --keep) KEEP=1 ;;
    --steps) shift ;;
    --steps=*) STEPS="${a#*=}" ;;
    *) echo "Unknown arg: $a"; exit 2 ;;
  esac
done

REPORT="$PROJ/reports/selftest.md"
LOG="$PROJ/logs/selftest.log"
SRC="$PROJ/inputs/source/_selftest_source.mp4"
MANIFEST="$PROJ/intermediate/chunk_manifest.json"
BACKUP="$PROJ/intermediate/chunk_manifest.selftest-backup.json"

pass=0; fail=0
declare -a LINES

step() {  # step "<name>" <command...>
  local name="$1"; shift
  echo "-----------------------------------------------------------------" | tee -a "$LOG"
  echo ">>> $name" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    echo "    PASS: $name"; pass=$((pass+1)); LINES+=("| $name | PASS |")
    return 0
  else
    echo "    FAIL: $name (see $LOG)"; fail=$((fail+1)); LINES+=("| $name | **FAIL** |")
    return 1
  fi
}

check_file() {  # check_file "<name>" <path>
  if [ -s "$2" ]; then
    echo "    PASS: $1 exists ($(du -h "$2" | cut -f1))"; pass=$((pass+1))
    LINES+=("| $1 | PASS |"); return 0
  else
    echo "    FAIL: $1 missing ($2)"; fail=$((fail+1))
    LINES+=("| $1 | **FAIL** |"); return 1
  fi
}

# ---- safety ------------------------------------------------------------------
# Only real MEDIA counts as "your files"; .gitkeep and README.md do not.
MEDIA_FIND=( -type f ! -name '_selftest_*' \(
  -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.avi'
  -o -iname '*.webm' -o -iname '*.m4v' -o -iname '*.mpg' -o -iname '*.mpeg'
  -o -iname '*.ts' -o -iname '*.wmv' -o -iname '*.flv'
  -o -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp'
  -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) )
user_src=$(find "$PROJ/inputs/source"     -maxdepth 1 "${MEDIA_FIND[@]}" | head -1)
user_ref=$(find "$PROJ/inputs/references" -maxdepth 1 "${MEDIA_FIND[@]}" | head -1)
if [ -n "$user_src" ] || [ -n "$user_ref" ]; then
  echo "REFUSING TO RUN: inputs/ already contains your files:"
  [ -n "$user_src" ] && echo "  $user_src"
  [ -n "$user_ref" ] && echo "  $user_ref"
  echo "The self-test would overwrite the chunk manifest built from them."
  echo "Run the real pipeline instead (see README.md)."
  exit 1
fi

mkdir -p "$PROJ/logs" "$PROJ/reports"
: > "$LOG"
[ -f "$MANIFEST" ] && cp "$MANIFEST" "$BACKUP"

echo "================================================================="
echo " SELF-TEST on synthetic media  ($(date -Iseconds))"
echo "================================================================="

# ---- 0. ComfyUI must be up ----------------------------------------------------
if ! "$PROJ/scripts/start_comfyui.sh" --status >/dev/null 2>&1; then
  echo ">>> Starting ComfyUI"
  "$PROJ/scripts/start_comfyui.sh" --daemon >>"$LOG" 2>&1
fi
step "ComfyUI API responds" "$PROJ/scripts/start_comfyui.sh" --status

# ---- 1. synthetic media -------------------------------------------------------
step "generate synthetic 240p source + references" \
  "$PY" "$PROJ/scripts/make_selftest_media.py" \
    --video "$SRC" --refs-dir "$PROJ/inputs/references" --seconds 20 --cut-at 10
check_file "synthetic source clip" "$SRC"

# ---- 2. inspection ------------------------------------------------------------
step "inspect_source.py" "$PY" "$PROJ/scripts/inspect_source.py" --source "$SRC" --exact-frames
check_file "reports/source_info.json" "$PROJ/reports/source_info.json"

# ---- 3. preprocessing ---------------------------------------------------------
# --auto-aspect: the synthetic clip is 4:3, so 832x480 would waste 25% on bars.
step "preprocess_source.py (CFR, scene cuts, 4n+1 chunking)" \
  "$PY" "$PROJ/scripts/preprocess_source.py" --source "$SRC" --auto-aspect
check_file "chunk manifest" "$MANIFEST"

step "manifest invariants (4n+1 lengths, dims %16, cuts respected)" \
  "$PY" - "$MANIFEST" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["chunks"], "no chunks produced"
for c in m["chunks"]:
    assert (c["n_frames"] - 1) % 4 == 0, f"{c['chunk_id']} length {c['n_frames']} not 4n+1"
    assert c["width"] % 16 == 0 and c["height"] % 16 == 0, f"{c['chunk_id']} dims not %16"
    assert c["end_frame"] - c["start_frame"] == c["n_frames"]
shots = {s["shot_id"]: s for s in m["shots"]}
for c in m["chunks"]:
    s = shots[c["shot_id"]]
    assert c["start_frame"] >= s["start_frame"] and c["end_frame"] <= s["end_frame"], \
        f"{c['chunk_id']} crosses its shot boundary"
assert len(m["shots"]) >= 2, f"expected the scene cut to be detected, got {len(m['shots'])} shot(s)"
print(f"OK: {len(m['shots'])} shots, {len(m['chunks'])} chunks, all 4n+1 and inside shots")
PYEOF

# ---- 4. references ------------------------------------------------------------
step "prepare_references.py" "$PY" "$PROJ/scripts/prepare_references.py"
check_file "reference sheet" "$PROJ/intermediate/reference_sheets/reference_sheet.png"
check_file "contact sheet" "$PROJ/intermediate/reference_sheets/contact_sheet.png"

# ---- 5. depth -----------------------------------------------------------------
step "make_depth.py" "$PY" "$PROJ/scripts/make_depth.py"
check_file "full depth video" "$PROJ/intermediate/depth/full_depth.mkv"

# ---- 6. tracking --------------------------------------------------------------
# Manual seed: a synthetic humanoid is not a person, so automatic identity
# matching is not meaningful here. SAM 2 propagation and mask export ARE.
# Every shot must be tracked, otherwise a pilot landing on an untracked shot has
# no mask. The synthetic figure restarts on the left after the cut, so the same
# relative box seeds both shots.
BOX=$("$PY" - "$MANIFEST" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
w, h = m["normalized"]["width"], m["normalized"]["height"]
# the figure starts near the left, centred vertically around 0.62*H
print(f"{int(w*0.06)},{int(h*0.28)},{int(w*0.34)},{int(h*0.92)}")
PYEOF
)
SHOTS=$("$PY" -c "import json;print(' '.join(s['shot_id'] for s in json.load(open('$MANIFEST'))['shots']))")
FIRST_SHOT=${SHOTS%% *}
for S in $SHOTS; do
  step "track_subject.py (SAM2, manual seed on $S)" \
    "$PY" "$PROJ/scripts/track_subject.py" --shot "$S" \
      --init-box "$BOX" --seed-frame 0 --force
  check_file "mask video $S" "$PROJ/intermediate/masks/${S}_mask.mkv"
  check_file "mask review sheet $S" "$PROJ/intermediate/masks/review/${S}_review.png"
done

step "mask alignment (frames/dims/fps match source and depth)" \
  "$PY" - "$MANIFEST" "$FIRST_SHOT" <<'PYEOF'
import json, subprocess, sys
from pathlib import Path
m = json.load(open(sys.argv[1])); shot = sys.argv[2]
root = Path(sys.argv[1]).parent.parent
def probe(p):
    n = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-count_packets",
        "-show_entries","stream=nb_read_packets","-of","csv=p=0",str(p)],
        capture_output=True,text=True).stdout.strip()
    d = subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height","-of","csv=p=0",str(p)],
        capture_output=True,text=True).stdout.strip()
    return int(n), d
cs = [c for c in m["chunks"] if c["shot_id"] == shot]
assert cs, "no chunks in shot"
bad = []
for c in cs:
    for key in ("depth_path","mask_path"):
        p = root / c[key]
        if not p.exists():
            bad.append(f"{c['chunk_id']} {key} missing {p}"); continue
        n, d = probe(p)
        if n != c["n_frames"]: bad.append(f"{c['chunk_id']} {key}: {n} != {c['n_frames']}")
        if d != f"{c['width']},{c['height']}": bad.append(f"{c['chunk_id']} {key}: {d}")
assert not bad, "\n".join(bad)
print(f"OK: {len(cs)} chunk(s) have frame-aligned depth and mask")
PYEOF

# ---- 7. mask polarity ---------------------------------------------------------
step "verify_mask_polarity.py (white = regenerate)" \
  "$PY" "$PROJ/scripts/verify_mask_polarity.py"

# ---- 8. pilot + generation ----------------------------------------------------
step "extract_pilot.py" "$PY" "$PROJ/scripts/extract_pilot.py" --seconds 5
step "run_chunks.py --pilot (real VACE generation)" \
  "$PY" "$PROJ/scripts/run_chunks.py" --pilot --limit 1 --steps "$STEPS"

step "generated chunk is valid and complete" "$PY" - "$MANIFEST" <<'PYEOF'
import json, subprocess, sys
from pathlib import Path
m = json.load(open(sys.argv[1]))
root = Path(sys.argv[1]).parent.parent
done = [c for c in m["chunks"] if c["status"] == "done"]
assert done, "no chunk completed"
c = done[0]
p = root / c["output_path"]
assert p.exists() and p.stat().st_size > 10_000, f"bad output {p}"
n = int(subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-count_packets",
    "-show_entries","stream=nb_read_packets","-of","csv=p=0",str(p)],
    capture_output=True,text=True).stdout.strip())
assert n == c["n_frames"], f"{n} frames != {c['n_frames']}"
print(f"OK: {c['chunk_id']} -> {n} frames, {p.stat().st_size/1e6:.2f} MB, "
      f"{c['duration_sec']}s, peak VRAM {c['peak_vram_mb']} MiB")
PYEOF

# ---- 9. assembly + audio ------------------------------------------------------
step "assemble.py --pilot (audio remux + A/V sync check)" \
  "$PY" "$PROJ/scripts/assemble.py" --pilot
check_file "pilot master" "$PROJ/outputs/final/pilot_master.mp4"

step "master has audio and matching duration" "$PY" - <<PYEOF
import json, os, subprocess, sys
p = "$PROJ/outputs/final/pilot_master.mp4"
if not os.path.exists(p):
    sys.exit(f"master not produced: {p}")
d = json.loads(subprocess.run(["ffprobe","-v","error","-print_format","json",
    "-show_streams","-show_format",p],capture_output=True,text=True).stdout)
v = [s for s in d["streams"] if s["codec_type"]=="video"]
a = [s for s in d["streams"] if s["codec_type"]=="audio"]
assert v, "no video stream"
assert a, "no audio stream - remux from the original failed"
dur = float(d["format"]["duration"])
assert dur > 1.0, f"suspiciously short: {dur}s"
print(f"OK: {v[0]['width']}x{v[0]['height']}, {dur:.2f}s, audio={a[0]['codec_name']}")
PYEOF

# ---- 10. comparisons ----------------------------------------------------------
step "make_comparisons.py" "$PY" "$PROJ/scripts/make_comparisons.py"

# ---- 11. every script is at least invocable -----------------------------------
echo "-----------------------------------------------------------------"
echo ">>> --help on every script"
allok=1
for f in "$PROJ"/scripts/*.py; do
  case "$(basename "$f")" in common.py|comfy_client.py) continue ;; esac
  if ! "$PY" "$f" --help >/dev/null 2>&1; then echo "    FAIL --help: $(basename "$f")"; allok=0; fi
done
for f in "$PROJ"/scripts/*.sh; do
  bash -n "$f" || { echo "    FAIL syntax: $(basename "$f")"; allok=0; }
done
if [ $allok -eq 1 ]; then echo "    PASS: all scripts invocable / syntactically valid"
  pass=$((pass+1)); LINES+=("| all scripts invocable | PASS |")
else fail=$((fail+1)); LINES+=("| all scripts invocable | **FAIL** |"); fi

# ---- 12. run_full.sh must refuse without the flag -----------------------------
echo "-----------------------------------------------------------------"
echo ">>> run_full.sh must refuse without --confirm-full-run"
"$PROJ/scripts/run_full.sh" >>"$LOG" 2>&1
rc=$?
if [ $rc -eq 3 ]; then echo "    PASS: refused (exit 3) and printed estimates"
  pass=$((pass+1)); LINES+=("| run_full.sh refuses without confirmation | PASS |")
else echo "    FAIL: expected exit 3, got $rc"
  fail=$((fail+1)); LINES+=("| run_full.sh refuses without confirmation | **FAIL** |"); fi

# ---- report -------------------------------------------------------------------
{
  echo "# Self-test report"
  echo
  echo "Generated: $(date -Iseconds)"
  echo
  echo "End-to-end validation of the pipeline on **synthetic** 240p media"
  echo "(20 s, 4:3, 24 fps, one hard scene cut at 10 s, with an audio tone),"
  echo "plus three synthetic reference stills."
  echo
  echo "| Check | Result |"
  echo "|---|---|"
  printf '%s\n' "${LINES[@]}"
  echo
  echo "**Passed: $pass — Failed: $fail**"
  echo
  echo "## What this does and does not prove"
  echo
  echo "Proven by this run:"
  echo
  echo "- ffprobe inspection, CFR normalization and square-pixel handling"
  echo "- scene-cut detection, and that no chunk crosses a cut"
  echo "- chunk lengths are all 4n+1 and dimensions are multiples of 16"
  echo "- Depth Anything V2 runs on CUDA and produces frame-aligned depth"
  echo "- SAM 2 propagates a seed through a shot and exports aligned masks"
  echo "- mask polarity: white regenerates, black is preserved (measured)"
  echo "- VACE 1.3B generates a real chunk at the configured size"
  echo "- chunks assemble, audio is remuxed from the original, A/V sync verified"
  echo "- run_full.sh refuses to start without explicit confirmation"
  echo
  echo "**Not** proven by this run, and only testable with your real footage:"
  echo
  echo "- identity matching quality (Grounding DINO + ArcFace + CLIP ReID)."
  echo "  A synthetic humanoid is not a person; this run seeds SAM 2 manually."
  echo "- restoration quality: whether VACE preserves your subject's face,"
  echo "  clothing and proportions convincingly. That is what the real pilot"
  echo "  and reports/pilot_results.md are for."
  echo
  echo "Full log: \`logs/selftest.log\`"
} > "$REPORT"

# ---- cleanup ------------------------------------------------------------------
if [ $KEEP -eq 0 ]; then
  echo "-----------------------------------------------------------------"
  echo ">>> Cleaning up synthetic media (use --keep to retain it)"
  rm -f "$SRC" "$PROJ"/inputs/references/_selftest_ref_*.png
  rm -rf "$PROJ/intermediate/normalized" "$PROJ/intermediate/depth" \
         "$PROJ/intermediate/masks" "$PROJ/intermediate/chunks" \
         "$PROJ/intermediate/shots" "$PROJ/intermediate/reference_sheets" \
         "$PROJ/intermediate/_assembly" "$PROJ/intermediate/_frames"
  rm -f "$PROJ"/outputs/restored_480p/*.mp4 "$PROJ"/outputs/final/pilot_master*.mp4 \
        "$PROJ"/outputs/comparisons/* "$PROJ"/outputs/pilots/pilot_source.mp4
  rm -f "$MANIFEST"
  mkdir -p "$PROJ"/intermediate/{normalized,shots,chunks,depth,masks,reference_sheets}
  for d in normalized shots chunks depth masks reference_sheets; do
    touch "$PROJ/intermediate/$d/.gitkeep"; done
  if [ -f "$BACKUP" ]; then mv "$BACKUP" "$MANIFEST"; echo "    restored your previous manifest"; fi
  echo "    synthetic media removed; inputs/ is clean and ready for your files"
else
  echo ">>> --keep: synthetic media and outputs retained"
fi

echo "================================================================="
echo " SELF-TEST: $pass passed, $fail failed"
echo " Report: $REPORT"
echo "================================================================="
[ $fail -eq 0 ]
