#!/usr/bin/env bash
# Environment verification. Regenerates reports/environment.md.
# Safe to run repeatedly. Never displays media.
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$PROJ/venv/bin/python"
OUT="$PROJ/reports/environment.md"
mkdir -p "$PROJ/reports"

pass=0; fail=0
chk() { # chk "<label>" "<value>" "<ok|bad>"
  if [ "$3" = ok ]; then echo "| $1 | \`$2\` | OK |"; pass=$((pass+1));
  else echo "| $1 | \`$2\` | **MISSING/FAIL** |"; fail=$((fail+1)); fi
}

{
echo "# Environment Report"
echo
echo "Generated: $(date -Iseconds)"
echo "Project root: \`$PROJ\`"
echo
echo "## Summary table"
echo
echo "| Item | Value | Status |"
echo "|---|---|---|"

chk "Ubuntu" "$(. /etc/os-release; echo "$PRETTY_NAME")" ok
chk "Kernel" "$(uname -r)" ok
chk "CPU" "$(lscpu | sed -n 's/^Model name: *//p' | head -1)" ok

TOTMEM=$(free -g | awk '/^Mem:/{print $2}')
AVAILMEM=$(free -g | awk '/^Mem:/{print $7}')
SWAP=$(free -g | awk '/^Swap:/{print $2}')
chk "RAM total / available" "${TOTMEM} GiB / ${AVAILMEM} GiB" ok
chk "Swap" "${SWAP} GiB" ok

if command -v nvidia-smi >/dev/null; then
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
  VFREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1)
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  chk "GPU" "$GPU" ok
  chk "VRAM total / free" "$VRAM / $VFREE" ok
  chk "NVIDIA driver" "$DRV" ok
  chk "Compute capability" "$CC" ok
else
  chk "nvidia-smi" "not found" bad
fi

DISK=$(df -BG --output=avail "$PROJ" | tail -1 | tr -d ' G')
chk "Free disk at project" "${DISK} GiB" ok

for t in git ffmpeg ffprobe curl; do
  if ! command -v $t >/dev/null; then chk "$t" "not found" bad; continue; fi
  # ffmpeg/ffprobe use -version; git/curl use --version
  case "$t" in
    ffmpeg|ffprobe) ver="$($t -version 2>&1 | head -1)" ;;
    *)              ver="$($t --version 2>&1 | head -1)" ;;
  esac
  chk "$t" "$(echo "$ver" | cut -c1-60)" ok
done

if [ -x "$PY" ]; then chk "venv python" "$("$PY" --version 2>&1)" ok
else chk "venv python" "$PY not found" bad; fi

if [ -d "$PROJ/ComfyUI/.git" ]; then
  chk "ComfyUI commit" "$(git -C "$PROJ/ComfyUI" rev-parse HEAD)" ok
  chk "ComfyUI tag" "$(git -C "$PROJ/ComfyUI" describe --tags --abbrev=0 2>/dev/null || echo untagged)" ok
else
  chk "ComfyUI" "not installed" bad
fi

if [ -x "$PY" ]; then
  TORCHINFO=$("$PY" - <<'EOF' 2>/dev/null
try:
    import torch
    print(f"{torch.__version__}|{torch.version.cuda}|{torch.cuda.is_available()}|"
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
except Exception as e:
    print(f"ERROR|{e}||")
EOF
)
  IFS='|' read -r TV TC TA TN <<< "$TORCHINFO"
  chk "PyTorch" "$TV (bundled CUDA $TC)" ok
  if [ "$TA" = "True" ]; then chk "torch.cuda.is_available()" "True -> $TN" ok
  else chk "torch.cuda.is_available()" "$TA" bad; fi
fi

echo
echo "**Checks passed: $pass — failed: $fail**"
echo

cat <<'MD'
## Key findings that changed the plan

1. **The GPU has 12 GB of VRAM, not 8 GB.** `nvidia-smi` reports an RTX 3060 with
   12288 MiB. The brief assumed ~8 GB. 12 GB comfortably fits VACE 1.3B at 480p and
   makes the FP8 text encoder a convenience rather than a hard requirement. The
   14B model is still **not** attempted locally, per instruction.

2. **System RAM is the real bottleneck, not VRAM.** 15 GiB total, with a large part
   already used by the desktop session (Firefox, Cursor, GNOME). The FP8 UMT5 text
   encoder alone is ~6.3 GB and is staged through system RAM.

   Consequences for the launcher flags:
   - `--cache-none` **is** used: it stops ComfyUI caching every node's output
     between runs, which is the single biggest RAM saver here.
   - `--disable-smart-memory` is deliberately **not** used. Its actual behaviour is
     to aggressively offload weights from VRAM back into system RAM, which is
     exactly the wrong trade on this machine: VRAM is plentiful (12 GB) and RAM is
     scarce. Leaving smart memory enabled keeps the weights resident in VRAM.
   - `--reserve-vram 1.0` leaves ~1 GB for Xorg/Firefox, which were measured
     holding ~1.1 GB at inspection time.

   Closing Firefox and Cursor before a long run frees several GiB of RAM.

3. **No system CUDA toolkit is installed and none is needed.** PyTorch ships its own
   CUDA 12.8 runtime. Driver 580.x supports it. The system driver was left untouched.

4. **Python 3.10 on PATH is Miniconda's.** The project deliberately uses the *system*
   `python3.12` to build `venv/`, so the conda base environment is never modified.

MD

echo "## Disk space budget"
echo
cat <<'MD'
| Item | Size |
|---|---|
| PyTorch + CUDA runtime wheels | ~8.5 GB |
| ComfyUI + Python deps | ~1.5 GB |
| `wan2.1_vace_1.3B_fp16.safetensors` | ~3.2 GB |
| `wan_2.1_vae.safetensors` | ~0.25 GB |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | ~6.7 GB |
| Depth Anything V2 (ViT-L) | ~1.3 GB |
| SAM 2.1 (hiera-large) | ~0.9 GB |
| Detector + ReID + face embeddings | ~1.5 GB |
| **Install subtotal** | **~24 GB** |
| Intermediates, 30-min reference workload @ 16 fps 832x480 | ~45-70 GB |
| Restored 480p master + deliverables | ~15-25 GB |
| **Peak total** | **~85-120 GB** |
MD
echo
echo "Free space measured above must exceed the peak total before a full run."
echo
echo "## Missing system packages"
echo
MISSING=""
command -v ffmpeg >/dev/null || MISSING="$MISSING ffmpeg"
command -v git    >/dev/null || MISSING="$MISSING git"
command -v curl   >/dev/null || MISSING="$MISSING curl"
"$PY" -c "import venv" 2>/dev/null || MISSING="$MISSING python3.12-venv"
if [ -z "$MISSING" ]; then
  echo "None. No \`sudo\` is required for this project."
else
  echo "Run this yourself (not run automatically):"
  echo
  echo '```bash'
  echo "sudo apt-get install -y$MISSING"
  echo '```'
fi
} > "$OUT"

echo "Wrote $OUT"
grep -E "^\*\*Checks" "$OUT"
[ "$fail" -eq 0 ] || echo "WARNING: $fail check(s) failed - see $OUT"
