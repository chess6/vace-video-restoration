#!/usr/bin/env bash
# Enforce CLAUDE.md rule 1: nothing in this project may display media on screen.
#
# Greps the project's own code for viewer/player calls and verifies the installed
# OpenCV genuinely has no GUI support compiled in. Exits non-zero on any hit, so
# it can be wired into CI or a pre-commit hook.
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$PROJ/venv/bin/python"
rc=0

# Only our own code; not ComfyUI, not site-packages.
FILES=$(find "$PROJ/scripts" -type f \( -name '*.py' -o -name '*.sh' \))

echo "=== 1. Forbidden viewer/player calls in scripts/ ==="
# Patterns are anchored so that mere mentions in comments/docs do not trip it:
# each requires the call syntax that would actually open something.
PATTERNS=(
  'cv2\.imshow *\('
  'cv2\.namedWindow *\('
  'cv2\.startWindowThread *\('
  '\.show *\( *\)'
  'pyplot\.show *\('
  'plt\.show *\('
  'webbrowser\.open'
  'IPython\.display'
  '\bxdg-open\b'
  '\bgio +open\b'
  '\bgnome-open\b'
  '\bffplay\b'
  '\bmpv\b'
  '\bvlc\b'
  '\bmplayer\b'
  '\btotem\b'
  '\beog\b'
  '\bfeh\b'
)
hits=0
for f in $FILES; do
  case "$(basename "$f")" in check_no_display.sh) continue ;; esac
  # Strip whole-line comments and anything inside backticks before matching, so
  # that documentation ABOUT the rule (which necessarily names the forbidden
  # tools) is not mistaken for a call TO them. Line numbers are preserved by
  # blanking rather than deleting.
  code=$(sed -e 's/^[[:space:]]*#.*$//' -e 's/`[^`]*`//g' "$f")
  for p in "${PATTERNS[@]}"; do
    out=$(printf '%s\n' "$code" | grep -nPH --label="$f" "$p" 2>/dev/null || true)
    if [ -n "$out" ]; then
      echo "  VIOLATION [$p]"; echo "$out" | sed 's/^/    /'; hits=$((hits+1))
    fi
  done
done
if [ $hits -eq 0 ]; then echo "  OK: no viewer or player calls found"; else rc=1; fi

echo
echo "=== 2. ComfyUI launcher must disable browser auto-launch ==="
if grep -q -- '--disable-auto-launch' "$PROJ/scripts/start_comfyui.sh"; then
  echo "  OK: start_comfyui.sh passes --disable-auto-launch"
else
  echo "  VIOLATION: start_comfyui.sh does not pass --disable-auto-launch"; rc=1
fi
if grep -qE -- '(^|[^-])--auto-launch' "$PROJ/scripts/start_comfyui.sh"; then
  echo "  VIOLATION: start_comfyui.sh passes --auto-launch"; rc=1
fi

echo
echo "=== 3. OpenCV must be the headless distribution ==="
if [ -x "$PY" ]; then
  "$PY" - <<'PYEOF'
import sys
import importlib.metadata as md

installed = {d.metadata["Name"].lower() for d in md.distributions()
             if d.metadata.get("Name")}
gui_pkg = "opencv-python" in installed or "opencv-contrib-python" in installed
headless = "opencv-python-headless" in installed or \
           "opencv-contrib-python-headless" in installed

if gui_pkg:
    print("  VIOLATION: the GUI OpenCV distribution is installed.")
    print("             pip uninstall -y opencv-python opencv-contrib-python")
    print("             pip install opencv-python-headless")
    sys.exit(1)
if not headless:
    print("  WARNING: no OpenCV distribution found"); sys.exit(0)

# The headless wheels still EXPOSE imshow as a stub in OpenCV 5.x; it raises at
# call time rather than being absent. So presence of the attribute is not the
# test - the installed distribution is. Confirm the stub really does refuse.
import cv2, numpy as np
try:
    cv2.imshow("x", np.zeros((2, 2, 3), np.uint8))
    cv2.destroyAllWindows()
    print("  VIOLATION: cv2.imshow actually opened a window - not a headless build")
    sys.exit(1)
except cv2.error as e:
    msg = str(e).replace("\n", " ")[:90]
    print(f"  OK: cv2 {cv2.__version__} headless; imshow refuses at runtime ({msg}...)")
except Exception as e:
    print(f"  OK: cv2 {cv2.__version__} headless; imshow unusable ({type(e).__name__})")
PYEOF
  [ $? -ne 0 ] && rc=1
else
  echo "  SKIP: venv python not found"
fi

echo
echo "=== 4. matplotlib must not use an interactive backend ==="
if [ -x "$PY" ]; then
  "$PY" - <<'PYEOF'
import sys
try:
    import matplotlib
except Exception:
    print("  SKIP: matplotlib not installed"); sys.exit(0)
b = matplotlib.get_backend().lower()
if b in ("agg", "pdf", "svg", "ps", "cairo", "template"):
    print(f"  OK: matplotlib backend is non-interactive ({b})")
else:
    print(f"  WARNING: matplotlib backend is {b}. Set MPLBACKEND=Agg.")
PYEOF
fi

echo
if [ $rc -eq 0 ]; then echo "NO-DISPLAY CHECK: PASSED"
else echo "NO-DISPLAY CHECK: FAILED - see violations above"; fi
exit $rc
