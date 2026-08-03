#!/usr/bin/env bash
# Start ComfyUI with a conservative profile tuned for this machine
# (RTX 3060 12 GB VRAM / 15 GiB system RAM).
#
# Flags were chosen by reading ComfyUI/comfy/cli_args.py in the installed revision,
# not guessed. Rationale for each is inline below.
#
# NEVER opens a browser or displays media: --disable-auto-launch is mandatory here.
#
# Usage:
#   scripts/start_comfyui.sh              # foreground
#   scripts/start_comfyui.sh --daemon     # background, writes logs/comfyui.log + .pid
#   scripts/start_comfyui.sh --stop       # stop a daemonised instance
#   scripts/start_comfyui.sh --status     # is the API up?
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$PROJ/venv/bin/python"
PORT="${COMFY_PORT:-8188}"
HOST="127.0.0.1"
LOG="$PROJ/logs/comfyui.log"
PIDFILE="$PROJ/logs/comfyui.pid"

ARGS=(
  "$PROJ/ComfyUI/main.py"
  --listen "$HOST"
  --port "$PORT"
  # Never pop open a browser window. Required by the project's no-display rule.
  --disable-auto-launch
  # Latent previews decode extra VAE frames every step; pure overhead for batch work.
  --preview-method none
  # Biggest RAM saver on this box: do not retain every node's output between runs.
  --cache-none
  # Leave ~1 GB of VRAM for Xorg/Firefox, measured at ~1.1 GB during inspection.
  --reserve-vram 1.0
  # sm_86 has fast SDPA/flash kernels in torch 2.9; xformers is not installed.
  --use-pytorch-cross-attention
  # Keep model/output dirs inside the project.
  --output-directory "$PROJ/ComfyUI/output"
  --input-directory  "$PROJ/ComfyUI/input"
)

api_up() { curl -s -m 3 -o /dev/null -w "%{http_code}" "http://$HOST:$PORT/system_stats" 2>/dev/null | grep -q 200; }

case "${1:-}" in
  --status)
    if api_up; then echo "ComfyUI API is UP on http://$HOST:$PORT"; exit 0
    else echo "ComfyUI API is DOWN on http://$HOST:$PORT"; exit 1; fi ;;
  --stop)
    if [ -f "$PIDFILE" ]; then
      pid=$(cat "$PIDFILE")
      kill "$pid" 2>/dev/null && echo "Stopped ComfyUI (pid $pid)" || echo "No live process for pid $pid"
      rm -f "$PIDFILE"
    else echo "No pidfile at $PIDFILE"; fi
    exit 0 ;;
esac

if api_up; then
  echo "ComfyUI is already running on http://$HOST:$PORT — nothing to do."
  exit 0
fi

mkdir -p "$PROJ/logs" "$PROJ/ComfyUI/input" "$PROJ/ComfyUI/output"

if [ "${1:-}" = "--daemon" ]; then
  echo "Starting ComfyUI (daemon) -> $LOG"
  nohup "$PY" "${ARGS[@]}" > "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  echo "pid $(cat "$PIDFILE")"
  for i in $(seq 1 90); do
    if api_up; then echo "API is up after ${i}s: http://$HOST:$PORT"; exit 0; fi
    if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "ERROR: ComfyUI exited during startup. Last 30 log lines:"; tail -30 "$LOG"; exit 1
    fi
    sleep 1
  done
  echo "ERROR: API did not come up within 90s. Last 30 log lines:"; tail -30 "$LOG"; exit 1
else
  echo "Starting ComfyUI (foreground) on http://$HOST:$PORT"
  exec "$PY" "${ARGS[@]}"
fi
