#!/usr/bin/env bash
set -euo pipefail

run_startup_probe() {
  local label="$1"
  shift
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "GOAIDO_STARTUP_GATE_FAILED: ${label} rc=${rc}" >&2
    echo "Holding failed container for 90s so RunPod container logs remain inspectable." >&2
    sleep 90
    exit "$rc"
  fi
}

if [[ "${SKIP_GPU_RUNTIME_PROBE:-0}" != "1" ]]; then
  run_startup_probe "gpu_runtime" python /opt/trellis-worker/verify-gpu-runtime.py
fi

cd /opt/ComfyUI
python main.py --listen 127.0.0.1 --port 8188 &
comfy_pid=$!
trap 'kill "$comfy_pid" 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  if curl --silent --fail http://127.0.0.1:8188/system_stats >/dev/null; then
    run_startup_probe "comfy_workflow" python /opt/trellis-worker/verify-comfy-workflow.py
    exec python /opt/trellis-worker/handler.py
  fi
  if ! kill -0 "$comfy_pid" 2>/dev/null; then
    echo "GOAIDO_STARTUP_GATE_FAILED: comfyui exited before readiness" >&2
    sleep 90
    exit 1
  fi
  sleep 2
done
echo "GOAIDO_STARTUP_GATE_FAILED: comfyui readiness timed out" >&2
sleep 90
exit 1
