#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/quantvla-reproduction}"
ISAAC_ROOT="${ISAAC_ROOT:-/root/autodl-tmp/Isaac-GR00T-n1.5}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain}"
PORT="${PORT:-5555}"
CASE_LIST="${CASE_LIST:-8:7,8:9,4:10,0:3,6:1,9:9,8:0}"
MODES="${MODES:-fp16 none ohb atm_ohb}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-/tmp/logs/phase5_trace_${RUN_ID}}"
TRACE_ROOT="${TRACE_ROOT:-/tmp/quantvla_trace_cases_${RUN_ID}}"

mkdir -p "${LOG_ROOT}" "${TRACE_ROOT}"

wait_for_port() {
  local port="$1"
  local attempts="${2:-120}"
  "${PYTHON_BIN}" - "${port}" "${attempts}" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
attempts = int(sys.argv[2])
for _ in range(attempts):
    sock = socket.socket()
    sock.settimeout(1.0)
    try:
        sock.connect(("127.0.0.1", port))
    except OSError:
        time.sleep(1)
    else:
        sock.close()
        raise SystemExit(0)
raise SystemExit(1)
PY
}

run_eval() {
  local mode="$1"
  cd "${REPO_ROOT}"
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "${PYTHON_BIN}" toy_quantvla/libero_eval_init_range.py \
    --task-suite-name libero_10 \
    --init-start 0 \
    --num-inits 1 \
    --case-list "${CASE_LIST}" \
    --trace-dir "${TRACE_ROOT}/${mode}" \
    --headless \
    --port "${PORT}" \
    --log-file "${LOG_ROOT}/eval_${mode}.log"
}

start_server() {
  local mode="$1"
  local log_file="${LOG_ROOT}/server_${mode}.log"
  if [[ "${mode}" == "fp16" ]]; then
    cd "${ISAAC_ROOT}"
    "${PYTHON_BIN}" scripts/inference_service.py \
      --model_path "${MODEL_PATH}" \
      --server \
      --data_config examples.Libero.custom_data_config:LiberoDataConfig \
      --denoising-steps 8 \
      --port "${PORT}" \
      --embodiment-tag new_embodiment \
      >"${log_file}" 2>&1 &
  else
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" toy_quantvla/quantized_inference_service.py \
      --config llm_dit_mlp \
      --mode "${mode}" \
      --port "${PORT}" \
      --output-json "toy_quantvla/results/phase5_trace_prepare_${mode}_${RUN_ID}.json" \
      >"${log_file}" 2>&1 &
  fi
  echo "$!"
}

echo "RUN_ID=${RUN_ID}"
echo "CASE_LIST=${CASE_LIST}"
echo "MODES=${MODES}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "TRACE_ROOT=${TRACE_ROOT}"

for mode in ${MODES}; do
  echo "=== $(date --iso-8601=seconds) START mode=${mode} ==="
  server_pid="$(start_server "${mode}")"
  echo "server_pid=${server_pid}"
  if ! wait_for_port "${PORT}" 180; then
    echo "Server did not open port ${PORT}; see ${LOG_ROOT}/server_${mode}.log" >&2
    kill "${server_pid}" 2>/dev/null || true
    exit 1
  fi
  echo "=== $(date --iso-8601=seconds) EVAL mode=${mode} ==="
  run_eval "${mode}"
  echo "=== $(date --iso-8601=seconds) EVAL_DONE mode=${mode} ==="
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  sleep 5
done

echo "=== $(date --iso-8601=seconds) ALL_DONE ==="
echo "Logs: ${LOG_ROOT}"
echo "Traces: ${TRACE_ROOT}"
