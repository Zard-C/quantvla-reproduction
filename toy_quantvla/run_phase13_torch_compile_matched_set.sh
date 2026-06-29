#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p /tmp/logs toy_quantvla/results

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain}"
DATA_CONFIG="${DATA_CONFIG:-examples.Libero.custom_data_config:LiberoDataConfig}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
CASE_LIST="${CASE_LIST:-6:1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_COMPILED="${RUN_COMPILED:-1}"
BASELINE_PORT="${BASELINE_PORT:-5590}"
COMPILED_PORT="${COMPILED_PORT:-5591}"
TAG="${TAG:-phase13_torch_compile_action_head_model_smoke}"
DETERMINISTIC_POLICY_SEEDS="${DETERMINISTIC_POLICY_SEEDS:-1}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260613}"
COMPILE_TARGET="${COMPILE_TARGET:-action_head_model}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
COMPILE_BACKEND="${COMPILE_BACKEND:-inductor}"
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"
COMPILE_DYNAMIC="${COMPILE_DYNAMIC:-}"
COMPILE_CUDAGRAPH_MARK_STEP="${COMPILE_CUDAGRAPH_MARK_STEP:-0}"
COMPILED_EXTRA_ARGS="${COMPILED_EXTRA_ARGS:-}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"

TASK4_DESC="put the white mug on the left plate and put the yellow and white mug on the right plate"
TASK6_DESC="put the white mug on the plate and put the chocolate pudding to the right of the plate"
TASK8_DESC="put both moka pots on the stove"

SEED_ARGS=()
if [ "${DETERMINISTIC_POLICY_SEEDS}" = "1" ]; then
  SEED_ARGS=(--deterministic-policy-seeds --policy-seed-base "${POLICY_SEED_BASE}")
fi

COMPILE_ARGS=(
  --torch-compile-target "${COMPILE_TARGET}"
  --torch-compile-mode "${COMPILE_MODE}"
  --torch-compile-backend "${COMPILE_BACKEND}"
)
if [ "${COMPILE_FULLGRAPH}" = "1" ]; then
  COMPILE_ARGS+=(--torch-compile-fullgraph)
fi
if [ -n "${COMPILE_DYNAMIC}" ]; then
  COMPILE_ARGS+=(--torch-compile-dynamic "${COMPILE_DYNAMIC}")
fi
if [ "${COMPILE_CUDAGRAPH_MARK_STEP}" = "1" ]; then
  COMPILE_ARGS+=(--torch-compile-cudagraph-mark-step)
fi
COMPILED_EXTRA_ARGS_ARRAY=()
if [ -n "${COMPILED_EXTRA_ARGS}" ]; then
  read -r -a COMPILED_EXTRA_ARGS_ARRAY <<< "${COMPILED_EXTRA_ARGS}"
fi
EVAL_EXTRA_ARGS_ARRAY=()
if [ -n "${EVAL_EXTRA_ARGS}" ]; then
  read -r -a EVAL_EXTRA_ARGS_ARRAY <<< "${EVAL_EXTRA_ARGS}"
fi

kill_if_running() {
  local pid_file="$1"
  if [ -f "${pid_file}" ]; then
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  fi
}

wait_for_log_line() {
  local pid_file="$1"
  local log_file="$2"
  local pattern="$3"
  local limit="$4"
  for _ in $(seq 1 "${limit}"); do
    if grep -q "${pattern}" "${log_file}"; then
      return 0
    fi
    if ! kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
      tail -160 "${log_file}"
      return 1
    fi
    sleep 1
  done
  tail -160 "${log_file}"
  return 1
}

run_eval() {
  local port="$1"
  local label="$2"
  local eval_log="/tmp/logs/${TAG}_${label}_eval.log"
  local eval_pid_file="/tmp/logs/${TAG}_${label}_eval.pid"
  : > "${eval_log}"
  env \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NO_ALBUMENTATIONS_UPDATE=1 \
    "${PYTHON_BIN}" toy_quantvla/libero_eval_init_range.py \
      --task-suite-name libero_10 \
      --case-list "${CASE_LIST}" \
      --headless \
      --port "${port}" \
      --trace-dir "/tmp/${TAG}_${label}_trace" \
      --log-file "toy_quantvla/results/${TAG}_${label}_client.log" \
      --latency-json "toy_quantvla/results/${TAG}_${label}_client_latency.json" \
      "${SEED_ARGS[@]}" \
      "${EVAL_EXTRA_ARGS_ARRAY[@]}" \
    > "${eval_log}" 2>&1 &
  echo $! > "${eval_pid_file}"
  echo "${label}_EVAL_PID=$(cat "${eval_pid_file}")"
  echo "${label}_EVAL_LOG=${eval_log}"
  wait "$(cat "${eval_pid_file}")"
  cp "${eval_log}" "toy_quantvla/results/${TAG}_${label}_eval.log"
}

run_server_case() {
  local port="$1"
  local label="$2"
  shift 2
  local server_log="/tmp/logs/${TAG}_${label}_server.log"
  local server_pid_file="/tmp/logs/${TAG}_${label}_server.pid"

  kill_if_running "${server_pid_file}"
  pkill -f "timed_fp16_inference_service.py.*--port ${port}" 2>/dev/null || true
  pkill -f "libero_eval_init_range.py.*--port ${port}" 2>/dev/null || true
  : > "${server_log}"
  : > "toy_quantvla/results/${TAG}_${label}_request_trace.jsonl"

  env \
    NO_ALBUMENTATIONS_UPDATE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "${PYTHON_BIN}" toy_quantvla/timed_fp16_inference_service.py \
      --model-path "${MODEL_PATH}" \
      --data-config "${DATA_CONFIG}" \
      --embodiment-tag "${EMBODIMENT_TAG}" \
      --denoising-steps 8 \
      --port "${port}" \
      --prewarm-observations 1 \
      --prewarm-indices 115 \
      --prewarm-task-description "${TASK4_DESC}" \
      --prewarm-task-description "${TASK6_DESC}" \
      --prewarm-task-description "${TASK8_DESC}" \
      --output-json "toy_quantvla/results/${TAG}_${label}_server_prepare.json" \
      --server-latency-json "toy_quantvla/results/${TAG}_${label}_server_latency.json" \
      --server-latency-flush-every 50 \
      --server-request-trace-jsonl "toy_quantvla/results/${TAG}_${label}_request_trace.jsonl" \
      "$@" \
    > "${server_log}" 2>&1 &
  echo $! > "${server_pid_file}"
  echo "${label}_SERVER_PID=$(cat "${server_pid_file}")"
  echo "${label}_SERVER_LOG=${server_log}"

  wait_for_log_line "${server_pid_file}" "${server_log}" "Starting timed FP16 server on port ${port}" 900
  run_eval "${port}" "${label}"
  sleep 2
  kill_if_running "${server_pid_file}"
  cp "${server_log}" "toy_quantvla/results/${TAG}_${label}_server.log"
}

echo "TAG=${TAG}"
echo "CASE_LIST=${CASE_LIST}"
echo "RUN_BASELINE=${RUN_BASELINE}"
echo "RUN_COMPILED=${RUN_COMPILED}"
echo "COMPILE_TARGET=${COMPILE_TARGET}"
echo "COMPILE_MODE=${COMPILE_MODE}"
echo "COMPILE_BACKEND=${COMPILE_BACKEND}"
echo "COMPILE_FULLGRAPH=${COMPILE_FULLGRAPH}"
echo "COMPILE_DYNAMIC=${COMPILE_DYNAMIC}"
echo "COMPILE_CUDAGRAPH_MARK_STEP=${COMPILE_CUDAGRAPH_MARK_STEP}"
echo "COMPILED_EXTRA_ARGS=${COMPILED_EXTRA_ARGS}"
echo "EVAL_EXTRA_ARGS=${EVAL_EXTRA_ARGS}"
echo "DETERMINISTIC_POLICY_SEEDS=${DETERMINISTIC_POLICY_SEEDS}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"

if [ "${RUN_BASELINE}" = "1" ]; then
  run_server_case "${BASELINE_PORT}" "baseline"
fi

if [ "${RUN_COMPILED}" = "1" ]; then
  run_server_case "${COMPILED_PORT}" "compiled" "${COMPILE_ARGS[@]}" "${COMPILED_EXTRA_ARGS_ARRAY[@]}"
fi

echo "Phase 13 torch.compile matched set complete"
