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
CASE_LIST="${CASE_LIST:-4:9,6:8}"
COMPILED_PORT="${COMPILED_PORT:-5598}"
GRIPPER_ORACLE_PORT="${GRIPPER_ORACLE_PORT:-5599}"
TAG="${TAG:-phase15_gripper_oracle_dual_server_smoke}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260613}"
ORACLE_ACTION_KEYS="${ORACLE_ACTION_KEYS:-gripper}"
COMPILE_TARGET="${COMPILE_TARGET:-action_head_model_blocks_8_15_eager}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
COMPILE_BACKEND="${COMPILE_BACKEND:-inductor}"

TASK4_DESC="put the white mug on the left plate and put the yellow and white mug on the right plate"
TASK6_DESC="put the white mug on the plate and put the chocolate pudding to the right of the plate"
TASK8_DESC="put both moka pots on the stove"

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

start_server() {
  local port="$1"
  local label="$2"
  shift 2
  local server_log="/tmp/logs/${TAG}_${label}_server.log"
  local server_pid_file="/tmp/logs/${TAG}_${label}_server.pid"

  kill_if_running "${server_pid_file}"
  pkill -f "timed_fp16_inference_service.py.*--port ${port}" 2>/dev/null || true
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
}

run_eval() {
  local eval_log="/tmp/logs/${TAG}_eval.log"
  local eval_pid_file="/tmp/logs/${TAG}_eval.pid"
  : > "${eval_log}"
  env \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NO_ALBUMENTATIONS_UPDATE=1 \
    "${PYTHON_BIN}" toy_quantvla/libero_eval_init_range.py \
      --task-suite-name libero_10 \
      --case-list "${CASE_LIST}" \
      --headless \
      --port "${COMPILED_PORT}" \
      --gripper-oracle-port "${GRIPPER_ORACLE_PORT}" \
      --oracle-action-keys "${ORACLE_ACTION_KEYS}" \
      --trace-dir "/tmp/${TAG}_compiled_gripper_oracle_trace" \
      --log-file "toy_quantvla/results/${TAG}_compiled_gripper_oracle_client.log" \
      --latency-json "toy_quantvla/results/${TAG}_compiled_gripper_oracle_client_latency.json" \
      --deterministic-policy-seeds \
      --policy-seed-base "${POLICY_SEED_BASE}" \
    > "${eval_log}" 2>&1 &
  echo $! > "${eval_pid_file}"
  echo "EVAL_PID=$(cat "${eval_pid_file}")"
  echo "EVAL_LOG=${eval_log}"
  wait "$(cat "${eval_pid_file}")"
  cp "${eval_log}" "toy_quantvla/results/${TAG}_compiled_gripper_oracle_eval.log"
}

cleanup() {
  kill_if_running "/tmp/logs/${TAG}_compiled_server.pid"
  kill_if_running "/tmp/logs/${TAG}_gripper_oracle_server.pid"
}
trap cleanup EXIT

echo "TAG=${TAG}"
echo "CASE_LIST=${CASE_LIST}"
echo "COMPILED_PORT=${COMPILED_PORT}"
echo "GRIPPER_ORACLE_PORT=${GRIPPER_ORACLE_PORT}"
echo "ORACLE_ACTION_KEYS=${ORACLE_ACTION_KEYS}"
echo "COMPILE_TARGET=${COMPILE_TARGET}"
echo "COMPILE_MODE=${COMPILE_MODE}"
echo "COMPILE_BACKEND=${COMPILE_BACKEND}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"

start_server "${GRIPPER_ORACLE_PORT}" "gripper_oracle"
start_server "${COMPILED_PORT}" "compiled" \
  --torch-compile-target "${COMPILE_TARGET}" \
  --torch-compile-mode "${COMPILE_MODE}" \
  --torch-compile-backend "${COMPILE_BACKEND}"

wait_for_log_line "/tmp/logs/${TAG}_gripper_oracle_server.pid" \
  "/tmp/logs/${TAG}_gripper_oracle_server.log" \
  "Starting timed FP16 server on port ${GRIPPER_ORACLE_PORT}" \
  900
wait_for_log_line "/tmp/logs/${TAG}_compiled_server.pid" \
  "/tmp/logs/${TAG}_compiled_server.log" \
  "Starting timed FP16 server on port ${COMPILED_PORT}" \
  900

run_eval

echo "Phase 15 gripper oracle dual-server run complete"
