#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/quantvla-reproduction
mkdir -p /tmp/logs toy_quantvla/results

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain}"
DATA_CONFIG="${DATA_CONFIG:-examples.Libero.custom_data_config:LiberoDataConfig}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
CASE_LIST="${CASE_LIST:-4:6,4:7,4:8,4:9,4:10,6:0,6:1,6:2,6:3,6:4,8:6,8:7,8:8,8:9,8:10}"
RUN_FP16="${RUN_FP16:-1}"
RUN_FP4="${RUN_FP4:-1}"
FP16_PORT="${FP16_PORT:-5580}"
FP4_PORT="${FP4_PORT:-5581}"
TAG="${TAG:-phase9_up_proj_matched_t4_6_8_15}"
DETERMINISTIC_POLICY_SEEDS="${DETERMINISTIC_POLICY_SEEDS:-0}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260609}"

TASK4_DESC="put the white mug on the left plate and put the yellow and white mug on the right plate"
TASK6_DESC="put the white mug on the plate and put the chocolate pudding to the right of the plate"
TASK8_DESC="put both moka pots on the stove"

SEED_ARGS=()
if [ "${DETERMINISTIC_POLICY_SEEDS}" = "1" ]; then
  SEED_ARGS=(--deterministic-policy-seeds --policy-seed-base "${POLICY_SEED_BASE}")
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

run_fp16() {
  local server_log="/tmp/logs/${TAG}_fp16_server.log"
  local eval_log="/tmp/logs/${TAG}_fp16_eval.log"
  local server_pid_file="/tmp/logs/${TAG}_fp16_server.pid"
  local eval_pid_file="/tmp/logs/${TAG}_fp16_eval.pid"

  kill_if_running "${server_pid_file}"
  pkill -f "timed_fp16_inference_service.py.*--port ${FP16_PORT}" 2>/dev/null || true
  pkill -f "libero_eval_init_range.py.*--port ${FP16_PORT}" 2>/dev/null || true
  : > "${server_log}"
  : > "${eval_log}"

  env \
    NO_ALBUMENTATIONS_UPDATE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "${PYTHON_BIN}" toy_quantvla/timed_fp16_inference_service.py \
      --model-path "${MODEL_PATH}" \
      --data-config "${DATA_CONFIG}" \
      --embodiment-tag "${EMBODIMENT_TAG}" \
      --denoising-steps 8 \
      --port "${FP16_PORT}" \
      --output-json "toy_quantvla/results/${TAG}_fp16_server_prepare.json" \
      --server-latency-json "toy_quantvla/results/${TAG}_fp16_server_latency.json" \
      --server-latency-flush-every 50 \
    > "${server_log}" 2>&1 &
  echo $! > "${server_pid_file}"
  echo "FP16_SERVER_PID=$(cat "${server_pid_file}")"
  echo "FP16_SERVER_LOG=${server_log}"

  wait_for_log_line "${server_pid_file}" "${server_log}" "Starting timed FP16 server on port ${FP16_PORT}" 180

  env \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NO_ALBUMENTATIONS_UPDATE=1 \
    "${PYTHON_BIN}" toy_quantvla/libero_eval_init_range.py \
      --task-suite-name libero_10 \
      --case-list "${CASE_LIST}" \
      --headless \
      --port "${FP16_PORT}" \
      --trace-dir "/tmp/${TAG}_fp16_trace" \
      --log-file "toy_quantvla/results/${TAG}_fp16_client.log" \
      --latency-json "toy_quantvla/results/${TAG}_fp16_client_latency.json" \
      "${SEED_ARGS[@]}" \
    > "${eval_log}" 2>&1 &
  echo $! > "${eval_pid_file}"
  echo "FP16_EVAL_PID=$(cat "${eval_pid_file}")"
  echo "FP16_EVAL_LOG=${eval_log}"

  wait "$(cat "${eval_pid_file}")"
  sleep 2
  kill_if_running "${server_pid_file}"
  cp "${server_log}" "toy_quantvla/results/${TAG}_fp16_server.log"
  cp "${eval_log}" "toy_quantvla/results/${TAG}_fp16_eval.log"
}

run_fp4() {
  local server_log="/tmp/logs/${TAG}_fp4_up_proj_warmdesc_server.log"
  local eval_log="/tmp/logs/${TAG}_fp4_up_proj_warmdesc_eval.log"
  local server_pid_file="/tmp/logs/${TAG}_fp4_up_proj_warmdesc_server.pid"
  local eval_pid_file="/tmp/logs/${TAG}_fp4_up_proj_warmdesc_eval.pid"

  kill_if_running "${server_pid_file}"
  pkill -f "cutlass_fp4_inference_service.py.*--port ${FP4_PORT}" 2>/dev/null || true
  pkill -f "libero_eval_init_range.py.*--port ${FP4_PORT}" 2>/dev/null || true
  : > "${server_log}"
  : > "${eval_log}"
  : > "toy_quantvla/results/${TAG}_fp4_up_proj_warmdesc_request_trace.jsonl"

  env \
    NO_ALBUMENTATIONS_UPDATE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "${PYTHON_BIN}" toy_quantvla/cutlass_fp4_inference_service.py \
      --model-path "${MODEL_PATH}" \
      --data-config "${DATA_CONFIG}" \
      --embodiment-tag "${EMBODIMENT_TAG}" \
      --denoising-steps 8 \
      --port "${FP4_PORT}" \
      --scope llm_mlp_only \
      --name-contains up_proj \
      --max-modules 0 \
      --pack-backend triton \
      --prewarm-observations 1 \
      --prewarm-indices 115 \
      --prewarm-task-description "${TASK4_DESC}" \
      --prewarm-task-description "${TASK6_DESC}" \
      --prewarm-task-description "${TASK8_DESC}" \
      --output-json "toy_quantvla/results/${TAG}_fp4_up_proj_warmdesc_server_prepare.json" \
      --server-latency-json "toy_quantvla/results/${TAG}_fp4_up_proj_warmdesc_server_latency.json" \
      --server-latency-flush-every 50 \
      --server-request-trace-jsonl "toy_quantvla/results/${TAG}_fp4_up_proj_warmdesc_request_trace.jsonl" \
      --server-request-trace-module-deltas \
    > "${server_log}" 2>&1 &
  echo $! > "${server_pid_file}"
  echo "FP4_SERVER_PID=$(cat "${server_pid_file}")"
  echo "FP4_SERVER_LOG=${server_log}"

  wait_for_log_line "${server_pid_file}" "${server_log}" "Starting CUTLASS FP4 server on port ${FP4_PORT}" 360

  env \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NO_ALBUMENTATIONS_UPDATE=1 \
    "${PYTHON_BIN}" toy_quantvla/libero_eval_init_range.py \
      --task-suite-name libero_10 \
      --case-list "${CASE_LIST}" \
      --headless \
      --port "${FP4_PORT}" \
      --trace-dir "/tmp/${TAG}_fp4_up_proj_warmdesc_trace" \
      --log-file "toy_quantvla/results/${TAG}_fp4_up_proj_warmdesc_client.log" \
      --latency-json "toy_quantvla/results/${TAG}_fp4_up_proj_warmdesc_client_latency.json" \
      "${SEED_ARGS[@]}" \
    > "${eval_log}" 2>&1 &
  echo $! > "${eval_pid_file}"
  echo "FP4_EVAL_PID=$(cat "${eval_pid_file}")"
  echo "FP4_EVAL_LOG=${eval_log}"

  wait "$(cat "${eval_pid_file}")"
  sleep 2
  kill_if_running "${server_pid_file}"
  cp "${server_log}" "toy_quantvla/results/${TAG}_fp4_up_proj_warmdesc_server.log"
  cp "${eval_log}" "toy_quantvla/results/${TAG}_fp4_up_proj_warmdesc_eval.log"
}

echo "TAG=${TAG}"
echo "CASE_LIST=${CASE_LIST}"
echo "RUN_FP16=${RUN_FP16}"
echo "RUN_FP4=${RUN_FP4}"
echo "DETERMINISTIC_POLICY_SEEDS=${DETERMINISTIC_POLICY_SEEDS}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"

if [ "${RUN_FP16}" = "1" ]; then
  run_fp16
fi

if [ "${RUN_FP4}" = "1" ]; then
  run_fp4
fi

echo "Phase 9 matched set complete"
