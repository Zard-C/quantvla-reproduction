#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/quantvla-reproduction
mkdir -p /tmp/logs toy_quantvla/results

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain}"
DATA_CONFIG="${DATA_CONFIG:-examples.Libero.custom_data_config:LiberoDataConfig}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
CASE_LIST="${CASE_LIST:-4:6,4:7,4:8,4:9,4:10,6:0,6:1,6:2,6:3,6:4,8:6,8:7,8:8,8:9,8:10}"
FP4_PORT="${FP4_PORT:-5581}"
TAG="${TAG:-phase10_fp4_scope_matched_t4_6_8_15}"
FP4_SCOPE="${FP4_SCOPE:-dit_mlp_only}"
FP4_NAME_CONTAINS="${FP4_NAME_CONTAINS:-}"
FP4_MAX_MODULES="${FP4_MAX_MODULES:-0}"
FP4_SUFFIX="${FP4_SUFFIX:-fp4_${FP4_SCOPE}_warmdesc}"
PACK_BACKEND="${PACK_BACKEND:-triton}"
PREWARM_OBSERVATIONS="${PREWARM_OBSERVATIONS:-1}"
PREWARM_INDICES="${PREWARM_INDICES:-115}"
DETERMINISTIC_POLICY_SEEDS="${DETERMINISTIC_POLICY_SEEDS:-1}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260609}"

TASK4_DESC="put the white mug on the left plate and put the yellow and white mug on the right plate"
TASK6_DESC="put the white mug on the plate and put the chocolate pudding to the right of the plate"
TASK8_DESC="put both moka pots on the stove"

SEED_ARGS=()
if [ "${DETERMINISTIC_POLICY_SEEDS}" = "1" ]; then
  SEED_ARGS=(--deterministic-policy-seeds --policy-seed-base "${POLICY_SEED_BASE}")
fi

NAME_ARGS=()
if [ -n "${FP4_NAME_CONTAINS}" ]; then
  NAME_ARGS=(--name-contains "${FP4_NAME_CONTAINS}")
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

server_log="/tmp/logs/${TAG}_${FP4_SUFFIX}_server.log"
eval_log="/tmp/logs/${TAG}_${FP4_SUFFIX}_eval.log"
server_pid_file="/tmp/logs/${TAG}_${FP4_SUFFIX}_server.pid"
eval_pid_file="/tmp/logs/${TAG}_${FP4_SUFFIX}_eval.pid"

echo "TAG=${TAG}"
echo "CASE_LIST=${CASE_LIST}"
echo "FP4_SCOPE=${FP4_SCOPE}"
echo "FP4_NAME_CONTAINS=${FP4_NAME_CONTAINS}"
echo "FP4_MAX_MODULES=${FP4_MAX_MODULES}"
echo "FP4_SUFFIX=${FP4_SUFFIX}"
echo "PACK_BACKEND=${PACK_BACKEND}"
echo "PREWARM_OBSERVATIONS=${PREWARM_OBSERVATIONS}"
echo "PREWARM_INDICES=${PREWARM_INDICES}"
echo "DETERMINISTIC_POLICY_SEEDS=${DETERMINISTIC_POLICY_SEEDS}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"

kill_if_running "${server_pid_file}"
pkill -f "cutlass_fp4_inference_service.py.*--port ${FP4_PORT}" 2>/dev/null || true
pkill -f "libero_eval_init_range.py.*--port ${FP4_PORT}" 2>/dev/null || true
: > "${server_log}"
: > "${eval_log}"
: > "toy_quantvla/results/${TAG}_${FP4_SUFFIX}_request_trace.jsonl"

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
    --scope "${FP4_SCOPE}" \
    "${NAME_ARGS[@]}" \
    --max-modules "${FP4_MAX_MODULES}" \
    --pack-backend "${PACK_BACKEND}" \
    --prewarm-observations "${PREWARM_OBSERVATIONS}" \
    --prewarm-indices "${PREWARM_INDICES}" \
    --prewarm-task-description "${TASK4_DESC}" \
    --prewarm-task-description "${TASK6_DESC}" \
    --prewarm-task-description "${TASK8_DESC}" \
    --output-json "toy_quantvla/results/${TAG}_${FP4_SUFFIX}_server_prepare.json" \
    --server-latency-json "toy_quantvla/results/${TAG}_${FP4_SUFFIX}_server_latency.json" \
    --server-latency-flush-every 50 \
    --server-request-trace-jsonl "toy_quantvla/results/${TAG}_${FP4_SUFFIX}_request_trace.jsonl" \
    --server-request-trace-module-deltas \
  > "${server_log}" 2>&1 &
echo $! > "${server_pid_file}"
echo "FP4_SERVER_PID=$(cat "${server_pid_file}")"
echo "FP4_SERVER_LOG=${server_log}"

wait_for_log_line "${server_pid_file}" "${server_log}" "Starting CUTLASS FP4 server on port ${FP4_PORT}" 480

env \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  NO_ALBUMENTATIONS_UPDATE=1 \
  "${PYTHON_BIN}" toy_quantvla/libero_eval_init_range.py \
    --task-suite-name libero_10 \
    --case-list "${CASE_LIST}" \
    --headless \
    --port "${FP4_PORT}" \
    --trace-dir "/tmp/${TAG}_${FP4_SUFFIX}_trace" \
    --log-file "toy_quantvla/results/${TAG}_${FP4_SUFFIX}_client.log" \
    --latency-json "toy_quantvla/results/${TAG}_${FP4_SUFFIX}_client_latency.json" \
    "${SEED_ARGS[@]}" \
  > "${eval_log}" 2>&1 &
echo $! > "${eval_pid_file}"
echo "FP4_EVAL_PID=$(cat "${eval_pid_file}")"
echo "FP4_EVAL_LOG=${eval_log}"

wait "$(cat "${eval_pid_file}")"
sleep 2
kill_if_running "${server_pid_file}"
cp "${server_log}" "toy_quantvla/results/${TAG}_${FP4_SUFFIX}_server.log"
cp "${eval_log}" "toy_quantvla/results/${TAG}_${FP4_SUFFIX}_eval.log"

echo "Phase 10 FP4 scope matched set complete"
