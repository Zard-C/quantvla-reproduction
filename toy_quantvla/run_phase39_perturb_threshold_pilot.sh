#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p /tmp/logs toy_quantvla/results

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
ISAAC_ROOT="${ISAAC_ROOT:-/root/autodl-tmp/Isaac-GR00T-n1.5}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain}"
DATA_CONFIG="${DATA_CONFIG:-examples.Libero.custom_data_config:LiberoDataConfig}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
DENOISING_STEPS="${DENOISING_STEPS:-8}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260715}"
PORT="${PORT:-6200}"
TAG_PREFIX="${TAG_PREFIX:-phase39_threshold_pilot_reduced_v1}"
RUN_MODE="${RUN_MODE:-reduced}"
CASE_LIST="${CASE_LIST:-4:9,6:8}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_PERTURB="${RUN_PERTURB:-1}"
RUN_REAL_DRIFT="${RUN_REAL_DRIFT:-0}"
REAL_VECTOR_JSON="${REAL_VECTOR_JSON:-toy_quantvla/results/phase39_real_drift_directions.json}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
RESUME="${RESUME:-1}"

if [ "${RUN_MODE}" = "full" ]; then
  DIRECTIONS="${DIRECTIONS:-y,z,yaw,continuous_6d}"
  WINDOWS="${WINDOWS:-full,early,mid,late}"
  EPSILONS="${EPSILONS:-0.003,0.006,0.01,0.02,0.03,0.05,0.08}"
else
  DIRECTIONS="${DIRECTIONS:-y,z,yaw}"
  WINDOWS="${WINDOWS:-full,early,late}"
  EPSILONS="${EPSILONS:-0.006,0.01,0.03,0.05}"
fi

TRACE_ROOT="${TRACE_ROOT:-toy_quantvla/results/${TAG_PREFIX}_traces}"
MANIFEST_JSONL="toy_quantvla/results/${TAG_PREFIX}_manifest.jsonl"

TASK4_DESC="put the white mug on the left plate and put the yellow and white mug on the right plate"
TASK6_DESC="put the white mug on the plate and put the chocolate pudding to the right of the plate"
TASK8_DESC="put both moka pots on the stove"

SERVER_PID_FILE="/tmp/logs/${TAG_PREFIX}_server.pid"
SERVER_LOG="/tmp/logs/${TAG_PREFIX}_server.log"

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

cleanup() {
  kill_if_running "${SERVER_PID_FILE}"
}
trap cleanup EXIT

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
      tail -160 "${log_file}" || true
      return 1
    fi
    sleep 1
  done
  tail -160 "${log_file}" || true
  return 1
}

epsilon_tag() {
  local value="$1"
  echo "${value}" | sed 's/-/m/g; s/\./p/g'
}

direction_keys() {
  local direction="$1"
  if [ "${direction}" = "continuous_6d" ]; then
    echo "continuous"
  else
    echo "${direction}"
  fi
}

window_bounds() {
  local task_id="$1"
  local window="$2"
  case "${task_id}:${window}" in
    4:early) echo "0 75" ;;
    4:mid) echo "75 150" ;;
    4:late) echo "150 225" ;;
    6:early) echo "0 200" ;;
    6:mid) echo "200 450" ;;
    6:late) echo "450 700" ;;
    *:full) echo "" ;;
    *)
      echo "Unsupported Phase39 window ${window} for task ${task_id}" >&2
      exit 2
      ;;
  esac
}

append_manifest() {
  local tag="$1"
  local kind="$2"
  local case_list="$3"
  local case_item="$4"
  local task_id="$5"
  local init_index="$6"
  local direction="$7"
  local action_keys="$8"
  local window="$9"
  local step_start="${10}"
  local step_end="${11}"
  local epsilon="${12}"
  local trace_dir="${13}"
  printf '{"tag":"%s","kind":"%s","case_list":"%s","case":"%s","task_id":"%s","init_index":"%s","direction":"%s","action_keys":"%s","window":"%s","step_start":"%s","step_end":"%s","epsilon":"%s","trace_dir":"%s"}\n' \
    "${tag}" "${kind}" "${case_list}" "${case_item}" "${task_id}" "${init_index}" \
    "${direction}" "${action_keys}" "${window}" "${step_start}" "${step_end}" \
    "${epsilon}" "${trace_dir}" >> "${MANIFEST_JSONL}"
}

run_eval() {
  local tag="$1"
  local case_list="$2"
  shift 2
  local trace_dir="${TRACE_ROOT}/${tag}"
  local eval_log="/tmp/logs/${tag}_eval.log"
  local client_json="toy_quantvla/results/${tag}_client_latency.json"

  if [ "${RESUME}" = "1" ] && [ -f "${client_json}" ]; then
    echo "SKIP existing ${client_json}"
    return 0
  fi

  echo "=== Phase39 eval: ${tag} cases=${case_list} ==="
  : > "${eval_log}"
  env \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NO_ALBUMENTATIONS_UPDATE=1 \
    "${PYTHON_BIN}" toy_quantvla/libero_eval_init_range.py \
      --task-suite-name "${TASK_SUITE}" \
      --case-list "${case_list}" \
      --headless \
      --port "${PORT}" \
      --trace-dir "${trace_dir}" \
      --log-file "toy_quantvla/results/${tag}_client.log" \
      --latency-json "${client_json}" \
      --deterministic-policy-seeds \
      --policy-seed-base "${POLICY_SEED_BASE}" \
      "$@" \
    > "${eval_log}" 2>&1
  cp "${eval_log}" "toy_quantvla/results/${tag}_eval.log"
}

start_server() {
  kill_if_running "${SERVER_PID_FILE}"
  pkill -f "timed_fp16_inference_service.py.*--port ${PORT}" 2>/dev/null || true
  pkill -f "libero_eval_init_range.py.*--port ${PORT}" 2>/dev/null || true
  : > "${SERVER_LOG}"
  : > "toy_quantvla/results/${TAG_PREFIX}_request_trace.jsonl"

  env \
    NO_ALBUMENTATIONS_UPDATE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "${PYTHON_BIN}" toy_quantvla/timed_fp16_inference_service.py \
      --isaac-root "${ISAAC_ROOT}" \
      --model-path "${MODEL_PATH}" \
      --data-config "${DATA_CONFIG}" \
      --embodiment-tag "${EMBODIMENT_TAG}" \
      --denoising-steps "${DENOISING_STEPS}" \
      --port "${PORT}" \
      --prewarm-observations 1 \
      --prewarm-indices 115 \
      --prewarm-task-description "${TASK4_DESC}" \
      --prewarm-task-description "${TASK6_DESC}" \
      --prewarm-task-description "${TASK8_DESC}" \
      --output-json "toy_quantvla/results/${TAG_PREFIX}_server_prepare.json" \
      --server-latency-json "toy_quantvla/results/${TAG_PREFIX}_server_latency.json" \
      --server-latency-flush-every 50 \
      --server-request-trace-jsonl "toy_quantvla/results/${TAG_PREFIX}_request_trace.jsonl" \
    > "${SERVER_LOG}" 2>&1 &
  echo $! > "${SERVER_PID_FILE}"
  echo "SERVER_PID=$(cat "${SERVER_PID_FILE}")"
  echo "SERVER_LOG=${SERVER_LOG}"
  wait_for_log_line "${SERVER_PID_FILE}" "${SERVER_LOG}" "Starting timed FP16 server on port ${PORT}" 900
}

echo "TAG_PREFIX=${TAG_PREFIX}"
echo "RUN_MODE=${RUN_MODE}"
echo "CASE_LIST=${CASE_LIST}"
echo "DIRECTIONS=${DIRECTIONS}"
echo "WINDOWS=${WINDOWS}"
echo "EPSILONS=${EPSILONS}"
echo "RUN_REAL_DRIFT=${RUN_REAL_DRIFT}"
echo "REAL_VECTOR_JSON=${REAL_VECTOR_JSON}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"
echo "PORT=${PORT}"
echo "TRACE_ROOT=${TRACE_ROOT}"
echo "MANIFEST_JSONL=${MANIFEST_JSONL}"

: > "${MANIFEST_JSONL}"
start_server

if [ "${RUN_BASELINE}" = "1" ]; then
  baseline_tag="${TAG_PREFIX}_baseline"
  append_manifest "${baseline_tag}" "baseline" "${CASE_LIST}" "" "" "" "" "" "baseline" "" "" "" "${TRACE_ROOT}/${baseline_tag}"
  run_eval "${baseline_tag}" "${CASE_LIST}"
fi

if [ "${RUN_PERTURB}" = "1" ]; then
  IFS=',' read -r -a CASE_ITEMS <<< "${CASE_LIST}"
  IFS=',' read -r -a DIRECTION_ITEMS <<< "${DIRECTIONS}"
  IFS=',' read -r -a WINDOW_ITEMS <<< "${WINDOWS}"
  IFS=',' read -r -a EPSILON_ITEMS <<< "${EPSILONS}"

  for case_item in "${CASE_ITEMS[@]}"; do
    task_id="${case_item%%:*}"
    init_index="${case_item##*:}"
    for direction in "${DIRECTION_ITEMS[@]}"; do
      action_keys="$(direction_keys "${direction}")"
      for window in "${WINDOW_ITEMS[@]}"; do
        bounds="$(window_bounds "${task_id}" "${window}")"
        step_start=""
        step_end=""
        window_args=()
        if [ -n "${bounds}" ]; then
          read -r step_start step_end <<< "${bounds}"
          window_args=(--action-perturb-step-start "${step_start}" --action-perturb-step-end "${step_end}")
        fi
        for epsilon in "${EPSILON_ITEMS[@]}"; do
          eps_tag="$(epsilon_tag "${epsilon}")"
          tag="${TAG_PREFIX}_case_t${task_id}_i${init_index}_dir_${direction}_win_${window}_eps_${eps_tag}"
          append_manifest "${tag}" "perturb" "${case_item}" "${case_item}" "${task_id}" "${init_index}" \
            "${direction}" "${action_keys}" "${window}" "${step_start}" "${step_end}" "${epsilon}" "${TRACE_ROOT}/${tag}"
          run_eval \
            "${tag}" \
            "${case_item}" \
            --action-perturb-keys "${action_keys}" \
            --action-perturb-amplitude "${epsilon}" \
            "${window_args[@]}"
        done
      done
    done
  done
fi

if [ "${RUN_REAL_DRIFT}" = "1" ]; then
  if [ ! -f "${REAL_VECTOR_JSON}" ]; then
    echo "RUN_REAL_DRIFT=1 but REAL_VECTOR_JSON does not exist: ${REAL_VECTOR_JSON}" >&2
    exit 2
  fi
  IFS=',' read -r -a EPSILON_ITEMS <<< "${EPSILONS}"
  while IFS=$'\t' read -r case_item task_id init_index direction window step_start step_end vector; do
    [ -n "${case_item}" ] || continue
    window_args=()
    if [ "${step_start}" != "-" ] && [ "${step_end}" != "-" ]; then
      window_args=(--action-perturb-step-start "${step_start}" --action-perturb-step-end "${step_end}")
    fi
    for epsilon in "${EPSILON_ITEMS[@]}"; do
      eps_tag="$(epsilon_tag "${epsilon}")"
      tag="${TAG_PREFIX}_case_t${task_id}_i${init_index}_dir_${direction}_win_${window}_eps_${eps_tag}"
      append_manifest "${tag}" "perturb" "${case_item}" "${case_item}" "${task_id}" "${init_index}" \
        "${direction}" "vector" "${window}" "${step_start/-/}" "${step_end/-/}" "${epsilon}" "${TRACE_ROOT}/${tag}"
      run_eval \
        "${tag}" \
        "${case_item}" \
        --action-perturb-vector "${vector}" \
        --action-perturb-amplitude "${epsilon}" \
        "${window_args[@]}"
    done
  done < <("${PYTHON_BIN}" toy_quantvla/phase39_real_drift_directions.py \
    --input-json "${REAL_VECTOR_JSON}" \
    --emit-runner-tsv \
    --case-list "${CASE_LIST}" \
    --windows "${WINDOWS}")
fi

cleanup
trap - EXIT

cp "${SERVER_LOG}" "toy_quantvla/results/${TAG_PREFIX}_server.log"

if [ "${RUN_SUMMARY}" = "1" ]; then
  TAG_PREFIX="${TAG_PREFIX}" \
  CASE_LIST="${CASE_LIST}" \
  DIRECTIONS="${DIRECTIONS}" \
  WINDOWS="${WINDOWS}" \
  EPSILONS="${EPSILONS}" \
  POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
  OUT_MD="${OUT_MD:-docs/phase39_closed_loop_perturbation_budget_pilot_zh.md}" \
  "${PYTHON_BIN}" toy_quantvla/phase39_perturb_threshold_summary.py
fi

echo "Phase39 perturbation threshold pilot complete."
