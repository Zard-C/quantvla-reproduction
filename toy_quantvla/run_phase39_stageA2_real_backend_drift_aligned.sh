#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p /tmp/logs toy_quantvla/results docs

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
ISAAC_ROOT="${ISAAC_ROOT:-/root/autodl-tmp/Isaac-GR00T-n1.5}"
COMPAT_STUBS="${COMPAT_STUBS:-toy_quantvla/compat_stubs}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
DENOISING_STEPS="${DENOISING_STEPS:-8}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
CASE_LIST="${CASE_LIST:-4:9,6:8}"
WINDOWS="${WINDOWS:-full,early}"
EPSILONS="${EPSILONS:-0.00001,0.00002,0.00003}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260613}"
BASE_SEED="${BASE_SEED:-${POLICY_SEED_BASE}}"
COMPILE_TARGET="${COMPILE_TARGET:-action_head_model}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
COMPILE_BACKEND="${COMPILE_BACKEND:-inductor}"
COMPILE_DYNAMIC="${COMPILE_DYNAMIC:-}"
COMPILE_FALLBACK_STEP_START="${COMPILE_FALLBACK_STEP_START:-}"
COMPILE_FALLBACK_STEP_END="${COMPILE_FALLBACK_STEP_END:-}"
MAX_POLICY_STEPS="${MAX_POLICY_STEPS:-0}"
PORT="${PORT:-6204}"
TAG_PREFIX="${TAG_PREFIX:-phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1}"
RUN_CAPTURE="${RUN_CAPTURE:-1}"
RUN_VECTOR="${RUN_VECTOR:-1}"
RUN_SWEEP="${RUN_SWEEP:-1}"

DRIFT_JSON="toy_quantvla/results/${TAG_PREFIX}_sameobs_online_drift.json"
DRIFT_MD="docs/${TAG_PREFIX}_sameobs_online_drift.md"
DRIFT_TRACE_DIR="toy_quantvla/results/${TAG_PREFIX}_sameobs_trace"
DRIFT_LOG="/tmp/logs/${TAG_PREFIX}_sameobs_online_drift.log"
VECTOR_JSON="toy_quantvla/results/${TAG_PREFIX}_sameobs_real_drift_directions.json"
VECTOR_MD="docs/${TAG_PREFIX}_sameobs_real_drift_directions_zh.md"
SWEEP_TAG="${TAG_PREFIX}_threshold"
SWEEP_MD="docs/${TAG_PREFIX}_threshold_zh.md"

echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "WINDOWS=${WINDOWS}"
echo "EPSILONS=${EPSILONS}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"
echo "COMPILE_TARGET=${COMPILE_TARGET}"
echo "COMPILE_MODE=${COMPILE_MODE}"
echo "COMPILE_BACKEND=${COMPILE_BACKEND}"
echo "COMPILE_FALLBACK_STEP_START=${COMPILE_FALLBACK_STEP_START}"
echo "COMPILE_FALLBACK_STEP_END=${COMPILE_FALLBACK_STEP_END}"
echo "MAX_POLICY_STEPS=${MAX_POLICY_STEPS}"
echo "PORT=${PORT}"

if [ "${RUN_CAPTURE}" = "1" ]; then
  echo "=== Stage A2 capture: same-observation reference-vs-tactic drift ==="
  compile_dynamic_args=()
  if [ -n "${COMPILE_DYNAMIC}" ]; then
    compile_dynamic_args=(--torch-compile-dynamic "${COMPILE_DYNAMIC}")
  fi
  compile_fallback_args=()
  if [ -n "${COMPILE_FALLBACK_STEP_START}" ] || [ -n "${COMPILE_FALLBACK_STEP_END}" ]; then
    if [ -z "${COMPILE_FALLBACK_STEP_START}" ] || [ -z "${COMPILE_FALLBACK_STEP_END}" ]; then
      echo "Both COMPILE_FALLBACK_STEP_START and COMPILE_FALLBACK_STEP_END are required" >&2
      exit 2
    fi
    compile_fallback_args=(
      --torch-compile-fallback-step-start "${COMPILE_FALLBACK_STEP_START}"
      --torch-compile-fallback-step-end "${COMPILE_FALLBACK_STEP_END}"
    )
  fi
  env \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NO_ALBUMENTATIONS_UPDATE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "${PYTHON_BIN}" toy_quantvla/phase13_torch_compile_online_drift.py \
      --isaac-root "${ISAAC_ROOT}" \
      --compat-stubs "${COMPAT_STUBS}" \
      --model-path "${MODEL_PATH}" \
      --task-suite-name "${TASK_SUITE}" \
      --case-list "${CASE_LIST}" \
      --embodiment-tag "${EMBODIMENT_TAG}" \
      --denoising-steps "${DENOISING_STEPS}" \
      --base-seed "${BASE_SEED}" \
      --torch-compile-target "${COMPILE_TARGET}" \
      --torch-compile-mode "${COMPILE_MODE}" \
      --torch-compile-backend "${COMPILE_BACKEND}" \
      "${compile_dynamic_args[@]}" \
      "${compile_fallback_args[@]}" \
      --max-policy-steps "${MAX_POLICY_STEPS}" \
      --headless \
      --no-video \
      --trace-dir "${DRIFT_TRACE_DIR}" \
      --log-file "${DRIFT_LOG}" \
      --output-json "${DRIFT_JSON}" \
      --output-md "${DRIFT_MD}"
fi

if [ "${RUN_VECTOR}" = "1" ]; then
  echo "=== Stage A2 vector: estimate same-observation real drift directions ==="
  "${PYTHON_BIN}" toy_quantvla/phase39_same_observation_drift_directions.py \
    --online-drift-json "${DRIFT_JSON}" \
    --variant-name speed_only \
    --case-list "${CASE_LIST}" \
    --windows "${WINDOWS}" \
    --output-json "${VECTOR_JSON}" \
    --output-md "${VECTOR_MD}"
fi

if [ "${RUN_SWEEP}" = "1" ]; then
  echo "=== Stage A2 sweep: closed-loop threshold along same-observation real drift ==="
  TAG_PREFIX="${SWEEP_TAG}" \
  RUN_MODE=reduced \
  RUN_BASELINE=1 \
  RUN_PERTURB=0 \
  RUN_REAL_DRIFT=1 \
  REAL_VECTOR_JSON="${VECTOR_JSON}" \
  CASE_LIST="${CASE_LIST}" \
  DIRECTIONS=real_speed_only_sameobs_mean \
  WINDOWS="${WINDOWS}" \
  EPSILONS="${EPSILONS}" \
  PORT="${PORT}" \
  POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  ISAAC_ROOT="${ISAAC_ROOT}" \
  COMPAT_STUBS="${COMPAT_STUBS}" \
  MODEL_PATH="${MODEL_PATH}" \
  EMBODIMENT_TAG="${EMBODIMENT_TAG}" \
  DENOISING_STEPS="${DENOISING_STEPS}" \
  OUT_MD="${SWEEP_MD}" \
  bash toy_quantvla/run_phase39_perturb_threshold_pilot.sh
fi

echo "Stage A2 real-backend-drift-aligned pilot complete."
