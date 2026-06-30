#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

CASE_LIST="${CASE_LIST:-0:18,0:19,0:20,1:18,1:19,1:20,2:18,2:19,2:20,3:18,3:19,3:20,4:18,4:19,4:20,5:18,5:19,5:20,6:18,6:19,6:20,7:18,7:19,7:20,8:18,8:19,8:20,9:18,9:19,9:20}"
TAG_PREFIX="${TAG_PREFIX:-phase32_tactic_validation_30case_v1}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260702}"
PORT_BASE="${PORT_BASE:-6000}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"

RUN_BASELINE_VARIANT="${RUN_BASELINE_VARIANT:-1}"
RUN_SPEED_ONLY="${RUN_SPEED_ONLY:-1}"
RUN_WINDOW_0_120="${RUN_WINDOW_0_120:-1}"
RUN_COMBO_BLOCKS0_3_WINDOW_0_120="${RUN_COMBO_BLOCKS0_3_WINDOW_0_120:-1}"

run_baseline() {
  local tag="${TAG_PREFIX}_baseline"
  echo "=== Phase32 tactic validation baseline: ${tag} ==="
  TAG="${tag}" \
  CASE_LIST="${CASE_LIST}" \
  POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
  BASELINE_PORT="$((PORT_BASE + 0))" \
  RUN_BASELINE=1 \
  RUN_COMPILED=0 \
  bash toy_quantvla/run_phase13_torch_compile_matched_set.sh
}

run_compiled_variant() {
  local tag="$1"
  local target="$2"
  local port="$3"
  local extra_args="${4:-}"
  local eval_extra_args="${5:-}"
  echo "=== Phase32 tactic validation candidate: ${tag} target=${target} extra=${extra_args} eval_extra=${eval_extra_args} ==="
  TAG="${tag}" \
  CASE_LIST="${CASE_LIST}" \
  POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
  COMPILED_PORT="${port}" \
  RUN_BASELINE=0 \
  RUN_COMPILED=1 \
  COMPILE_TARGET="${target}" \
  COMPILED_EXTRA_ARGS="${extra_args}" \
  EVAL_EXTRA_ARGS="${eval_extra_args}" \
  bash toy_quantvla/run_phase13_torch_compile_matched_set.sh
}

echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"
echo "PORT_BASE=${PORT_BASE}"
echo "RUN_BASELINE_VARIANT=${RUN_BASELINE_VARIANT}"
echo "RUN_SPEED_ONLY=${RUN_SPEED_ONLY}"
echo "RUN_WINDOW_0_120=${RUN_WINDOW_0_120}"
echo "RUN_COMBO_BLOCKS0_3_WINDOW_0_120=${RUN_COMBO_BLOCKS0_3_WINDOW_0_120}"

if [ "${RUN_BASELINE_VARIANT}" = "1" ]; then
  run_baseline
fi

if [ "${RUN_SPEED_ONLY}" = "1" ]; then
  run_compiled_variant \
    "${TAG_PREFIX}_speed_only_action_head_model" \
    "action_head_model" \
    "$((PORT_BASE + 1))"
fi

if [ "${RUN_WINDOW_0_120}" = "1" ]; then
  run_compiled_variant \
    "${TAG_PREFIX}_window_0_120" \
    "action_head_model" \
    "$((PORT_BASE + 2))" \
    "--torch-compile-fallback-step-start 0 --torch-compile-fallback-step-end 120" \
    "--send-policy-step-key"
fi

if [ "${RUN_COMBO_BLOCKS0_3_WINDOW_0_120}" = "1" ]; then
  run_compiled_variant \
    "${TAG_PREFIX}_combo_blocks0_3_window_0_120" \
    "action_head_model_blocks_0_3_eager" \
    "$((PORT_BASE + 3))" \
    "--torch-compile-fallback-step-start 0 --torch-compile-fallback-step-end 120" \
    "--send-policy-step-key"
fi

TAG_PREFIX="${TAG_PREFIX}" \
CASE_LIST="${CASE_LIST}" \
POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
REPORT_TITLE="${REPORT_TITLE:-Phase 32: Held-Out Tactic Validation v2}" \
OUT_MD="${OUT_MD:-docs/phase32_tactic_validation_report_zh.md}" \
"${PYTHON_BIN}" toy_quantvla/phase32_tactic_validation_summary.py

echo "Phase32 tactic validation complete."
