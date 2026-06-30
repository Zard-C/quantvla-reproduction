#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

CASE_LIST="${CASE_LIST:-0:15,0:16,0:17,1:15,1:16,1:17,2:15,2:16,2:17,3:15,3:16,3:17,4:15,4:16,4:17,5:15,5:16,5:17,6:15,6:16,6:17,7:15,7:16,7:17,8:15,8:16,8:17,9:15,9:16,9:17}"
TAG_PREFIX="${TAG_PREFIX:-phase30_heldout_sanity_30case_v1}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260701}"
PORT_BASE="${PORT_BASE:-5900}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"

RUN_BASELINE_VARIANT="${RUN_BASELINE_VARIANT:-1}"
RUN_SPEED_ONLY="${RUN_SPEED_ONLY:-1}"
RUN_BLOCKS0_3="${RUN_BLOCKS0_3:-1}"
RUN_WINDOW_0_120="${RUN_WINDOW_0_120:-1}"

run_baseline() {
  local tag="${TAG_PREFIX}_baseline"
  echo "=== Phase30 held-out sanity baseline: ${tag} ==="
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
  echo "=== Phase30 held-out sanity candidate: ${tag} target=${target} extra=${extra_args} eval_extra=${eval_extra_args} ==="
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
echo "RUN_BLOCKS0_3=${RUN_BLOCKS0_3}"
echo "RUN_WINDOW_0_120=${RUN_WINDOW_0_120}"

if [ "${RUN_BASELINE_VARIANT}" = "1" ]; then
  run_baseline
fi

if [ "${RUN_SPEED_ONLY}" = "1" ]; then
  run_compiled_variant \
    "${TAG_PREFIX}_speed_only_action_head_model" \
    "action_head_model" \
    "$((PORT_BASE + 1))"
fi

if [ "${RUN_BLOCKS0_3}" = "1" ]; then
  run_compiled_variant \
    "${TAG_PREFIX}_proxy_blocks0_3_eager" \
    "action_head_model_blocks_0_3_eager" \
    "$((PORT_BASE + 2))"
fi

if [ "${RUN_WINDOW_0_120}" = "1" ]; then
  run_compiled_variant \
    "${TAG_PREFIX}_window_0_120" \
    "action_head_model" \
    "$((PORT_BASE + 3))" \
    "--torch-compile-fallback-step-start 0 --torch-compile-fallback-step-end 120" \
    "--send-policy-step-key"
fi

TAG_PREFIX="${TAG_PREFIX}" \
CASE_LIST="${CASE_LIST}" \
POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
REPORT_TITLE="${REPORT_TITLE:-Phase 30: Held-Out Sanity Set}" \
OUT_MD="${OUT_MD:-docs/phase30_heldout_sanity_report_zh.md}" \
"${PYTHON_BIN}" toy_quantvla/phase30_heldout_sanity_summary.py

echo "Phase30 held-out sanity complete."
