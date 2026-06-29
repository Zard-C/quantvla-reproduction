#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

CASE_LIST="${CASE_LIST:-4:0,4:1,4:2,4:3,4:4,4:5,4:6,4:7,4:8,4:9,4:10,6:0,6:1,6:2,6:3,6:4,6:5,6:6,6:7,6:8,6:9,6:10,8:0,8:1,8:2,8:3,8:4,8:5,8:6,8:7,8:8,8:9,8:10}"
TAG_PREFIX="${TAG_PREFIX:-phase28C_proxy_guided_33case_v1}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260613}"
PORT_BASE="${PORT_BASE:-5720}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"

run_compiled_variant() {
  local tag="$1"
  local target="$2"
  local port="$3"
  local extra_args="${4:-}"
  local eval_extra_args="${5:-}"
  echo "=== Phase28C candidate: ${tag} target=${target} extra=${extra_args} eval_extra=${eval_extra_args} ==="
  TAG="${tag}"   CASE_LIST="${CASE_LIST}"   POLICY_SEED_BASE="${POLICY_SEED_BASE}"   COMPILED_PORT="${port}"   RUN_BASELINE=0   RUN_COMPILED=1   COMPILE_TARGET="${target}"   COMPILED_EXTRA_ARGS="${extra_args}"   bash toy_quantvla/run_phase13_torch_compile_matched_set.sh
}

echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"
echo "PORT_BASE=${PORT_BASE}"

run_compiled_variant "${TAG_PREFIX}_proxy_block0_eager" "action_head_model_blocks_0_0_eager" "$((PORT_BASE + 0))"
run_compiled_variant "${TAG_PREFIX}_proxy_block0_blocks8_15_eager" "action_head_model_blocks_0_0_8_15_eager" "$((PORT_BASE + 1))"
run_compiled_variant "${TAG_PREFIX}_proxy_blocks0_3_eager" "action_head_model_blocks_0_3_eager" "$((PORT_BASE + 2))"
run_compiled_variant "${TAG_PREFIX}_duration_window_eager_120_320" "action_head_model" "$((PORT_BASE + 3))" "--torch-compile-fallback-step-start 120 --torch-compile-fallback-step-end 320" "--send-policy-step-key"

TAG_PREFIX="${TAG_PREFIX}" CASE_LIST="${CASE_LIST}" POLICY_SEED_BASE="${POLICY_SEED_BASE}" REPORT_TITLE="${REPORT_TITLE:-Phase 28C: Proxy-Guided Candidate Search}" OUT_MD="${OUT_MD:-docs/phase28c_proxy_guided_33case_report_zh.md}" "${PYTHON_BIN}" toy_quantvla/phase28c_proxy_guided_summary.py

echo "Phase28C complete."
