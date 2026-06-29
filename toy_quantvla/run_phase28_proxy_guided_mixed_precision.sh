#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

CASE_LIST="${CASE_LIST:-4:6,4:7,4:8,4:9,4:10,6:0,6:1,6:2,6:3,6:4,8:6,8:7,8:8,8:9,8:10}"
TAG_PREFIX="${TAG_PREFIX:-phase28A_proxy_guided_15case_v1}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260613}"
PORT_BASE="${PORT_BASE:-5620}"
PHASE_LABEL="${PHASE_LABEL:-Phase28 proxy-guided mixed precision}"

RUN_BASELINE_VARIANT="${RUN_BASELINE_VARIANT:-1}"
RUN_SPEED_ONLY="${RUN_SPEED_ONLY:-1}"
RUN_PROXY_BLOCK0="${RUN_PROXY_BLOCK0:-1}"
RUN_PROXY_BLOCKS8_15="${RUN_PROXY_BLOCKS8_15:-1}"
RUN_RANDOM_BLOCK1="${RUN_RANDOM_BLOCK1:-1}"

run_baseline() {
  local tag="${TAG_PREFIX}_baseline"
  echo "=== ${PHASE_LABEL} baseline: ${tag} ==="
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
  echo "=== ${PHASE_LABEL} compiled variant: ${tag} target=${target} ==="
  TAG="${tag}" \
  CASE_LIST="${CASE_LIST}" \
  POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
  COMPILED_PORT="${port}" \
  RUN_BASELINE=0 \
  RUN_COMPILED=1 \
  COMPILE_TARGET="${target}" \
  bash toy_quantvla/run_phase13_torch_compile_matched_set.sh
}

echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"
echo "PORT_BASE=${PORT_BASE}"
echo "RUN_BASELINE_VARIANT=${RUN_BASELINE_VARIANT}"
echo "RUN_SPEED_ONLY=${RUN_SPEED_ONLY}"
echo "RUN_PROXY_BLOCK0=${RUN_PROXY_BLOCK0}"
echo "RUN_PROXY_BLOCKS8_15=${RUN_PROXY_BLOCKS8_15}"
echo "RUN_RANDOM_BLOCK1=${RUN_RANDOM_BLOCK1}"

if [ "${RUN_BASELINE_VARIANT}" = "1" ]; then
  run_baseline
fi

if [ "${RUN_SPEED_ONLY}" = "1" ]; then
  run_compiled_variant \
    "${TAG_PREFIX}_speed_only_action_head_model" \
    "action_head_model" \
    "$((PORT_BASE + 1))"
fi

if [ "${RUN_PROXY_BLOCK0}" = "1" ]; then
  run_compiled_variant \
    "${TAG_PREFIX}_proxy_block0_eager" \
    "action_head_model_blocks_0_0_eager" \
    "$((PORT_BASE + 2))"
fi

if [ "${RUN_PROXY_BLOCKS8_15}" = "1" ]; then
  run_compiled_variant \
    "${TAG_PREFIX}_proxy_blocks8_15_eager" \
    "action_head_model_blocks_8_15_eager" \
    "$((PORT_BASE + 3))"
fi

if [ "${RUN_RANDOM_BLOCK1}" = "1" ]; then
  run_compiled_variant \
    "${TAG_PREFIX}_random_block1_eager" \
    "action_head_model_blocks_1_1_eager" \
    "$((PORT_BASE + 4))"
fi

echo "${PHASE_LABEL} run complete."
