#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

CASE_LIST="${CASE_LIST:-4:0,4:1,4:2,4:3,4:4,4:5,4:6,4:7,4:8,4:9,4:10,6:0,6:1,6:2,6:3,6:4,6:5,6:6,6:7,6:8,6:9,6:10,8:0,8:1,8:2,8:3,8:4,8:5,8:6,8:7,8:8,8:9,8:10}"
TAG_PREFIX="${TAG_PREFIX:-phase29_finer_duration_proxy_33case_v1}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260613}"
PORT_BASE="${PORT_BASE:-5800}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"

# Default pass focuses on the likely grasp-critical prefix discovered by Phase28D.
# Override WINDOWS to run a broader atomic sweep, for example:
# WINDOWS="0:80,80:160,160:240,240:320,320:500,0:120,0:180,0:220,80:240,120:280"
WINDOWS="${WINDOWS:-0:120,0:180,0:220,80:240,120:280,160:240,240:320}"

run_window() {
  local start="$1"
  local end="$2"
  local port="$3"
  local tag="${TAG_PREFIX}_window_${start}_${end}"
  echo "=== Phase29 finer duration proxy: ${tag} steps=[${start},${end}) ==="
  TAG="${tag}" \
  CASE_LIST="${CASE_LIST}" \
  POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
  COMPILED_PORT="${port}" \
  RUN_BASELINE=0 \
  RUN_COMPILED=1 \
  COMPILE_TARGET="action_head_model" \
  COMPILED_EXTRA_ARGS="--torch-compile-fallback-step-start ${start} --torch-compile-fallback-step-end ${end}" \
  EVAL_EXTRA_ARGS="--send-policy-step-key" \
  bash toy_quantvla/run_phase13_torch_compile_matched_set.sh
}

echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"
echo "PORT_BASE=${PORT_BASE}"
echo "WINDOWS=${WINDOWS}"

idx=0
IFS=',' read -r -a WINDOW_ITEMS <<< "${WINDOWS}"
for item in "${WINDOW_ITEMS[@]}"; do
  start="${item%%:*}"
  end="${item##*:}"
  if [ -z "${start}" ] || [ -z "${end}" ] || [ "${start}" = "${end}" ]; then
    echo "Bad window item: ${item}" >&2
    exit 2
  fi
  run_window "${start}" "${end}" "$((PORT_BASE + idx))"
  idx="$((idx + 1))"
done

TAG_PREFIX="${TAG_PREFIX}" \
CASE_LIST="${CASE_LIST}" \
POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
WINDOWS="${WINDOWS}" \
REPORT_TITLE="${REPORT_TITLE:-Phase 29: Finer Duration Proxy}" \
OUT_MD="${OUT_MD:-docs/phase29_finer_duration_proxy_report_zh.md}" \
"${PYTHON_BIN}" toy_quantvla/phase29_finer_duration_proxy_summary.py

echo "Phase29 complete."
