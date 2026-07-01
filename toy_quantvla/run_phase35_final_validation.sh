#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

CASE_LIST="${CASE_LIST:-0:21,0:22,0:23,1:21,1:22,1:23,2:21,2:22,2:23,3:21,3:22,3:23,4:21,4:22,4:23,5:21,5:22,5:23,6:21,6:22,6:23,7:21,7:22,7:23,8:21,8:22,8:23,9:21,9:22,9:23}"
TAG_PREFIX="${TAG_PREFIX:-phase35_final_validation_30case_v1}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260704}"
PORT_BASE="${PORT_BASE:-6200}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"

echo "Phase35 final validation wrapper"
echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"
echo "PORT_BASE=${PORT_BASE}"

TAG_PREFIX="${TAG_PREFIX}" \
CASE_LIST="${CASE_LIST}" \
POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
PORT_BASE="${PORT_BASE}" \
PYTHON_BIN="${PYTHON_BIN}" \
REPORT_TITLE="${REPORT_TITLE:-Phase 35: Final Held-Out Tactic Validation}" \
OUT_MD="${OUT_MD:-docs/phase35_final_validation_report_zh.md}" \
bash toy_quantvla/run_phase32_tactic_validation.sh

echo "Phase35 final validation complete."
