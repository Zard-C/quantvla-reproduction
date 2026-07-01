#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

CASE_LIST="${CASE_LIST:-0:15,0:16,0:17,1:15,1:16,1:17,2:15,2:16,2:17,3:15,3:16,3:17,4:15,4:16,4:17,5:15,5:16,5:17,6:15,6:16,6:17,7:15,7:16,7:17,8:15,8:16,8:17,9:15,9:16,9:17}"
TAG_PREFIX="${TAG_PREFIX:-phase33_combo_backtest_phase30_30case_v1}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260701}"
PORT_BASE="${PORT_BASE:-6100}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
REFERENCE_SUMMARY="${REFERENCE_SUMMARY:-toy_quantvla/results/phase30_heldout_sanity_30case_v1_summary.json}"

echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"
echo "PORT_BASE=${PORT_BASE}"
echo "REFERENCE_SUMMARY=${REFERENCE_SUMMARY}"

TAG="${TAG_PREFIX}_combo_blocks0_3_window_0_120" \
CASE_LIST="${CASE_LIST}" \
POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
COMPILED_PORT="$((PORT_BASE + 0))" \
RUN_BASELINE=0 \
RUN_COMPILED=1 \
COMPILE_TARGET="action_head_model_blocks_0_3_eager" \
COMPILED_EXTRA_ARGS="--torch-compile-fallback-step-start 0 --torch-compile-fallback-step-end 120" \
EVAL_EXTRA_ARGS="--send-policy-step-key" \
bash toy_quantvla/run_phase13_torch_compile_matched_set.sh

TAG_PREFIX="${TAG_PREFIX}" \
CASE_LIST="${CASE_LIST}" \
POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
REFERENCE_SUMMARY="${REFERENCE_SUMMARY}" \
REPORT_TITLE="${REPORT_TITLE:-Phase 33: Combo Backtest on Phase30 Slice}" \
OUT_MD="${OUT_MD:-docs/phase33_combo_backtest_phase30_report_zh.md}" \
"${PYTHON_BIN}" toy_quantvla/phase33_combo_backtest_summary.py

echo "Phase33 combo backtest complete."
