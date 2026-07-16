#!/usr/bin/env bash
set -euo pipefail

# Phase49 extends the N1.5 Phase48 transfer probe to an all-task confirmation
# fold. It keeps the same tactic set and evaluates whether the Phase48
# repair/no-regression signal survives broader task coverage.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export PYTHONPATH="${REPO_ROOT}/toy_quantvla/compat_stubs:${PYTHONPATH:-}"

export TAG_PREFIX="${TAG_PREFIX:-phase49_n15_alltask_confirmation_30case_v1}"
export CASE_LIST="${CASE_LIST:-0:24,0:25,0:26,1:24,1:25,1:26,2:24,2:25,2:26,3:24,3:25,3:26,4:24,4:25,4:26,5:24,5:25,5:26,6:24,6:25,6:26,7:24,7:25,7:26,8:24,8:25,8:26,9:24,9:25,9:26}"
export POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260717}"
export PORT_BASE="${PORT_BASE:-7900}"
export PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"

export RUN_BASELINE_VARIANT="${RUN_BASELINE_VARIANT:-1}"
export RUN_SPEED_ONLY="${RUN_SPEED_ONLY:-1}"
export RUN_WINDOW_0_120="${RUN_WINDOW_0_120:-1}"
export RUN_COMBO_BLOCKS0_3_WINDOW_0_120="${RUN_COMBO_BLOCKS0_3_WINDOW_0_120:-1}"

export REPORT_TITLE="${REPORT_TITLE:-Phase49: N1.5 All-task Confirmation Fold}"
export OUT_MD="${OUT_MD:-docs/phase49_n15_alltask_confirmation_report_zh.md}"

echo "Phase49 N1.5 all-task confirmation fold"
echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"
echo "PORT_BASE=${PORT_BASE}"

bash toy_quantvla/run_phase32_tactic_validation.sh
