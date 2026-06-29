#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

CASE_LIST="${CASE_LIST:-4:0,4:1,4:2,4:3,4:4,4:5,4:6,4:7,4:8,4:9,4:10,6:0,6:1,6:2,6:3,6:4,6:5,6:6,6:7,6:8,6:9,6:10,8:0,8:1,8:2,8:3,8:4,8:5,8:6,8:7,8:8,8:9,8:10}"
TAG_PREFIX="${TAG_PREFIX:-phase28B_proxy_guided_33case_v1}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260613}"
PORT_BASE="${PORT_BASE:-5680}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"

RUN_BASELINE_VARIANT="${RUN_BASELINE_VARIANT:-1}"
RUN_SPEED_ONLY="${RUN_SPEED_ONLY:-1}"
RUN_PROXY_BLOCK0="${RUN_PROXY_BLOCK0:-0}"
RUN_PROXY_BLOCKS8_15="${RUN_PROXY_BLOCKS8_15:-1}"
RUN_RANDOM_BLOCK1="${RUN_RANDOM_BLOCK1:-1}"

export CASE_LIST
export TAG_PREFIX
export POLICY_SEED_BASE
export PORT_BASE
export PYTHON_BIN
export RUN_BASELINE_VARIANT
export RUN_SPEED_ONLY
export RUN_PROXY_BLOCK0
export RUN_PROXY_BLOCKS8_15
export RUN_RANDOM_BLOCK1
export PHASE_LABEL="Phase28B proxy-guided 33-case"

bash toy_quantvla/run_phase28_proxy_guided_mixed_precision.sh

REPORT_TITLE="${REPORT_TITLE:-Phase 28B: Proxy-Guided 33-Case Mixed Precision}" \
OUT_MD="${OUT_MD:-docs/phase28b_proxy_guided_33case_report_zh.md}" \
"${PYTHON_BIN}" toy_quantvla/phase28_proxy_guided_summary.py

echo "Phase28B summary written to ${OUT_MD:-docs/phase28b_proxy_guided_33case_report_zh.md}"
