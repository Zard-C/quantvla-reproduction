#!/usr/bin/env bash
set -euo pipefail

# Phase48 switches back to the GR00T N1.5 LIBERO checkpoint and runs a small
# CLSG-TS v2 probe with the same behavior/speed anchors used in the N1.5
# multi-fold studies. It is a model-transfer check for the methodology, not a
# final deployment benchmark.

export TAG_PREFIX="${TAG_PREFIX:-phase48_n15_methodology_probe_15case_v1}"
export CASE_LIST="${CASE_LIST:-0:21,0:22,0:23,1:21,1:22,1:23,4:21,4:22,4:23,6:21,6:22,6:23,8:21,8:22,8:23}"
export POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260716}"
export PORT_BASE="${PORT_BASE:-7800}"
export PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"

export RUN_BASELINE_VARIANT="${RUN_BASELINE_VARIANT:-1}"
export RUN_SPEED_ONLY="${RUN_SPEED_ONLY:-1}"
export RUN_WINDOW_0_120="${RUN_WINDOW_0_120:-1}"
export RUN_COMBO_BLOCKS0_3_WINDOW_0_120="${RUN_COMBO_BLOCKS0_3_WINDOW_0_120:-1}"

export REPORT_TITLE="${REPORT_TITLE:-Phase48: N1.5 Methodology Transfer Probe}"
export OUT_MD="${OUT_MD:-docs/phase48_n15_methodology_probe_report_zh.md}"

echo "Phase48 N1.5 methodology transfer probe"
echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"
echo "PORT_BASE=${PORT_BASE}"

bash toy_quantvla/run_phase32_tactic_validation.sh
