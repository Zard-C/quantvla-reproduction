#!/usr/bin/env bash
set -euo pipefail

export TAG_PREFIX="${TAG_PREFIX:-phase42_n17_bo_heldout_validation_15case_v1}"
export CASE_LIST="${CASE_LIST:-0:30,0:31,0:32,1:30,1:31,1:32,4:30,4:31,4:32,6:30,6:31,6:32,8:30,8:31,8:32}"
export TACTICS="${TACTICS:-fp16 speed_only window_0_20 window_2_12 window_4_9 window_6_11}"
export REPORT_TITLE="${REPORT_TITLE:-Phase 42: N1.7 CLSG-BO Held-out Validation}"
export OUT_MD="${OUT_MD:-docs/phase42_n17_bo_heldout_validation_report_zh.md}"
export PORT_BASE="${PORT_BASE:-7000}"
export RECORD_VIDEO="${RECORD_VIDEO:-0}"
export SERVER_TRACE_CUDA_SYNC="${SERVER_TRACE_CUDA_SYNC:-1}"
export SERVER_LATENCY_FLUSH_EVERY="${SERVER_LATENCY_FLUSH_EVERY:-1}"
export POLICY_CLIENT_TIMEOUT_MS="${POLICY_CLIENT_TIMEOUT_MS:-600000}"
export PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"

echo "Phase42 N1.7 CLSG-BO held-out validation"
echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "TACTICS=${TACTICS}"
echo "OUT_MD=${OUT_MD}"
echo "PORT_BASE=${PORT_BASE}"

bash toy_quantvla/run_phase36_n17_tactic_probe.sh
