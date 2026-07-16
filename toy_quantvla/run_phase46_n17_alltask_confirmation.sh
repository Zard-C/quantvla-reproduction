#!/usr/bin/env bash
set -euo pipefail

export TAG_PREFIX="${TAG_PREFIX:-phase46_n17_alltask_confirmation_30case_v1}"
export CASE_LIST="${CASE_LIST:-0:40,0:41,0:42,1:40,1:41,1:42,2:40,2:41,2:42,3:40,3:41,3:42,4:40,4:41,4:42,5:40,5:41,5:42,6:40,6:41,6:42,7:40,7:41,7:42,8:40,8:41,8:42,9:40,9:41,9:42}"
export TACTICS="${TACTICS:-fp16 speed_only window_0_20 window_2_12 blocks0_3_window_2_12}"
export REPORT_TITLE="${REPORT_TITLE:-Phase 46: N1.7 All-task Confirmation Fold}"
export OUT_MD="${OUT_MD:-docs/phase46_n17_alltask_confirmation_report_zh.md}"
export PORT_BASE="${PORT_BASE:-7500}"
export RECORD_VIDEO="${RECORD_VIDEO:-0}"
export SERVER_TRACE_CUDA_SYNC="${SERVER_TRACE_CUDA_SYNC:-1}"
export SERVER_LATENCY_FLUSH_EVERY="${SERVER_LATENCY_FLUSH_EVERY:-1}"
export POLICY_CLIENT_TIMEOUT_MS="${POLICY_CLIENT_TIMEOUT_MS:-600000}"
export PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"

echo "Phase46 N1.7 all-task confirmation fold"
echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "TACTICS=${TACTICS}"
echo "OUT_MD=${OUT_MD}"
echo "PORT_BASE=${PORT_BASE}"

bash toy_quantvla/run_phase36_n17_tactic_probe.sh
