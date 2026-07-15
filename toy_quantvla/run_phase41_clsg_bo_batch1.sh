#!/usr/bin/env bash
set -euo pipefail

export TAG_PREFIX="${TAG_PREFIX:-phase41_clsg_bo_batch1_from_phase40_15case_v1}"
export CASE_LIST="${CASE_LIST:-0:27,0:28,0:29,1:27,1:28,1:29,4:27,4:28,4:29,6:27,6:28,6:29,8:27,8:28,8:29}"
export TACTICS="${TACTICS:-window_0_18 window_0_25 window_0_10 window_2_12 window_24_29 window_18_30}"
export REPORT_TITLE="${REPORT_TITLE:-Phase 41: CLSG-BO Batch 1 from Phase40}"
export OUT_MD="${OUT_MD:-docs/phase41_clsg_bo_batch1_from_phase40_report_zh.md}"
export PORT_BASE="${PORT_BASE:-6900}"
export RECORD_VIDEO="${RECORD_VIDEO:-0}"
export SERVER_TRACE_CUDA_SYNC="${SERVER_TRACE_CUDA_SYNC:-1}"
export SERVER_LATENCY_FLUSH_EVERY="${SERVER_LATENCY_FLUSH_EVERY:-1}"
export POLICY_CLIENT_TIMEOUT_MS="${POLICY_CLIENT_TIMEOUT_MS:-600000}"
export PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"

echo "Phase41 CLSG-BO batch1"
echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "TACTICS=${TACTICS}"
echo "OUT_MD=${OUT_MD}"
echo "PORT_BASE=${PORT_BASE}"

bash toy_quantvla/run_phase36_n17_tactic_probe.sh
