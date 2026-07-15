#!/usr/bin/env bash
set -euo pipefail

export TAG_PREFIX="${TAG_PREFIX:-phase43_n17_hybrid_bo_probe_10case_v1}"
export CASE_LIST="${CASE_LIST:-0:33,0:34,1:33,1:34,4:33,4:34,6:33,6:34,8:33,8:34}"
export TACTICS="${TACTICS:-fp16 speed_only window_0_20 window_2_12 blocks0_3 blocks0_3_window_0_20 blocks0_3_window_2_12 blocks8_15_window_2_12 blocks16_31_window_2_12}"
export REPORT_TITLE="${REPORT_TITLE:-Phase 43: N1.7 Hybrid CLSG-BO Probe}"
export OUT_MD="${OUT_MD:-docs/phase43_n17_hybrid_bo_probe_report_zh.md}"
export PORT_BASE="${PORT_BASE:-7200}"
export RECORD_VIDEO="${RECORD_VIDEO:-0}"
export SERVER_TRACE_CUDA_SYNC="${SERVER_TRACE_CUDA_SYNC:-1}"
export SERVER_LATENCY_FLUSH_EVERY="${SERVER_LATENCY_FLUSH_EVERY:-1}"
export POLICY_CLIENT_TIMEOUT_MS="${POLICY_CLIENT_TIMEOUT_MS:-600000}"
export PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"

echo "Phase43 N1.7 hybrid CLSG-BO probe"
echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "TACTICS=${TACTICS}"
echo "OUT_MD=${OUT_MD}"
echo "PORT_BASE=${PORT_BASE}"

bash toy_quantvla/run_phase36_n17_tactic_probe.sh
