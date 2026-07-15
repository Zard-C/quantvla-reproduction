#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
PHASE44_SUMMARY="${PHASE44_SUMMARY:-toy_quantvla/results/phase44_n17_hybrid_heldout_15case_v1_summary.json}"

if [ ! -f "${PHASE44_SUMMARY}" ]; then
  echo "Missing Phase44 summary: ${PHASE44_SUMMARY}" >&2
  exit 1
fi

SELECTED_TACTICS="$("${PYTHON_BIN}" toy_quantvla/phase43_select_followup_tactics.py \
  --summary-json "${PHASE44_SUMMARY}" \
  --max-tactics "${MAX_TACTICS:-6}" \
  --min-speedup "${MIN_SPEEDUP:-1.15}" \
  --out-json "toy_quantvla/results/phase45_n17_hybrid_alltask_stress_selection.json" \
  --out-md "docs/phase45_n17_hybrid_alltask_stress_selection_zh.md")"

export TAG_PREFIX="${TAG_PREFIX:-phase45_n17_hybrid_alltask_stress_20case_v1}"
export CASE_LIST="${CASE_LIST:-0:38,0:39,1:38,1:39,2:38,2:39,3:38,3:39,4:38,4:39,5:38,5:39,6:38,6:39,7:38,7:39,8:38,8:39,9:38,9:39}"
export TACTICS="${TACTICS:-${SELECTED_TACTICS}}"
export REPORT_TITLE="${REPORT_TITLE:-Phase 45: N1.7 Hybrid All-task Stress Check}"
export OUT_MD="${OUT_MD:-docs/phase45_n17_hybrid_alltask_stress_report_zh.md}"
export PORT_BASE="${PORT_BASE:-7400}"
export RECORD_VIDEO="${RECORD_VIDEO:-0}"
export SERVER_TRACE_CUDA_SYNC="${SERVER_TRACE_CUDA_SYNC:-1}"
export SERVER_LATENCY_FLUSH_EVERY="${SERVER_LATENCY_FLUSH_EVERY:-1}"
export POLICY_CLIENT_TIMEOUT_MS="${POLICY_CLIENT_TIMEOUT_MS:-600000}"

echo "Phase45 N1.7 hybrid all-task stress check"
echo "PHASE44_SUMMARY=${PHASE44_SUMMARY}"
echo "SELECTED_TACTICS=${SELECTED_TACTICS}"
echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "TACTICS=${TACTICS}"
echo "OUT_MD=${OUT_MD}"
echo "PORT_BASE=${PORT_BASE}"

bash toy_quantvla/run_phase36_n17_tactic_probe.sh
