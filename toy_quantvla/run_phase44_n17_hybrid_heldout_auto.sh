#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
PHASE43_SUMMARY="${PHASE43_SUMMARY:-toy_quantvla/results/phase43_n17_hybrid_bo_probe_10case_v1_summary.json}"

if [ ! -f "${PHASE43_SUMMARY}" ]; then
  echo "Missing Phase43 summary: ${PHASE43_SUMMARY}" >&2
  exit 1
fi

SELECTED_TACTICS="$("${PYTHON_BIN}" toy_quantvla/phase43_select_followup_tactics.py \
  --summary-json "${PHASE43_SUMMARY}" \
  --max-tactics "${MAX_TACTICS:-7}" \
  --min-speedup "${MIN_SPEEDUP:-1.20}" \
  --out-json "toy_quantvla/results/phase44_n17_hybrid_heldout_selection.json" \
  --out-md "docs/phase44_n17_hybrid_heldout_selection_zh.md")"

export TAG_PREFIX="${TAG_PREFIX:-phase44_n17_hybrid_heldout_15case_v1}"
export CASE_LIST="${CASE_LIST:-0:35,0:36,0:37,1:35,1:36,1:37,4:35,4:36,4:37,6:35,6:36,6:37,8:35,8:36,8:37}"
export TACTICS="${TACTICS:-${SELECTED_TACTICS}}"
export REPORT_TITLE="${REPORT_TITLE:-Phase 44: N1.7 Hybrid Held-out Validation}"
export OUT_MD="${OUT_MD:-docs/phase44_n17_hybrid_heldout_report_zh.md}"
export PORT_BASE="${PORT_BASE:-7300}"
export RECORD_VIDEO="${RECORD_VIDEO:-0}"
export SERVER_TRACE_CUDA_SYNC="${SERVER_TRACE_CUDA_SYNC:-1}"
export SERVER_LATENCY_FLUSH_EVERY="${SERVER_LATENCY_FLUSH_EVERY:-1}"
export POLICY_CLIENT_TIMEOUT_MS="${POLICY_CLIENT_TIMEOUT_MS:-600000}"

echo "Phase44 N1.7 hybrid held-out validation"
echo "PHASE43_SUMMARY=${PHASE43_SUMMARY}"
echo "SELECTED_TACTICS=${SELECTED_TACTICS}"
echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "TACTICS=${TACTICS}"
echo "OUT_MD=${OUT_MD}"
echo "PORT_BASE=${PORT_BASE}"

bash toy_quantvla/run_phase36_n17_tactic_probe.sh
