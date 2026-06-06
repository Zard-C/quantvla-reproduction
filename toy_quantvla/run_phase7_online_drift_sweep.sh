#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/quantvla-reproduction}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

CASE_LIST="${CASE_LIST:-8:7,8:9,4:10,4:11,0:3,0:7,6:1,9:9}"
SCOPES="${SCOPES:-dit_mlp_only llm_mlp_only llm_mlp_dit_mlp llm_dit_mlp}"
MODES="${MODES:-none ohb atm_ohb}"
MAX_POLICY_STEPS="${MAX_POLICY_STEPS:-80}"
NUM_CALIBRATION_OBSERVATIONS="${NUM_CALIBRATION_OBSERVATIONS:-3}"

TRACE_ROOT="${TRACE_ROOT:-/tmp/quantvla_phase7_online_drift_${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-/tmp/logs/phase7_online_drift_${RUN_ID}}"

mkdir -p "${TRACE_ROOT}" "${LOG_ROOT}"

echo "RUN_ID=${RUN_ID}"
echo "CASE_LIST=${CASE_LIST}"
echo "SCOPES=${SCOPES}"
echo "MODES=${MODES}"
echo "MAX_POLICY_STEPS=${MAX_POLICY_STEPS}"
echo "TRACE_ROOT=${TRACE_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"

cd "${REPO_ROOT}"
for scope in ${SCOPES}; do
  for mode in ${MODES}; do
    tag="${scope}_${mode}_${MAX_POLICY_STEPS}_${RUN_ID}"
    echo "=== $(date --iso-8601=seconds) START scope=${scope} mode=${mode} ==="
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "${PYTHON_BIN}" toy_quantvla/phase7_online_drift.py \
      --case-list "${CASE_LIST}" \
      --scope "${scope}" \
      --mode "${mode}" \
      --num-calibration-observations "${NUM_CALIBRATION_OBSERVATIONS}" \
      --max-policy-steps "${MAX_POLICY_STEPS}" \
      --headless \
      --no-video \
      --trace-dir "${TRACE_ROOT}/${scope}/${mode}" \
      --log-file "${LOG_ROOT}/${tag}.log" \
      --output-json "toy_quantvla/results/phase7_online_drift_${tag}.json" \
      --output-md "docs/phase7_online_drift_${tag}.md"
    echo "=== $(date --iso-8601=seconds) DONE scope=${scope} mode=${mode} ==="
  done
done

echo "=== $(date --iso-8601=seconds) ALL_DONE ==="
echo "Logs: ${LOG_ROOT}"
echo "Traces: ${TRACE_ROOT}"
