#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/quantvla-reproduction

CASES="${CASES:-4:6 6:3 8:8 8:10 6:4}"
FP16_PORT="${FP16_PORT:-5582}"
FP4_PORT="${FP4_PORT:-5583}"

echo "Isolated recheck cases: ${CASES}"
echo "FP16_PORT=${FP16_PORT}"
echo "FP4_PORT=${FP4_PORT}"

for case_id in ${CASES}; do
  tag_case="${case_id/:/_}"
  tag="phase9_up_proj_isolated_recheck_task${tag_case}"
  echo "============================================================"
  echo "CASE=${case_id}"
  echo "TAG=${tag}"
  echo "============================================================"
  CASE_LIST="${case_id}" \
    TAG="${tag}" \
    FP16_PORT="${FP16_PORT}" \
    FP4_PORT="${FP4_PORT}" \
    RUN_FP16=1 \
    RUN_FP4=1 \
    bash toy_quantvla/run_phase9_up_proj_matched_set.sh
done

echo "Phase 9 isolated recheck complete"
