#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/quantvla-reproduction

CASES="${CASES:-4:6 6:3 8:8 8:10 6:4}"
FP16_PORT="${FP16_PORT:-5582}"
FP4_PORT="${FP4_PORT:-5583}"
TAG_PREFIX="${TAG_PREFIX:-phase9_up_proj_isolated_recheck}"
DETERMINISTIC_POLICY_SEEDS="${DETERMINISTIC_POLICY_SEEDS:-0}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260609}"

echo "Isolated recheck cases: ${CASES}"
echo "FP16_PORT=${FP16_PORT}"
echo "FP4_PORT=${FP4_PORT}"
echo "TAG_PREFIX=${TAG_PREFIX}"
echo "DETERMINISTIC_POLICY_SEEDS=${DETERMINISTIC_POLICY_SEEDS}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"

for case_id in ${CASES}; do
  tag_case="${case_id/:/_}"
  tag="${TAG_PREFIX}_task${tag_case}"
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
    DETERMINISTIC_POLICY_SEEDS="${DETERMINISTIC_POLICY_SEEDS}" \
    POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
    bash toy_quantvla/run_phase9_up_proj_matched_set.sh
done

echo "Phase 9 isolated recheck complete"
