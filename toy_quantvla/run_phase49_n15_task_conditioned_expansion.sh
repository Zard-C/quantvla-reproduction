#!/usr/bin/env bash
set -euo pipefail

# Phase49 expands the N1.5 transfer probe around the tasks where Phase48
# showed different repair profiles: task 4, task 6, and task 8.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/toy_quantvla/compat_stubs:${PYTHONPATH:-}"

export TAG_PREFIX="${TAG_PREFIX:-phase49_n15_task_conditioned_expansion_15case_v1}"
export CASE_LIST="${CASE_LIST:-4:21,4:22,4:23,4:24,4:25,6:21,6:22,6:23,6:24,6:25,8:21,8:22,8:23,8:24,8:25}"
export TACTICS="${TACTICS:-fp16 speed_only window_0_60 window_0_120 window_60_180 window_120_260 window_0_240 blocks0_3 combo_blocks0_3_window_0_120 combo_blocks0_3_window_120_260}"
export POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260716}"
export PORT_BASE="${PORT_BASE:-7900}"
export PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
export REPORT_TITLE="${REPORT_TITLE:-Phase49: N1.5 Task-Conditioned Tactic Expansion}"
export OUT_MD="${OUT_MD:-docs/phase49_n15_task_conditioned_expansion_report_zh.md}"

configure_tactic() {
  local tactic="$1"
  RUN_BASELINE_VARIANT=0
  RUN_COMPILED_VARIANT=1
  COMPILE_TARGET="action_head_model"
  COMPILED_EXTRA_ARGS=""
  EVAL_EXTRA_ARGS=""

  case "${tactic}" in
    fp16)
      RUN_BASELINE_VARIANT=1
      RUN_COMPILED_VARIANT=0
      ;;
    speed_only)
      COMPILE_TARGET="action_head_model"
      ;;
    blocks0_3)
      COMPILE_TARGET="action_head_model_blocks_0_3_eager"
      ;;
    window_[0-9]*_[0-9]*)
      local bounds="${tactic#window_}"
      local start="${bounds%%_*}"
      local end="${bounds##*_}"
      COMPILE_TARGET="action_head_model"
      COMPILED_EXTRA_ARGS="--torch-compile-fallback-step-start ${start} --torch-compile-fallback-step-end ${end}"
      EVAL_EXTRA_ARGS="--send-policy-step-key"
      ;;
    combo_blocks0_3_window_[0-9]*_[0-9]*)
      local bounds="${tactic#combo_blocks0_3_window_}"
      local start="${bounds%%_*}"
      local end="${bounds##*_}"
      COMPILE_TARGET="action_head_model_blocks_0_3_eager"
      COMPILED_EXTRA_ARGS="--torch-compile-fallback-step-start ${start} --torch-compile-fallback-step-end ${end}"
      EVAL_EXTRA_ARGS="--send-policy-step-key"
      ;;
    *)
      echo "Unknown Phase49 tactic: ${tactic}" >&2
      return 2
      ;;
  esac
}

run_tactic() {
  local tactic="$1"
  local idx="$2"
  configure_tactic "${tactic}"
  local tag="${TAG_PREFIX}_${tactic}"
  local port="$((PORT_BASE + idx))"
  echo "=== Phase49 tactic=${tactic} tag=${tag} port=${port} ==="
  echo "RUN_BASELINE_VARIANT=${RUN_BASELINE_VARIANT}"
  echo "RUN_COMPILED_VARIANT=${RUN_COMPILED_VARIANT}"
  echo "COMPILE_TARGET=${COMPILE_TARGET}"
  echo "COMPILED_EXTRA_ARGS=${COMPILED_EXTRA_ARGS}"
  echo "EVAL_EXTRA_ARGS=${EVAL_EXTRA_ARGS}"

  if [ "${RUN_BASELINE_VARIANT}" = "1" ]; then
    TAG="${tag}" \
    CASE_LIST="${CASE_LIST}" \
    POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
    BASELINE_PORT="${port}" \
    RUN_BASELINE=1 \
    RUN_COMPILED=0 \
    bash toy_quantvla/run_phase13_torch_compile_matched_set.sh
  else
    TAG="${tag}" \
    CASE_LIST="${CASE_LIST}" \
    POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
    COMPILED_PORT="${port}" \
    RUN_BASELINE=0 \
    RUN_COMPILED=1 \
    COMPILE_TARGET="${COMPILE_TARGET}" \
    COMPILED_EXTRA_ARGS="${COMPILED_EXTRA_ARGS}" \
    EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS}" \
    bash toy_quantvla/run_phase13_torch_compile_matched_set.sh
  fi
}

echo "Phase49 N1.5 task-conditioned tactic expansion"
echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "TACTICS=${TACTICS}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"
echo "PORT_BASE=${PORT_BASE}"

idx=0
for tactic in ${TACTICS}; do
  run_tactic "${tactic}" "${idx}"
  idx="$((idx + 1))"
done

TAG_PREFIX="${TAG_PREFIX}" \
CASE_LIST="${CASE_LIST}" \
TACTICS="${TACTICS}" \
POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
REPORT_TITLE="${REPORT_TITLE}" \
OUT_MD="${OUT_MD}" \
"${PYTHON_BIN}" toy_quantvla/phase49_n15_task_conditioned_summary.py

echo "Phase49 complete."
