#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p /tmp/logs toy_quantvla/results docs

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
TAG="${TAG:-phase16_step253_block03_bisect_v1}"
CASE_LIST="${CASE_LIST:-6:8}"
FOCUS_STEP="${FOCUS_STEP:-253}"
BASE_SEED="${BASE_SEED:-20260613}"
REPEATS="${REPEATS:-2}"
WARMUP_REPEATS="${WARMUP_REPEATS:-1}"
SEED_OFFSETS="${SEED_OFFSETS:--2 0 2}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
COMPILE_BACKEND="${COMPILE_BACKEND:-inductor}"
TARGETS="${TARGETS:-action_head_model action_head_model_blocks_0_3_eager action_head_model_blocks_0_1_eager action_head_model_blocks_2_3_eager action_head_model_blocks_0_0_eager action_head_model_blocks_1_1_eager action_head_model_blocks_2_2_eager action_head_model_blocks_3_3_eager}"

echo "TAG=${TAG}"
echo "CASE_LIST=${CASE_LIST}"
echo "FOCUS_STEP=${FOCUS_STEP}"
echo "BASE_SEED=${BASE_SEED}"
echo "REPEATS=${REPEATS}"
echo "WARMUP_REPEATS=${WARMUP_REPEATS}"
echo "SEED_OFFSETS=${SEED_OFFSETS}"
echo "COMPILE_MODE=${COMPILE_MODE}"
echo "COMPILE_BACKEND=${COMPILE_BACKEND}"
echo "TARGETS=${TARGETS}"

result_jsons=()
for target in ${TARGETS}; do
  slug="${target//[^A-Za-z0-9_]/_}"
  result_json="toy_quantvla/results/${TAG}_${slug}.json"
  result_md="docs/${TAG}_${slug}.md"
  trace_dir="toy_quantvla/results/${TAG}_${slug}_trace"
  log_file="/tmp/logs/${TAG}_${slug}.log"
  result_jsons+=("${result_json}")

  echo "=== Running ${target} ==="
  rm -rf "${trace_dir}"
  env \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NO_ALBUMENTATIONS_UPDATE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "${PYTHON_BIN}" toy_quantvla/phase16_step_focused_replay.py \
      --case-list "${CASE_LIST}" \
      --focus-policy-steps "${FOCUS_STEP}" \
      --base-seed "${BASE_SEED}" \
      --seed-offsets ${SEED_OFFSETS} \
      --repeats "${REPEATS}" \
      --warmup-repeats "${WARMUP_REPEATS}" \
      --torch-compile-target "${target}" \
      --torch-compile-mode "${COMPILE_MODE}" \
      --torch-compile-backend "${COMPILE_BACKEND}" \
      --headless \
      --trace-dir "${trace_dir}" \
      --log-file "${log_file}" \
      --output-json "${result_json}" \
      --output-md "${result_md}"
done

summary_args=()
for result_json in "${result_jsons[@]}"; do
  summary_args+=(--result-json "${result_json}")
done

"${PYTHON_BIN}" toy_quantvla/phase16_step_focused_summary.py \
  "${summary_args[@]}" \
  --output-json "toy_quantvla/results/${TAG}_summary.json" \
  --output-md "docs/${TAG}_summary.md"

echo "Phase 16.6 block 0-3 bisect complete"
