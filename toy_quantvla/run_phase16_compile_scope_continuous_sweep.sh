#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p /tmp/logs toy_quantvla/results docs

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
TAG="${TAG:-phase16_compile_scope_continuous_4_9_6_8_s260_v1}"
CASE_LIST="${CASE_LIST:-4:9,6:8}"
MAX_POLICY_STEPS="${MAX_POLICY_STEPS:-260}"
BASE_SEED="${BASE_SEED:-20260613}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
COMPILE_BACKEND="${COMPILE_BACKEND:-inductor}"
TARGETS="${TARGETS:-action_head_model action_head_model_blocks_8_15_eager action_head_model_blocks_6_15_eager action_head_dit_blocks_0_7 action_head_dit_attn_all action_head_dit_ff_all}"

echo "TAG=${TAG}"
echo "CASE_LIST=${CASE_LIST}"
echo "MAX_POLICY_STEPS=${MAX_POLICY_STEPS}"
echo "BASE_SEED=${BASE_SEED}"
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
    "${PYTHON_BIN}" toy_quantvla/phase13_torch_compile_online_drift.py \
      --case-list "${CASE_LIST}" \
      --max-policy-steps "${MAX_POLICY_STEPS}" \
      --base-seed "${BASE_SEED}" \
      --torch-compile-target "${target}" \
      --torch-compile-mode "${COMPILE_MODE}" \
      --torch-compile-backend "${COMPILE_BACKEND}" \
      --headless \
      --no-video \
      --trace-dir "${trace_dir}" \
      --log-file "${log_file}" \
      --output-json "${result_json}" \
      --output-md "${result_md}"
done

summary_args=()
for result_json in "${result_jsons[@]}"; do
  summary_args+=(--result-json "${result_json}")
done

"${PYTHON_BIN}" toy_quantvla/phase16_compile_scope_continuous_summary.py \
  "${summary_args[@]}" \
  --output-json "toy_quantvla/results/${TAG}_summary.json" \
  --output-md "docs/${TAG}_summary.md"

echo "Phase 16 compile scope continuous sweep complete"
