#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

CASE_LIST="${CASE_LIST:-0:21,0:22,0:23,1:21,1:22,1:23,4:21,4:22,4:23,6:21,6:22,6:23,8:21,8:22,8:23}"
TACTICS="${TACTICS:-fp16 speed_only window_0_10 window_0_20 window_0_30 window_10_30 window_20_50}"
TAG_PREFIX="${TAG_PREFIX:-phase36_n17_tactic_probe_15case_v1}"
PORT_BASE="${PORT_BASE:-6520}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
QWEN3_SNAPSHOT="${QWEN3_SNAPSHOT:-/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203}"
RECORD_VIDEO="${RECORD_VIDEO:-0}"
SERVER_TRACE_CUDA_SYNC="${SERVER_TRACE_CUDA_SYNC:-1}"
SERVER_LATENCY_FLUSH_EVERY="${SERVER_LATENCY_FLUSH_EVERY:-1}"
POLICY_CLIENT_TIMEOUT_MS="${POLICY_CLIENT_TIMEOUT_MS:-600000}"

env_name_for_task() {
  case "$1" in
    0) echo "libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket" ;;
    1) echo "libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket" ;;
    2) echo "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it" ;;
    3) echo "libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it" ;;
    4) echo "libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate" ;;
    5) echo "libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy" ;;
    6) echo "libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate" ;;
    7) echo "libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket" ;;
    8) echo "libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" ;;
    9) echo "libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it" ;;
    *) echo "Unknown LIBERO-10 task id: $1" >&2; return 1 ;;
  esac
}

configure_tactic() {
  TORCH_COMPILE_TARGET="none"
  TORCH_COMPILE_FALLBACK_STEP_START=""
  TORCH_COMPILE_FALLBACK_STEP_END=""
  case "$1" in
    fp16)
      TORCH_COMPILE_TARGET="none"
      ;;
    speed_only)
      TORCH_COMPILE_TARGET="action_head_model"
      ;;
    window_0_10)
      TORCH_COMPILE_TARGET="action_head_model"
      TORCH_COMPILE_FALLBACK_STEP_START="0"
      TORCH_COMPILE_FALLBACK_STEP_END="10"
      ;;
    window_0_20)
      TORCH_COMPILE_TARGET="action_head_model"
      TORCH_COMPILE_FALLBACK_STEP_START="0"
      TORCH_COMPILE_FALLBACK_STEP_END="20"
      ;;
    window_0_30)
      TORCH_COMPILE_TARGET="action_head_model"
      TORCH_COMPILE_FALLBACK_STEP_START="0"
      TORCH_COMPILE_FALLBACK_STEP_END="30"
      ;;
    window_10_30)
      TORCH_COMPILE_TARGET="action_head_model"
      TORCH_COMPILE_FALLBACK_STEP_START="10"
      TORCH_COMPILE_FALLBACK_STEP_END="30"
      ;;
    window_20_50)
      TORCH_COMPILE_TARGET="action_head_model"
      TORCH_COMPILE_FALLBACK_STEP_START="20"
      TORCH_COMPILE_FALLBACK_STEP_END="50"
      ;;
    *)
      echo "Unknown tactic: $1" >&2
      return 1
      ;;
  esac
}

write_runner_status() {
  local tag="$1"
  local tactic="$2"
  local task="$3"
  local init="$4"
  local status="$5"
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
payload = {
    "tag": ${tag@Q},
    "tactic": ${tactic@Q},
    "task_id": int(${task@Q}),
    "init_index": int(${init@Q}),
    "exit_status": int(${status@Q}),
}
path = Path("toy_quantvla/results") / f"{payload['tag']}_runner_status.json"
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

echo "Phase36 N1.7 tactic probe"
echo "TAG_PREFIX=${TAG_PREFIX}"
echo "CASE_LIST=${CASE_LIST}"
echo "TACTICS=${TACTICS}"
echo "PORT_BASE=${PORT_BASE}"
echo "QWEN3_SNAPSHOT=${QWEN3_SNAPSHOT}"
echo "RECORD_VIDEO=${RECORD_VIDEO}"
echo "SERVER_TRACE_CUDA_SYNC=${SERVER_TRACE_CUDA_SYNC}"
echo "SERVER_LATENCY_FLUSH_EVERY=${SERVER_LATENCY_FLUSH_EVERY}"
echo "POLICY_CLIENT_TIMEOUT_MS=${POLICY_CLIENT_TIMEOUT_MS}"

run_index=0
for tactic in ${TACTICS}; do
  configure_tactic "${tactic}"
  IFS=',' read -ra cases <<< "${CASE_LIST}"
  for case_id in "${cases[@]}"; do
    task_id="${case_id%%:*}"
    init_index="${case_id##*:}"
    env_name="$(env_name_for_task "${task_id}")"
    tag="${TAG_PREFIX}_${tactic}_task${task_id}_init${init_index}"
    port="$((PORT_BASE + (run_index % 50)))"
    run_index="$((run_index + 1))"

    echo "=== ${tag} tactic=${tactic} task=${task_id} init=${init_index} port=${port} ==="
    set +e
    GR00T_QWEN3_INIT_FROM_CONFIG=1 \
    GR00T_QWEN3_CONFIG_NAME="${QWEN3_SNAPSHOT}" \
    GR00T_QWEN3_PROCESSOR_NAME="${QWEN3_SNAPSHOT}" \
    GR00T_DISABLE_VIDEO_RECORDING=1 \
    TAG="${tag}" \
    PORT="${port}" \
    RECORD_VIDEO="${RECORD_VIDEO}" \
    ENV_NAME="${env_name}" \
    SEED="${init_index}" \
    LIBERO_USE_BENCHMARK_INIT_STATES=1 \
    TORCH_COMPILE_TARGET="${TORCH_COMPILE_TARGET}" \
    TORCH_COMPILE_FALLBACK_STEP_START="${TORCH_COMPILE_FALLBACK_STEP_START}" \
    TORCH_COMPILE_FALLBACK_STEP_END="${TORCH_COMPILE_FALLBACK_STEP_END}" \
    SERVER_TRACE_CUDA_SYNC="${SERVER_TRACE_CUDA_SYNC}" \
    SERVER_LATENCY_FLUSH_EVERY="${SERVER_LATENCY_FLUSH_EVERY}" \
    POLICY_CLIENT_TIMEOUT_MS="${POLICY_CLIENT_TIMEOUT_MS}" \
    bash toy_quantvla/run_phase36_n17_timed_rollout.sh
    status=$?
    set -e
    write_runner_status "${tag}" "${tactic}" "${task_id}" "${init_index}" "${status}"
    if [ "${status}" -ne 0 ]; then
      echo "WARNING: ${tag} exited with status ${status}" >&2
    fi
  done
done

TAG_PREFIX="${TAG_PREFIX}" \
CASE_LIST="${CASE_LIST}" \
TACTICS="${TACTICS}" \
REPORT_TITLE="${REPORT_TITLE:-Phase 36B: N1.7 Tactic Probe}" \
OUT_MD="${OUT_MD:-docs/phase36b_n17_tactic_probe_report_zh.md}" \
"${PYTHON_BIN}" toy_quantvla/phase36_n17_tactic_probe_summary.py

echo "Phase36 N1.7 tactic probe complete."
