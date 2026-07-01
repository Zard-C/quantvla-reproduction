#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p /tmp/logs toy_quantvla/results

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
ISAAC_ROOT="${ISAAC_ROOT:-/root/autodl-tmp/Isaac-GR00T}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/GR00T-N1.7-LIBERO/libero_10}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-LIBERO_PANDA}"
PORT="${PORT:-6510}"
TAG="${TAG:-phase36_n17_timed_rollout_smoke_v1}"
ENV_NAME="${ENV_NAME:-libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket}"
N_EPISODES="${N_EPISODES:-1}"
N_ENVS="${N_ENVS:-1}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
SEED="${SEED:-20260705}"

TORCH_COMPILE_TARGET="${TORCH_COMPILE_TARGET:-none}"
TORCH_COMPILE_BACKEND="${TORCH_COMPILE_BACKEND:-inductor}"
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-reduce-overhead}"
TORCH_COMPILE_FULLGRAPH="${TORCH_COMPILE_FULLGRAPH:-0}"
TORCH_COMPILE_DYNAMIC="${TORCH_COMPILE_DYNAMIC:-}"
TORCH_COMPILE_FALLBACK_STEP_START="${TORCH_COMPILE_FALLBACK_STEP_START:-}"
TORCH_COMPILE_FALLBACK_STEP_END="${TORCH_COMPILE_FALLBACK_STEP_END:-}"
SERVER_TRACE_CUDA_SYNC="${SERVER_TRACE_CUDA_SYNC:-0}"

SERVER_LOG="/tmp/logs/${TAG}_server.log"
CLIENT_LOG="/tmp/logs/${TAG}_client.log"
SERVER_PID_FILE="/tmp/logs/${TAG}_server.pid"
VIDEO_DIR="/tmp/${TAG}_videos"

kill_server() {
  if [ -f "${SERVER_PID_FILE}" ]; then
    local pid
    pid="$(cat "${SERVER_PID_FILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  fi
}

wait_for_server() {
  local limit="${1:-1200}"
  for _ in $(seq 1 "${limit}"); do
    if grep -q "Server is ready and listening" "${SERVER_LOG}"; then
      return 0
    fi
    if ! kill -0 "$(cat "${SERVER_PID_FILE}")" 2>/dev/null; then
      tail -200 "${SERVER_LOG}" || true
      return 1
    fi
    sleep 1
  done
  tail -200 "${SERVER_LOG}" || true
  return 1
}

COMPILE_ARGS=(
  --torch-compile-target "${TORCH_COMPILE_TARGET}"
  --torch-compile-backend "${TORCH_COMPILE_BACKEND}"
  --torch-compile-mode "${TORCH_COMPILE_MODE}"
)
if [ "${TORCH_COMPILE_FULLGRAPH}" = "1" ]; then
  COMPILE_ARGS+=(--torch-compile-fullgraph)
fi
if [ -n "${TORCH_COMPILE_DYNAMIC}" ]; then
  COMPILE_ARGS+=(--torch-compile-dynamic "${TORCH_COMPILE_DYNAMIC}")
fi
if [ -n "${TORCH_COMPILE_FALLBACK_STEP_START}" ] || [ -n "${TORCH_COMPILE_FALLBACK_STEP_END}" ]; then
  COMPILE_ARGS+=(
    --torch-compile-fallback-step-start "${TORCH_COMPILE_FALLBACK_STEP_START}"
    --torch-compile-fallback-step-end "${TORCH_COMPILE_FALLBACK_STEP_END}"
  )
fi
if [ "${SERVER_TRACE_CUDA_SYNC}" = "1" ]; then
  COMPILE_ARGS+=(--server-request-trace-cuda-sync)
fi

echo "Phase36 N1.7 timed rollout"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "ISAAC_ROOT=${ISAAC_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "EMBODIMENT_TAG=${EMBODIMENT_TAG}"
echo "PORT=${PORT}"
echo "TAG=${TAG}"
echo "ENV_NAME=${ENV_NAME}"
echo "N_EPISODES=${N_EPISODES}"
echo "N_ENVS=${N_ENVS}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS}"
echo "SEED=${SEED}"
echo "TORCH_COMPILE_TARGET=${TORCH_COMPILE_TARGET}"
echo "TORCH_COMPILE_BACKEND=${TORCH_COMPILE_BACKEND}"
echo "TORCH_COMPILE_MODE=${TORCH_COMPILE_MODE}"
echo "TORCH_COMPILE_FULLGRAPH=${TORCH_COMPILE_FULLGRAPH}"
echo "TORCH_COMPILE_DYNAMIC=${TORCH_COMPILE_DYNAMIC}"
echo "TORCH_COMPILE_FALLBACK_STEP_START=${TORCH_COMPILE_FALLBACK_STEP_START}"
echo "TORCH_COMPILE_FALLBACK_STEP_END=${TORCH_COMPILE_FALLBACK_STEP_END}"
echo "SERVER_TRACE_CUDA_SYNC=${SERVER_TRACE_CUDA_SYNC}"

kill_server
pkill -f "n17_timed_gr00t_server.py.*--port ${PORT}" 2>/dev/null || true
rm -rf "${VIDEO_DIR}"
: > "${SERVER_LOG}"
: > "${CLIENT_LOG}"

(
  export PYTHONPATH="${REPO_ROOT}:${ISAAC_ROOT}:${PYTHONPATH:-}"
  export NO_ALBUMENTATIONS_UPDATE=1
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  "${PYTHON_BIN}" toy_quantvla/n17_timed_gr00t_server.py \
    --isaac-root "${ISAAC_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --embodiment-tag "${EMBODIMENT_TAG}" \
    --port "${PORT}" \
    --output-json "toy_quantvla/results/${TAG}_server_prepare.json" \
    --server-latency-json "toy_quantvla/results/${TAG}_server_latency.json" \
    --server-latency-flush-every 20 \
    --server-request-trace-jsonl "toy_quantvla/results/${TAG}_request_trace.jsonl" \
    "${COMPILE_ARGS[@]}"
) > "${SERVER_LOG}" 2>&1 &
echo $! > "${SERVER_PID_FILE}"
echo "SERVER_PID=$(cat "${SERVER_PID_FILE}")"
echo "SERVER_LOG=${SERVER_LOG}"

wait_for_server 1200

set +e
(
  cd "${ISAAC_ROOT}"
  export PYTHONPATH="${ISAAC_ROOT}:${PYTHONPATH:-}"
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export NO_ALBUMENTATIONS_UPDATE=1
  "${PYTHON_BIN}" gr00t/eval/rollout_policy.py \
    --n-episodes "${N_EPISODES}" \
    --policy-client-host 127.0.0.1 \
    --policy-client-port "${PORT}" \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --env-name "${ENV_NAME}" \
    --n-action-steps "${N_ACTION_STEPS}" \
    --n-envs "${N_ENVS}" \
    --seed "${SEED}" \
    --video-dir "${VIDEO_DIR}"
) > "${CLIENT_LOG}" 2>&1
CLIENT_STATUS=$?
set -e

kill_server

cp "${SERVER_LOG}" "toy_quantvla/results/${TAG}_server.log"
cp "${CLIENT_LOG}" "toy_quantvla/results/${TAG}_client.log"

"${PYTHON_BIN}" - <<PY
import json, re
from pathlib import Path
tag = ${TAG@Q}
client = Path(${CLIENT_LOG@Q}).read_text(errors="replace")
server = Path(${SERVER_LOG@Q}).read_text(errors="replace")
match = re.search(r"success rate:\\s+([0-9.]+)", client)
summary = {
    "tag": tag,
    "model_path": ${MODEL_PATH@Q},
    "isaac_root": ${ISAAC_ROOT@Q},
    "embodiment_tag": ${EMBODIMENT_TAG@Q},
    "env_name": ${ENV_NAME@Q},
    "n_episodes": int(${N_EPISODES@Q}),
    "n_envs": int(${N_ENVS@Q}),
    "n_action_steps": int(${N_ACTION_STEPS@Q}),
    "max_episode_steps": int(${MAX_EPISODE_STEPS@Q}),
    "seed": int(${SEED@Q}),
    "client_status": int(${CLIENT_STATUS}),
    "success_rate": float(match.group(1)) if match else None,
    "server_ready": "Server is ready and listening" in server,
    "torch_compile_target": ${TORCH_COMPILE_TARGET@Q},
    "torch_compile_fallback_step_start": ${TORCH_COMPILE_FALLBACK_STEP_START@Q},
    "torch_compile_fallback_step_end": ${TORCH_COMPILE_FALLBACK_STEP_END@Q},
    "video_dir": ${VIDEO_DIR@Q},
    "client_log": f"toy_quantvla/results/{tag}_client.log",
    "server_log": f"toy_quantvla/results/{tag}_server.log",
    "server_latency_json": f"toy_quantvla/results/{tag}_server_latency.json",
    "server_prepare_json": f"toy_quantvla/results/{tag}_server_prepare.json",
}
out = Path("toy_quantvla/results") / f"{tag}_summary.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

exit "${CLIENT_STATUS}"
