#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

DOWNLOAD_SESSION="${DOWNLOAD_SESSION:-phase36_n17_aria}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/GR00T-N1.7-LIBERO/libero_10}"
WEIGHT_1="${WEIGHT_1:-${MODEL_DIR}/model-00001-of-00002.safetensors}"
WEIGHT_2="${WEIGHT_2:-${MODEL_DIR}/model-00002-of-00002.safetensors}"
WEIGHT_1_BYTES="${WEIGHT_1_BYTES:-4990519232}"
WEIGHT_2_BYTES="${WEIGHT_2_BYTES:-1919980184}"
RUN_SMOKE="${RUN_SMOKE:-1}"
SMOKE_LOG="${SMOKE_LOG:-phase36_n17_official_smoke_tmux.log}"

file_size() {
  local path="$1"
  stat -c %s "${path}" 2>/dev/null || echo 0
}

echo "[phase36] waiting for download session: ${DOWNLOAD_SESSION}"
while tmux has-session -t "${DOWNLOAD_SESSION}" 2>/dev/null; do
  date
  du -h "${WEIGHT_1}" 2>/dev/null || true
  du -h "${WEIGHT_2}" 2>/dev/null || true
  sleep "${POLL_SECONDS}"
done

echo "[phase36] download session ended"
size_1="$(file_size "${WEIGHT_1}")"
size_2="$(file_size "${WEIGHT_2}")"
echo "[phase36] ${WEIGHT_1}: ${size_1} bytes"
echo "[phase36] ${WEIGHT_2}: ${size_2} bytes"

if [ "${size_1}" != "${WEIGHT_1_BYTES}" ]; then
  echo "[phase36] size mismatch for ${WEIGHT_1}: expected ${WEIGHT_1_BYTES}, got ${size_1}" >&2
  exit 1
fi
if [ "${size_2}" != "${WEIGHT_2_BYTES}" ]; then
  echo "[phase36] size mismatch for ${WEIGHT_2}: expected ${WEIGHT_2_BYTES}, got ${size_2}" >&2
  exit 1
fi

if [ "${RUN_SMOKE}" != "1" ]; then
  echo "[phase36] RUN_SMOKE=${RUN_SMOKE}; stopping after size validation"
  exit 0
fi

echo "[phase36] starting official N1.7 smoke"
bash toy_quantvla/run_phase36_n17_official_smoke.sh 2>&1 | tee "${SMOKE_LOG}"
