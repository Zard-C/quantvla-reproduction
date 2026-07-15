#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p /tmp/logs toy_quantvla/results docs

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
CASE_LIST="${CASE_LIST:-4:9,6:8}"
WINDOWS="${WINDOWS:-full,early}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260613}"
MEAN_EPSILONS="${MEAN_EPSILONS:-0.00001,0.00002,0.00003}"
REPLAY_LAMBDAS="${REPLAY_LAMBDAS:-0.25,0.5,1.0}"

SPEED_TAG="${SPEED_TAG:-phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1}"
SPEED_VECTOR_JSON="toy_quantvla/results/${SPEED_TAG}_sameobs_real_drift_directions.json"
SPEED_ONLINE_JSON="toy_quantvla/results/${SPEED_TAG}_sameobs_online_drift.json"
SPEED_SEQUENCE_JSON="toy_quantvla/results/${SPEED_TAG}_sameobs_real_drift_sequences.json"
SPEED_SEQUENCE_MD="docs/${SPEED_TAG}_sameobs_real_drift_sequences_zh.md"
SPEED_SEQUENCE_DIR="toy_quantvla/results/${SPEED_TAG}_sameobs_real_drift_sequences"

COMBO_TAG="${COMBO_TAG:-phase39_stageA2_sameobs_realdrift_combo_blocks0_3_window0_120_seed20260613_v1}"
COMBO_SEQUENCE_JSON="toy_quantvla/results/${COMBO_TAG}_sameobs_real_drift_sequences.json"
COMBO_SEQUENCE_MD="docs/${COMBO_TAG}_sameobs_real_drift_sequences_zh.md"
COMBO_SEQUENCE_DIR="toy_quantvla/results/${COMBO_TAG}_sameobs_real_drift_sequences"

echo "CASE_LIST=${CASE_LIST}"
echo "WINDOWS=${WINDOWS}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"
echo "MEAN_EPSILONS=${MEAN_EPSILONS}"
echo "REPLAY_LAMBDAS=${REPLAY_LAMBDAS}"
echo "SPEED_TAG=${SPEED_TAG}"
echo "COMBO_TAG=${COMBO_TAG}"

if [ ! -f "${SPEED_VECTOR_JSON}" ]; then
  echo "Missing ${SPEED_VECTOR_JSON}. Run Stage A2 speed-only capture/vector first." >&2
  exit 2
fi
if [ ! -f "${SPEED_ONLINE_JSON}" ]; then
  echo "Missing ${SPEED_ONLINE_JSON}. Run Stage A2 speed-only capture first." >&2
  exit 2
fi

echo "=== A2.2 sign symmetry: negative mean real-drift direction ==="
TAG_PREFIX=phase39_stageA2_sign_neg_speedonly_mean_seed20260613_v1_threshold \
RUN_MODE=reduced \
RUN_BASELINE=1 \
RUN_PERTURB=0 \
RUN_REAL_DRIFT=1 \
RUN_REAL_SEQUENCE=0 \
REAL_VECTOR_JSON="${SPEED_VECTOR_JSON}" \
ACTION_PERTURB_SIGN=-1 \
CASE_LIST="${CASE_LIST}" \
DIRECTIONS=real_speed_only_sameobs_mean \
WINDOWS="${WINDOWS}" \
EPSILONS="${MEAN_EPSILONS}" \
PORT=6210 \
POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
PYTHON_BIN="${PYTHON_BIN}" \
OUT_MD=docs/phase39_stageA2_sign_neg_speedonly_mean_threshold_zh.md \
bash toy_quantvla/run_phase39_perturb_threshold_pilot.sh

echo "=== A2.3 speed-only replay sequence: build per-step drift sequences ==="
"${PYTHON_BIN}" toy_quantvla/phase39_same_observation_drift_sequences.py \
  --online-drift-json "${SPEED_ONLINE_JSON}" \
  --variant-name speed_only \
  --case-list "${CASE_LIST}" \
  --windows "${WINDOWS}" \
  --sequence-dir "${SPEED_SEQUENCE_DIR}" \
  --output-json "${SPEED_SEQUENCE_JSON}" \
  --output-md "${SPEED_SEQUENCE_MD}"

echo "=== A2.3 speed-only replay sequence: lambda sweep ==="
TAG_PREFIX=phase39_stageA2_replay_speedonly_lambda_seed20260613_v1_threshold \
RUN_MODE=reduced \
RUN_BASELINE=1 \
RUN_PERTURB=0 \
RUN_REAL_DRIFT=0 \
RUN_REAL_SEQUENCE=1 \
REAL_SEQUENCE_JSON="${SPEED_SEQUENCE_JSON}" \
CASE_LIST="${CASE_LIST}" \
DIRECTIONS=real_speed_only_sameobs_sequence \
WINDOWS="${WINDOWS}" \
EPSILONS="${REPLAY_LAMBDAS}" \
PORT=6211 \
POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
PYTHON_BIN="${PYTHON_BIN}" \
OUT_MD=docs/phase39_stageA2_replay_speedonly_lambda_threshold_zh.md \
bash toy_quantvla/run_phase39_perturb_threshold_pilot.sh

echo "=== A2.4 protected tactic: same-observation drift capture/vector ==="
TAG_PREFIX="${COMBO_TAG}" \
CASE_LIST="${CASE_LIST}" \
WINDOWS="${WINDOWS}" \
EPSILONS="${MEAN_EPSILONS}" \
POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
BASE_SEED="${POLICY_SEED_BASE}" \
MAX_POLICY_STEPS=0 \
COMPILE_TARGET=action_head_model_blocks_0_3_eager \
COMPILE_FALLBACK_STEP_START=0 \
COMPILE_FALLBACK_STEP_END=120 \
RUN_CAPTURE=1 \
RUN_VECTOR=1 \
RUN_SWEEP=0 \
PYTHON_BIN="${PYTHON_BIN}" \
bash toy_quantvla/run_phase39_stageA2_real_backend_drift_aligned.sh

echo "=== A2.4 protected tactic replay sequence: build per-step drift sequences ==="
"${PYTHON_BIN}" toy_quantvla/phase39_same_observation_drift_sequences.py \
  --online-drift-json "toy_quantvla/results/${COMBO_TAG}_sameobs_online_drift.json" \
  --variant-name combo_blocks0_3_window0_120 \
  --case-list "${CASE_LIST}" \
  --windows "${WINDOWS}" \
  --sequence-dir "${COMBO_SEQUENCE_DIR}" \
  --output-json "${COMBO_SEQUENCE_JSON}" \
  --output-md "${COMBO_SEQUENCE_MD}"

echo "=== A2.4 protected tactic replay sequence: lambda sweep ==="
TAG_PREFIX=phase39_stageA2_replay_combo_blocks0_3_window0_120_lambda_seed20260613_v1_threshold \
RUN_MODE=reduced \
RUN_BASELINE=1 \
RUN_PERTURB=0 \
RUN_REAL_DRIFT=0 \
RUN_REAL_SEQUENCE=1 \
REAL_SEQUENCE_JSON="${COMBO_SEQUENCE_JSON}" \
CASE_LIST="${CASE_LIST}" \
DIRECTIONS=real_combo_blocks0_3_window0_120_sameobs_sequence \
WINDOWS="${WINDOWS}" \
EPSILONS="${REPLAY_LAMBDAS}" \
PORT=6212 \
POLICY_SEED_BASE="${POLICY_SEED_BASE}" \
PYTHON_BIN="${PYTHON_BIN}" \
OUT_MD=docs/phase39_stageA2_replay_combo_blocks0_3_window0_120_lambda_threshold_zh.md \
bash toy_quantvla/run_phase39_perturb_threshold_pilot.sh

echo "Phase39 Stage A2 followups complete."
