#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/gr00t-libero-py310/bin/python}"
CASE_LIST="${CASE_LIST:-4:9,6:8}"
WINDOWS="${WINDOWS:-full,early}"
POLICY_SEED_BASE="${POLICY_SEED_BASE:-20260613}"
REPLAY_LAMBDAS="${REPLAY_LAMBDAS:-0.25,0.5,1.0}"

SPEED_TAG="${SPEED_TAG:-phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1}"
COMBO_TAG="${COMBO_TAG:-phase39_stageA2_sameobs_realdrift_combo_blocks0_3_window0_120_seed20260613_v1}"
SPEED_SEQUENCE_JSON="toy_quantvla/results/${SPEED_TAG}_sameobs_real_drift_sequences.json"
COMBO_SEQUENCE_JSON="toy_quantvla/results/${COMBO_TAG}_sameobs_real_drift_sequences.json"

echo "CASE_LIST=${CASE_LIST}"
echo "WINDOWS=${WINDOWS}"
echo "POLICY_SEED_BASE=${POLICY_SEED_BASE}"
echo "REPLAY_LAMBDAS=${REPLAY_LAMBDAS}"
echo "SPEED_SEQUENCE_JSON=${SPEED_SEQUENCE_JSON}"
echo "COMBO_SEQUENCE_JSON=${COMBO_SEQUENCE_JSON}"

echo "=== Resume A2.3 speed-only replay sequence: lambda sweep ==="
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

echo "=== Resume A2.4 protected tactic replay sequence: lambda sweep ==="
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

echo "Phase39 Stage A2 replay resume complete."
