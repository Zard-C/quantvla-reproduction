#!/usr/bin/env bash
set -euo pipefail

WAIT_SESSION="${WAIT_SESSION:-phase43_hybrid_bo}"
POLL_SECONDS="${POLL_SECONDS:-60}"

echo "Phase43 follow-up queue"
echo "WAIT_SESSION=${WAIT_SESSION}"
echo "POLL_SECONDS=${POLL_SECONDS}"
echo "STARTED_AT=$(date -Is)"

while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  echo "[$(date -Is)] waiting for ${WAIT_SESSION} to finish..."
  sleep "${POLL_SECONDS}"
done

echo "[$(date -Is)] ${WAIT_SESSION} finished; starting Phase44."
bash toy_quantvla/run_phase44_n17_hybrid_heldout_auto.sh

echo "[$(date -Is)] Phase44 finished; starting Phase45."
bash toy_quantvla/run_phase45_n17_hybrid_alltask_stress_auto.sh

echo "[$(date -Is)] Phase43 follow-up queue complete."
