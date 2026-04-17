#!/usr/bin/env bash
set -euo pipefail

# One-click HMARL eval launcher.
# Usage:
#   scripts/run_hmarl_eval.sh <experiment_name> [episodes] [opponent]

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_hmarl_eval.sh <experiment_name> [episodes] [opponent]"
  exit 1
fi

EXP_NAME="$1"
EPISODES="${2:-50}"
OPPONENT="${3:-fix_rule}"

OUT_JSON="train_dir/mappo/${EXP_NAME}/eval/eval_manual_${OPPONENT}_${EPISODES}.json"

echo "[run_hmarl_eval] experiment=${EXP_NAME} episodes=${EPISODES} opponent=${OPPONENT}"

conda run -n maca-py37-min python scripts/eval_mappo_maca.py \
  --experiment "${EXP_NAME}" \
  --episodes "${EPISODES}" \
  --deterministic True \
  --maca_opponent "${OPPONENT}" \
  --output_json "${OUT_JSON}"

# Do not fail pipeline on acceptance levels below target; checker exits 2 in that case.
set +e
conda run -n maca-py37-min python scripts/check_hmarl_acceptance.py --eval_json "${OUT_JSON}"
CHECK_RC=$?
set -e

echo "[run_hmarl_eval] eval_json=${OUT_JSON} checker_rc=${CHECK_RC}"
