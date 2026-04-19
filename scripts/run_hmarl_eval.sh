#!/usr/bin/env bash
set -euo pipefail

# One-click HMARL eval launcher.
# Usage:
#   scripts/run_hmarl_eval.sh <experiment_name> [episodes] [opponent] [extra eval args...]
#
# Examples:
#   scripts/run_hmarl_eval.sh exp_a 20 fix_rule
#   scripts/run_hmarl_eval.sh exp_a 20 fix_rule --maca_render True
#   scripts/run_hmarl_eval.sh exp_a 20 fix_rule --maca_render True --output_json train_dir/mappo/exp_a/eval/eval_gui.json

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_hmarl_eval.sh <experiment_name> [episodes] [opponent] [extra eval args...]"
  exit 1
fi

EXP_NAME="$1"
EPISODES="${2:-50}"
OPPONENT="${3:-fix_rule}"
EXTRA_EVAL_ARGS=()
if [[ $# -ge 4 ]]; then
  EXTRA_EVAL_ARGS=("${@:4}")
fi

OUT_JSON="train_dir/mappo/${EXP_NAME}/eval/eval_manual_${OPPONENT}_${EPISODES}.json"
HAS_CUSTOM_OUT_JSON=false
for ((i=0; i<${#EXTRA_EVAL_ARGS[@]}; i++)); do
  arg="${EXTRA_EVAL_ARGS[$i]}"
  if [[ "${arg}" == "--output_json" ]]; then
    if (( i + 1 >= ${#EXTRA_EVAL_ARGS[@]} )); then
      echo "[run_hmarl_eval] --output_json requires a path"
      exit 1
    fi
    OUT_JSON="${EXTRA_EVAL_ARGS[$((i + 1))]}"
    HAS_CUSTOM_OUT_JSON=true
  elif [[ "${arg}" == --output_json=* ]]; then
    OUT_JSON="${arg#--output_json=}"
    HAS_CUSTOM_OUT_JSON=true
  fi
done

echo "[run_hmarl_eval] experiment=${EXP_NAME} episodes=${EPISODES} opponent=${OPPONENT} output_json=${OUT_JSON}"

EVAL_CMD=(
  conda run -n maca-py37-min python scripts/eval_mappo_maca.py
  --experiment "${EXP_NAME}"
  --episodes "${EPISODES}"
  --deterministic True
  --maca_opponent "${OPPONENT}"
)

if [[ "${HAS_CUSTOM_OUT_JSON}" != "true" ]]; then
  EVAL_CMD+=(--output_json "${OUT_JSON}")
fi

if (( ${#EXTRA_EVAL_ARGS[@]} > 0 )); then
  EVAL_CMD+=("${EXTRA_EVAL_ARGS[@]}")
fi

"${EVAL_CMD[@]}"

# Do not fail pipeline on acceptance levels below target; checker exits 2 in that case.
set +e
conda run -n maca-py37-min python scripts/check_hmarl_acceptance.py --eval_json "${OUT_JSON}"
CHECK_RC=$?
set -e

echo "[run_hmarl_eval] eval_json=${OUT_JSON} checker_rc=${CHECK_RC}"
