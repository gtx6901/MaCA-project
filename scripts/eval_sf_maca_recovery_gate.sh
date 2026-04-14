#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXP_NAME="${1:?usage: scripts/eval_sf_maca_recovery_gate.sh <experiment> <opponent> [episodes] [output_json] }"
OPPONENT="${2:?usage: scripts/eval_sf_maca_recovery_gate.sh <experiment> <opponent> [episodes] [output_json] }"
EPISODES="${3:-5}"
OUTPUT_JSON="${4:-log/${EXP_NAME}.eval${EPISODES}.${OPPONENT}.json}"
TRAIN_DIR="${TRAIN_DIR:-train_dir/sample_factory}"
DEVICE="${DEVICE:-cpu}"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment${PYTHONPATH:+:$PYTHONPATH}"
export MACA_ENABLE_SF_BUFFER_SQUEEZE_PATCH="${MACA_ENABLE_SF_BUFFER_SQUEEZE_PATCH:-0}"
export MACA_REWARD_RADAR_FIGHTER_DETECTOR="${MACA_REWARD_RADAR_FIGHTER_DETECTOR:-8}"
export MACA_REWARD_RADAR_FIGHTER_FIGHTER="${MACA_REWARD_RADAR_FIGHTER_FIGHTER:-8}"
export MACA_REWARD_STRIKE_FIGHTER_SUCCESS="${MACA_REWARD_STRIKE_FIGHTER_SUCCESS:-1200}"
export MACA_REWARD_STRIKE_FIGHTER_FAIL="${MACA_REWARD_STRIKE_FIGHTER_FAIL:--2}"
export MACA_REWARD_STRIKE_ACT_VALID="${MACA_REWARD_STRIKE_ACT_VALID:-40}"
export MACA_REWARD_KEEP_ALIVE_STEP="${MACA_REWARD_KEEP_ALIVE_STEP:--2}"
export MACA_REWARD_DRAW="${MACA_REWARD_DRAW:--2000}"
export MACA_MISSED_ATTACK_PENALTY="${MACA_MISSED_ATTACK_PENALTY:-60}"
export MACA_FIRE_LOGIT_BIAS="${MACA_FIRE_LOGIT_BIAS:-0.0}"
export MACA_FIRE_PROB_FLOOR="${MACA_FIRE_PROB_FLOOR:-0.03}"
export MACA_EVAL_FIRE_PROB_FLOOR="${MACA_EVAL_FIRE_PROB_FLOOR:-0.0}"

conda run --no-capture-output -n maca-py37-min python scripts/eval_sf_maca.py \
  --experiment="$EXP_NAME" \
  --train_dir="$TRAIN_DIR" \
  --episodes="$EPISODES" \
  --device="$DEVICE" \
  --maca_opponent="$OPPONENT" \
  --output_json="$OUTPUT_JSON"
