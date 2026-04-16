#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-mappo_maca_${RUN_ID}}"
TRAIN_DIR="${TRAIN_DIR:-train_dir/mappo}"
DEVICE="${DEVICE:-gpu}"
SEED="${SEED:-1}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-20000000}"
NUM_ENVS="${NUM_ENVS:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ROLLOUT="${ROLLOUT:-128}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
ROLE_EMBED_DIM="${ROLE_EMBED_DIM:-8}"
SAVE_EVERY_SEC="${SAVE_EVERY_SEC:-900}"
LOG_EVERY_SEC="${LOG_EVERY_SEC:-30}"
MAX_VISIBLE_ENEMIES="${MAX_VISIBLE_ENEMIES:-4}"
OPPONENT="${OPPONENT:-fix_rule}"
MAX_STEP="${MAX_STEP:-1000}"
FRIENDLY_ATTRITION_PENALTY="${FRIENDLY_ATTRITION_PENALTY:-200}"
ENEMY_ATTRITION_REWARD="${ENEMY_ATTRITION_REWARD:-100}"
FRESH_START="${FRESH_START:-1}"

mkdir -p log "$TRAIN_DIR"
if [[ "$FRESH_START" == "1" && -d "$TRAIN_DIR/$EXP_NAME" ]]; then
  echo "Fresh start enabled, removing existing experiment dir: $TRAIN_DIR/$EXP_NAME"
  rm -rf "$TRAIN_DIR/$EXP_NAME"
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment${PYTHONPATH:+:$PYTHONPATH}"

cmd=(
  conda run --no-capture-output -n maca-py37-min python scripts/train_mappo_maca.py
  --experiment="$EXP_NAME"
  --train_dir="$TRAIN_DIR"
  --device="$DEVICE"
  --seed="$SEED"
  --num_envs="$NUM_ENVS"
  --num_workers="$NUM_WORKERS"
  --rollout="$ROLLOUT"
  --train_for_env_steps="$TOTAL_ENV_STEPS"
  --ppo_epochs="$PPO_EPOCHS"
  --learning_rate="$LEARNING_RATE"
  --hidden_size="$HIDDEN_SIZE"
  --role_embed_dim="$ROLE_EMBED_DIM"
  --save_every_sec="$SAVE_EVERY_SEC"
  --log_every_sec="$LOG_EVERY_SEC"
  --maca_opponent="$OPPONENT"
  --maca_max_step="$MAX_STEP"
  --maca_max_visible_enemies="$MAX_VISIBLE_ENEMIES"
  --maca_friendly_attrition_penalty="$FRIENDLY_ATTRITION_PENALTY"
  --maca_enemy_attrition_reward="$ENEMY_ATTRITION_REWARD"
)

echo "Experiment: $EXP_NAME"
echo "Log: log/${EXP_NAME}.train.log"
printf 'Command: '
printf '%q ' "${cmd[@]}"
echo

"${cmd[@]}" 2>&1 | tee "log/${EXP_NAME}.train.log"
