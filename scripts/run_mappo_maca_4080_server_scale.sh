#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-mappo_maca_4080_server_scale_${RUN_ID}}"
export DEVICE="${DEVICE:-gpu}"
export TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-120000000}"
export NUM_ENVS="${NUM_ENVS:-16}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export ROLLOUT="${ROLLOUT:-160}"
export PPO_EPOCHS="${PPO_EPOCHS:-4}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
export ROLE_EMBED_DIM="${ROLE_EMBED_DIM:-8}"
export SAVE_EVERY_SEC="${SAVE_EVERY_SEC:-900}"
export LOG_EVERY_SEC="${LOG_EVERY_SEC:-30}"
export MAX_VISIBLE_ENEMIES="${MAX_VISIBLE_ENEMIES:-4}"
export OPPONENT="${OPPONENT:-fix_rule}"
export MAX_STEP="${MAX_STEP:-1000}"

exec bash scripts/run_mappo_maca_train.sh "$@"
