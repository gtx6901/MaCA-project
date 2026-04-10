#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-sf_maca_gpu_smoke_${RUN_ID}}"
TRAIN_DIR="${TRAIN_DIR:-train_dir/sample_factory}"
ENV_NAME="${ENV_NAME:-maca_aircombat}"
OPPONENT="${OPPONENT:-fix_rule_no_att}"
MAX_STEP="${MAX_STEP:-120}"
SEED="${SEED:-1}"

mkdir -p log "$TRAIN_DIR"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment${PYTHONPATH:+:$PYTHONPATH}"

CMD=(
  conda run --no-capture-output -n maca-py37-min python scripts/train_sf_maca.py
  --algo=APPO
  --env="$ENV_NAME"
  --experiment="$EXP_NAME"
  --train_dir="$TRAIN_DIR"
  --device=gpu
  --seed="$SEED"
  --num_workers=1
  --num_envs_per_worker=1
  --worker_num_splits=1
  --rollout=8
  --recurrence=1
  --batch_size=32
  --num_batches_per_iteration=1
  --train_for_env_steps=320
  --ppo_epochs=1
  --save_every_sec=3600
  --experiment_summaries_interval=10
  --decorrelate_envs_on_one_worker=False
  --train_in_background_thread=False
  --with_vtrace=False
  --use_rnn=False
  --maca_opponent="$OPPONENT"
  --maca_max_step="$MAX_STEP"
  --maca_render=False
)

LOG_FILE="log/${EXP_NAME}.launcher.log"

echo "Launching GPU smoke test: $EXP_NAME"
echo "Log: $LOG_FILE"
printf 'Command: '
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
