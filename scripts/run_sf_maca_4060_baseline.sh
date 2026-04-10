#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-sf_maca_4060_fixrule_${RUN_ID}}"
TRAIN_DIR="${TRAIN_DIR:-train_dir/sample_factory}"
ENV_NAME="${ENV_NAME:-maca_aircombat}"
OPPONENT="${OPPONENT:-fix_rule}"
MAX_STEP="${MAX_STEP:-1000}"
SEED="${SEED:-1}"
TRAIN_SECONDS="${TRAIN_SECONDS:-7200}"
NUM_WORKERS="${NUM_WORKERS:-4}"

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
  --num_workers="$NUM_WORKERS"
  --num_envs_per_worker=1
  --worker_num_splits=1
  --rollout=16
  --recurrence=16
  --batch_size=512
  --num_batches_per_iteration=1
  --num_minibatches_to_accumulate=1
  --train_for_seconds="$TRAIN_SECONDS"
  --ppo_epochs=1
  --save_every_sec=900
  --experiment_summaries_interval=30
  --decorrelate_envs_on_one_worker=False
  --train_in_background_thread=True
  --use_rnn=True
  --rnn_type=lstm
  --hidden_size=128
  --gamma=0.995
  --reward_scale=0.1
  --exploration_loss_coeff=0.003
  --max_policy_lag=40
  --with_vtrace=True
  --maca_opponent="$OPPONENT"
  --maca_max_step="$MAX_STEP"
  --maca_render=False
)

LOG_FILE="log/${EXP_NAME}.launcher.log"

echo "Launching 4060 baseline run: $EXP_NAME"
echo "Log: $LOG_FILE"
printf 'Command: '
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
