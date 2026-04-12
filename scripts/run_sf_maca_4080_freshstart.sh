#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-sf_maca_4080_fresh_${RUN_ID}}"
TRAIN_DIR="${TRAIN_DIR:-train_dir/sample_factory}"
ENV_NAME="${ENV_NAME:-maca_aircombat}"
OPPONENT="${OPPONENT:-fix_rule}"
MAX_STEP="${MAX_STEP:-1000}"
SEED="${SEED:-1}"

TRAIN_SECONDS="${TRAIN_SECONDS:-21600}"
TRAIN_ENV_STEPS="${TRAIN_ENV_STEPS:-100000000}"
NUM_WORKERS="${NUM_WORKERS:-12}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-1}"
ROLLOUT="${ROLLOUT:-64}"
RECURRENCE="${RECURRENCE:-64}"
BATCH_SIZE="${BATCH_SIZE:-$((NUM_WORKERS * ROLLOUT * 10))}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GAMMA="${GAMMA:-0.999}"
REWARD_SCALE="${REWARD_SCALE:-0.005}"
REWARD_CLIP="${REWARD_CLIP:-50.0}"
EXPLORATION_LOSS_COEFF="${EXPLORATION_LOSS_COEFF:-0.05}"
MAX_POLICY_LAG="${MAX_POLICY_LAG:-15}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-12}"
SAVE_EVERY_SEC="${SAVE_EVERY_SEC:-900}"
TRAIN_IN_BACKGROUND_THREAD="${TRAIN_IN_BACKGROUND_THREAD:-False}"
LEARNER_MAIN_LOOP_NUM_CORES="${LEARNER_MAIN_LOOP_NUM_CORES:-3}"
TRAJ_BUFFERS_EXCESS_RATIO="${TRAJ_BUFFERS_EXCESS_RATIO:-4.0}"

FRESH_START="${FRESH_START:-1}"

mkdir -p log "$TRAIN_DIR"

if [[ "$FRESH_START" == "1" && -d "$TRAIN_DIR/$EXP_NAME" ]]; then
  echo "Fresh start enabled, removing existing experiment dir: $TRAIN_DIR/$EXP_NAME"
  rm -rf "$TRAIN_DIR/$EXP_NAME"
fi

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
  --num_envs_per_worker="$NUM_ENVS_PER_WORKER"
  --worker_num_splits=1
  --rollout="$ROLLOUT"
  --recurrence="$RECURRENCE"
  --batch_size="$BATCH_SIZE"
  --num_batches_per_iteration=1
  --num_minibatches_to_accumulate=1
  --train_for_seconds="$TRAIN_SECONDS"
  --train_for_env_steps="$TRAIN_ENV_STEPS"
  --ppo_epochs="$PPO_EPOCHS"
  --save_every_sec="$SAVE_EVERY_SEC"
  --keep_checkpoints="$KEEP_CHECKPOINTS"
  --experiment_summaries_interval=60
  --decorrelate_envs_on_one_worker=False
  --set_workers_cpu_affinity=False
  --force_envs_single_thread=True
  --train_in_background_thread="$TRAIN_IN_BACKGROUND_THREAD"
  --learner_main_loop_num_cores="$LEARNER_MAIN_LOOP_NUM_CORES"
  --traj_buffers_excess_ratio="$TRAJ_BUFFERS_EXCESS_RATIO"
  --with_vtrace=True
  --use_rnn=True
  --rnn_type=lstm
  --hidden_size="$HIDDEN_SIZE"
  --learning_rate="$LEARNING_RATE"
  --gamma="$GAMMA"
  --reward_scale="$REWARD_SCALE"
  --reward_clip="$REWARD_CLIP"
  --exploration_loss_coeff="$EXPLORATION_LOSS_COEFF"
  --max_policy_lag="$MAX_POLICY_LAG"
  --maca_opponent="$OPPONENT"
  --maca_max_step="$MAX_STEP"
  --maca_render=False
)

LOG_FILE="log/${EXP_NAME}.launcher.log"

echo "Launching 4080 fresh-start run: $EXP_NAME"
echo "Log: $LOG_FILE"
printf 'Command: '
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
