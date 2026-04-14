#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-sf_maca_recovery_progress_gate_${RUN_ID}}"
TRAIN_DIR="${TRAIN_DIR:-train_dir/sample_factory}"
ENV_NAME="${ENV_NAME:-maca_aircombat}"
OPPONENT="${OPPONENT:-fix_rule}"
MAX_STEP="${MAX_STEP:-1000}"
SEED="${SEED:-1}"

# Default gate: 4h fresh start against the real target.
TRAIN_SECONDS="${TRAIN_SECONDS:-14400}"
TRAIN_ENV_STEPS="${TRAIN_ENV_STEPS:-120000000}"
NUM_WORKERS="${NUM_WORKERS:-10}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-1}"
ROLLOUT="${ROLLOUT:-64}"
RECURRENCE="${RECURRENCE:-64}"
BATCH_SIZE="${BATCH_SIZE:-6400}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GAMMA="${GAMMA:-0.999}"
REWARD_SCALE="${REWARD_SCALE:-0.002}"
REWARD_CLIP="${REWARD_CLIP:-50.0}"
EXPLORATION_LOSS_COEFF="${EXPLORATION_LOSS_COEFF:-0.06}"
MAX_POLICY_LAG="${MAX_POLICY_LAG:-12}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-12}"
SAVE_EVERY_SEC="${SAVE_EVERY_SEC:-900}"
TRAIN_IN_BACKGROUND_THREAD="${TRAIN_IN_BACKGROUND_THREAD:-False}"
LEARNER_MAIN_LOOP_NUM_CORES="${LEARNER_MAIN_LOOP_NUM_CORES:-3}"
TRAJ_BUFFERS_EXCESS_RATIO="${TRAJ_BUFFERS_EXCESS_RATIO:-4.0}"
MACA_EXTENDED_OBSERVATION="${MACA_EXTENDED_OBSERVATION:-True}"

mkdir -p log "$TRAIN_DIR"

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

# Structural rescue defaults: reward contact acquisition, closing distance,
# and first entry into a valid attack envelope.
export MACA_CONTACT_REWARD="${MACA_CONTACT_REWARD:-20}"
export MACA_PROGRESS_REWARD_SCALE="${MACA_PROGRESS_REWARD_SCALE:-0.5}"
export MACA_PROGRESS_REWARD_CAP="${MACA_PROGRESS_REWARD_CAP:-20}"
export MACA_ATTACK_WINDOW_REWARD="${MACA_ATTACK_WINDOW_REWARD:-30}"

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
  --maca_extended_observation="$MACA_EXTENDED_OBSERVATION"
)

LOG_FILE="log/${EXP_NAME}.fresh_gate.log"

echo "Launching fresh gate run: $EXP_NAME"
echo "Log: $LOG_FILE"
printf 'Command: '
printf '%q ' "${CMD[@]}"
echo
echo "Runtime tweaks:"
env | grep '^MACA_' | sort

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
