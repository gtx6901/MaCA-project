#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-sf_maca_recovery_${RUN_ID}}"
TRAIN_DIR="${TRAIN_DIR:-train_dir/sample_factory}"
ENV_NAME="${ENV_NAME:-maca_aircombat}"
MAX_STEP="${MAX_STEP:-1000}"
SEED="${SEED:-1}"

# Phase-1: build proactive attack habit against easier opponent
PHASE1_SECONDS="${PHASE1_SECONDS:-5400}"
PHASE1_OPPONENT="${PHASE1_OPPONENT:-fix_rule_no_att}"

# Phase-2: fine-tune against target opponent
PHASE2_SECONDS="${PHASE2_SECONDS:-16200}"
PHASE2_OPPONENT="${PHASE2_OPPONENT:-fix_rule}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-100000000}"

NUM_WORKERS="${NUM_WORKERS:-10}"
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
MAX_POLICY_LAG="${MAX_POLICY_LAG:-15}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-12}"
SAVE_EVERY_SEC="${SAVE_EVERY_SEC:-900}"
TRAIN_IN_BACKGROUND_THREAD="${TRAIN_IN_BACKGROUND_THREAD:-False}"
LEARNER_MAIN_LOOP_NUM_CORES="${LEARNER_MAIN_LOOP_NUM_CORES:-3}"
TRAJ_BUFFERS_EXCESS_RATIO="${TRAJ_BUFFERS_EXCESS_RATIO:-4.0}"

# phase-specific entropy to avoid collapse in early stage
PHASE1_EXPLORATION="${PHASE1_EXPLORATION:-0.06}"
PHASE2_EXPLORATION="${PHASE2_EXPLORATION:-0.03}"

FRESH_START="${FRESH_START:-1}"

mkdir -p log "$TRAIN_DIR"

if [[ "$FRESH_START" == "1" && -d "$TRAIN_DIR/$EXP_NAME" ]]; then
  echo "Fresh start enabled, removing existing experiment dir: $TRAIN_DIR/$EXP_NAME"
  rm -rf "$TRAIN_DIR/$EXP_NAME"
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment${PYTHONPATH:+:$PYTHONPATH}"

run_phase() {
  local phase_tag="$1"
  local opponent="$2"
  local train_seconds="$3"
  local exploration="$4"
  local log_file="log/${EXP_NAME}.${phase_tag}.log"

  local -a cmd=(
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
    --train_for_seconds="$train_seconds"
    --train_for_env_steps="$TOTAL_ENV_STEPS"
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
    --exploration_loss_coeff="$exploration"
    --max_policy_lag="$MAX_POLICY_LAG"
    --maca_opponent="$opponent"
    --maca_max_step="$MAX_STEP"
    --maca_render=False
  )

  echo "=== ${phase_tag} | opponent=${opponent} | train_for_seconds=${train_seconds} | exploration=${exploration} ==="
  echo "Log: ${log_file}"
  printf 'Command: '
  printf '%q ' "${cmd[@]}"
  echo

  "${cmd[@]}" 2>&1 | tee "$log_file"
}

# Phase 1: easier opponent, stronger entropy
run_phase "phase1_no_att" "$PHASE1_OPPONENT" "$PHASE1_SECONDS" "$PHASE1_EXPLORATION"

# Phase 2: target opponent, reduced entropy for consolidation
run_phase "phase2_fixrule" "$PHASE2_OPPONENT" "$PHASE2_SECONDS" "$PHASE2_EXPLORATION"

echo "Recovery curriculum finished for experiment: $EXP_NAME"
