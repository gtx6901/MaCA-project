#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-sf_maca_decoupled_8h_${RUN_ID}}"
TRAIN_DIR="${TRAIN_DIR:-train_dir/sample_factory}"
ENV_NAME="${ENV_NAME:-maca_aircombat}"
MAX_STEP="${MAX_STEP:-1000}"
SEED="${SEED:-1}"
DEVICE="${DEVICE:-gpu}"

# About 8h total:
# warmup 45m (no_att) + 6h pulse blocks + final 75m fix_rule consolidation.
PHASE1_SECONDS="${PHASE1_SECONDS:-2700}"
PHASE1_OPPONENT="${PHASE1_OPPONENT:-fix_rule_no_att}"
PHASE2_CYCLES="${PHASE2_CYCLES:-6}"
PHASE2_PULSE_SECONDS="${PHASE2_PULSE_SECONDS:-600}"
PHASE2_MAIN_SECONDS="${PHASE2_MAIN_SECONDS:-3000}"
PHASE3_SECONDS="${PHASE3_SECONDS:-4500}"
PHASE3_OPPONENT="${PHASE3_OPPONENT:-fix_rule}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-180000000}"
MIN_PHASE_ENV_STEP_HEADROOM="${MIN_PHASE_ENV_STEP_HEADROOM:-500000}"

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
MAX_POLICY_LAG="${MAX_POLICY_LAG:-12}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-12}"
SAVE_EVERY_SEC="${SAVE_EVERY_SEC:-900}"
TRAIN_IN_BACKGROUND_THREAD="${TRAIN_IN_BACKGROUND_THREAD:-False}"
LEARNER_MAIN_LOOP_NUM_CORES="${LEARNER_MAIN_LOOP_NUM_CORES:-3}"
TRAJ_BUFFERS_EXCESS_RATIO="${TRAJ_BUFFERS_EXCESS_RATIO:-4.0}"

PHASE1_EXPLORATION="${PHASE1_EXPLORATION:-0.07}"
PHASE2_PULSE_EXPLORATION="${PHASE2_PULSE_EXPLORATION:-0.06}"
PHASE2_MAIN_EXPLORATION="${PHASE2_MAIN_EXPLORATION:-0.045}"
PHASE3_EXPLORATION="${PHASE3_EXPLORATION:-0.04}"

FRESH_START="${FRESH_START:-1}"
MACA_EXTENDED_OBSERVATION="${MACA_EXTENDED_OBSERVATION:-True}"
MACA_DECOUPLED_ACTION_HEADS="${MACA_DECOUPLED_ACTION_HEADS:-True}"
MACA_ADAPTIVE_SUPPORT_POLICY="${MACA_ADAPTIVE_SUPPORT_POLICY:-False}"
MACA_SUPPORT_SEARCH_HOLD="${MACA_SUPPORT_SEARCH_HOLD:-6}"
MACA_RADAR_TRACKING_OBSERVATION="${MACA_RADAR_TRACKING_OBSERVATION:-False}"
MACA_TRACK_MEMORY_STEPS="${MACA_TRACK_MEMORY_STEPS:-12}"
MACA_SEMANTIC_SCREEN_OBSERVATION="${MACA_SEMANTIC_SCREEN_OBSERVATION:-False}"
MACA_SCREEN_TRACK_MEMORY_STEPS="${MACA_SCREEN_TRACK_MEMORY_STEPS:-12}"
MACA_RELATIVE_MOTION_OBSERVATION="${MACA_RELATIVE_MOTION_OBSERVATION:-False}"
MACA_DELTA_COURSE_ACTION="${MACA_DELTA_COURSE_ACTION:-False}"
MACA_COURSE_DELTA_DEG="${MACA_COURSE_DELTA_DEG:-45}"
MACA_TACTICAL_MODE_OBSERVATION="${MACA_TACTICAL_MODE_OBSERVATION:-False}"
MACA_COURSE_PRIOR_OBSERVATION="${MACA_COURSE_PRIOR_OBSERVATION:-False}"
MACA_COURSE_PRIOR_STRENGTH="${MACA_COURSE_PRIOR_STRENGTH:-0.0}"
MACA_COURSE_HOLD_STEPS="${MACA_COURSE_HOLD_STEPS:-1}"
MACA_MAX_COURSE_CHANGE_BINS="${MACA_MAX_COURSE_CHANGE_BINS:-15}"
MACA_LOCK_STATE_OBSERVATION="${MACA_LOCK_STATE_OBSERVATION:-False}"
MACA_TEAM_STATUS_OBSERVATION="${MACA_TEAM_STATUS_OBSERVATION:-False}"
MACA_THREAT_STATE_OBSERVATION="${MACA_THREAT_STATE_OBSERVATION:-False}"
MACA_INTERCEPT_COURSE_ASSIST="${MACA_INTERCEPT_COURSE_ASSIST:-False}"
MACA_INTERCEPT_COURSE_BLEND="${MACA_INTERCEPT_COURSE_BLEND:-0.0}"
MACA_INTERCEPT_BREAK_HOLD_BINS="${MACA_INTERCEPT_BREAK_HOLD_BINS:-0}"
MACA_INTERCEPT_LEAD_DEG="${MACA_INTERCEPT_LEAD_DEG:-20}"
MACA_COMMIT_DISTANCE="${MACA_COMMIT_DISTANCE:-140}"
MACA_ATTACK_PRIOR_STRENGTH="${MACA_ATTACK_PRIOR_STRENGTH:-0.0}"

mkdir -p log "$TRAIN_DIR"

if [[ "$FRESH_START" == "1" && -d "$TRAIN_DIR/$EXP_NAME" ]]; then
  echo "Fresh start enabled, removing existing experiment dir: $TRAIN_DIR/$EXP_NAME"
  rm -rf "$TRAIN_DIR/$EXP_NAME"
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment${PYTHONPATH:+:$PYTHONPATH}"
export MACA_ENABLE_SF_BUFFER_SQUEEZE_PATCH="${MACA_ENABLE_SF_BUFFER_SQUEEZE_PATCH:-0}"
export MACA_REWARD_RADAR_FIGHTER_DETECTOR="${MACA_REWARD_RADAR_FIGHTER_DETECTOR:-0}"
export MACA_REWARD_RADAR_FIGHTER_FIGHTER="${MACA_REWARD_RADAR_FIGHTER_FIGHTER:-0}"
export MACA_REWARD_STRIKE_FIGHTER_SUCCESS="${MACA_REWARD_STRIKE_FIGHTER_SUCCESS:-900}"
export MACA_REWARD_STRIKE_FIGHTER_FAIL="${MACA_REWARD_STRIKE_FIGHTER_FAIL:--6}"
export MACA_REWARD_STRIKE_ACT_VALID="${MACA_REWARD_STRIKE_ACT_VALID:-0}"
export MACA_REWARD_KEEP_ALIVE_STEP="${MACA_REWARD_KEEP_ALIVE_STEP:--1}"
export MACA_REWARD_DRAW="${MACA_REWARD_DRAW:--1500}"
export MACA_MISSED_ATTACK_PENALTY="${MACA_MISSED_ATTACK_PENALTY:-0}"
export MACA_FIRE_LOGIT_BIAS="${MACA_FIRE_LOGIT_BIAS:-0.0}"
export MACA_FIRE_PROB_FLOOR="${MACA_FIRE_PROB_FLOOR:-0.0}"
export MACA_EVAL_FIRE_PROB_FLOOR="${MACA_EVAL_FIRE_PROB_FLOOR:-0.0}"
export MACA_CONTACT_REWARD="${MACA_CONTACT_REWARD:-0}"
export MACA_PROGRESS_REWARD_SCALE="${MACA_PROGRESS_REWARD_SCALE:-0}"
export MACA_PROGRESS_REWARD_CAP="${MACA_PROGRESS_REWARD_CAP:-20}"
export MACA_ATTACK_WINDOW_REWARD="${MACA_ATTACK_WINDOW_REWARD:-0}"
export MACA_FRIENDLY_ATTRITION_PENALTY="${MACA_FRIENDLY_ATTRITION_PENALTY:-0}"
export MACA_ENEMY_ATTRITION_REWARD="${MACA_ENEMY_ATTRITION_REWARD:-0}"
export MACA_ATTACK_PRIOR_STRENGTH

latest_env_steps() {
  local checkpoint_dir="${TRAIN_DIR}/${EXP_NAME}/checkpoint_p0"
  local latest_checkpoint

  if [[ ! -d "$checkpoint_dir" ]]; then
    echo 0
    return
  fi

  latest_checkpoint="$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'checkpoint_*.pth' | sort | tail -n 1)"
  if [[ -z "$latest_checkpoint" ]]; then
    echo 0
    return
  fi

  local checkpoint_name
  checkpoint_name="$(basename "$latest_checkpoint")"
  if [[ "$checkpoint_name" =~ ^checkpoint_[0-9]+_([0-9]+)\.pth$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi

  echo 0
}

run_phase() {
  local phase_tag="$1"
  local opponent="$2"
  local cumulative_train_seconds="$3"
  local exploration="$4"
  local log_file="log/${EXP_NAME}.${phase_tag}.log"
  local current_env_steps
  current_env_steps="$(latest_env_steps)"
  local headroom=$((TOTAL_ENV_STEPS - current_env_steps))

  if (( headroom < MIN_PHASE_ENV_STEP_HEADROOM )); then
    echo "Skipping ${phase_tag}: insufficient env-step headroom (${headroom} < ${MIN_PHASE_ENV_STEP_HEADROOM}, current=${current_env_steps}, cap=${TOTAL_ENV_STEPS})"
    return 2
  fi

  local -a cmd=(
    conda run --no-capture-output -n maca-py37-min python scripts/train_sf_maca.py
    --algo=APPO
    --env="$ENV_NAME"
    --experiment="$EXP_NAME"
    --train_dir="$TRAIN_DIR"
    --device="$DEVICE"
    --seed="$SEED"
    --num_workers="$NUM_WORKERS"
    --num_envs_per_worker="$NUM_ENVS_PER_WORKER"
    --worker_num_splits=1
    --rollout="$ROLLOUT"
    --recurrence="$RECURRENCE"
    --batch_size="$BATCH_SIZE"
    --num_batches_per_iteration=1
    --num_minibatches_to_accumulate=1
    --train_for_seconds="$cumulative_train_seconds"
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
    --maca_extended_observation="$MACA_EXTENDED_OBSERVATION"
    --maca_decoupled_action_heads="$MACA_DECOUPLED_ACTION_HEADS"
    --maca_adaptive_support_policy="$MACA_ADAPTIVE_SUPPORT_POLICY"
    --maca_support_search_hold="$MACA_SUPPORT_SEARCH_HOLD"
    --maca_radar_tracking_observation="$MACA_RADAR_TRACKING_OBSERVATION"
    --maca_track_memory_steps="$MACA_TRACK_MEMORY_STEPS"
    --maca_semantic_screen_observation="$MACA_SEMANTIC_SCREEN_OBSERVATION"
    --maca_screen_track_memory_steps="$MACA_SCREEN_TRACK_MEMORY_STEPS"
    --maca_relative_motion_observation="$MACA_RELATIVE_MOTION_OBSERVATION"
    --maca_delta_course_action="$MACA_DELTA_COURSE_ACTION"
    --maca_course_delta_deg="$MACA_COURSE_DELTA_DEG"
    --maca_tactical_mode_observation="$MACA_TACTICAL_MODE_OBSERVATION"
    --maca_course_prior_observation="$MACA_COURSE_PRIOR_OBSERVATION"
    --maca_course_prior_strength="$MACA_COURSE_PRIOR_STRENGTH"
    --maca_course_hold_steps="$MACA_COURSE_HOLD_STEPS"
    --maca_max_course_change_bins="$MACA_MAX_COURSE_CHANGE_BINS"
    --maca_lock_state_observation="$MACA_LOCK_STATE_OBSERVATION"
    --maca_team_status_observation="$MACA_TEAM_STATUS_OBSERVATION"
    --maca_threat_state_observation="$MACA_THREAT_STATE_OBSERVATION"
    --maca_intercept_course_assist="$MACA_INTERCEPT_COURSE_ASSIST"
    --maca_intercept_course_blend="$MACA_INTERCEPT_COURSE_BLEND"
    --maca_intercept_break_hold_bins="$MACA_INTERCEPT_BREAK_HOLD_BINS"
    --maca_intercept_lead_deg="$MACA_INTERCEPT_LEAD_DEG"
    --maca_commit_distance="$MACA_COMMIT_DISTANCE"
    --maca_attack_prior_strength="$MACA_ATTACK_PRIOR_STRENGTH"
  )

  echo "=== ${phase_tag} | opponent=${opponent} | train_for_seconds=${cumulative_train_seconds} | exploration=${exploration} ==="
  echo "Log: ${log_file}"
  printf 'Command: '
  printf '%q ' "${cmd[@]}"
  echo

  "${cmd[@]}" 2>&1 | tee "$log_file"
}

remove_done_if_exists() {
  local done_file="${TRAIN_DIR}/${EXP_NAME}/done"
  if [[ -f "$done_file" ]]; then
    echo "Removing done file: $done_file"
    rm -f "$done_file"
  fi
}

current_target_seconds=0

current_target_seconds=$((current_target_seconds + PHASE1_SECONDS))
run_phase "phase1_noatt_warmup" "$PHASE1_OPPONENT" "$current_target_seconds" "$PHASE1_EXPLORATION" || exit $?
remove_done_if_exists

for ((cycle=1; cycle<=PHASE2_CYCLES; cycle++)); do
  current_target_seconds=$((current_target_seconds + PHASE2_PULSE_SECONDS))
  run_phase "phase2_pulse_noatt_c${cycle}" "$PHASE1_OPPONENT" "$current_target_seconds" "$PHASE2_PULSE_EXPLORATION" || exit $?
  remove_done_if_exists

  current_target_seconds=$((current_target_seconds + PHASE2_MAIN_SECONDS))
  run_phase "phase2_fixrule_c${cycle}" "fix_rule" "$current_target_seconds" "$PHASE2_MAIN_EXPLORATION" || exit $?
  remove_done_if_exists
done

current_target_seconds=$((current_target_seconds + PHASE3_SECONDS))
run_phase "phase3_fixrule_consolidate" "$PHASE3_OPPONENT" "$current_target_seconds" "$PHASE3_EXPLORATION" || exit $?

echo "Decoupled 8h curriculum finished for experiment: $EXP_NAME"
