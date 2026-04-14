#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXP_NAME="${EXP_NAME:-sf_maca_recovery_20260412_163435}"
TRAIN_DIR="${TRAIN_DIR:-train_dir/sample_factory}"
EXP_DIR="${TRAIN_DIR}/${EXP_NAME}"
CFG_PATH="${EXP_DIR}/cfg.json"
CHECKPOINT_DIR="${EXP_DIR}/checkpoint_p0"
DONE_FILE="${EXP_DIR}/done"

RUN_TAG="${RUN_TAG:-${EXP_NAME}_validation_fixrule}"
MASTER_LOG="log/${RUN_TAG}.master.log"
TRAIN_LOG="log/${RUN_TAG}.train.log"
MONITOR_LOG="log/${RUN_TAG}.monitor.log"
POST_FIXRULE_JSON="log/${RUN_TAG}.eval20.fix_rule.json"
POST_NOATT_JSON="log/${RUN_TAG}.eval20.fix_rule_no_att.json"

VALIDATION_DURATION_SECONDS="${VALIDATION_DURATION_SECONDS:-18000}"
POST_EVAL_EPISODES="${POST_EVAL_EPISODES:-20}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-3600}"
MONITOR_EVAL_EPISODES="${MONITOR_EVAL_EPISODES:-5}"

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
MAX_POLICY_LAG="${MAX_POLICY_LAG:-15}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-12}"
SAVE_EVERY_SEC="${SAVE_EVERY_SEC:-900}"
TRAIN_IN_BACKGROUND_THREAD="${TRAIN_IN_BACKGROUND_THREAD:-False}"
LEARNER_MAIN_LOOP_NUM_CORES="${LEARNER_MAIN_LOOP_NUM_CORES:-3}"
TRAJ_BUFFERS_EXCESS_RATIO="${TRAJ_BUFFERS_EXCESS_RATIO:-4.0}"
TRAIN_FOR_ENV_STEPS="${TRAIN_FOR_ENV_STEPS:-290000000}"
MIN_ENV_STEP_DELTA="${MIN_ENV_STEP_DELTA:-50000}"
REQUIRED_ENV_STEPS_HEADROOM="${REQUIRED_ENV_STEPS_HEADROOM:-30000000}"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
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
export MACA_AUTO_EVAL_LOW_FIRE_THRESHOLD="${MACA_AUTO_EVAL_LOW_FIRE_THRESHOLD:-0.003}"
export MACA_AUTO_EVAL_MAX_LOW_FIRE_STREAK="${MACA_AUTO_EVAL_MAX_LOW_FIRE_STREAK:-4}"
export MACA_AUTO_EVAL_MIN_FIRE_OPP_RATIO="${MACA_AUTO_EVAL_MIN_FIRE_OPP_RATIO:-0.01}"
export MACA_AUTO_EVAL_MAX_LOW_RATIO_STREAK="${MACA_AUTO_EVAL_MAX_LOW_RATIO_STREAK:-4}"
export MACA_AUTO_EVAL_MIN_OPPORTUNITY_FOR_RATIO="${MACA_AUTO_EVAL_MIN_OPPORTUNITY_FOR_RATIO:-0.35}"
export MACA_AUTO_EVAL_ANOMALY_FIRE_THRESHOLD="${MACA_AUTO_EVAL_ANOMALY_FIRE_THRESHOLD:-0.003}"

mkdir -p log

log_msg() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$MASTER_LOG"
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    log_msg "fatal missing_path=${path}"
    exit 1
  fi
}

latest_checkpoint_path() {
  python3 - "$CHECKPOINT_DIR" <<'PY'
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
candidates = []
for ckpt in ckpt_dir.glob("checkpoint_*.pth"):
    try:
        stat = ckpt.stat()
    except FileNotFoundError:
        continue
    if stat.st_size <= 0:
        continue
    candidates.append((stat.st_mtime, ckpt))
if not candidates:
    print("")
    raise SystemExit(0)
candidates.sort(key=lambda item: item[0], reverse=True)
print(candidates[0][1])
PY
}

checkpoint_env_steps() {
  local checkpoint_path="$1"
  python3 - "$checkpoint_path" <<'PY'
import re
import sys
from pathlib import Path

name = Path(sys.argv[1]).name
match = re.match(r"checkpoint_\d+_(\d+)\.pth$", name)
print(match.group(1) if match else "")
PY
}

target_train_seconds() {
  python3 - "$CFG_PATH" "$VALIDATION_DURATION_SECONDS" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = json.load(handle)
base_seconds = int(cfg.get("train_for_seconds", 0))
duration = int(sys.argv[2])
target = max(base_seconds, 24000) + duration
print(target)
PY
}

eval_summary_line() {
  local json_path="$1"
  python3 - "$json_path" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    summary = json.load(handle)["summary"]
parts = [
    f"episodes={summary.get('episodes')}",
    f"win_rate={summary.get('win_rate')}",
    f"round_reward_mean={summary.get('round_reward_mean')}",
    f"opponent_round_reward_mean={summary.get('opponent_round_reward_mean')}",
    f"true_reward_mean={summary.get('true_reward_mean')}",
    f"fire_action_frac_mean={summary.get('fire_action_frac_mean')}",
    f"attack_opportunity_frac_mean={summary.get('attack_opportunity_frac_mean')}",
    f"missed_attack_frac_mean={summary.get('missed_attack_frac_mean')}",
    f"episode_len_mean={summary.get('episode_len_mean')}",
]
print(" ".join(parts))
PY
}

run_eval() {
  local opponent="$1"
  local output_json="$2"
  conda run --no-capture-output -n maca-py37-min python scripts/eval_sf_maca.py \
    --experiment="$EXP_NAME" \
    --train_dir="$TRAIN_DIR" \
    --episodes="$POST_EVAL_EPISODES" \
    --device=cpu \
    --maca_opponent="$opponent" \
    --output_json="$output_json" >> "$MASTER_LOG" 2>&1
}

require_path "$EXP_DIR"
require_path "$CFG_PATH"
require_path "$CHECKPOINT_DIR"

if [[ -f "$DONE_FILE" ]]; then
  log_msg "removing_done_file path=${DONE_FILE}"
  rm -f "$DONE_FILE"
fi

RESUME_CHECKPOINT="$(latest_checkpoint_path)"
if [[ -z "$RESUME_CHECKPOINT" ]]; then
  log_msg "fatal reason=no_checkpoint_available"
  exit 1
fi

START_ENV_STEPS="$(checkpoint_env_steps "$RESUME_CHECKPOINT")"
if [[ -z "$START_ENV_STEPS" ]]; then
  log_msg "fatal reason=unparseable_checkpoint_env_steps checkpoint=${RESUME_CHECKPOINT}"
  exit 1
fi

HEADROOM=$((TRAIN_FOR_ENV_STEPS - START_ENV_STEPS))
if (( HEADROOM < REQUIRED_ENV_STEPS_HEADROOM )); then
  log_msg "fatal reason=insufficient_env_step_headroom current_env_steps=${START_ENV_STEPS} train_for_env_steps=${TRAIN_FOR_ENV_STEPS} required_headroom=${REQUIRED_ENV_STEPS_HEADROOM} actual_headroom=${HEADROOM}"
  exit 1
fi

TARGET_TRAIN_SECONDS="$(target_train_seconds)"

: > "$MASTER_LOG"
: > "$TRAIN_LOG"
: > "$MONITOR_LOG"

log_msg "validation_start exp=${EXP_NAME} resume_checkpoint=${RESUME_CHECKPOINT}"
log_msg "validation_targets train_for_seconds=${TARGET_TRAIN_SECONDS} train_for_env_steps=${TRAIN_FOR_ENV_STEPS} start_env_steps=${START_ENV_STEPS} duration_seconds=${VALIDATION_DURATION_SECONDS}"
log_msg "runtime_tweaks radar_ff=${MACA_REWARD_RADAR_FIGHTER_FIGHTER} radar_fd=${MACA_REWARD_RADAR_FIGHTER_DETECTOR} hit=${MACA_REWARD_STRIKE_FIGHTER_SUCCESS} miss=${MACA_REWARD_STRIKE_FIGHTER_FAIL} valid_fire=${MACA_REWARD_STRIKE_ACT_VALID} keep_alive=${MACA_REWARD_KEEP_ALIVE_STEP} draw=${MACA_REWARD_DRAW} missed_attack_penalty=${MACA_MISSED_ATTACK_PENALTY} fire_logit_bias=${MACA_FIRE_LOGIT_BIAS} fire_prob_floor=${MACA_FIRE_PROB_FLOOR} eval_fire_prob_floor=${MACA_EVAL_FIRE_PROB_FLOOR}"
log_msg "monitor_guards low_fire_threshold=${MACA_AUTO_EVAL_LOW_FIRE_THRESHOLD} low_fire_max_streak=${MACA_AUTO_EVAL_MAX_LOW_FIRE_STREAK} min_fire_opp_ratio=${MACA_AUTO_EVAL_MIN_FIRE_OPP_RATIO} low_ratio_max_streak=${MACA_AUTO_EVAL_MAX_LOW_RATIO_STREAK} min_opp_for_ratio=${MACA_AUTO_EVAL_MIN_OPPORTUNITY_FOR_RATIO} anomaly_fire_threshold=${MACA_AUTO_EVAL_ANOMALY_FIRE_THRESHOLD}"

{
  echo "resume_experiment=${EXP_NAME}"
  echo "resume_checkpoint=${RESUME_CHECKPOINT}"
  echo "opponent=fix_rule"
  echo "train_for_seconds=${TARGET_TRAIN_SECONDS}"
  echo "train_for_env_steps=${TRAIN_FOR_ENV_STEPS}"
  echo "command=conda run --no-capture-output -n maca-py37-min python scripts/train_sf_maca.py --algo=APPO --env=maca_aircombat --experiment=${EXP_NAME} --train_dir=${TRAIN_DIR} --device=gpu --seed=1 --num_workers=${NUM_WORKERS} --num_envs_per_worker=${NUM_ENVS_PER_WORKER} --worker_num_splits=1 --rollout=${ROLLOUT} --recurrence=${RECURRENCE} --batch_size=${BATCH_SIZE} --num_batches_per_iteration=1 --num_minibatches_to_accumulate=1 --train_for_seconds=${TARGET_TRAIN_SECONDS} --train_for_env_steps=${TRAIN_FOR_ENV_STEPS} --ppo_epochs=${PPO_EPOCHS} --save_every_sec=${SAVE_EVERY_SEC} --keep_checkpoints=${KEEP_CHECKPOINTS} --experiment_summaries_interval=60 --decorrelate_envs_on_one_worker=False --set_workers_cpu_affinity=False --force_envs_single_thread=True --train_in_background_thread=${TRAIN_IN_BACKGROUND_THREAD} --learner_main_loop_num_cores=${LEARNER_MAIN_LOOP_NUM_CORES} --traj_buffers_excess_ratio=${TRAJ_BUFFERS_EXCESS_RATIO} --with_vtrace=True --use_rnn=True --rnn_type=lstm --hidden_size=${HIDDEN_SIZE} --learning_rate=${LEARNING_RATE} --gamma=${GAMMA} --reward_scale=${REWARD_SCALE} --reward_clip=${REWARD_CLIP} --exploration_loss_coeff=${EXPLORATION_LOSS_COEFF} --max_policy_lag=${MAX_POLICY_LAG} --maca_opponent=fix_rule --maca_max_step=1000 --maca_render=False"
  conda run --no-capture-output -n maca-py37-min python scripts/train_sf_maca.py \
    --algo=APPO \
    --env=maca_aircombat \
    --experiment="$EXP_NAME" \
    --train_dir="$TRAIN_DIR" \
    --device=gpu \
    --seed=1 \
    --num_workers="$NUM_WORKERS" \
    --num_envs_per_worker="$NUM_ENVS_PER_WORKER" \
    --worker_num_splits=1 \
    --rollout="$ROLLOUT" \
    --recurrence="$RECURRENCE" \
    --batch_size="$BATCH_SIZE" \
    --num_batches_per_iteration=1 \
    --num_minibatches_to_accumulate=1 \
    --train_for_seconds="$TARGET_TRAIN_SECONDS" \
    --train_for_env_steps="$TRAIN_FOR_ENV_STEPS" \
    --ppo_epochs="$PPO_EPOCHS" \
    --save_every_sec="$SAVE_EVERY_SEC" \
    --keep_checkpoints="$KEEP_CHECKPOINTS" \
    --experiment_summaries_interval=60 \
    --decorrelate_envs_on_one_worker=False \
    --set_workers_cpu_affinity=False \
    --force_envs_single_thread=True \
    --train_in_background_thread="$TRAIN_IN_BACKGROUND_THREAD" \
    --learner_main_loop_num_cores="$LEARNER_MAIN_LOOP_NUM_CORES" \
    --traj_buffers_excess_ratio="$TRAJ_BUFFERS_EXCESS_RATIO" \
    --with_vtrace=True \
    --use_rnn=True \
    --rnn_type=lstm \
    --hidden_size="$HIDDEN_SIZE" \
    --learning_rate="$LEARNING_RATE" \
    --gamma="$GAMMA" \
    --reward_scale="$REWARD_SCALE" \
    --reward_clip="$REWARD_CLIP" \
    --exploration_loss_coeff="$EXPLORATION_LOSS_COEFF" \
    --max_policy_lag="$MAX_POLICY_LAG" \
    --maca_opponent=fix_rule \
    --maca_max_step=1000 \
    --maca_render=False
} >> "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!

bash scripts/auto_eval_during_training.sh \
  "$EXP_NAME" \
  "validation_fixrule" \
  "$TRAIN_PID" \
  "fix_rule" \
  "fix_rule_no_att" \
  "$MONITOR_INTERVAL_SEC" \
  "$MONITOR_LOG" \
  "$MONITOR_EVAL_EPISODES" &
MONITOR_PID=$!

TRAIN_STATUS=0
wait "$TRAIN_PID" || TRAIN_STATUS=$?
wait "$MONITOR_PID" || true

if [[ $TRAIN_STATUS -ne 0 ]]; then
  log_msg "validation_failed exit_code=${TRAIN_STATUS}"
  exit "$TRAIN_STATUS"
fi

END_CHECKPOINT="$(latest_checkpoint_path)"
END_ENV_STEPS="$(checkpoint_env_steps "$END_CHECKPOINT")"
ENV_STEP_DELTA=$((END_ENV_STEPS - START_ENV_STEPS))
if (( ENV_STEP_DELTA < MIN_ENV_STEP_DELTA )); then
  log_msg "validation_failed reason=insufficient_env_progress start_env_steps=${START_ENV_STEPS} end_env_steps=${END_ENV_STEPS} env_step_delta=${ENV_STEP_DELTA} min_required=${MIN_ENV_STEP_DELTA}"
  exit 1
fi

run_eval "fix_rule" "$POST_FIXRULE_JSON"
run_eval "fix_rule_no_att" "$POST_NOATT_JSON"

log_msg "post_eval opponent=fix_rule $(eval_summary_line "$POST_FIXRULE_JSON")"
log_msg "post_eval opponent=fix_rule_no_att $(eval_summary_line "$POST_NOATT_JSON")"
log_msg "validation_finished latest_checkpoint=${END_CHECKPOINT} end_env_steps=${END_ENV_STEPS} env_step_delta=${ENV_STEP_DELTA}"
