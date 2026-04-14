#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXP_NAME="sf_maca_recovery_20260412_163435"
TRAIN_DIR="train_dir/sample_factory"
EXP_DIR="${TRAIN_DIR}/${EXP_NAME}"
CFG_PATH="${EXP_DIR}/cfg.json"
DONE_FILE="${EXP_DIR}/done"
CHECKPOINT_DIR="${EXP_DIR}/checkpoint_p0"
BEST_STAGE_DIR="${EXP_DIR}/best_stage_snapshots"

RUN_TAG="sf_maca_recovery_night"
MASTER_LOG="log/${RUN_TAG}_master.log"
PRE_FIXRULE_JSON="log/${RUN_TAG}_pre_fixrule.eval20.json"
PRE_NOATT_JSON="log/${RUN_TAG}_pre_noatt.eval20.json"
STAGE1_LOG="log/${RUN_TAG}_stage1_noatt.log"
STAGE1_MONITOR_LOG="log/${RUN_TAG}_stage1_noatt.monitor.log"
STAGE1_PRIMARY_JSON="log/${RUN_TAG}_stage1_noatt.eval20.json"
STAGE1_SECONDARY_JSON="log/${RUN_TAG}_stage1_noatt.vs_fix_rule.eval20.json"
STAGE2_LOG="log/${RUN_TAG}_stage2_fixrule.log"
STAGE2_MONITOR_LOG="log/${RUN_TAG}_stage2_fixrule.monitor.log"
STAGE2_PRIMARY_JSON="log/${RUN_TAG}_stage2_fixrule.eval20.json"
STAGE2_SECONDARY_JSON="log/${RUN_TAG}_stage2_fixrule.vs_noatt.eval20.json"
STAGE3_LOG="log/${RUN_TAG}_stage3_fixrule.log"
STAGE3_MONITOR_LOG="log/${RUN_TAG}_stage3_fixrule.monitor.log"
STAGE3_PRIMARY_JSON="log/${RUN_TAG}_stage3_fixrule.eval20.json"
STAGE3_SECONDARY_JSON="log/${RUN_TAG}_stage3_fixrule.vs_noatt.eval20.json"
FINAL_REPORT="log/${RUN_TAG}_final_report.txt"

ORIGINAL_TRAIN_FOR_SECONDS="${ORIGINAL_TRAIN_FOR_SECONDS:-16000}"
STAGE1_DURATION_SECONDS="${STAGE1_DURATION_SECONDS:-7200}"
STAGE2_DURATION_SECONDS="${STAGE2_DURATION_SECONDS:-10800}"
STAGE3_DURATION_SECONDS="${STAGE3_DURATION_SECONDS:-10800}"
STAGE1_TARGET_SECONDS=$((ORIGINAL_TRAIN_FOR_SECONDS + STAGE1_DURATION_SECONDS))
STAGE2_TARGET_SECONDS=$((STAGE1_TARGET_SECONDS + STAGE2_DURATION_SECONDS))
STAGE3_TARGET_SECONDS=$((STAGE2_TARGET_SECONDS + STAGE3_DURATION_SECONDS))

NUM_WORKERS="${NUM_WORKERS:-10}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-1}"
ROLLOUT="${ROLLOUT:-64}"
RECURRENCE="${RECURRENCE:-64}"
BATCH_SIZE="${BATCH_SIZE:-6400}"
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
TRAIN_FOR_ENV_STEPS="${TRAIN_FOR_ENV_STEPS:-250000000}"
MIN_STAGE_ENV_STEP_DELTA="${MIN_STAGE_ENV_STEP_DELTA:-50000}"
REQUIRED_ENV_STEPS_HEADROOM="${REQUIRED_ENV_STEPS_HEADROOM:-100000000}"

STAGE1_EXPLORATION="${STAGE1_EXPLORATION:-0.07}"
STAGE2_EXPLORATION="${STAGE2_EXPLORATION:-0.055}"
STAGE3_EXPLORATION="${STAGE3_EXPLORATION:-0.045}"

MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-1800}"
MONITOR_EVAL_EPISODES="${MONITOR_EVAL_EPISODES:-5}"
POST_STAGE_EVAL_EPISODES="${POST_STAGE_EVAL_EPISODES:-20}"

EXPECTED_RESUME_CHECKPOINT="checkpoint_000047139_72171946.pth"

mkdir -p log "$BEST_STAGE_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MACA_ENABLE_SF_BUFFER_SQUEEZE_PATCH="${MACA_ENABLE_SF_BUFFER_SQUEEZE_PATCH:-0}"
export MACA_REWARD_RADAR_FIGHTER_DETECTOR="${MACA_REWARD_RADAR_FIGHTER_DETECTOR:-2}"
export MACA_REWARD_RADAR_FIGHTER_FIGHTER="${MACA_REWARD_RADAR_FIGHTER_FIGHTER:-2}"
export MACA_REWARD_STRIKE_FIGHTER_SUCCESS="${MACA_REWARD_STRIKE_FIGHTER_SUCCESS:-1200}"
export MACA_REWARD_STRIKE_FIGHTER_FAIL="${MACA_REWARD_STRIKE_FIGHTER_FAIL:--2}"
export MACA_REWARD_STRIKE_ACT_VALID="${MACA_REWARD_STRIKE_ACT_VALID:-10}"
export MACA_REWARD_KEEP_ALIVE_STEP="${MACA_REWARD_KEEP_ALIVE_STEP:--2}"
export MACA_REWARD_DRAW="${MACA_REWARD_DRAW:--2000}"
export MACA_MISSED_ATTACK_PENALTY="${MACA_MISSED_ATTACK_PENALTY:-2}"

backup_if_exists() {
  local path="$1"
  if [[ -f "$path" ]]; then
    mv "$path" "${path}.bak.$(date '+%Y%m%d_%H%M%S')"
  fi
}

for path in \
  "$MASTER_LOG" \
  "$PRE_FIXRULE_JSON" \
  "$PRE_NOATT_JSON" \
  "$STAGE1_LOG" \
  "$STAGE1_MONITOR_LOG" \
  "$STAGE1_PRIMARY_JSON" \
  "$STAGE1_SECONDARY_JSON" \
  "$STAGE2_LOG" \
  "$STAGE2_MONITOR_LOG" \
  "$STAGE2_PRIMARY_JSON" \
  "$STAGE2_SECONDARY_JSON" \
  "$STAGE3_LOG" \
  "$STAGE3_MONITOR_LOG" \
  "$STAGE3_PRIMARY_JSON" \
  "$STAGE3_SECONDARY_JSON" \
  "$FINAL_REPORT"; do
  backup_if_exists "$path"
done

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
  python3 - "$CHECKPOINT_DIR" "${1:-0}" <<'PY'
import sys
import time
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
min_age = int(sys.argv[2])
now = time.time()
candidates = []
for ckpt in ckpt_dir.glob("checkpoint_*.pth"):
    try:
        stat = ckpt.stat()
    except FileNotFoundError:
        continue
    if stat.st_size <= 0:
        continue
    if min_age > 0 and now - stat.st_mtime < min_age:
        continue
    candidates.append((stat.st_mtime, ckpt))

if not candidates:
    print("")
    sys.exit(0)

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
if not match:
    print("")
    sys.exit(0)
print(match.group(1))
PY
}

json_metric() {
  local json_path="$1"
  local metric_name="$2"
  python3 - "$json_path" "$metric_name" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
value = payload["summary"].get(sys.argv[2])
print(value)
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

remove_done_if_exists() {
  if [[ -f "$DONE_FILE" ]]; then
    log_msg "removing_done_file path=${DONE_FILE}"
    rm -f "$DONE_FILE"
  fi
}

save_stage_checkpoint_snapshot() {
  local stage_name="$1"
  local checkpoint_path="$2"
  local snapshot_path="${BEST_STAGE_DIR}/${stage_name}.pth"
  cp -f "$checkpoint_path" "$snapshot_path"
  log_msg "saved_stage_checkpoint stage=${stage_name} path=${snapshot_path}"
}

assert_env_budget() {
  local current_checkpoint="$1"
  local current_env_steps="$2"
  local remaining=$((TRAIN_FOR_ENV_STEPS - current_env_steps))
  if (( remaining <= 0 )); then
    log_msg "fatal reason=env_step_cap_already_reached current_env_steps=${current_env_steps} train_for_env_steps=${TRAIN_FOR_ENV_STEPS} checkpoint=${current_checkpoint}"
    exit 1
  fi
}

assert_total_headroom() {
  local current_checkpoint="$1"
  local current_env_steps="$2"
  local headroom=$((TRAIN_FOR_ENV_STEPS - current_env_steps))
  if (( headroom < REQUIRED_ENV_STEPS_HEADROOM )); then
    log_msg "fatal reason=insufficient_env_step_headroom current_env_steps=${current_env_steps} train_for_env_steps=${TRAIN_FOR_ENV_STEPS} required_headroom=${REQUIRED_ENV_STEPS_HEADROOM} actual_headroom=${headroom} checkpoint=${current_checkpoint}"
    exit 1
  fi
}

run_eval() {
  local opponent="$1"
  local output_json="$2"
  local output_log="$3"

  {
    echo "=== eval opponent=${opponent} output=${output_json} ==="
    conda run --no-capture-output -n maca-py37-min python scripts/eval_sf_maca.py \
      --experiment="$EXP_NAME" \
      --train_dir="$TRAIN_DIR" \
      --episodes="$POST_STAGE_EVAL_EPISODES" \
      --device=cpu \
      --maca_opponent="$opponent" \
      --output_json="$output_json"
  } >> "$output_log" 2>&1
}

run_eval_pair() {
  local stage_name="$1"
  local stage_log="$2"
  local primary_opponent="$3"
  local primary_json="$4"
  local secondary_opponent="$5"
  local secondary_json="$6"

  run_eval "$primary_opponent" "$primary_json" "$stage_log"
  run_eval "$secondary_opponent" "$secondary_json" "$stage_log"

  log_msg "stage_eval_complete stage=${stage_name} opponent=${primary_opponent} $(eval_summary_line "$primary_json")"
  log_msg "stage_eval_complete stage=${stage_name} opponent=${secondary_opponent} $(eval_summary_line "$secondary_json")"
}

run_stage() {
  local stage_name="$1"
  local opponent="$2"
  local exploration="$3"
  local cumulative_train_seconds="$4"
  local stage_log="$5"
  local monitor_log="$6"
  local primary_eval_opponent="$7"
  local primary_eval_json="$8"
  local secondary_eval_opponent="$9"
  local secondary_eval_json="${10}"

  remove_done_if_exists

  local resume_checkpoint
  resume_checkpoint="$(latest_checkpoint_path 0)"
  if [[ -z "$resume_checkpoint" ]]; then
    log_msg "fatal stage=${stage_name} reason=no_checkpoint_available"
    exit 1
  fi
  local start_env_steps
  start_env_steps="$(checkpoint_env_steps "$resume_checkpoint")"
  if [[ -z "$start_env_steps" ]]; then
    log_msg "fatal stage=${stage_name} reason=unparseable_checkpoint_env_steps checkpoint=${resume_checkpoint}"
    exit 1
  fi
  assert_env_budget "$resume_checkpoint" "$start_env_steps"

  log_msg "stage_start name=${stage_name} opponent=${opponent} exploration=${exploration} train_for_seconds=${cumulative_train_seconds} train_for_env_steps=${TRAIN_FOR_ENV_STEPS} start_env_steps=${start_env_steps} resume_checkpoint=${resume_checkpoint}"

  {
    echo "=== stage=${stage_name} start ==="
    echo "resume_experiment=${EXP_NAME}"
    echo "resume_checkpoint=${resume_checkpoint}"
    echo "opponent=${opponent}"
    echo "exploration=${exploration}"
    echo "train_for_seconds=${cumulative_train_seconds}"
    echo "train_for_env_steps=${TRAIN_FOR_ENV_STEPS}"
    echo "done_file=${DONE_FILE}"
    echo "command=conda run --no-capture-output -n maca-py37-min python scripts/train_sf_maca.py --algo=APPO --env=maca_aircombat --experiment=${EXP_NAME} --train_dir=${TRAIN_DIR} --device=gpu --seed=1 --num_workers=${NUM_WORKERS} --num_envs_per_worker=${NUM_ENVS_PER_WORKER} --worker_num_splits=1 --rollout=${ROLLOUT} --recurrence=${RECURRENCE} --batch_size=${BATCH_SIZE} --num_batches_per_iteration=1 --num_minibatches_to_accumulate=1 --train_for_seconds=${cumulative_train_seconds} --train_for_env_steps=${TRAIN_FOR_ENV_STEPS} --ppo_epochs=${PPO_EPOCHS} --save_every_sec=${SAVE_EVERY_SEC} --keep_checkpoints=${KEEP_CHECKPOINTS} --experiment_summaries_interval=60 --decorrelate_envs_on_one_worker=False --set_workers_cpu_affinity=False --force_envs_single_thread=True --train_in_background_thread=${TRAIN_IN_BACKGROUND_THREAD} --learner_main_loop_num_cores=${LEARNER_MAIN_LOOP_NUM_CORES} --traj_buffers_excess_ratio=${TRAJ_BUFFERS_EXCESS_RATIO} --with_vtrace=True --use_rnn=True --rnn_type=lstm --hidden_size=${HIDDEN_SIZE} --learning_rate=${LEARNING_RATE} --gamma=${GAMMA} --reward_scale=${REWARD_SCALE} --reward_clip=${REWARD_CLIP} --exploration_loss_coeff=${exploration} --max_policy_lag=${MAX_POLICY_LAG} --maca_opponent=${opponent} --maca_max_step=1000 --maca_render=False"
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
      --train_for_seconds="$cumulative_train_seconds" \
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
      --exploration_loss_coeff="$exploration" \
      --max_policy_lag="$MAX_POLICY_LAG" \
      --maca_opponent="$opponent" \
      --maca_max_step=1000 \
      --maca_render=False
  } >> "$stage_log" 2>&1 &
  local train_pid=$!

  bash scripts/auto_eval_during_training.sh \
    "$EXP_NAME" \
    "$stage_name" \
    "$train_pid" \
    "$primary_eval_opponent" \
    "$secondary_eval_opponent" \
    "$MONITOR_INTERVAL_SEC" \
    "$monitor_log" \
    "$MONITOR_EVAL_EPISODES" &
  local monitor_pid=$!

  local train_status=0
  wait "$train_pid" || train_status=$?
  wait "$monitor_pid" || true

  if [[ $train_status -ne 0 ]]; then
    log_msg "stage_failed name=${stage_name} exit_code=${train_status}"
    exit "$train_status"
  fi

  if rg -n "out of memory|CUDA error" "$stage_log" >/dev/null 2>&1; then
    log_msg "stage_failed name=${stage_name} reason=gpu_oom_or_cuda_error"
    exit 1
  fi

  local latest_checkpoint
  latest_checkpoint="$(latest_checkpoint_path 0)"
  if [[ -z "$latest_checkpoint" ]]; then
    log_msg "stage_failed name=${stage_name} reason=no_checkpoint_after_stage"
    exit 1
  fi
  local end_env_steps
  end_env_steps="$(checkpoint_env_steps "$latest_checkpoint")"
  if [[ -z "$end_env_steps" ]]; then
    log_msg "stage_failed name=${stage_name} reason=unparseable_checkpoint_env_steps_after_stage checkpoint=${latest_checkpoint}"
    exit 1
  fi
  local env_step_delta=$((end_env_steps - start_env_steps))
  if (( env_step_delta < MIN_STAGE_ENV_STEP_DELTA )); then
    log_msg "stage_failed name=${stage_name} reason=insufficient_stage_env_progress start_env_steps=${start_env_steps} end_env_steps=${end_env_steps} env_step_delta=${env_step_delta} min_required=${MIN_STAGE_ENV_STEP_DELTA}"
    exit 1
  fi

  save_stage_checkpoint_snapshot "$stage_name" "$latest_checkpoint"
  run_eval_pair "$stage_name" "$stage_log" "$primary_eval_opponent" "$primary_eval_json" "$secondary_eval_opponent" "$secondary_eval_json"
  log_msg "stage_finished name=${stage_name} start_env_steps=${start_env_steps} end_env_steps=${end_env_steps} env_step_delta=${env_step_delta} latest_checkpoint=${latest_checkpoint}"
}

require_path "$EXP_DIR"
require_path "$CFG_PATH"
require_path "$CHECKPOINT_DIR"

log_msg "run_start exp=${EXP_NAME}"
log_msg "config cumulative_targets stage1=${STAGE1_TARGET_SECONDS} stage2=${STAGE2_TARGET_SECONDS} stage3=${STAGE3_TARGET_SECONDS}"
log_msg "runtime_tweaks radar_ff=${MACA_REWARD_RADAR_FIGHTER_FIGHTER} radar_fd=${MACA_REWARD_RADAR_FIGHTER_DETECTOR} hit=${MACA_REWARD_STRIKE_FIGHTER_SUCCESS} miss=${MACA_REWARD_STRIKE_FIGHTER_FAIL} valid_fire=${MACA_REWARD_STRIKE_ACT_VALID} keep_alive=${MACA_REWARD_KEEP_ALIVE_STEP} draw=${MACA_REWARD_DRAW} missed_attack_penalty=${MACA_MISSED_ATTACK_PENALTY} buffer_patch=${MACA_ENABLE_SF_BUFFER_SQUEEZE_PATCH}"

current_checkpoint="$(latest_checkpoint_path 0)"
if [[ -z "$current_checkpoint" ]]; then
  log_msg "fatal no_checkpoint_found_under=${CHECKPOINT_DIR}"
  exit 1
fi
current_env_steps="$(checkpoint_env_steps "$current_checkpoint")"
if [[ -z "$current_env_steps" ]]; then
  log_msg "fatal reason=unparseable_initial_checkpoint_env_steps checkpoint=${current_checkpoint}"
  exit 1
fi

if [[ ! -f "${CHECKPOINT_DIR}/${EXPECTED_RESUME_CHECKPOINT}" ]]; then
  log_msg "warning expected_resume_checkpoint_missing expected=${CHECKPOINT_DIR}/${EXPECTED_RESUME_CHECKPOINT}"
fi
assert_total_headroom "$current_checkpoint" "$current_env_steps"
log_msg "resume_checkpoint=${current_checkpoint}"
log_msg "resume_env_steps=${current_env_steps} train_for_env_steps=${TRAIN_FOR_ENV_STEPS} required_headroom=${REQUIRED_ENV_STEPS_HEADROOM} min_stage_env_step_delta=${MIN_STAGE_ENV_STEP_DELTA}"

run_eval_pair "pre_night" "$MASTER_LOG" "fix_rule" "$PRE_FIXRULE_JSON" "fix_rule_no_att" "$PRE_NOATT_JSON"

run_stage \
  "stage1_noatt" \
  "fix_rule_no_att" \
  "$STAGE1_EXPLORATION" \
  "$STAGE1_TARGET_SECONDS" \
  "$STAGE1_LOG" \
  "$STAGE1_MONITOR_LOG" \
  "fix_rule_no_att" \
  "$STAGE1_PRIMARY_JSON" \
  "fix_rule" \
  "$STAGE1_SECONDARY_JSON"

stage1_fire_primary="$(json_metric "$STAGE1_PRIMARY_JSON" "fire_action_frac_mean")"
stage1_fire_secondary="$(json_metric "$STAGE1_SECONDARY_JSON" "fire_action_frac_mean")"
if python3 - "$stage1_fire_primary" "$stage1_fire_secondary" <<'PY'
import sys
values = [float(v) for v in sys.argv[1:] if v not in ("", "None")]
sys.exit(7 if values and max(values) < 0.005 else 0)
PY
then
  :
else
  override_status=$?
  if [[ $override_status -eq 7 ]]; then
  STAGE2_EXPLORATION=0.08
  log_msg "stage2_exploration_override new_value=${STAGE2_EXPLORATION} reason=stage1_fire_action_below_0.005"
  else
    exit "$override_status"
  fi
fi

run_stage \
  "stage2_fixrule" \
  "fix_rule" \
  "$STAGE2_EXPLORATION" \
  "$STAGE2_TARGET_SECONDS" \
  "$STAGE2_LOG" \
  "$STAGE2_MONITOR_LOG" \
  "fix_rule" \
  "$STAGE2_PRIMARY_JSON" \
  "fix_rule_no_att" \
  "$STAGE2_SECONDARY_JSON"

run_stage \
  "stage3_fixrule" \
  "fix_rule" \
  "$STAGE3_EXPLORATION" \
  "$STAGE3_TARGET_SECONDS" \
  "$STAGE3_LOG" \
  "$STAGE3_MONITOR_LOG" \
  "fix_rule" \
  "$STAGE3_PRIMARY_JSON" \
  "fix_rule_no_att" \
  "$STAGE3_SECONDARY_JSON"

python3 scripts/generate_night_report.py --experiment "$EXP_NAME" --output "$FINAL_REPORT" --master-log "$MASTER_LOG" >> "$MASTER_LOG" 2>&1

final_checkpoint="$(latest_checkpoint_path 0)"
log_msg "run_finished exp=${EXP_NAME} latest_checkpoint=${final_checkpoint}"
log_msg "final_report=${FINAL_REPORT}"
