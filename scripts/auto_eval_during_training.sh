#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 7 ]]; then
  echo "usage: $0 <exp_name> <stage_name> <train_pid> <primary_opponent> <secondary_opponent> <interval_sec> <monitor_log> [episodes]"
  exit 2
fi

EXP_NAME="$1"
STAGE_NAME="$2"
TRAIN_PID="$3"
PRIMARY_OPPONENT="$4"
SECONDARY_OPPONENT="$5"
INTERVAL_SEC="$6"
MONITOR_LOG="$7"
EPISODES="${8:-5}"

TRAIN_DIR="train_dir/sample_factory"
EXP_DIR="${TRAIN_DIR}/${EXP_NAME}"
TMP_PREFIX="__tmp_eval_${EXP_NAME}_${STAGE_NAME}"
LOW_FIRE_THRESHOLD="${MACA_AUTO_EVAL_LOW_FIRE_THRESHOLD:-0}"
LOW_FIRE_MAX_STREAK="${MACA_AUTO_EVAL_MAX_LOW_FIRE_STREAK:-0}"
LOW_FIRE_STREAK=0
MIN_FIRE_OPP_RATIO="${MACA_AUTO_EVAL_MIN_FIRE_OPP_RATIO:-0}"
LOW_RATIO_MAX_STREAK="${MACA_AUTO_EVAL_MAX_LOW_RATIO_STREAK:-0}"
MIN_OPPORTUNITY_FOR_RATIO="${MACA_AUTO_EVAL_MIN_OPPORTUNITY_FOR_RATIO:-0.2}"
ANOMALY_FIRE_THRESHOLD="${MACA_AUTO_EVAL_ANOMALY_FIRE_THRESHOLD:-0.001}"
LOW_RATIO_STREAK=0

mkdir -p log
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment${PYTHONPATH:+:$PYTHONPATH}"

log_msg() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$MONITOR_LOG"
}

cleanup_tmp() {
  python3 - "$ROOT_DIR" "$TMP_PREFIX" <<'PY'
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
prefix = sys.argv[2]
base = root / "train_dir" / "sample_factory"
for path in base.glob(f"{prefix}_*"):
    shutil.rmtree(path, ignore_errors=True)
PY
}

make_eval_clone() {
  python3 - "$ROOT_DIR" "$EXP_NAME" "$TMP_PREFIX" <<'PY'
import json
import shutil
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
exp_name = sys.argv[2]
tmp_prefix = sys.argv[3]
exp_dir = root / "train_dir" / "sample_factory" / exp_name
cfg_path = exp_dir / "cfg.json"
ckpt_dir = exp_dir / "checkpoint_p0"
now = time.time()

checkpoints = []
for ckpt in ckpt_dir.glob("checkpoint_*.pth"):
    try:
        stat = ckpt.stat()
    except FileNotFoundError:
        continue
    if stat.st_size <= 0:
        continue
    if now - stat.st_mtime < 120:
        continue
    checkpoints.append((stat.st_mtime, ckpt))

if not checkpoints:
    print("")
    sys.exit(0)

checkpoints.sort(key=lambda item: item[0], reverse=True)
checkpoint = checkpoints[0][1]
temp_exp = f"{tmp_prefix}_{int(now)}"
temp_dir = root / "train_dir" / "sample_factory" / temp_exp
temp_ckpt_dir = temp_dir / "checkpoint_p0"
temp_ckpt_dir.mkdir(parents=True, exist_ok=False)

with cfg_path.open("r", encoding="utf-8") as handle:
    cfg = json.load(handle)
cfg["experiment"] = temp_exp

with (temp_dir / "cfg.json").open("w", encoding="utf-8") as handle:
    json.dump(cfg, handle, ensure_ascii=False, indent=2)

git_diff = exp_dir / "git.diff"
if git_diff.exists():
    shutil.copy2(git_diff, temp_dir / "git.diff")

shutil.copy2(checkpoint, temp_ckpt_dir / checkpoint.name)
print(temp_exp)
PY
}

read_metric() {
  local json_path="$1"
  local metric_name="$2"
  python3 - "$json_path" "$metric_name" <<'PY'
import json
import math
import sys

path = sys.argv[1]
metric = sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
value = payload["summary"].get(metric)
if isinstance(value, float) and math.isnan(value):
    print("nan")
else:
    print(value)
PY
}

summarize_eval() {
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
    f"executed_fire_action_frac_mean={summary.get('executed_fire_action_frac_mean')}",
    f"attack_opportunity_frac_mean={summary.get('attack_opportunity_frac_mean')}",
    f"missed_attack_frac_mean={summary.get('missed_attack_frac_mean')}",
    f"course_change_frac_mean={summary.get('course_change_frac_mean')}",
    f"course_unique_frac_mean={summary.get('course_unique_frac_mean')}",
    f"visible_enemy_count_mean={summary.get('visible_enemy_count_mean')}",
    f"contact_frac_mean={summary.get('contact_frac_mean')}",
    f"nearest_enemy_distance_mean={summary.get('nearest_enemy_distance_mean')}",
    f"engagement_progress_reward_mean={summary.get('engagement_progress_reward_mean')}",
    f"episode_len_mean={summary.get('episode_len_mean')}",
]
print(" ".join(parts))
PY
}

run_progress_eval() {
  local opponent="$1"
  local ts="$2"
  local safe_tag="$3"
  local out_json="log/${EXP_NAME}_${STAGE_NAME}.${safe_tag}.progress_${ts}.json"
  local temp_exp
  temp_exp="$(make_eval_clone)"

  if [[ -z "$temp_exp" ]]; then
    log_msg "progress_eval_skip opponent=${opponent} reason=no_stable_checkpoint"
    return 0
  fi

  if conda run --no-capture-output -n maca-py37-min python scripts/eval_sf_maca.py \
    --experiment="$temp_exp" \
    --train_dir="$TRAIN_DIR" \
    --episodes="$EPISODES" \
    --device=cpu \
    --maca_opponent="$opponent" \
    --output_json="$out_json" >> "$MONITOR_LOG" 2>&1; then
    local summary
    local fire_frac
    local opp_frac
    local fire_opp_ratio
    local win_rate
    summary="$(summarize_eval "$out_json")"
    fire_frac="$(read_metric "$out_json" "fire_action_frac_mean")"
    opp_frac="$(read_metric "$out_json" "attack_opportunity_frac_mean")"
    fire_opp_ratio="$(python3 - "$fire_frac" "$opp_frac" <<'PY'
import sys
try:
    fire = float(sys.argv[1])
    opp = float(sys.argv[2])
except Exception:
    print("nan")
    raise SystemExit(0)
if opp <= 1e-9:
    print(0.0)
else:
    print(fire / opp)
PY
)"
    win_rate="$(read_metric "$out_json" "win_rate")"
    log_msg "progress_eval opponent=${opponent} ${summary} fire_to_opportunity_ratio=${fire_opp_ratio}"
    if [[ "$fire_frac" == "nan" || "$win_rate" == "nan" ]]; then
      log_msg "anomaly opponent=${opponent} reason=nan_metric json=${out_json}"
    fi
    if python3 - "$fire_frac" "$ANOMALY_FIRE_THRESHOLD" <<'PY'
import sys
try:
    value = float(sys.argv[1])
    threshold = float(sys.argv[2])
except Exception:
    sys.exit(0)
if threshold > 0 and value < threshold:
    sys.exit(7)
PY
    then
      :
    else
      local fire_status=$?
      if [[ $fire_status -eq 7 ]]; then
        log_msg "anomaly opponent=${opponent} reason=fire_action_too_low threshold=${ANOMALY_FIRE_THRESHOLD} value=${fire_frac}"
      fi
    fi

    if [[ "$LOW_FIRE_MAX_STREAK" != "0" && "$opponent" == "$PRIMARY_OPPONENT" ]]; then
      if python3 - "$fire_frac" "$opp_frac" "$LOW_FIRE_THRESHOLD" "$MIN_OPPORTUNITY_FOR_RATIO" <<'PY'
import sys
try:
    value = float(sys.argv[1])
    opp = float(sys.argv[2])
    threshold = float(sys.argv[3])
    min_opp = float(sys.argv[4])
except Exception:
    sys.exit(0)
sys.exit(7 if threshold > 0 and opp >= min_opp and value < threshold else 0)
PY
      then
        LOW_FIRE_STREAK=0
      else
        low_fire_status=$?
        if [[ $low_fire_status -eq 7 ]]; then
          LOW_FIRE_STREAK=$((LOW_FIRE_STREAK + 1))
          log_msg "primary_low_fire_streak opponent=${opponent} threshold=${LOW_FIRE_THRESHOLD} min_opp=${MIN_OPPORTUNITY_FOR_RATIO} streak=${LOW_FIRE_STREAK}/${LOW_FIRE_MAX_STREAK} value=${fire_frac} opp=${opp_frac}"
          if (( LOW_FIRE_STREAK >= LOW_FIRE_MAX_STREAK )); then
            log_msg "fatal_monitor reason=primary_fire_action_stuck_below_threshold opponent=${opponent} threshold=${LOW_FIRE_THRESHOLD} streak=${LOW_FIRE_STREAK} train_pid=${TRAIN_PID}"
            kill "$TRAIN_PID" 2>/dev/null || true
          fi
        fi
      fi
    fi

    if [[ "$LOW_RATIO_MAX_STREAK" != "0" && "$MIN_FIRE_OPP_RATIO" != "0" && "$opponent" == "$PRIMARY_OPPONENT" ]]; then
      if python3 - "$fire_opp_ratio" "$opp_frac" "$MIN_FIRE_OPP_RATIO" "$MIN_OPPORTUNITY_FOR_RATIO" <<'PY'
import sys
try:
    ratio = float(sys.argv[1])
    opp = float(sys.argv[2])
    ratio_threshold = float(sys.argv[3])
    min_opp = float(sys.argv[4])
except Exception:
    sys.exit(0)
sys.exit(7 if opp >= min_opp and ratio < ratio_threshold else 0)
PY
      then
        LOW_RATIO_STREAK=0
      else
        low_ratio_status=$?
        if [[ $low_ratio_status -eq 7 ]]; then
          LOW_RATIO_STREAK=$((LOW_RATIO_STREAK + 1))
          log_msg "primary_low_fire_ratio_streak opponent=${opponent} threshold=${MIN_FIRE_OPP_RATIO} min_opp=${MIN_OPPORTUNITY_FOR_RATIO} streak=${LOW_RATIO_STREAK}/${LOW_RATIO_MAX_STREAK} ratio=${fire_opp_ratio} opp=${opp_frac}"
          if (( LOW_RATIO_STREAK >= LOW_RATIO_MAX_STREAK )); then
            log_msg "fatal_monitor reason=primary_fire_opportunity_ratio_stuck_low opponent=${opponent} threshold=${MIN_FIRE_OPP_RATIO} streak=${LOW_RATIO_STREAK} train_pid=${TRAIN_PID}"
            kill "$TRAIN_PID" 2>/dev/null || true
          fi
        fi
      fi
    fi
  else
    log_msg "progress_eval_failed opponent=${opponent} json=${out_json}"
  fi

  python3 - "$ROOT_DIR" "$temp_exp" <<'PY'
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
temp_exp = sys.argv[2]
temp_dir = root / "train_dir" / "sample_factory" / temp_exp
shutil.rmtree(temp_dir, ignore_errors=True)
PY
}

trap cleanup_tmp EXIT
log_msg "monitor_start stage=${STAGE_NAME} pid=${TRAIN_PID} primary=${PRIMARY_OPPONENT} secondary=${SECONDARY_OPPONENT} interval_sec=${INTERVAL_SEC} episodes=${EPISODES}"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep "$INTERVAL_SEC" || true
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    break
  fi

  ts="$(date '+%Y%m%d_%H%M%S')"
  run_progress_eval "$PRIMARY_OPPONENT" "$ts" "${PRIMARY_OPPONENT}"
  run_progress_eval "$SECONDARY_OPPONENT" "$ts" "${SECONDARY_OPPONENT}"
done

log_msg "monitor_stop stage=${STAGE_NAME}"
