#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_curriculum_autotrain.sh [options]

Options:
  --run-id <id>                 Run id used in output filenames.
  --seed <int>                  Random seed. Default: 42
  --base-resume <path>          Start checkpoint for the first block.
                                Default: model/simple/model.pkl
  --fresh-start-first-block     Start the first no_att block from scratch.
                                When set, --base-resume is ignored.
  --cycles <int>                Number of curriculum cycles. Default: 10
  --noatt-epochs <int>          Epochs for each no_att block. Default: 2
  --noatt-epochs-after-first <int>
                                Epochs for no_att block from cycle 2 onward.
                                Default: same as --noatt-epochs.
  --fix-epochs <int>            Epochs for each fix_rule block. Default: 80
  --noatt-max-step <int>        max_step for no_att block. Default: 500
  --fix-step-start <int>        Initial max_step for fix_rule block. Default: 500
  --fix-step-final <int>        Max fix_rule max_step cap. Default: 1100
  --fix-step-inc <int>          Step increase when gate passes. Default: 100
  --gate-win-last10 <float>     Promotion gate on fix block last10 win rate. Default: 0.05
  --force-step-ramp <0|1>       Always increase fix max_step each cycle. Default: 0
  --lr-noatt <float>            Learning rate for no_att block. Default: 0.0005
  --lr-fix <float>              Learning rate for fix_rule block. Default: 0.00025
  --gamma-noatt <float>         Gamma for no_att block. Default: 0.99
  --gamma-fix <float>           Gamma for fix_rule block. Default: 0.99
  --learn-interval-noatt <int>  learn_interval for no_att block. Default: 50
  --learn-interval-fix <int>    learn_interval for fix_rule block. Default: 8
  --batch-size <int>            Replay batch size. Default: 128
  --memory-size <int>           Replay memory size. Default: 12000
  --min-replay-size <int>       Start learning after this many transitions. Default: 2000
  --epsilon-noatt <float>       Epsilon for no_att block. Default: 0.12
  --epsilon-fix <float>         Epsilon for fix_rule block. Default: 0.55
  --epsilon-inc <float>         Epsilon increment. Default: -0.00005
  --keep-checkpoints <int>      Keep latest N model_*.pkl checkpoints. Default: 24
  --headless <0|1>              1=dummy SDL, 0=render-capable env. Default: 1
  --help                        Show this help message.
EOF
}

RUN_ID="$(date +%Y%m%d_%H%M%S)"
SEED=42
BASE_RESUME="model/simple/model.pkl"
FRESH_START_FIRST_BLOCK=0
CYCLES=10
NOATT_EPOCHS=2
NOATT_EPOCHS_AFTER_FIRST=""
FIX_EPOCHS=80
NOATT_MAX_STEP=500
FIX_STEP_START=500
FIX_STEP_FINAL=1100
FIX_STEP_INC=100
GATE_WIN_LAST10=0.05
FORCE_STEP_RAMP=0
LR_NOATT=0.0005
LR_FIX=0.00025
GAMMA_NOATT=0.99
GAMMA_FIX=0.99
LEARN_INTERVAL_NOATT=50
LEARN_INTERVAL_FIX=8
BATCH_SIZE=128
MEMORY_SIZE=12000
MIN_REPLAY_SIZE=2000
EPSILON_NOATT=0.12
EPSILON_FIX=0.55
EPSILON_INC=-0.00005
KEEP_CHECKPOINTS=24
HEADLESS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --base-resume) BASE_RESUME="$2"; shift 2 ;;
    --fresh-start-first-block) FRESH_START_FIRST_BLOCK=1; shift ;;
    --cycles) CYCLES="$2"; shift 2 ;;
    --noatt-epochs) NOATT_EPOCHS="$2"; shift 2 ;;
    --noatt-epochs-after-first) NOATT_EPOCHS_AFTER_FIRST="$2"; shift 2 ;;
    --fix-epochs) FIX_EPOCHS="$2"; shift 2 ;;
    --noatt-max-step) NOATT_MAX_STEP="$2"; shift 2 ;;
    --fix-step-start) FIX_STEP_START="$2"; shift 2 ;;
    --fix-step-final) FIX_STEP_FINAL="$2"; shift 2 ;;
    --fix-step-inc) FIX_STEP_INC="$2"; shift 2 ;;
    --gate-win-last10) GATE_WIN_LAST10="$2"; shift 2 ;;
    --force-step-ramp) FORCE_STEP_RAMP="$2"; shift 2 ;;
    --lr-noatt) LR_NOATT="$2"; shift 2 ;;
    --lr-fix) LR_FIX="$2"; shift 2 ;;
    --gamma-noatt) GAMMA_NOATT="$2"; shift 2 ;;
    --gamma-fix) GAMMA_FIX="$2"; shift 2 ;;
    --learn-interval-noatt) LEARN_INTERVAL_NOATT="$2"; shift 2 ;;
    --learn-interval-fix) LEARN_INTERVAL_FIX="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --memory-size) MEMORY_SIZE="$2"; shift 2 ;;
    --min-replay-size) MIN_REPLAY_SIZE="$2"; shift 2 ;;
    --epsilon-noatt) EPSILON_NOATT="$2"; shift 2 ;;
    --epsilon-fix) EPSILON_FIX="$2"; shift 2 ;;
    --epsilon-inc) EPSILON_INC="$2"; shift 2 ;;
    --keep-checkpoints) KEEP_CHECKPOINTS="$2"; shift 2 ;;
    --headless) HEADLESS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$NOATT_EPOCHS_AFTER_FIRST" ]]; then
  NOATT_EPOCHS_AFTER_FIRST="$NOATT_EPOCHS"
fi

if [[ "$FRESH_START_FIRST_BLOCK" != "1" ]]; then
  if [[ ! -f "$BASE_RESUME" ]]; then
    echo "Base resume checkpoint not found: $BASE_RESUME" >&2
    exit 1
  fi
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment:${PYTHONPATH:-}"
# Avoid delayed epoch logs when running under nohup/tee.
export PYTHONUNBUFFERED=1
if [[ "$HEADLESS" == "1" ]]; then
  export SDL_VIDEODRIVER=dummy
  export SDL_AUDIODRIVER=dummy
else
  unset SDL_VIDEODRIVER || true
  unset SDL_AUDIODRIVER || true
fi

mkdir -p log model/backup model/simple

BACKUP_DIR="model/backup/simple_before_${RUN_ID}"
cp -a model/simple "$BACKUP_DIR"

MASTER_LOG="log/auto_curriculum_${RUN_ID}.log"
touch "$MASTER_LOG"

echo "[AutoTrain] start: $(date)" | tee -a "$MASTER_LOG"
echo "[AutoTrain] run_id=$RUN_ID seed=$SEED cycles=$CYCLES base_resume=$BASE_RESUME fresh_start_first_block=$FRESH_START_FIRST_BLOCK" | tee -a "$MASTER_LOG"
echo "[AutoTrain] noatt_epochs_first=$NOATT_EPOCHS noatt_epochs_after_first=$NOATT_EPOCHS_AFTER_FIRST fix_epochs=$FIX_EPOCHS noatt_max_step=$NOATT_MAX_STEP fix_step_start=$FIX_STEP_START fix_step_final=$FIX_STEP_FINAL fix_step_inc=$FIX_STEP_INC gate_win_last10=$GATE_WIN_LAST10 force_step_ramp=$FORCE_STEP_RAMP keep_checkpoints=$KEEP_CHECKPOINTS" | tee -a "$MASTER_LOG"
echo "[AutoTrain] lr_noatt=$LR_NOATT lr_fix=$LR_FIX gamma_noatt=$GAMMA_NOATT gamma_fix=$GAMMA_FIX learn_interval_noatt=$LEARN_INTERVAL_NOATT learn_interval_fix=$LEARN_INTERVAL_FIX batch_size=$BATCH_SIZE memory_size=$MEMORY_SIZE min_replay_size=$MIN_REPLAY_SIZE epsilon_noatt=$EPSILON_NOATT epsilon_fix=$EPSILON_FIX epsilon_inc=$EPSILON_INC" | tee -a "$MASTER_LOG"
echo "[AutoTrain] backup=$BACKUP_DIR" | tee -a "$MASTER_LOG"

run_and_log() {
  local -a cmd=("$@")
  printf '[AutoTrain] cmd:' | tee -a "$MASTER_LOG"
  for item in "${cmd[@]}"; do
    printf ' %q' "$item" | tee -a "$MASTER_LOG"
  done
  printf '\n' | tee -a "$MASTER_LOG"
  "${cmd[@]}" 2>&1 | tee -a "$MASTER_LOG"
}

prune_old_checkpoints() {
  local keep="$1"
  if (( keep <= 0 )); then
    return
  fi

  local ckpts=()
  while IFS= read -r f; do
    ckpts+=("$f")
  done < <(ls -1t model/simple/model_*.pkl 2>/dev/null || true)

  local total="${#ckpts[@]}"
  if (( total <= keep )); then
    return
  fi

  local removed=0
  local i
  for ((i=keep; i<total; i++)); do
    rm -f "${ckpts[i]}"
    removed=$((removed + 1))
  done
  echo "[AutoTrain] prune checkpoints: keep=$keep removed=$removed total_before=$total" | tee -a "$MASTER_LOG"
}

analyze_metrics_json() {
  local metrics_csv="$1"
  python - "$metrics_csv" <<'PY'
import csv
import json
import os
import re
import sys

p = sys.argv[1]
rows = list(csv.DictReader(open(p)))
if not rows:
    print(json.dumps({
        "epochs": 0,
        "win_rate_all": 0.0,
        "win_rate_last10": 0.0,
        "win_rate_last20": 0.0,
        "reward_last10": 0.0,
        "reward_last20": 0.0,
        "avg_steps": 0.0,
        "timeout_rate": 0.0,
        "totally_lose_rate": 0.0,
        "lose_rate": 0.0,
        "draw_rate": 0.0,
    }))
    raise SystemExit(0)

win = [int(r["red_win"]) for r in rows]
tr = [float(r["total_reward"]) for r in rows]
steps = [int(r["steps"]) for r in rows]
rr = [int(r["red_round_reward"]) for r in rows]
k10 = min(10, len(rows))
k20 = min(20, len(rows))
base = os.path.basename(p)
m = re.search(r"_s(\d+)\.csv$", base)
cap = int(m.group(1)) if m else max(steps)
out = {
    "epochs": len(rows),
    "win_rate_all": sum(win) / len(rows),
    "win_rate_last10": sum(win[-k10:]) / k10,
    "win_rate_last20": sum(win[-k20:]) / k20,
    "reward_last10": sum(tr[-k10:]) / k10,
    "reward_last20": sum(tr[-k20:]) / k20,
    "avg_steps": sum(steps) / len(rows),
    "timeout_rate": sum(1 for s in steps if s >= cap) / len(rows),
    "totally_lose_rate": sum(1 for x in rr if x == -2000) / len(rows),
    "lose_rate": sum(1 for x in rr if x == -1000) / len(rows),
    "draw_rate": sum(1 for x in rr if x == -1500) / len(rows),
}
print(json.dumps(out))
PY
}

json_get() {
  local json_str="$1"
  local key="$2"
  python - "$json_str" "$key" <<'PY'
import json
import sys
d = json.loads(sys.argv[1])
print(d[sys.argv[2]])
PY
}

float_ge() {
  local a="$1"
  local b="$2"
  python - "$a" "$b" <<'PY'
import sys
print("1" if float(sys.argv[1]) >= float(sys.argv[2]) else "0")
PY
}

fix_step="$FIX_STEP_START"
initial_flag_applied=0

for ((cycle = 1; cycle <= CYCLES; cycle++)); do
  echo "[AutoTrain] ===== Cycle $cycle/$CYCLES =====" | tee -a "$MASTER_LOG"

  noatt_epochs_this_cycle="$NOATT_EPOCHS_AFTER_FIRST"
  if (( cycle == 1 )); then
    noatt_epochs_this_cycle="$NOATT_EPOCHS"
  fi

  if (( noatt_epochs_this_cycle > 0 )); then
    tag_noatt="${RUN_ID}_cycle${cycle}_a_noatt_e${noatt_epochs_this_cycle}_s${NOATT_MAX_STEP}"
    metrics_noatt="log/train_dqn_metrics_${tag_noatt}.csv"
    summary_noatt="log/train_dqn_summary_${tag_noatt}.json"

    cmd_noatt=(
      python scripts/train_dqn_pipeline.py
      --epochs "$noatt_epochs_this_cycle"
      --max_step "$NOATT_MAX_STEP"
      --seed "$SEED"
      --opponent fix_rule_no_att
      --lr "$LR_NOATT"
      --gamma "$GAMMA_NOATT"
      --learn_interval "$LEARN_INTERVAL_NOATT"
      --batch_size "$BATCH_SIZE"
      --memory_size "$MEMORY_SIZE"
      --min_replay_size "$MIN_REPLAY_SIZE"
      --epsilon "$EPSILON_NOATT"
      --epsilon_increment "$EPSILON_INC"
      --metrics_csv "$metrics_noatt"
      --summary_json "$summary_noatt"
    )
    if [[ "$initial_flag_applied" == "0" ]]; then
      if [[ "$FRESH_START_FIRST_BLOCK" == "1" ]]; then
        cmd_noatt+=(--fresh_start)
      else
        cmd_noatt+=(--resume "$BASE_RESUME")
      fi
      initial_flag_applied=1
    fi
    run_and_log "${cmd_noatt[@]}"
    prune_old_checkpoints "$KEEP_CHECKPOINTS"
  else
    echo "[AutoTrain] skip no_att block in cycle=$cycle (noatt_epochs_this_cycle=$noatt_epochs_this_cycle)" | tee -a "$MASTER_LOG"
  fi

  tag_fix="${RUN_ID}_cycle${cycle}_b_fix_e${FIX_EPOCHS}_s${fix_step}"
  metrics_fix="log/train_dqn_metrics_${tag_fix}.csv"
  summary_fix="log/train_dqn_summary_${tag_fix}.json"

  cmd_fix=(
    python scripts/train_dqn_pipeline.py
    --epochs "$FIX_EPOCHS"
    --max_step "$fix_step"
    --seed "$SEED"
    --opponent fix_rule
    --lr "$LR_FIX"
    --gamma "$GAMMA_FIX"
    --learn_interval "$LEARN_INTERVAL_FIX"
    --batch_size "$BATCH_SIZE"
    --memory_size "$MEMORY_SIZE"
    --min_replay_size "$MIN_REPLAY_SIZE"
    --epsilon "$EPSILON_FIX"
    --epsilon_increment "$EPSILON_INC"
    --metrics_csv "$metrics_fix"
    --summary_json "$summary_fix"
  )
  if [[ "$initial_flag_applied" == "0" ]]; then
    if [[ "$FRESH_START_FIRST_BLOCK" == "1" ]]; then
      cmd_fix+=(--fresh_start)
    else
      cmd_fix+=(--resume "$BASE_RESUME")
    fi
    initial_flag_applied=1
  fi
  run_and_log "${cmd_fix[@]}"
  prune_old_checkpoints "$KEEP_CHECKPOINTS"

  analysis_json="$(analyze_metrics_json "$metrics_fix")"
  win_all="$(json_get "$analysis_json" "win_rate_all")"
  win_last10="$(json_get "$analysis_json" "win_rate_last10")"
  win_last20="$(json_get "$analysis_json" "win_rate_last20")"
  reward_last10="$(json_get "$analysis_json" "reward_last10")"
  reward_last20="$(json_get "$analysis_json" "reward_last20")"
  avg_steps="$(json_get "$analysis_json" "avg_steps")"
  timeout_rate="$(json_get "$analysis_json" "timeout_rate")"
  totally_lose_rate="$(json_get "$analysis_json" "totally_lose_rate")"
  lose_rate="$(json_get "$analysis_json" "lose_rate")"
  draw_rate="$(json_get "$analysis_json" "draw_rate")"

  echo "[AutoTrain] fix_block_metrics cycle=$cycle max_step=$fix_step win_all=$win_all win_last10=$win_last10 win_last20=$win_last20 avg_steps=$avg_steps timeout_rate=$timeout_rate reward_last10=$reward_last10 reward_last20=$reward_last20 totally_lose=$totally_lose_rate lose=$lose_rate draw=$draw_rate" | tee -a "$MASTER_LOG"

  pass_gate="$(float_ge "$win_last10" "$GATE_WIN_LAST10")"
  should_ramp=0
  if [[ "$pass_gate" == "1" ]]; then
    should_ramp=1
    echo "[AutoTrain] gate passed by win_last10=$win_last10 (>= $GATE_WIN_LAST10)" | tee -a "$MASTER_LOG"
  elif [[ "$FORCE_STEP_RAMP" == "1" ]]; then
    should_ramp=1
    echo "[AutoTrain] gate not passed, but force ramp enabled" | tee -a "$MASTER_LOG"
  fi

  if [[ "$should_ramp" == "1" ]]; then
    if (( fix_step < FIX_STEP_FINAL )); then
      next_step=$((fix_step + FIX_STEP_INC))
      if (( next_step > FIX_STEP_FINAL )); then
        next_step="$FIX_STEP_FINAL"
      fi
      echo "[AutoTrain] promote fix max_step: $fix_step -> $next_step" | tee -a "$MASTER_LOG"
      fix_step="$next_step"
    else
      echo "[AutoTrain] fix max_step already at final=$fix_step" | tee -a "$MASTER_LOG"
    fi
  else
    echo "[AutoTrain] gate not passed, keep fix max_step=$fix_step" | tee -a "$MASTER_LOG"
  fi
done

prune_old_checkpoints "$KEEP_CHECKPOINTS"
echo "[AutoTrain] finished: $(date)" | tee -a "$MASTER_LOG"
echo "[AutoTrain] final model: model/simple/model.pkl" | tee -a "$MASTER_LOG"
echo "[AutoTrain] log file: $MASTER_LOG" | tee -a "$MASTER_LOG"
