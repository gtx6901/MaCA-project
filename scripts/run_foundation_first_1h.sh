#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_foundation_first_1h.sh [options]

Description:
  ~1 hour foundation-first launcher:
    Stage A: no_att fast-finish foundation
    Stage B: no_att pressure consolidation
    Stage C: no_att bridge preparation
    Stage D: very short fix_rule bridge
    Stage E: short fix_rule verification

Options:
  --run-id <id>             Master run id. Default: current timestamp.
  --seed <int>              Random seed. Default: 42
  --headless <0|1>          1=dummy SDL, 0=enable pygame render. Default: 1
  --fresh-start             Start Stage A from scratch.
  --clean-model             Delete model/*.pkl before start (use with --fresh-start).
  --keep-checkpoints <int>  Keep latest N checkpoints. Default: 16
  --help                    Show this help.
EOF
}

RUN_ID="$(date +%Y%m%d_%H%M%S)"
SEED=42
HEADLESS=1
FRESH_START=0
CLEAN_MODEL=0
KEEP_CHECKPOINTS=16
EST_GPU_HEADLESS_STEPS_PER_MIN="${EST_GPU_HEADLESS_STEPS_PER_MIN:-6500}"
EST_GPU_GUI_STEPS_PER_MIN="${EST_GPU_GUI_STEPS_PER_MIN:-4000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --headless) HEADLESS="$2"; shift 2 ;;
    --fresh-start) FRESH_START=1; shift ;;
    --clean-model) CLEAN_MODEL=1; shift ;;
    --keep-checkpoints) KEEP_CHECKPOINTS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$CLEAN_MODEL" == "1" && "$FRESH_START" != "1" ]]; then
  echo "Error: --clean-model requires --fresh-start." >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
if [[ "$HEADLESS" == "1" ]]; then
  export SDL_VIDEODRIVER=dummy
  export SDL_AUDIODRIVER=dummy
else
  unset SDL_VIDEODRIVER || true
  unset SDL_AUDIODRIVER || true
fi

mkdir -p log model/backup model/simple
MASTER_LOG="log/foundation_first_1h_${RUN_ID}.log"
touch "$MASTER_LOG"

log() {
  echo "[Foundation1h] $*" | tee -a "$MASTER_LOG"
}

format_minutes() {
  local minutes="$1"
  printf '%dh%02dm' "$((minutes / 60))" "$((minutes % 60))"
}

log_runtime_budget() {
  local total_steps="$1"
  local headless_min=$(((total_steps + EST_GPU_HEADLESS_STEPS_PER_MIN - 1) / EST_GPU_HEADLESS_STEPS_PER_MIN))
  local gui_min=$(((total_steps + EST_GPU_GUI_STEPS_PER_MIN - 1) / EST_GPU_GUI_STEPS_PER_MIN))
  log "planned_step_budget=$total_steps"
  log "estimated_runtime_gpu_headless=$(format_minutes "$headless_min") at ~${EST_GPU_HEADLESS_STEPS_PER_MIN} steps/min"
  log "estimated_runtime_gpu_gui=$(format_minutes "$gui_min") at ~${EST_GPU_GUI_STEPS_PER_MIN} steps/min"
}

run_and_log() {
  local -a cmd=("$@")
  printf '[Foundation1h] cmd:' | tee -a "$MASTER_LOG"
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
  log "prune checkpoints: keep=$keep removed=$removed total_before=$total"
}

summarize_metrics() {
  local label="$1"
  local metrics_csv="$2"
  local summary_json="$3"
  python - "$label" "$metrics_csv" "$summary_json" <<'PY'
import csv
import json
import os
import re
import sys

label = sys.argv[1]
path = sys.argv[2]
summary_path = sys.argv[3]
rows = list(csv.DictReader(open(path)))
if not rows:
    print(f"[Foundation1h] summary {label}: empty")
    raise SystemExit(0)

steps = [int(r["steps"]) for r in rows]
rewards = [float(r["total_reward"]) for r in rows]
wins = [int(r["red_win"]) for r in rows]
rr = [int(r["red_round_reward"]) for r in rows]
k10 = min(10, len(rows))
k20 = min(20, len(rows))
base = os.path.basename(path)
cap = int(re.search(r"_s(\d+)\.csv$", base).group(1))
timeout_rate = sum(1 for s in steps if s >= cap) / len(rows)
totally_lose = sum(1 for x in rr if x == -2000) / len(rows)
lose = sum(1 for x in rr if x == -1000) / len(rows)
draw = sum(1 for x in rr if x == -1500) / len(rows)
elapsed_sec = None
if os.path.exists(summary_path):
    with open(summary_path) as f:
        elapsed_sec = float(json.load(f).get("elapsed_sec", 0) or 0)
steps_per_min = (sum(steps) * 60.0 / elapsed_sec) if elapsed_sec else 0.0
print(
    f"[Foundation1h] summary {label}: "
    f"n={len(rows)} win={sum(wins)/len(rows):.3f} "
    f"win_last10={sum(wins[-k10:])/k10:.3f} "
    f"win_last20={sum(wins[-k20:])/k20:.3f} "
    f"avg_steps={sum(steps)/len(rows):.1f} "
    f"timeout_rate={timeout_rate:.3f} "
    f"reward_last10={sum(rewards[-k10:])/k10:.1f} "
    f"reward_last20={sum(rewards[-k20:])/k20:.1f} "
    f"totally_lose={totally_lose:.3f} lose={lose:.3f} draw={draw:.3f} "
    f"elapsed_min={(elapsed_sec / 60.0):.2f} "
    f"steps_per_min={steps_per_min:.1f}"
)
PY
}

run_stage() {
  local stage_tag="$1"
  local opponent="$2"
  local epochs="$3"
  local max_step="$4"
  local lr="$5"
  local gamma="$6"
  local learn_interval="$7"
  local epsilon="$8"
  local epsilon_inc="$9"

  local metrics_csv="log/train_dqn_metrics_${RUN_ID}_${stage_tag}.csv"
  local summary_json="log/train_dqn_summary_${RUN_ID}_${stage_tag}.json"
  local stage_step_budget=$((epochs * max_step))
  log "stage=$stage_tag opponent=$opponent epochs=$epochs max_step=$max_step step_budget=$stage_step_budget lr=$lr gamma=$gamma learn_interval=$learn_interval epsilon=$epsilon epsilon_increment=$epsilon_inc"

  local -a cmd=(
    python scripts/train_dqn_pipeline.py
    --epochs "$epochs"
    --max_step "$max_step"
    --seed "$SEED"
    --opponent "$opponent"
    --lr "$lr"
    --gamma "$gamma"
    --learn_interval "$learn_interval"
    --batch_size 128
    --memory_size 12000
    --min_replay_size 4000
    --epsilon "$epsilon"
    --epsilon_increment "$epsilon_inc"
    --metrics_csv "$metrics_csv"
    --summary_json "$summary_json"
  )

  if [[ "$HEADLESS" == "0" ]]; then
    cmd+=(--render)
  fi

  if [[ "$stage_tag" == "stageA_noatt_foundation_s300" && "$FRESH_START" == "1" ]]; then
    cmd+=(--fresh_start)
  fi

  run_and_log "${cmd[@]}"
  prune_old_checkpoints "$KEEP_CHECKPOINTS"
  summarize_metrics "$stage_tag" "$metrics_csv" "$summary_json" | tee -a "$MASTER_LOG"
}

if [[ "$CLEAN_MODEL" == "1" ]]; then
  log "cleaning model checkpoints before start"
  find model -type f -name '*.pkl' -delete
fi

log "start run_id=$RUN_ID seed=$SEED headless=$HEADLESS fresh_start=$FRESH_START keep_checkpoints=$KEEP_CHECKPOINTS"
log "strategy: much longer no_att finish-pressure block with a still-short fix bridge"
log_runtime_budget 240000

run_stage "stageA_noatt_foundation_s300" fix_rule_no_att 220 300 0.00035 0.99 18 0.10 -0.000004
run_stage "stageB_noatt_pressure_s320" fix_rule_no_att 180 320 0.00028 0.99 14 0.08 -0.000004
run_stage "stageC_noatt_bridgeprep_s360" fix_rule_no_att 140 360 0.00022 0.99 12 0.06 -0.000003
run_stage "stageD_fix_bridge_s350" fix_rule 60 350 0.00018 0.99 10 0.10 -0.000003
run_stage "stageE_fix_verify_s450" fix_rule 100 450 0.00016 0.99 10 0.08 -0.000003

log "finished all stages"
log "master log: $MASTER_LOG"
