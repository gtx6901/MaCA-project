#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_fix_mixed_1to2h.sh [options]

Description:
  Fix-rule-oriented mixed curriculum for about 1.5-2 hours:
    Phase 1: short no_att foundation from scratch
    Phase 2+: many fix_rule-heavy cycles with small no_att refresh blocks

Options:
  --run-id <id>             Master run id. Default: current timestamp.
  --seed <int>              Random seed. Default: 42
  --headless <0|1>          1=dummy SDL, 0=enable pygame render. Default: 1
  --fresh-start             Start the first block from scratch.
  --clean-model             Delete model/*.pkl before start (use with --fresh-start).
  --keep-checkpoints <int>  Keep latest N checkpoints. Default: 24
  --help                    Show this help.
EOF
}

RUN_ID="$(date +%Y%m%d_%H%M%S)"
SEED=42
HEADLESS=1
FRESH_START=0
CLEAN_MODEL=0
KEEP_CHECKPOINTS=24
EST_GPU_HEADLESS_STEPS_PER_MIN="${EST_GPU_HEADLESS_STEPS_PER_MIN:-6500}"
EST_GPU_GUI_STEPS_PER_MIN="${EST_GPU_GUI_STEPS_PER_MIN:-4000}"

# Tuned profile: longer no_att episodes for real contact, but lower no_att share overall.
CYCLES=12
NOATT_EPOCHS_FIRST=80
NOATT_EPOCHS_AFTER_FIRST=4
FIX_EPOCHS=40
NOATT_MAX_STEP=1000
FIX_STEP_START=550
FIX_STEP_FINAL=650
FIX_STEP_INC=25

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

mkdir -p log model/backup model/simple
MASTER_LOG="log/fix_mixed_1to2h_${RUN_ID}.log"
touch "$MASTER_LOG"

log() {
  echo "[FixMixed] $*" | tee -a "$MASTER_LOG"
}

format_minutes() {
  local minutes="$1"
  printf '%dh%02dm' "$((minutes / 60))" "$((minutes % 60))"
}

compute_fix_budget() {
  local fix_step="$FIX_STEP_START"
  local total_fix_steps=0
  local cycle
  for ((cycle = 1; cycle <= CYCLES; cycle++)); do
    total_fix_steps=$((total_fix_steps + FIX_EPOCHS * fix_step))
    if (( fix_step < FIX_STEP_FINAL )); then
      fix_step=$((fix_step + FIX_STEP_INC))
      if (( fix_step > FIX_STEP_FINAL )); then
        fix_step="$FIX_STEP_FINAL"
      fi
    fi
  done
  echo "$total_fix_steps"
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
  printf '[FixMixed] cmd:' | tee -a "$MASTER_LOG"
  for item in "${cmd[@]}"; do
    printf ' %q' "$item" | tee -a "$MASTER_LOG"
  done
  printf '\n' | tee -a "$MASTER_LOG"
  "${cmd[@]}" 2>&1 | tee -a "$MASTER_LOG"
}

if [[ "$CLEAN_MODEL" == "1" ]]; then
  log "cleaning model checkpoints before start"
  find model -type f -name '*.pkl' -delete
fi

NOATT_BUDGET=$((NOATT_EPOCHS_FIRST * NOATT_MAX_STEP + (CYCLES - 1) * NOATT_EPOCHS_AFTER_FIRST * NOATT_MAX_STEP))
FIX_BUDGET="$(compute_fix_budget)"
TOTAL_BUDGET=$((NOATT_BUDGET + FIX_BUDGET))

log "start run_id=$RUN_ID seed=$SEED headless=$HEADLESS fresh_start=$FRESH_START keep_checkpoints=$KEEP_CHECKPOINTS"
log "strategy: longer no_att episodes for contact quality, but fix_rule-heavy alternation overall"
log "cycles=$CYCLES noatt_first=$NOATT_EPOCHS_FIRST noatt_after_first=$NOATT_EPOCHS_AFTER_FIRST fix_epochs=$FIX_EPOCHS"
log "noatt_max_step=$NOATT_MAX_STEP fix_step_start=$FIX_STEP_START fix_step_final=$FIX_STEP_FINAL fix_step_inc=$FIX_STEP_INC"
log_runtime_budget "$TOTAL_BUDGET"

cmd=(
  bash scripts/run_curriculum_autotrain.sh
  --run-id "$RUN_ID"
  --seed "$SEED"
  --cycles "$CYCLES"
  --noatt-epochs "$NOATT_EPOCHS_FIRST"
  --noatt-epochs-after-first "$NOATT_EPOCHS_AFTER_FIRST"
  --fix-epochs "$FIX_EPOCHS"
  --noatt-max-step "$NOATT_MAX_STEP"
  --fix-step-start "$FIX_STEP_START"
  --fix-step-final "$FIX_STEP_FINAL"
  --fix-step-inc "$FIX_STEP_INC"
  --gate-win-last10 0.02
  --force-step-ramp 1
  --lr-noatt 0.00030
  --lr-fix 0.00016
  --gamma-noatt 0.99
  --gamma-fix 0.99
  --learn-interval-noatt 16
  --learn-interval-fix 10
  --batch-size 128
  --memory-size 12000
  --min-replay-size 4000
  --epsilon-noatt 0.08
  --epsilon-fix 0.12
  --epsilon-inc -0.000003
  --keep-checkpoints "$KEEP_CHECKPOINTS"
  --headless "$HEADLESS"
)

if [[ "$FRESH_START" == "1" ]]; then
  cmd+=(--fresh-start-first-block)
fi

run_and_log "${cmd[@]}"

log "finished curriculum"
log "wrapper log: $MASTER_LOG"
log "autotrain log: log/auto_curriculum_${RUN_ID}.log"
