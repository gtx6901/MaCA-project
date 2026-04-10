#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_overnight_2to3h.sh [options]

Description:
  2-3 hour curriculum launcher (3 phases, fix_rule heavy):
    Phase A: short bootstrap (no_att calibration + short fix)
    Phase B: main training (fix-heavy, progressive horizon)
    Phase C: consolidation (fix-heavy, lower exploration)

Options:
  --run-id <id>             Master run id. Default: current timestamp.
  --seed <int>              Random seed. Default: 42
  --headless <0|1>          1=dummy SDL. Default: 1
  --fresh-start             Start phase A from scratch.
  --clean-model             Delete model/*.pkl before start (use with --fresh-start).
  --keep-checkpoints <int>  Keep latest N checkpoints. Default: 20
  --help                    Show this help.
EOF
}

RUN_ID="$(date +%Y%m%d_%H%M%S)"
SEED=42
HEADLESS=1
FRESH_START=0
CLEAN_MODEL=0
KEEP_CHECKPOINTS=20

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

mkdir -p log
MASTER_LOG="log/overnight_2to3h_${RUN_ID}.log"
touch "$MASTER_LOG"

log() {
  echo "[Overnight2to3h] $*" | tee -a "$MASTER_LOG"
}

run_phase() {
  local phase_tag="$1"
  shift
  local -a cmd=(bash scripts/run_curriculum_autotrain.sh --run-id "${RUN_ID}_${phase_tag}" --seed "$SEED" --headless "$HEADLESS" --keep-checkpoints "$KEEP_CHECKPOINTS" "$@")
  printf '[Overnight2to3h] phase=%s cmd:' "$phase_tag" | tee -a "$MASTER_LOG"
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

log "start run_id=$RUN_ID seed=$SEED headless=$HEADLESS fresh_start=$FRESH_START keep_checkpoints=$KEEP_CHECKPOINTS"
log "target wall-time: about 2-3 hours (machine-dependent)"
log "phase plan: A(bootstrap) -> B(main) -> C(consolidation)"

# Phase A: bootstrap (~20-35 min)
PHASE_A_ARGS=(
  --cycles 3
  --noatt-epochs 2
  --noatt-epochs-after-first 0
  --fix-epochs 35
  --noatt-max-step 300
  --fix-step-start 450
  --fix-step-final 650
  --fix-step-inc 50
  --gate-win-last10 0.05
  --force-step-ramp 0
  --lr-noatt 0.0005
  --lr-fix 0.00035
  --gamma-noatt 0.99
  --gamma-fix 0.985
  --learn-interval-noatt 50
  --learn-interval-fix 12
  --batch-size 128
  --memory-size 12000
  --min-replay-size 2500
  --epsilon-noatt 0.10
  --epsilon-fix 0.65
  --epsilon-inc -0.00003
)
if [[ "$FRESH_START" == "1" ]]; then
  PHASE_A_ARGS+=(--fresh-start-first-block)
fi
run_phase "phaseA" "${PHASE_A_ARGS[@]}"

# Phase B: main training (~85-120 min)
run_phase "phaseB" \
  --cycles 12 \
  --noatt-epochs 0 \
  --fix-epochs 85 \
  --noatt-max-step 300 \
  --fix-step-start 650 \
  --fix-step-final 850 \
  --fix-step-inc 50 \
  --gate-win-last10 0.06 \
  --force-step-ramp 0 \
  --lr-noatt 0.00045 \
  --lr-fix 0.00025 \
  --gamma-noatt 0.99 \
  --gamma-fix 0.985 \
  --learn-interval-noatt 50 \
  --learn-interval-fix 10 \
  --batch-size 128 \
  --memory-size 12000 \
  --min-replay-size 2500 \
  --epsilon-noatt 0.10 \
  --epsilon-fix 0.55 \
  --epsilon-inc -0.00003

# Phase C: consolidation (~45-70 min)
run_phase "phaseC" \
  --cycles 8 \
  --noatt-epochs 0 \
  --fix-epochs 95 \
  --noatt-max-step 300 \
  --fix-step-start 850 \
  --fix-step-final 950 \
  --fix-step-inc 50 \
  --gate-win-last10 0.08 \
  --force-step-ramp 0 \
  --lr-noatt 0.0004 \
  --lr-fix 0.0002 \
  --gamma-noatt 0.99 \
  --gamma-fix 0.985 \
  --learn-interval-noatt 50 \
  --learn-interval-fix 10 \
  --batch-size 128 \
  --memory-size 12000 \
  --min-replay-size 2500 \
  --epsilon-noatt 0.08 \
  --epsilon-fix 0.45 \
  --epsilon-inc -0.00002

log "finished all phases"
log "master log: $MASTER_LOG"
log "phase logs:"
log "  log/auto_curriculum_${RUN_ID}_phaseA.log"
log "  log/auto_curriculum_${RUN_ID}_phaseB.log"
log "  log/auto_curriculum_${RUN_ID}_phaseC.log"
