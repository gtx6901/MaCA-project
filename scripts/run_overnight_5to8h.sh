#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_overnight_5to8h.sh [options]

Description:
  5-8 hour overnight curriculum launcher (3 phases):
    Phase A: warm-up (no_att-heavy, short fix horizon)
    Phase B: main training (fix-heavy, progressive horizon)
    Phase C: consolidation (fix at full horizon, lower exploration)

Options:
  --run-id <id>             Master run id. Default: current timestamp.
  --seed <int>              Random seed. Default: 42
  --headless <0|1>          1=dummy SDL. Default: 1
  --fresh-start             Start phase A from scratch (ignore prior model).
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
MASTER_LOG="log/overnight_5to8h_${RUN_ID}.log"
touch "$MASTER_LOG"

log() {
  echo "[Overnight] $*" | tee -a "$MASTER_LOG"
}

run_phase() {
  local phase_tag="$1"
  shift
  local -a cmd=(bash scripts/run_curriculum_autotrain.sh --run-id "${RUN_ID}_${phase_tag}" --seed "$SEED" --headless "$HEADLESS" --keep-checkpoints "$KEEP_CHECKPOINTS" "$@")
  printf '[Overnight] phase=%s cmd:' "$phase_tag" | tee -a "$MASTER_LOG"
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
log "phase plan: A(warm-up) -> B(main) -> C(consolidation)"

# Phase A: warm-up (~1h target depending machine)
PHASE_A_ARGS=(
  --cycles 3
  --noatt-epochs 6
  --fix-epochs 24
  --noatt-max-step 900
  --fix-step-start 500
  --fix-step-final 900
  --fix-step-inc 100
  --gate-win-last10 0.05
  --force-step-ramp 0
  --lr-noatt 0.0005
  --lr-fix 0.00025
  --gamma-noatt 0.99
  --gamma-fix 0.99
  --learn-interval-noatt 50
  --learn-interval-fix 10
  --batch-size 128
  --memory-size 12000
  --min-replay-size 4000
  --epsilon-noatt 0.15
  --epsilon-fix 0.45
  --epsilon-inc -0.00008
)
if [[ "$FRESH_START" == "1" ]]; then
  PHASE_A_ARGS+=(--fresh-start-first-block)
fi
run_phase "phaseA" "${PHASE_A_ARGS[@]}"

# Phase B: main training (~3-5h target depending machine)
run_phase "phaseB" \
  --cycles 7 \
  --noatt-epochs 2 \
  --fix-epochs 42 \
  --noatt-max-step 900 \
  --fix-step-start 900 \
  --fix-step-final 1100 \
  --fix-step-inc 100 \
  --gate-win-last10 0.02 \
  --force-step-ramp 0 \
  --lr-noatt 0.0005 \
  --lr-fix 0.00025 \
  --gamma-noatt 0.99 \
  --gamma-fix 0.99 \
  --learn-interval-noatt 50 \
  --learn-interval-fix 8 \
  --batch-size 128 \
  --memory-size 12000 \
  --min-replay-size 4000 \
  --epsilon-noatt 0.12 \
  --epsilon-fix 0.50 \
  --epsilon-inc -0.00008

# Phase C: consolidation (~1-2h target depending machine)
run_phase "phaseC" \
  --cycles 3 \
  --noatt-epochs 2 \
  --fix-epochs 30 \
  --noatt-max-step 900 \
  --fix-step-start 1100 \
  --fix-step-final 1100 \
  --fix-step-inc 100 \
  --gate-win-last10 0.05 \
  --force-step-ramp 0 \
  --lr-noatt 0.0004 \
  --lr-fix 0.0002 \
  --gamma-noatt 0.99 \
  --gamma-fix 0.99 \
  --learn-interval-noatt 50 \
  --learn-interval-fix 8 \
  --batch-size 128 \
  --memory-size 12000 \
  --min-replay-size 4000 \
  --epsilon-noatt 0.10 \
  --epsilon-fix 0.30 \
  --epsilon-inc -0.00005

log "finished all phases"
log "master log: $MASTER_LOG"
log "phase logs:"
log "  log/auto_curriculum_${RUN_ID}_phaseA.log"
log "  log/auto_curriculum_${RUN_ID}_phaseB.log"
log "  log/auto_curriculum_${RUN_ID}_phaseC.log"
