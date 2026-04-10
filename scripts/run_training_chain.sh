#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment:${PYTHONPATH:-}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"

EPOCHS="${EPOCHS:-100}"
MAX_STEP="${MAX_STEP:-1500}"
LEARN_INTERVAL="${LEARN_INTERVAL:-150}"
BATCH_SIZE="${BATCH_SIZE:-320}"
SEED="${SEED:-42}"
EVAL_ROUNDS="${EVAL_ROUNDS:-50}"

python scripts/train_dqn_pipeline.py \
  --epochs "$EPOCHS" \
  --max_step "$MAX_STEP" \
  --learn_interval "$LEARN_INTERVAL" \
  --batch_size "$BATCH_SIZE" \
  --seed "$SEED" \
  --opponent fix_rule_no_att \
  --metrics_csv log/train_dqn_metrics.csv \
  --summary_json log/train_dqn_summary.json

python scripts/eval_dqn_model.py \
  --agent1 simple \
  --agent2 fix_rule_no_att \
  --rounds "$EVAL_ROUNDS" \
  --max_step "$MAX_STEP" \
  --seed "$SEED" \
  --metrics_csv log/eval_simple_vs_fix_rule_no_att.csv \
  --summary_json log/eval_simple_vs_fix_rule_no_att_summary.json

python scripts/eval_dqn_model.py \
  --agent1 simple \
  --agent2 fix_rule \
  --rounds "$EVAL_ROUNDS" \
  --max_step "$MAX_STEP" \
  --seed "$SEED" \
  --metrics_csv log/eval_simple_vs_fix_rule.csv \
  --summary_json log/eval_simple_vs_fix_rule_summary.json

printf '\nTraining chain finished.\n'
printf '  train summary: %s\n' "log/train_dqn_summary.json"
printf '  eval summary : %s\n' "log/eval_simple_vs_fix_rule_no_att_summary.json"
printf '  eval summary : %s\n' "log/eval_simple_vs_fix_rule_summary.json"
