#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXP_NAME="${1:-}"
EPISODES="${EPISODES:-20}"
DEVICE="${DEVICE:-gpu}"
OPPONENT="${OPPONENT:-fix_rule}"
DETERMINISTIC="${DETERMINISTIC:-True}"
OUTPUT_JSON="${OUTPUT_JSON:-}"

if [[ -z "$EXP_NAME" ]]; then
  EXP_NAME="$(ls -1dt train_dir/mappo/* 2>/dev/null | head -n 1 | xargs -r basename)"
fi

if [[ -z "$EXP_NAME" ]]; then
  echo "No experiment found under train_dir/mappo"
  exit 1
fi

cmd=(
  conda run --no-capture-output -n maca-py37-min
  python scripts/eval_mappo_maca.py
  --experiment "$EXP_NAME"
  --train_dir train_dir/mappo
  --episodes "$EPISODES"
  --device "$DEVICE"
  --maca_opponent "$OPPONENT"
  --deterministic "$DETERMINISTIC"
  --progress True
)

if [[ -n "$OUTPUT_JSON" ]]; then
  cmd+=(--output_json "$OUTPUT_JSON")
fi

printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
