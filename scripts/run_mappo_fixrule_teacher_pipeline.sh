#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-mappo_maca_fixrule_teacher_${RUN_ID}}"
CONFIG_PATH="${CONFIG_PATH:-configs/mappo.yaml}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/log}"
BACKGROUND="${BACKGROUND:-1}"
FRESH_START="${FRESH_START:-0}"

mkdir -p "$LOG_DIR"

if [[ "$FRESH_START" == "1" && -d "train_dir/mappo/$EXP_NAME" ]]; then
  rm -rf "train_dir/mappo/$EXP_NAME"
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment${PYTHONPATH:+:$PYTHONPATH}"

cmd=(
  conda run --no-capture-output -n maca-py37-min
  python scripts/train.py
  --config "$CONFIG_PATH"
  --experiment "$EXP_NAME"
)

log_path="$LOG_DIR/${EXP_NAME}.launcher.log"

echo "experiment=$EXP_NAME"
echo "config=$CONFIG_PATH"
echo "log=$log_path"

if [[ "$BACKGROUND" == "1" ]]; then
  setsid env EXP_NAME="$EXP_NAME" "${cmd[@]}" >"$log_path" 2>&1 < /dev/null &
  echo "pid=$!"
else
  env EXP_NAME="$EXP_NAME" "${cmd[@]}" 2>&1 | tee "$log_path"
fi
