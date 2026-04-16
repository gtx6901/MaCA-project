#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXP_NAME="${1:-}"

if [[ -n "$EXP_NAME" ]]; then
  LOG_PATH="$ROOT_DIR/log/${EXP_NAME}.launcher.log"
else
  LOG_PATH="$(ls -1t "$ROOT_DIR"/log/*.launcher.log 2>/dev/null | head -n 1)"
fi

if [[ -z "${LOG_PATH:-}" || ! -f "$LOG_PATH" ]]; then
  echo "No launcher log found."
  exit 1
fi

echo "log=$LOG_PATH"
echo
echo "[process]"
pgrep -af "scripts/train.py --config|scripts/train_mappo_maca.py" || true
echo
echo "[gpu]"
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader || true
echo
echo "[tail]"
tail -n 80 "$LOG_PATH"
