#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment${PYTHONPATH:+:$PYTHONPATH}"

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "maca-py37-min"; then
  exec conda run --no-capture-output -n maca-py37-min python scripts/train.py --config configs/mappo.yaml "$@"
fi

exec python scripts/train.py --config configs/mappo.yaml "$@"
