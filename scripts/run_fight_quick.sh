#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment:${PYTHONPATH:-}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
# fight_mp.py asks for ENTER at exit; feed one line to avoid hanging in non-interactive runs.
printf '\n' | python fight_mp.py --round 1 --max_step 200
