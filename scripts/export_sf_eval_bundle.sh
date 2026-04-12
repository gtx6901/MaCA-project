#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiment_name> [train_dir]"
  exit 1
fi

EXP_NAME="$1"
TRAIN_DIR="${2:-train_dir/sample_factory}"
EXP_DIR="${TRAIN_DIR}/${EXP_NAME}"
CKPT_DIR="${EXP_DIR}/checkpoint_p0"

if [[ ! -f "${EXP_DIR}/cfg.json" ]]; then
  echo "cfg.json not found: ${EXP_DIR}/cfg.json"
  exit 2
fi

LATEST_CKPT="$(ls -1t "${CKPT_DIR}"/checkpoint_*.pth 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "No checkpoint found under: ${CKPT_DIR}"
  exit 3
fi

PACK_ROOT="exports/${EXP_NAME}_eval_latest"
ARCHIVE="exports/${EXP_NAME}_eval_latest.tar.gz"

rm -rf "$PACK_ROOT"
mkdir -p "${PACK_ROOT}/${TRAIN_DIR}/${EXP_NAME}/checkpoint_p0"

cp "${EXP_DIR}/cfg.json" "${PACK_ROOT}/${TRAIN_DIR}/${EXP_NAME}/cfg.json"
cp "${LATEST_CKPT}" "${PACK_ROOT}/${TRAIN_DIR}/${EXP_NAME}/checkpoint_p0/"
cp "scripts/eval_sf_maca.py" "${PACK_ROOT}/eval_sf_maca.py"

cat > "${PACK_ROOT}/README_eval.txt" <<EOF
Experiment: ${EXP_NAME}
Train dir : ${TRAIN_DIR}
Checkpoint: $(basename "${LATEST_CKPT}")

Unpack this archive at project root, then run on local machine:
python scripts/eval_sf_maca.py \
  --algo=APPO \
  --env=maca_aircombat \
  --experiment=${EXP_NAME} \
  --train_dir=${TRAIN_DIR} \
  --device=cpu \
  --episodes=1 \
  --maca_render=True \
  --maca_opponent=fix_rule \
  --maca_max_step=1000
EOF

tar -czf "$ARCHIVE" -C "$PACK_ROOT" .

echo "Bundle created: $ARCHIVE"
ls -lh "$ARCHIVE"
