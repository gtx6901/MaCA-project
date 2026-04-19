#!/usr/bin/env bash
set -euo pipefail

# Unified HMARL process control.
# Usage:
#   scripts/hmarl_ctl.sh start <exp_name> [resume_from]
#   scripts/hmarl_ctl.sh start-fg <exp_name> [resume_from]
#   scripts/hmarl_ctl.sh stop <exp_name>
#   scripts/hmarl_ctl.sh status <exp_name>
#   scripts/hmarl_ctl.sh logs <exp_name>
#   scripts/hmarl_ctl.sh tb [port] [exp_name|all]
#
# Behavior:
#   - If resume_from is omitted: fresh start (no --resume)
#   - If resume_from is provided: resume from target env steps / checkpoint path
#
# Agent variant selection:
#   - Export HMARL_AGENT_VARIANT=baseline|rule_fire before invoking this script.

LOG_ROOT="train_dir/mappo/_manual_logs"
mkdir -p "${LOG_ROOT}"

cmd="${1:-}"
if [[ -z "${cmd}" ]]; then
  echo "Usage: scripts/hmarl_ctl.sh <start|start-fg|stop|status|logs|tb> ..."
  exit 1
fi

is_running() {
  local pid="$1"
  if [[ -z "${pid}" ]]; then
    return 1
  fi
  kill -0 "${pid}" >/dev/null 2>&1
}

start_bg() {
  local exp_name="$1"
  local resume_from="${2:-}"
  local pid_file="${LOG_ROOT}/${exp_name}.pid"
  local run_tag="fresh"
  if [[ -n "${resume_from}" ]]; then
    run_tag="resume_${resume_from}"
  fi
  local log_file="${LOG_ROOT}/train_${exp_name}_${run_tag}_$(date +%Y%m%d_%H%M%S).log"

  if [[ -f "${pid_file}" ]]; then
    local old_pid
    old_pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if is_running "${old_pid}"; then
      echo "[hmarl_ctl] already running exp=${exp_name} pid=${old_pid}"
      echo "[hmarl_ctl] stop first: scripts/hmarl_ctl.sh stop ${exp_name}"
      exit 1
    fi
  fi

  if [[ -n "${resume_from}" ]]; then
    echo "[hmarl_ctl] mode=resume exp=${exp_name} resume_from=${resume_from}"
    nohup bash scripts/run_hmarl_train.sh "${exp_name}" "" resume "${resume_from}" > "${log_file}" 2>&1 &
  else
    echo "[hmarl_ctl] mode=fresh exp=${exp_name}"
    nohup bash scripts/run_hmarl_train.sh "${exp_name}" > "${log_file}" 2>&1 &
  fi
  local new_pid=$!
  echo "${new_pid}" > "${pid_file}"

  echo "[hmarl_ctl] started exp=${exp_name} pid=${new_pid}"
  echo "[hmarl_ctl] log=${log_file}"
}

start_fg() {
  local exp_name="$1"
  local resume_from="${2:-}"
  if [[ -n "${resume_from}" ]]; then
    echo "[hmarl_ctl] foreground run exp=${exp_name} mode=resume resume_from=${resume_from}"
    exec bash scripts/run_hmarl_train.sh "${exp_name}" "" resume "${resume_from}"
  fi
  echo "[hmarl_ctl] foreground run exp=${exp_name} mode=fresh"
  exec bash scripts/run_hmarl_train.sh "${exp_name}"
}

stop_run() {
  local exp_name="$1"
  local pid_file="${LOG_ROOT}/${exp_name}.pid"

  if [[ ! -f "${pid_file}" ]]; then
    echo "[hmarl_ctl] no pid file for exp=${exp_name}"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if is_running "${pid}"; then
    kill "${pid}" >/dev/null 2>&1 || true
    sleep 1
    if is_running "${pid}"; then
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
    echo "[hmarl_ctl] stopped exp=${exp_name} pid=${pid}"
  else
    echo "[hmarl_ctl] stale pid file removed exp=${exp_name} pid=${pid}"
  fi

  rm -f "${pid_file}"
}

status_run() {
  local exp_name="$1"
  local pid_file="${LOG_ROOT}/${exp_name}.pid"

  if [[ ! -f "${pid_file}" ]]; then
    echo "[hmarl_ctl] status=stopped exp=${exp_name}"
    return 1
  fi

  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if is_running "${pid}"; then
    echo "[hmarl_ctl] status=running exp=${exp_name} pid=${pid}"
    ps -fp "${pid}" || true
    return 0
  fi

  echo "[hmarl_ctl] status=stopped(stale pid) exp=${exp_name} pid=${pid}"
  return 1
}

logs_run() {
  local exp_name="$1"
  local file
  file="$(ls -1t "${LOG_ROOT}"/train_${exp_name}_*.log 2>/dev/null | head -n 1 || true)"
  if [[ -z "${file}" ]]; then
    echo "[hmarl_ctl] no log found for exp=${exp_name}"
    exit 1
  fi
  echo "[hmarl_ctl] tail -f ${file}"
  tail -f "${file}"
}

run_tb() {
  local port="${1:-6007}"
  local exp_name="${2:-all}"

  if [[ "${exp_name}" != "all" ]]; then
    local single_dir="train_dir/mappo/${exp_name}/tb"
    if [[ ! -d "${single_dir}" ]]; then
      echo "[hmarl_ctl] tb dir not found: ${single_dir}"
      exit 1
    fi
    echo "[hmarl_ctl] tensorboard mode=single exp=${exp_name} port=${port} logdir=${single_dir}"
    exec conda run -n maca-py37-min python -m tensorboard.main --logdir "${single_dir}" --port "${port}" --host 0.0.0.0
  fi

  local specs=()
  local tb_dir
  for tb_dir in train_dir/mappo/*/tb; do
    if [[ ! -d "${tb_dir}" ]]; then
      continue
    fi
    if ls -1 "${tb_dir}"/events.out.tfevents.* >/dev/null 2>&1; then
      local exp
      exp="$(basename "$(dirname "${tb_dir}")")"
      specs+=("${exp}:${tb_dir}")
    fi
  done

  if [[ ${#specs[@]} -le 0 ]]; then
    echo "[hmarl_ctl] no event files found, fallback logdir=train_dir/mappo"
    exec conda run -n maca-py37-min python -m tensorboard.main --logdir train_dir/mappo --port "${port}" --host 0.0.0.0
  fi

  local logdir_spec
  logdir_spec="$(IFS=,; echo "${specs[*]}")"
  echo "[hmarl_ctl] tensorboard mode=all port=${port} runs=${#specs[@]}"
  exec conda run -n maca-py37-min python -m tensorboard.main --logdir_spec "${logdir_spec}" --port "${port}" --host 0.0.0.0
}

case "${cmd}" in
  start)
    if [[ $# -lt 2 ]]; then
      echo "Usage: scripts/hmarl_ctl.sh start <exp_name> [resume_from]"
      exit 1
    fi
    start_bg "$2" "${3:-}"
    ;;
  start-fg)
    if [[ $# -lt 2 ]]; then
      echo "Usage: scripts/hmarl_ctl.sh start-fg <exp_name> [resume_from]"
      exit 1
    fi
    start_fg "$2" "${3:-}"
    ;;
  stop)
    if [[ $# -lt 2 ]]; then
      echo "Usage: scripts/hmarl_ctl.sh stop <exp_name>"
      exit 1
    fi
    stop_run "$2"
    ;;
  status)
    if [[ $# -lt 2 ]]; then
      echo "Usage: scripts/hmarl_ctl.sh status <exp_name>"
      exit 1
    fi
    status_run "$2"
    ;;
  logs)
    if [[ $# -lt 2 ]]; then
      echo "Usage: scripts/hmarl_ctl.sh logs <exp_name>"
      exit 1
    fi
    logs_run "$2"
    ;;
  tb)
    run_tb "${2:-6007}" "${3:-all}"
    ;;
  *)
    echo "Unknown command: ${cmd}"
    echo "Usage: scripts/hmarl_ctl.sh <start|start-fg|stop|status|logs|tb> ..."
    exit 1
    ;;
esac
