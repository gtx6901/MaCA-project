#!/usr/bin/env bash
set -euo pipefail

# Unified HMARL process control.
# Usage:
#   scripts/hmarl_ctl.sh start <exp_name> [resume_from]
#   scripts/hmarl_ctl.sh start-fg <exp_name> [resume_from]
#   scripts/hmarl_ctl.sh stop <exp_name>
#   scripts/hmarl_ctl.sh status <exp_name>
#   scripts/hmarl_ctl.sh logs <exp_name>
#   scripts/hmarl_ctl.sh tb [port]

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
  local resume_from="${2:-200000}"
  local pid_file="${LOG_ROOT}/${exp_name}.pid"
  local log_file="${LOG_ROOT}/train_${exp_name}_resume_${resume_from}_$(date +%Y%m%d_%H%M%S).log"

  if [[ -f "${pid_file}" ]]; then
    local old_pid
    old_pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if is_running "${old_pid}"; then
      echo "[hmarl_ctl] already running exp=${exp_name} pid=${old_pid}"
      echo "[hmarl_ctl] stop first: scripts/hmarl_ctl.sh stop ${exp_name}"
      exit 1
    fi
  fi

  nohup bash scripts/run_hmarl_train.sh "${exp_name}" "" resume "${resume_from}" > "${log_file}" 2>&1 &
  local new_pid=$!
  echo "${new_pid}" > "${pid_file}"

  echo "[hmarl_ctl] started exp=${exp_name} pid=${new_pid}"
  echo "[hmarl_ctl] log=${log_file}"
}

start_fg() {
  local exp_name="$1"
  local resume_from="${2:-200000}"
  echo "[hmarl_ctl] foreground run exp=${exp_name} resume_from=${resume_from}"
  exec bash scripts/run_hmarl_train.sh "${exp_name}" "" resume "${resume_from}"
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
  echo "[hmarl_ctl] tensorboard port=${port}"
  exec conda run -n maca-py37-min python -m tensorboard.main --logdir train_dir/mappo --port "${port}" --host 0.0.0.0
}

case "${cmd}" in
  start)
    if [[ $# -lt 2 ]]; then
      echo "Usage: scripts/hmarl_ctl.sh start <exp_name> [resume_from]"
      exit 1
    fi
    start_bg "$2" "${3:-200000}"
    ;;
  start-fg)
    if [[ $# -lt 2 ]]; then
      echo "Usage: scripts/hmarl_ctl.sh start-fg <exp_name> [resume_from]"
      exit 1
    fi
    start_fg "$2" "${3:-200000}"
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
    run_tb "${2:-6007}"
    ;;
  *)
    echo "Unknown command: ${cmd}"
    echo "Usage: scripts/hmarl_ctl.sh <start|start-fg|stop|status|logs|tb> ..."
    exit 1
    ;;
esac
