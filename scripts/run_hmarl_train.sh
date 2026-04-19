#!/usr/bin/env bash
set -euo pipefail

# One-click HMARL train launcher.
# Usage:
#   scripts/run_hmarl_train.sh [experiment_name] [teacher_ckpt] [resume] [resume_from] [agent_variant] [extra_train_args...]
#
# resume_from:
#   - number: target env_steps (e.g. 200000)
#   - path: specific checkpoint file path
#
# agent_variant:
#   - baseline: full_discrete + no rule attack
#   - rule_fire: nearest_target + fire_or_not + disable high-level mode

EXP_NAME="${1:-hmarl_main_$(date +%Y%m%d_%H%M%S)}"
TEACHER_CKPT="${2:-}"
RESUME_MODE="${3:-}"
RESUME_FROM="${4:-}"
AGENT_VARIANT="${5:-${HMARL_AGENT_VARIANT:-baseline}}"
EXTRA_TRAIN_ARGS=()
if [[ $# -ge 6 ]]; then
  EXTRA_TRAIN_ARGS=("${@:6}")
fi

echo "[run_hmarl_train] experiment=${EXP_NAME} profile=conservative_stable agent_variant=${AGENT_VARIANT}"

TRAIN_ARGS=(
  --experiment "${EXP_NAME}"
  --train_for_env_steps 10000000
  --num_envs 12
  --num_workers 4
  --rollout 96
  --chunk_len 24
  --burn_in 8
  --ppo_epochs 6
  --num_mini_batches 6
  --learning_rate 2e-4
  --lr_schedule linear
  --lr_min_ratio 0.33
  --lr_second_half_start_frac 0.5
  --lr_second_half_ratio 0.33
  --clip_ratio 0.12
  --entropy_coeff 0.015
  --max_grad_norm 3.5
  --max_actor_grad_norm 1.5
  --max_critic_grad_norm 2.0
  --save_best_checkpoint true
  --best_checkpoint_source eval_then_train
  --mode_interval 6
  --high_level_loss_coeff 0.3
  --high_level_entropy_coeff 0.002
  --maca_mode_reward_scale 0.8
  --maca_exec_reward_scale 0.35
  --maca_disengage_penalty 0.08
  --maca_bearing_reward_scale 0.08
  --maca_progress_reward_scale 0.003
  --maca_attack_window_reward 0.15
  --maca_agent_aux_reward_scale 0.08
  --curriculum_enabled true
  --curriculum_easy_frac 0.15
  --curriculum_medium_frac 0.5
  --curriculum_easy_opponent fix_rule_no_att
  --curriculum_medium_opponent fix_rule
  --curriculum_full_opponent fix_rule
  --curriculum_easy_max_step 1000
  --curriculum_medium_max_step 1200
  --curriculum_full_max_step 1500
  --curriculum_easy_random_pos false
  --curriculum_medium_random_pos false
  --curriculum_full_random_pos false
  --eval_every_env_steps 100000
  --eval_episodes 30
  --save_every_sec 600
  --log_every_sec 20
  --tensorboard true
)

case "${AGENT_VARIANT}" in
  baseline)
    TRAIN_ARGS+=(
      --attack_rule_mode none
      --attack_policy_mode full_discrete
      --disable_high_level_mode false
    )
    ;;
  rule_fire)
    TRAIN_ARGS+=(
      --attack_rule_mode nearest_target
      --attack_policy_mode fire_or_not
      --disable_high_level_mode true
    )
    ;;
  *)
    echo "[run_hmarl_train] unknown agent_variant=${AGENT_VARIANT}, expected: baseline | rule_fire"
    exit 1
    ;;
esac

if [[ -n "${TEACHER_CKPT}" ]]; then
  echo "[run_hmarl_train] teacher_ckpt=${TEACHER_CKPT} imitation=enabled"
  TRAIN_ARGS+=(
    --teacher_bc_checkpoint "${TEACHER_CKPT}"
    --imitation_coef 0.08
    --imitation_warmup_updates 200
  )
else
  echo "[run_hmarl_train] teacher_ckpt=none imitation=disabled"
fi

if [[ "${RESUME_MODE}" == "resume" ]]; then
  echo "[run_hmarl_train] resume=true"
  TRAIN_ARGS+=(
    --resume
  )
  if [[ -n "${RESUME_FROM}" ]]; then
    if [[ -f "${RESUME_FROM}" ]]; then
      echo "[run_hmarl_train] resume_checkpoint=${RESUME_FROM}"
      TRAIN_ARGS+=(
        --resume_checkpoint "${RESUME_FROM}"
      )
    else
      echo "[run_hmarl_train] resume_env_steps=${RESUME_FROM}"
      TRAIN_ARGS+=(
        --resume_env_steps "${RESUME_FROM}"
      )
    fi
  fi
fi

if (( ${#EXTRA_TRAIN_ARGS[@]} > 0 )); then
  echo "[run_hmarl_train] extra_train_args=${EXTRA_TRAIN_ARGS[*]}"
  TRAIN_ARGS+=("${EXTRA_TRAIN_ARGS[@]}")
fi

conda run --no-capture-output -n maca-py37-min python scripts/train_mappo_maca.py \
  "${TRAIN_ARGS[@]}"

echo "[run_hmarl_train] done experiment=${EXP_NAME}"
