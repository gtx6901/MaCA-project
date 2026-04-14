#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-sf_maca_4060_8g_curriculum_${RUN_ID}}"
TRAIN_DIR="${TRAIN_DIR:-train_dir/sample_factory}"
ENV_NAME="${ENV_NAME:-maca_aircombat}"
MAX_STEP="${MAX_STEP:-1000}"
SEED="${SEED:-1}"

# Two-stage total schedule (4h default): 1h no_att + 3h pulsed fix_rule
# Each phase duration here is an incremental wall-clock budget for that phase.
PHASE1_SECONDS="${PHASE1_SECONDS:-3600}"
PHASE1_OPPONENT="${PHASE1_OPPONENT:-fix_rule_no_att}"
PHASE2_SECONDS="${PHASE2_SECONDS:-10800}"
PHASE2_OPPONENT="${PHASE2_OPPONENT:-fix_rule}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-70000000}"

# 8GB-safe trainer footprint
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-1}"
ROLLOUT="${ROLLOUT:-32}"
RECURRENCE="${RECURRENCE:-32}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-192}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GAMMA="${GAMMA:-0.999}"
REWARD_SCALE="${REWARD_SCALE:-0.005}"
REWARD_CLIP="${REWARD_CLIP:-50.0}"
MAX_POLICY_LAG="${MAX_POLICY_LAG:-20}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-8}"
SAVE_EVERY_SEC="${SAVE_EVERY_SEC:-1200}"
TRAJ_BUFFERS_EXCESS_RATIO="${TRAJ_BUFFERS_EXCESS_RATIO:-2.0}"

PHASE1_EXPLORATION="${PHASE1_EXPLORATION:-0.06}"
PHASE2_EXPLORATION="${PHASE2_EXPLORATION:-0.045}"
PHASE2_PULSE_EXPLORATION="${PHASE2_PULSE_EXPLORATION:-0.05}"

# Phase-2 pulse block: 15m no_att + 45m fix_rule, repeated 3 times by default
PHASE2_BLOCK_SECONDS="${PHASE2_BLOCK_SECONDS:-3600}"
PHASE2_PULSE_SECONDS="${PHASE2_PULSE_SECONDS:-900}"
PHASE2_MAIN_SECONDS="${PHASE2_MAIN_SECONDS:-2700}"
PHASE2_CYCLES="${PHASE2_CYCLES:-$((PHASE2_SECONDS / PHASE2_BLOCK_SECONDS))}"

FRESH_START="${FRESH_START:-1}"

mkdir -p log "$TRAIN_DIR"

if [[ "$FRESH_START" == "1" && -d "$TRAIN_DIR/$EXP_NAME" ]]; then
	echo "Fresh start enabled, removing existing experiment dir: $TRAIN_DIR/$EXP_NAME"
	rm -rf "$TRAIN_DIR/$EXP_NAME"
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/environment${PYTHONPATH:+:$PYTHONPATH}"

run_phase() {
	local phase_tag="$1"
	local opponent="$2"
	local train_seconds="$3"
	local exploration="$4"
	local log_file="log/${EXP_NAME}.${phase_tag}.log"

	local -a cmd=(
		conda run --no-capture-output -n maca-py37-min python scripts/train_sf_maca.py
		--algo=APPO
		--env="$ENV_NAME"
		--experiment="$EXP_NAME"
		--train_dir="$TRAIN_DIR"
		--device=gpu
		--seed="$SEED"
		--num_workers="$NUM_WORKERS"
		--num_envs_per_worker="$NUM_ENVS_PER_WORKER"
		--worker_num_splits=1
		--rollout="$ROLLOUT"
		--recurrence="$RECURRENCE"
		--batch_size="$BATCH_SIZE"
		--num_batches_per_iteration=1
		--num_minibatches_to_accumulate=1
		--train_for_seconds="$train_seconds"
		--train_for_env_steps="$TOTAL_ENV_STEPS"
		--ppo_epochs="$PPO_EPOCHS"
		--save_every_sec="$SAVE_EVERY_SEC"
		--keep_checkpoints="$KEEP_CHECKPOINTS"
		--experiment_summaries_interval=60
		--decorrelate_envs_on_one_worker=False
		--set_workers_cpu_affinity=False
		--force_envs_single_thread=True
		--train_in_background_thread=False
		--learner_main_loop_num_cores=2
		--traj_buffers_excess_ratio="$TRAJ_BUFFERS_EXCESS_RATIO"
		--with_vtrace=True
		--use_rnn=True
		--rnn_type=lstm
		--hidden_size="$HIDDEN_SIZE"
		--learning_rate="$LEARNING_RATE"
		--gamma="$GAMMA"
		--reward_scale="$REWARD_SCALE"
		--reward_clip="$REWARD_CLIP"
		--exploration_loss_coeff="$exploration"
		--max_policy_lag="$MAX_POLICY_LAG"
		--maca_opponent="$opponent"
		--maca_max_step="$MAX_STEP"
		--maca_render=False
	)

	echo "=== ${phase_tag} | opponent=${opponent} | train_for_seconds=${train_seconds} | exploration=${exploration} ==="
	echo "Log: ${log_file}"
	printf 'Command: '
	printf '%q ' "${cmd[@]}"
	echo

	"${cmd[@]}" 2>&1 | tee "$log_file"
}

remove_done_if_exists() {
	local done_file="${TRAIN_DIR}/${EXP_NAME}/done"
	if [[ -f "$done_file" ]]; then
		echo "Removing done file: $done_file"
		rm -f "$done_file"
	fi
}

if [[ "$PHASE2_CYCLES" -lt 1 ]]; then
	PHASE2_CYCLES=1
fi

if [[ "$PHASE2_MAIN_SECONDS" -lt 1 ]]; then
	echo "PHASE2_MAIN_SECONDS must be >= 1, got: $PHASE2_MAIN_SECONDS"
	exit 1
fi

run_phase "phase1_no_att" "$PHASE1_OPPONENT" "$PHASE1_SECONDS" "$PHASE1_EXPLORATION"
remove_done_if_exists

for ((cycle=1; cycle<=PHASE2_CYCLES; cycle++)); do
	run_phase "phase2_pulse_noatt_c${cycle}" "$PHASE1_OPPONENT" "$PHASE2_PULSE_SECONDS" "$PHASE2_PULSE_EXPLORATION"
	remove_done_if_exists

	run_phase "phase2_fixrule_c${cycle}" "$PHASE2_OPPONENT" "$PHASE2_MAIN_SECONDS" "$PHASE2_EXPLORATION"
	remove_done_if_exists
done

echo "4060 8G curriculum finished for experiment: $EXP_NAME"
