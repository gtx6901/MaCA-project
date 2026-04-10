# Sample Factory MARL Plan

This document fixes the next technical direction for MaCA on a single `RTX 4060 8GB`.

## Goal

Train a fighter policy that can achieve non-trivial win rate against `fix_rule`, then extend to stronger opponent pools.

## Why We Are Changing Direction

Recent runs show a repeatable failure mode:

- `fix_rule_no_att` can be learned.
- `fix_rule` remains near-zero win rate.
- Longer `no_att` pretraining improves survival and score, but does not transfer to real combat.

This is not a small hyperparameter problem. The current DQN route is structurally mismatched to:

- multi-agent control
- partial observability
- long horizons
- large discrete action spaces
- non-stationary opponents

## Target Stack

- Environment API: parallel-style multi-agent wrapper
- Training framework: `Sample Factory`
- Algorithm family: `APPO/PPO`
- Policy layout: shared fighter policy
- Memory: `LSTM`
- Invalid moves: action masking at env adapter level
- Opponent training: fixed rule first, then opponent pool / self-play

## Hardware Constraints

`RTX 4060 8GB` means:

- one shared fighter policy, not per-agent policies
- compact CNN encoder
- small recurrent core (`128` hidden size)
- moderate rollout lengths and batch sizes
- no heavy centralized image critic at the first stage

## Current Status

As of `2026-04-10`, the following pieces are complete:

- parallel multi-agent wrapper for red fighters
- Sample Factory env adapter
- compact custom encoder
- env/model registration and training entry point
- GPU smoke test on local host completed end-to-end
- compatibility patches added in `scripts/train_sf_maca.py` for:
  - shared-memory launch restrictions
  - single-trajectory squeeze bug in `Sample Factory 1.x`
  - checkpoint temp filename incompatibility with current `torch`

Smoke-test success criteria already achieved:

- learner initializes on GPU
- policy worker initializes on GPU
- at least one optimizer step completes
- checkpoints are saved successfully
- run exits cleanly

## Recommended Commands

Quick GPU smoke test:

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="sf_maca_gpu_smoke_${RUN_ID}" bash scripts/run_sf_maca_gpu_smoke.sh
```

2-hour 4060 baseline against `fix_rule`:

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="sf_maca_4060_fixrule_${RUN_ID}" \
TRAIN_SECONDS=7200 \
bash scripts/run_sf_maca_4060_baseline.sh
```

## Work Phases

### Phase 1: Standardize the Environment

Deliverables:

- `marl_env/maca_parallel_env.py`
- fixed fighter agent ids
- dict observations per fighter
- per-agent rewards
- action masks
- inactive-agent handling for dead fighters
- global low-dimensional state export

Status:

- complete

### Phase 2: Add a Framework Adapter

Deliverables:

- Sample Factory multi-agent adapter
- observation/action specs compatible with the framework
- smoke-test launch path

Status:

- complete

### Phase 3: Build the First PPO/APPO Baseline

Initial choices:

- shared policy for all red fighters
- compact CNN + info MLP + LSTM
- fixed opponent `fix_rule`
- no dependency on long DQN-style `no_att` pretraining

Status:

- baseline launch script prepared
- smoke path verified
- full training still needs iterative evaluation and tuning

### Phase 4: Improve Generalization

Deliverables:

- opponent pool
- periodic evaluation against fixed snapshots
- learned-vs-learned only after rule-opponent baseline works

Status:

- pending

### Phase 5: Optional Advanced Tuning

Only after Phase 4 works:

- PBT for reward shaping / entropy / lr
- stronger recurrent models
- richer critic state

Status:

- pending
