# Sample Factory MARL Plan

This document records the current engineering plan for MaCA on a single GPU host.

## Goal

Train a shared red-fighter policy against `fix_rule` with a stable, reproducible `Sample Factory` pipeline, then iterate through evaluation-driven tuning.

## Current Direction

The project has already switched its mainline to:

- environment: custom multi-agent wrapper in `marl_env/maca_parallel_env.py`
- framework: `Sample Factory 1.x`
- algorithm: `APPO` with `V-trace`
- policy: shared fighter policy
- memory: `LSTM`
- action validity: policy-level masking plus env-side fallback

Legacy DQN scripts still exist, but they are no longer the default path for new experiments.

## Current Status

As of `2026-04-11`, the following pieces are implemented and wired together:

- parallel-style red fighter wrapper
- Sample Factory environment adapter
- custom encoder for image + measurements
- env/model registration with MaCA-specific defaults
- training entry with compatibility patches
- standalone evaluation script
- GPU smoke launcher
- 4060 baseline launcher
- 4080 fresh-start launcher

## Important Code Truths

The following are true in the current codebase and should be treated as source of truth:

- agent count is fixed to 10 red fighters
- dead fighters stay in the agent set and only allow no-op
- action space size is `336`
- action masks are generated from missile count, target id, and distance
- measurements are now `7`-dimensional, not `6`
- heading is encoded as `sin/cos`, not as a linear scalar
- `reward_clip` is `50.0`
- `gamma` is `0.999`

## Current Default Hyperparameters

These are the current defaults in `marl_env/sample_factory_registration.py`, and `scripts/run_sf_maca_4060_baseline.sh` uses the same main settings unless explicitly overridden.

| Parameter | Current value | Notes |
|---|---|---|
| `algo` | `APPO` | async PPO with V-trace |
| `hidden_size` | `256` | single-layer LSTM core |
| `rollout` | `64` | trajectory chunk length |
| `recurrence` | `64` | BPTT length |
| `num_workers` | `6` | current code default |
| `batch_size` | `3840` | current code default |
| `ppo_epochs` | `4` | sample reuse per batch |
| `learning_rate` | `1e-4` | fresh-training default |
| `gamma` | `0.999` | effective horizon aligned with `max_step=1000` |
| `reward_scale` | `0.005` | scales raw env rewards |
| `reward_clip` | `50.0` | preserves fully-winning signal after scaling |
| `max_policy_lag` | `15` | stale-sample control |
| `exploration_loss_coeff` | `0.02` | entropy regularization |
| `keep_checkpoints` | `20` | baseline script default |

## Why Earlier Notes Look Different

Older notes in this repo mentioned settings such as:

- `num_workers = 4`
- `batch_size = 256`
- `rollout = 32`
- `recurrence = 32`

Those were intermediate audit recommendations from an earlier tuning stage. They do not match the current registered defaults or the current baseline launcher anymore. When in doubt, follow the code.

## Recommended Commands

### Smoke Test

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="sf_maca_gpu_smoke_${RUN_ID}" \
bash scripts/run_sf_maca_gpu_smoke.sh
```

### Baseline Training

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="sf_maca_4060_fixrule_${RUN_ID}" \
TRAIN_SECONDS=7200 \
bash scripts/run_sf_maca_4060_baseline.sh
```

### Evaluation

```bash
conda run --no-capture-output -n maca-py37-min \
  python scripts/eval_sf_maca.py \
  --experiment="$EXP_NAME" \
  --train_dir=train_dir/sample_factory \
  --episodes=30
```

## Work Phases

### Phase 1: Environment Standardization

Deliverables:

- fixed red fighter slots
- dict observations
- action masks
- dead-agent handling

Status:

- complete

### Phase 2: Framework Adapter

Deliverables:

- Sample Factory env wrapper
- compatible observation/action specs
- stable reset/step behavior

Status:

- complete

### Phase 3: Baseline Training Loop

Deliverables:

- GPU smoke path
- 4060 baseline launcher
- checkpoint save compatibility
- action mask patch
- evaluation script

Status:

- complete, but still being tuned

### Phase 4: Evaluation-Driven Tuning

Focus:

- compare worker count and batch size against learner backlog
- monitor invalid action fraction
- track win rate with fixed evaluation episodes
- tune against `fix_rule`, not only against training reward

Status:

- active

### Phase 5: Generalization

Possible next steps:

- opponent pool
- periodic cross-checkpoint evaluation
- self-play only after fixed-opponent baseline is reliable

Status:

- pending

## Practical Rule

For this repo, "baseline" means "the path that can be launched, resumed, checkpointed, and independently evaluated today", not "the best-known policy already found". This document should therefore follow the current scripts, not older tuning notes.
