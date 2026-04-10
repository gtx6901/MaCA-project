# Session Handoff: MaCA GPU Training (2026-04-09)

## 1. Goal

Run and optimize GPU-based RL training in this repository, while keeping simulation usable and stable.

## 2. Environment Status

### Host GPU / Docker status observed in this session

- Host `nvidia-smi` works (RTX 4060 Laptop visible).
- Docker runtime was configured with NVIDIA toolkit successfully.
- `sudo docker run --gpus all ... nvidia-smi` worked.
- User later chose to drop Docker route and use local Conda training directly.

### Local Conda status

- Active env during checks: `maca-py37-min` (Python 3.7.16).
- `scripts/check_maca_env.py` can run.
- A completed training run exists:
  - `log/train_dqn_summary_e20.json`
  - `log/train_dqn_metrics_e20.csv`
  - elapsed ~558.7s for 20 epochs.

## 3. Training Result Snapshot (first completed run)

From `log/train_dqn_summary_e20.json` and first lines of `log/train_dqn_metrics_e20.csv`:

- 20 epochs finished, checkpoint reached `model_000000100.pkl`.
- Win signal exists against `fix_rule_no_att`.
- Reward still fluctuates heavily (not yet stable convergence).

## 4. Code Changes Made In This Session

### Performance-focused changes retained

1. Replay buffer optimization in `dqn.py`
- Replaced list/pop replay memory with fixed-size ring buffer (`numpy` arrays).
- Sampling now uses direct array indexing + `torch.from_numpy`.
- Expected benefit: less Python overhead and less memory churn.

2. Batched action inference in `dqn.py` + training loop usage
- Added `choose_action_batch(...)` in `RLFighter`.
- `scripts/train_dqn_pipeline.py` now batches alive fighters each step and performs one model forward for the batch.
- Expected benefit: fewer tiny forward calls, better CPU/GPU utilization.

3. Reduced per-step Python object overhead in training loop
- `red_fighter_action` preallocated as `numpy` array instead of append+convert pattern.

4. Added `--memory_size` CLI arg to training script
- File: `scripts/train_dqn_pipeline.py`
- Default kept at `500` for behavior compatibility.

### Behavior/parameter consistency choices

- Training defaults were restored to original values:
  - `learn_interval=150`, `batch_size=320`, `lr=0.007`, `gamma=0.8`, `epsilon_increment=-0.0001`, `memory_size=500`.
- Epsilon-greedy semantics are kept compatible with the original code path (batch path follows same greedy/random condition style).

## 5. Validation Done After Latest Edits

- Only static syntax validation was run (no new training started, per user request):

```bash
python -m py_compile dqn.py scripts/train_dqn_pipeline.py obs_construct/simple/construct.py
```

- Result: `OK`.

## 6. Docs Updated In Session

- Added/updated local-env focused guide:
  - `doc/gpu_env_setup.md`
- Deprecated docker guide now points to local guide:
  - `doc/gpu_container_env_setup.md`

## 7. Current User Preference

- User does not want Docker complexity for now.
- User wants manual training runs and then assistant-side analysis/tuning.
- User requested that when they are actively training, assistant should avoid running smoke training automatically.

## 8. Suggested Next Step In Next Chat

1. Run one new training round with current optimized code (same baseline hyperparameters first).
2. Compare speed and curve vs previous `e20` logs:
- per-epoch wall time
- GPU utilization trend
- reward / win stability
3. If speed improves and behavior is stable, then apply staged hyperparameter tuning.

