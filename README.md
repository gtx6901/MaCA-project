# MaCA MAPPO Training Lane

This repository now exposes a research-oriented recurrent MAPPO training lane for MaCA. The active training path is the lightweight centralized-training/decentralized-execution implementation built around:

- shared-parameter recurrent actor
- centralized team critic
- raw-target invalid-action masking
- optional behavior cloning warm start
- periodic fixed-opponent evaluation

The intended one-click entrypoint for the full `fix_rule -> teacher -> BC -> PPO` pipeline is:

```bash
bash scripts/run_mappo_fixrule_teacher_pipeline.sh
```

The plain config-driven entrypoint remains:

```bash
bash run_train.sh
```

This runs:

```bash
python scripts/train.py --config configs/mappo.yaml
```

## Environment

Target runtime:

- Python 3.7
- PyTorch 1.13.1
- CUDA 11.7
- cuDNN 8.5

Install dependencies:

```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$(pwd)/environment:${PYTHONPATH}"
```

If you use the existing Conda environment in this repo, make sure it exposes the same Python and PyTorch versions.

## Training

Default training command:

```bash
bash scripts/run_mappo_fixrule_teacher_pipeline.sh
```

The default config is [configs/mappo.yaml](/home/lehan/MaCA-master/configs/mappo.yaml). Important defaults:

- `fix_rule` is the main training opponent
- `10` environments and `5` rollout workers target the measured throughput sweet spot on `13900H + RTX 4060 8GB`
- recurrent PPO uses `rollout=128`, `chunk_len=16`, `burn_in=8`
- BC warm start is enabled by default and now uses filtered `fix_rule` teacher data plus recurrent sequence cloning
- deterministic evaluation runs every `1,000,000` environment steps

Useful overrides:

```bash
python scripts/train.py --config configs/mappo.yaml
python scripts/train_mappo_maca.py --help
```

To resume an experiment, set `train.resume: true` in the YAML or pass `--resume` through the low-level trainer.

## Evaluation

Run evaluation against the config-defined checkpoint and opponent:

```bash
python scripts/evaluate.py --config configs/mappo.yaml
```

Override the experiment or output path when needed:

```bash
python scripts/evaluate.py \
  --config configs/mappo.yaml \
  --experiment mappo_maca_fixrule_main \
  --output_json log/mappo_eval.json
```

Training-time evaluations are written under `train_dir/mappo/<experiment>/eval/`. TensorBoard logs are written under `train_dir/mappo/<experiment>/tb/`.

## Behavior Cloning Warm Start

BC warm start is configured in `bc_warm_start` inside the YAML. The default setup now:

- collects `fix_rule` teacher trajectories
- keeps only winning / positive-exchange episodes
- runs recurrent sequence BC on the actor before PPO

When enabled, `scripts/train.py` will:

1. collect teacher trajectories with `scripts/collect_teacher_maca.py` if no dataset path is provided
2. pretrain the actor with `scripts/pretrain_bc_maca.py`
3. launch recurrent MAPPO training from that checkpoint

This keeps the warm start on the same experiment path so the trainer can resume directly from the BC checkpoint.

## Project Structure

- [configs/mappo.yaml](/home/lehan/MaCA-master/configs/mappo.yaml): top-level training config
- [run_train.sh](/home/lehan/MaCA-master/run_train.sh): one-click launcher
- [scripts/run_mappo_fixrule_teacher_pipeline.sh](/home/lehan/MaCA-master/scripts/run_mappo_fixrule_teacher_pipeline.sh): one-click teacher-to-PPO launcher
- [scripts/hmarl_ctl.sh](/home/lehan/MaCA-master/scripts/hmarl_ctl.sh): HMARL process control helper
- [scripts/run_hmarl_train.sh](/home/lehan/MaCA-master/scripts/run_hmarl_train.sh): HMARL training launcher
- [scripts/run_hmarl_eval.sh](/home/lehan/MaCA-master/scripts/run_hmarl_eval.sh): HMARL evaluation launcher
- [scripts/train.py](/home/lehan/MaCA-master/scripts/train.py): unified config-driven training entrypoint
- [scripts/evaluate.py](/home/lehan/MaCA-master/scripts/evaluate.py): unified config-driven evaluation entrypoint
- [scripts/train_mappo_maca.py](/home/lehan/MaCA-master/scripts/train_mappo_maca.py): recurrent MAPPO trainer
- [scripts/eval_mappo_maca.py](/home/lehan/MaCA-master/scripts/eval_mappo_maca.py): checkpoint evaluator
- [scripts/collect_teacher_maca.py](/home/lehan/MaCA-master/scripts/collect_teacher_maca.py): teacher dataset collector
- [scripts/pretrain_bc_maca.py](/home/lehan/MaCA-master/scripts/pretrain_bc_maca.py): behavior cloning warm start
- [marl_env/mappo_env.py](/home/lehan/MaCA-master/marl_env/mappo_env.py): structured environment wrapper with centralized state and target masks
- [marl_env/mappo_model.py](/home/lehan/MaCA-master/marl_env/mappo_model.py): shared recurrent actor and centralized critic

## Notes

- The repository has been cleaned to keep MAPPO as the only active training lane.
- Periodic evaluation is centered on `fix_rule`, which is the real acceptance gate for this project.
