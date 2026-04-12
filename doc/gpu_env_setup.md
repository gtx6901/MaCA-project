# MaCA GPU 环境配置（当前主线：Sample Factory）

目标：在本机 Conda 环境中跑通当前 `Sample Factory + APPO/PPO` 训练链路，而不是旧 DQN 主线。

## 1. 前置检查

```bash
nvidia-smi
```

能正常输出再继续。

## 2. 创建环境

MaCA 自带 PyArmor 依赖，当前建议继续固定 `Python 3.7`。

```bash
cd /home/lehan/MaCA-master
conda env create -f conda/maca-gpu-dev.yml
conda activate maca-py37-min
```

如果环境已存在且损坏，可删除后重建：

```bash
conda env remove -n maca-py37-min
conda env create -f conda/maca-gpu-dev.yml
conda activate maca-py37-min
```

## 3. 配置运行时变量

```bash
cd /home/lehan/MaCA-master
export PYTHONPATH="$(pwd):$(pwd)/environment:${PYTHONPATH}"
```

## 4. 环境自检

```bash
python scripts/check_maca_env.py
python -c "import torch; print(torch.__version__); print('cuda=', torch.cuda.is_available(), 'count=', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

通过标准：

- `MaCA interface import: OK`
- `cuda=True`
- `count>=1`

## 5. 先跑多智能体环境冒烟

这一步不依赖 Sample Factory，只验证并行环境包装是否正常。

```bash
conda activate maca-py37-min
export PYTHONPATH="$(pwd):$(pwd)/environment:${PYTHONPATH}"
python scripts/smoke_test_marl_env.py --steps 5 --opponent fix_rule
```

## 6. 跑 Sample Factory GPU 冒烟

当前最小闭环入口：

```bash
cd /home/lehan/MaCA-master
conda activate maca-py37-min
export PYTHONPATH="$(pwd):$(pwd)/environment:${PYTHONPATH}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="sf_maca_gpu_smoke_${RUN_ID}" \
bash scripts/run_sf_maca_gpu_smoke.sh
```

这个脚本当前默认会使用：

- `num_workers=1`
- `rollout=8`
- `recurrence=1`
- `batch_size=32`
- `train_for_env_steps=320`
- `use_rnn=False`
- `with_vtrace=False`
- `maca_opponent=fix_rule_no_att`

它的目标不是训练出策略，只是确认训练链路能完整运行。

## 7. 跑正式基线训练

当前正式入口：

```bash
cd /home/lehan/MaCA-master
conda activate maca-py37-min
export PYTHONPATH="$(pwd):$(pwd)/environment:${PYTHONPATH}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="sf_maca_4060_fixrule_${RUN_ID}" \
bash scripts/run_sf_maca_4060_baseline.sh
```

当前基线脚本默认值：

- `device=gpu`
- `maca_opponent=fix_rule`
- `maca_max_step=1000`
- `num_workers=8`
- `rollout=64`
- `recurrence=64`
- `batch_size=5120`
- `use_rnn=True`
- `rnn_type=lstm`
- `gamma=0.999`
- `reward_scale=0.005`
- `reward_clip=50.0`

## 8. 后台运行模板

```bash
cd /home/lehan/MaCA-master
conda activate maca-py37-min
export PYTHONPATH="$(pwd):$(pwd)/environment:${PYTHONPATH}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="sf_maca_4060_fixrule_${RUN_ID}"
nohup bash scripts/run_sf_maca_4060_baseline.sh \
  > "log/${EXP_NAME}.launcher.log" 2>&1 &
echo $! > "log/${EXP_NAME}.pid"
```

看日志：

```bash
tail -f "log/${EXP_NAME}.launcher.log"
```

停止：

```bash
kill "$(cat "log/${EXP_NAME}.pid")"
```

## 9. 训练后评估

```bash
conda run --no-capture-output -n maca-py37-min \
  python scripts/eval_sf_maca.py \
  --experiment="$EXP_NAME" \
  --train_dir=train_dir/sample_factory \
  --episodes=30
```

## 10. 常见问题

1. `torch.cuda.is_available() == False`

- 先确认 `nvidia-smi` 正常。
- 重新激活环境：`conda activate maca-py37-min`。
- 重建环境：按第 2 节重建。

2. `interface import failed` 或 `ModuleNotFoundError`

- 检查 `PYTHONPATH` 是否包含项目根目录和 `environment`。

3. `Sample Factory` 能启动但 learner 很慢

- 当前默认 `num_workers=8`、`batch_size=5120` 比较激进。
- 如果日志里持续出现 learner backlog，再考虑下调 worker 数或 batch。

4. 文档里看到旧 DQN 命令

- `scripts/train_dqn_pipeline.py` 仍在仓库，但不是当前默认主线。
- 现在优先使用 `run_sf_maca_gpu_smoke.sh` 和 `run_sf_maca_4060_baseline.sh`。
