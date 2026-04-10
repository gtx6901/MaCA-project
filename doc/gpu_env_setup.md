# MaCA GPU 环境配置（本机 Conda 版）

目标：不使用 Docker，直接在本机 Conda 环境跑 MaCA 的 GPU 训练。

## 1. 前置检查

```bash
nvidia-smi
```

能正常输出再继续。

## 2. 创建 GPU 训练环境

说明：MaCA 含 PyArmor 模块，建议固定 Python 3.7。

```bash
cd /home/lehan/MaCA-master
conda env create -f conda/maca-gpu-dev.yml
conda activate maca-py37-min
```

如果环境已存在，可先删除重建：

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

## 5. GPU 训练命令

训练脚本当前续训规则：

- 默认自动续训（若 `model/simple/model.pkl` 存在）
- 强制从头训练：加 `--fresh_start`
- 指定续训点：加 `--resume model/simple/model.pkl`（或其他 checkpoint）

先跑冒烟（推荐先通路）：

```bash
cd /home/lehan/MaCA-master
conda activate maca-py37-min
export PYTHONPATH="$(pwd):$(pwd)/environment:${PYTHONPATH}"
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
python scripts/train_dqn_pipeline.py \
  --epochs 1 \
  --max_step 300 \
  --seed 42 \
  --opponent fix_rule_no_att \
  --metrics_csv log/train_dqn_metrics_gpu_smoke.csv \
  --summary_json log/train_dqn_summary_gpu_smoke.json
```

冒烟通过后跑正式训练：

```bash
cd /home/lehan/MaCA-master
conda activate maca-py37-min
export PYTHONPATH="$(pwd):$(pwd)/environment:${PYTHONPATH}"
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
python scripts/train_dqn_pipeline.py \
  --epochs 100 \
  --max_step 1500 \
  --seed 42 \
  --opponent fix_rule_no_att \
  --metrics_csv log/train_dqn_metrics_stage1_e100.csv \
  --summary_json log/train_dqn_summary_stage1_e100.json
```

## 6. 结果查看

```bash
cat log/train_dqn_summary_gpu_smoke.json
head -n 5 log/train_dqn_metrics_gpu_smoke.csv
```

正式训练：

```bash
cat log/train_dqn_summary_stage1_e100.json
head -n 5 log/train_dqn_metrics_stage1_e100.csv
```

## 7. 常见问题

1. `torch.cuda.is_available() == False`

- 先确认 `nvidia-smi` 正常。
- 重新激活环境：`conda activate maca-py37-min`。
- 重建环境：第 2 节删除并重建。

2. `interface import failed` 或 `ModuleNotFoundError`

- 检查 `PYTHONPATH` 是否包含项目根目录和 `environment` 目录。

3. 训练很慢

- 先确认 `cuda=True`。
- 把 `--batch_size` 从 64 调到 128（显存允许时）。
