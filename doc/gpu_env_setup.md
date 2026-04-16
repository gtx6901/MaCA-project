# MaCA GPU 环境配置（当前主线：MAPPO）

目标：在本机 Conda 环境中跑通当前的 `recurrent MAPPO + centralized critic` 训练链路。

## 1. 前置检查

```bash
nvidia-smi
```

## 2. 创建环境

建议固定：

- Python 3.7
- PyTorch 1.13.1
- CUDA 11.7

```bash
cd <repo-root>
conda env create -f conda/maca-gpu-dev.yml
conda activate maca-py37-min
```

## 3. 运行时变量

```bash
cd <repo-root>
export PYTHONPATH="$(pwd):$(pwd)/environment:${PYTHONPATH}"
```

## 4. 环境自检

```bash
python -c "import interface, torch; print('interface ok'); print(torch.__version__); print('cuda=', torch.cuda.is_available())"
```

## 5. 一键训练

默认入口：

```bash
bash run_train.sh
```

该入口会优先使用 `maca-py37-min` Conda 环境，并加载 `configs/mappo.yaml`。

## 6. 直接运行配置入口

```bash
conda run --no-capture-output -n maca-py37-min \
  python scripts/train.py --config configs/mappo.yaml
```

## 7. 评估

```bash
conda run --no-capture-output -n maca-py37-min \
  python scripts/evaluate.py --config configs/mappo.yaml
```

## 8. 4060 建议参数

默认配置已经面向 `13900H + RTX 4060 Laptop 8GB` 调整：

- `num_envs=8`
- `num_workers=4`
- `rollout=128`
- `chunk_len=16`
- `burn_in=8`
- `hidden_size=256`
- `eval_every_env_steps=1000000`

## 9. 训练产物

训练会将产物写入：

- `train_dir/mappo/<experiment>/checkpoint/`
- `train_dir/mappo/<experiment>/eval/`
- `train_dir/mappo/<experiment>/tb/`

这些目录属于运行时产物，不应作为长期文档或审计记录保留在仓库中。

## 10. 常见问题

1. `torch.cuda.is_available() == False`

- 先确认 `nvidia-smi` 正常。
- 重新激活环境：`conda activate maca-py37-min`。
- 重建环境：按第 2 节重建。

2. `interface import failed` 或 `ModuleNotFoundError`

- 检查 `PYTHONPATH` 是否包含项目根目录和 `environment`。

3. 训练启动后速度偏低

- 优先检查环境步进是否成为瓶颈。
- 在 4060 上先维持默认 `num_envs=8`、`num_workers=4`，再按吞吐结果微调。

4. 文档里提到旧路线

- 当前仓库只保留 MAPPO 主线。
- 如果发现与当前主线无关的历史训练内容，说明文档没有清理干净，应继续删除或改写。
