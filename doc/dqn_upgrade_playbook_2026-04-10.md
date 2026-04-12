# DQN 升级学习手册（历史路线，2026-04-10）

本文档保留为历史记录，用来说明仓库中旧 DQN 训练链路曾做过哪些结构升级。它仍然有阅读价值，但不再代表当前主线。

## 1. 当前定位

当前项目主线已经切到：

- `Sample Factory`
- `APPO/PPO`
- `LSTM`
- `marl_env/*`
- `scripts/train_sf_maca.py`

本文件对应的是旧路线：

- `dqn.py`
- `scripts/train_dqn_pipeline.py`
- `scripts/run_curriculum_autotrain.sh`

如果你现在要做新实验，优先看：

- `doc/sample_factory_marl_plan.md`
- `doc/gpu_env_setup.md`
- `doc/rl_training_postrun_playbook.md`

## 2. 这条旧路线当时解决了什么

当时 DQN 主干主要补了这些问题：

1. 提高值函数训练稳定性：
- Double DQN
- Huber loss
- grad clip

2. 改善网络结构：
- 更紧凑的 CNN
- Dueling head
- 去掉 Dropout

3. 修正训练流程正确性：
- replay buffer 增加 `done`
- terminal 状态不再 bootstrap
- replay 预热 `min_replay_size`

4. 让自动训练脚本能传递新参数：
- `scripts/run_curriculum_autotrain.sh`

## 3. 关键代码位置

- 网络与学习器：`dqn.py`
- 单次训练入口：`scripts/train_dqn_pipeline.py`
- 自动课表脚本：`scripts/run_curriculum_autotrain.sh`
- 旧混合课表入口：`scripts/run_fix_mixed_1to2h.sh`
- 旧 foundation-first 入口：
  - `scripts/run_foundation_first_1h.sh`
  - `scripts/run_foundation_first_2to3h.sh`

## 4. 仍然成立的经验

虽然主线已经切换，但下面这些经验仍然有效：

- 终止状态处理必须正确
- 奖励与回报裁剪会强烈影响训练行为
- 观测结构变化通常会导致旧 checkpoint 失配
- 自动化训练脚本必须明确记录参数和日志
- 评估不能只看训练过程中单点 reward

## 5. 不再适合作为当前默认建议的内容

下面这些内容不要再当成“现在应该怎么做”的建议：

- 用 DQN 作为主训练框架
- 把 `train_dqn_pipeline.py` 当成默认训练入口
- 把旧 `model/simple/*.pkl` 视为当前主线 checkpoint
- 用 DQN 的 metrics/summary 文件格式指导 Sample Factory 实验

## 6. 如果你还要复现实验

只有在下面这些场景里，才建议回看或复跑 DQN 路线：

- 你要复现旧实验结果
- 你要比较主线切换前后的行为差异
- 你要学习值函数路线为何在这个任务上遇到结构性瓶颈

## 7. 历史命令模板

这类命令现在只适合做历史复现，不是当前默认入口：

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
cd /root/autodl-tmp/MaCA-project
conda activate maca-py37-min
export PYTHONPATH="$(pwd):$(pwd)/environment:${PYTHONPATH}"

nohup bash scripts/run_curriculum_autotrain.sh \
  --run-id "$RUN_ID" \
  --fresh-start-first-block \
  --headless 1 \
  > "log/auto_curriculum_${RUN_ID}_launcher.log" 2>&1 &
```

## 8. 一句话总结

这份文档记录的是“DQN 路线曾经怎样被修到更像一个合格实验系统”，而不是“当前项目应该怎样继续推进”。
