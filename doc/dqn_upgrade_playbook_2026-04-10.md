# DQN 升级学习手册（2026-04-10）

本文档记录本次对 DQN 主干代码的结构性升级，目标是提升训练稳定性与样本效率，便于后续复现与学习。

---

## 1. 升级目标

旧版训练中，`fix_rule` 阶段长期出现 `0` 胜率，并且对参数较敏感。  
本次升级聚焦三个方向：

1. 强化 DQN 算法稳定性（Double DQN、终止状态处理、鲁棒损失）。
2. 降低网络过大与随机噪声问题（去掉 Dropout，重构编码器）。
3. 改善训练流程质量（replay 预热、自动训练脚本接入新参数）。

---

## 2. 核心代码改动

### 2.1 网络结构：普通 CNN -> Dueling CNN

文件：`dqn.py`

- `NetFighter` 改为更紧凑的卷积编码器（多次 stride 下采样 + `AdaptiveAvgPool2d`）。
- 删除 Dropout，避免 Q 值估计在训练/推理阶段出现额外随机扰动。
- 增加 Dueling Head：
  - `value_head`
  - `adv_head`
  - `Q(s,a)=V(s)+A(s,a)-mean(A)`
- 输入归一化：
  - 图像输入 `img / 255.0`
  - `info` 各维按经验尺度归一化（航向、弹药、距离、ID、角度）。

学习点：
- Dueling 的核心是把“状态价值”和“动作优势”解耦，常见于稀疏奖励或动作多的场景。
- 在值函数方法中，Dropout 往往不如稳定目标网络和损失设计有效。

---

### 2.2 学习器：DQN -> Double DQN + Huber + Grad Clip

文件：`dqn.py`

- 损失：`MSELoss -> SmoothL1Loss(Huber)`
- 优化器：`RMSprop -> Adam`
- 目标值：支持 Double DQN
  - 在线网络选动作 `argmax_a Q_eval(s',a)`
  - 目标网络评估该动作 `Q_target(s', argmax(...))`
- 增加梯度裁剪：`clip_grad_norm_`
- 增加奖励裁剪（默认 `reward_clip=3000`）

学习点：
- Huber 对异常 TD 误差更稳。
- Double DQN 主要解决 Q 值过估计。
- 梯度裁剪对对抗环境中的突发误差很关键。

---

### 2.3 回放记忆：补齐终止状态 done

文件：`dqn.py`, `scripts/train_dqn_pipeline.py`

- replay buffer 新增 `done_memory`
- `store_transition(..., done=False)` 新增终止标记
- 计算目标值时使用：
  - `q_target = r + (1-done) * gamma * q_next`

学习点：
- 这是 DQN 的关键正确性条件。  
  若终局仍 bootstrap 下一状态，会污染目标并引发训练漂移。

---

### 2.4 训练流程：增加 replay 预热

文件：`scripts/train_dqn_pipeline.py`

- 新增参数：`--min_replay_size`（默认 2000）
- 仅当 `memory_counter >= min_replay_size` 才开始 `learn()`
- 在 step 处和 episode 结束处都使用该阈值

学习点：
- 预热能减少早期相关样本导致的过拟合与 Q 值偏移。

---

### 2.5 自动训练脚本接入新参数

文件：`scripts/run_curriculum_autotrain.sh`

- 新增参数：`--min-replay-size`（默认 4000）
- 自动传给训练脚本 `--min_replay_size`

---

## 3. 兼容性与注意事项

1. 本次 `NetFighter` 结构已变，旧 checkpoint 与新网络不兼容。  
   如果直接 `--resume` 旧模型，会报错并提示使用 `--fresh_start`。

2. 推荐本次升级后先从头训练：
   - 清理 `model/simple/*.pkl`
   - 使用 `--fresh_start` 启动首轮

3. 冒烟命令必须保证 `PYTHONPATH` 正确，否则会出现：
   - `ModuleNotFoundError: No module named 'dqn'`

---

## 4. 推荐启动模板（升级后）

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
cd /home/lehan/MaCA-master
conda activate maca-py37-min
export PYTHONPATH="$(pwd):$(pwd)/environment:${PYTHONPATH}"

nohup bash scripts/run_curriculum_autotrain.sh \
  --run-id "$RUN_ID" \
  --fresh-start-first-block \
  --headless 1 \
  > "log/auto_curriculum_${RUN_ID}_launcher.log" 2>&1 &

echo $! > "log/auto_curriculum_${RUN_ID}.pid"
echo "$RUN_ID"
```

---

## 5. 观察指标（建议）

优先看 `fix_rule` block：

1. `win_last10` 是否首次大于 `0`
2. `avg_steps` 是否逐步增加并稳定（不是乱涨乱跌）
3. `reward_last10` 是否稳步抬升（不是单次偶发）

建议每个 cycle 固定记录：

- `fix win_rate_all`
- `fix win_rate_last10`
- `fix reward_last10`
- `fix avg_steps`

---

## 6. 下一步可继续学习的改进点

1. Prioritized Replay（按 TD error 采样）。
2. N-step return（加快稀疏奖励传播）。
3. 动作掩码（非法攻击目标不参与 argmax）。
4. 更细的观测归一化（在 obs 构造阶段做标准化，而非仅网络侧）。

