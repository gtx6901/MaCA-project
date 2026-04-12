# MaCA Sample Factory 训练复盘（2026-04）

## 项目背景

将 MaCA 10v10 空战仿真从旧版 DQN 迁移到 **Sample Factory 1.x + APPO/PPO + LSTM** 的多智能体共享策略训练框架。红方 10 架战斗机共享一个策略网络，蓝方为固定规则 (`fix_rule`) 对手。

---

## 问题与解决方案

### 1. Batch size 与多 agent 错配

**现象**：Policy lag 偏高，GPU 大量空等。

**原因**：设置 `batch_size=256` 时忘记乘以 agent 数量。实际每步产生数据量为 `num_workers × rollout × num_agents = 4 × 64 × 10 = 2560`，batch 远小于数据生产速率。

**解决**：将 batch_size 对齐到 `num_workers × rollout × num_agents`，最终采用 8 workers → `batch_size=5120`。

---

### 2. `train_for_seconds` 续训失效

**现象**：续训命令启动后立即停止，日志显示时间已超限。

**原因**：Sample Factory 从原始实验开始时间计算已用秒数，不是从本次启动计算。续训时 7200s 早已耗尽。

**解决**：续训一律设 `TRAIN_SECONDS=999999`，用 `TRAIN_ENV_STEPS` 或手动 kill 控制训练时长。

---

### 3. 观测编码中的航向角表示

**现象**：早期训练策略对方向无感，动作随机性高。

**原因**：原始航向角直接线性编码，±180° 边界不连续（179° 与 -179° 在数值上差距极大，实际方向相邻）。

**解决**：将航向角改为 `sin(course) + cos(course)` 双分量编码，消除边界不连续问题。`measurements` 维度从 6 维扩展到 7 维。

---

### 4. Win rate 持续为 0

**现象**：训练数十万步，win_rate 始终为 0，episode reward 全负。

**原因（多个叠加）**：
- `gamma=0.995` 折扣过短，无法感知回合终结奖励
- `reward_clip=30` 截断了击杀奖励（击杀 fighter 原始奖励 420，缩放后 2.1，但正向信号仍被削弱）
- batch_size 不对齐导致 policy lag 高，梯度更新效率低

**解决**：
- `gamma: 0.995 → 0.999`
- `reward_clip: 30 → 50`
- rollout/recurrence 对齐（均为 64），batch_size 对齐

---

### 5. Agent 被动原地旋转（根本问题）

**现象**：通过 GUI 渲染观察发现 agent 在原地高频旋转，完全不主动进攻，等蓝方进入射程后才被动开火。win_rate 极低（0.02）。

**根本原因**：`reward_radar_fighter_fighter = 50` 每一步只要雷达探测到敌机就给 +50 原始奖励（缩放后 +0.25/step）。一局 943 步 × 探测约 5 架敌机 ≈ **累积雷达奖励 ~1179（缩放后）**，远超单次击杀的 2.1。策略学会了"原地转圈让敌方飞近、持续触发雷达奖励"比主动进攻性价比更高。

**解决（奖励结构重设计）**：

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `reward_radar_fighter_fighter` | 50 | 8 | 消除被动雷达刷分 |
| `reward_radar_fighter_detector` | 50 | 8 | 一致性 |
| `reward_strike_act_valid` | 5 | 0 | 消除被动开火奖励 |
| `reward_strike_fighter_success` | 420 | 650 | 提高主动击杀相对价值 |
| `reward_keep_alive_step` | -1 | -2 | 增加被动生存机会成本 |
| `reward_draw` | -500 | -1500 | 严惩耗时拖局 |

由于奖励结构变化幅度大，放弃续训、全新起跑重建值函数。

---

### 6. 评估脚本卡住

**现象**：运行 `eval_sf_maca.py` 后进程挂起不退出。

**原因**：传入 `--render=True` 而非正确的 `--maca_render=True`，导致参数未被识别，渲染逻辑异常阻塞。

**解决**：使用正确的参数名 `--maca_render=False`（评估时关闭渲染）。

---

### 7. PID 文件损坏

**现象**：`cat log/*.pid` 输出非数字内容，无法用于 kill。

**原因**：后台命令写入 pid 文件的方式有误，输出混入了 shell echo 文本。

**解决**：直接用 `pkill -f "train_sf_maca.py"` 终止进程，绕过 pid 文件。

---

## 关键经验总结

1. **多智能体 batch size 必须乘以 agent 数量**，否则 policy lag 爆炸、GPU 利用率低下。
2. **奖励工程是 MARL 最容易踩坑的地方**：任何每步触发的奖励项，哪怕单步很小，乘以几百步后都可能主导回合总回报，导致意料之外的策略涌现。必须在设计阶段就做量级估算。
3. **GUI 渲染是诊断行为的最快手段**，日志曲线无法揭示策略的具体行为模式。
4. **续训与奖励结构大改不兼容**：值函数已对旧结构校准，奖励大幅修改后必须全新起跑。
5. `train_for_seconds` 在 Sample Factory 中是绝对时间而非相对时间，续训场景下应禁用。
