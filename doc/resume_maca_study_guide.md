# MaCA 课程设计（DQN 多智能体对抗）——简历一致性学习指南

> 目标：让你的“简历描述”与仓库代码 **完全一致**，并且你能在面试中顺着代码把实现讲清楚。

## 1. 你在简历里可以怎么定位这个项目（真实、不夸大）

- **项目一句话**：基于 MaCA（Multi-agent Combat Arena）对抗仿真平台，在同构 10v10 战机对抗场景中实现/跑通 DQN 决策训练闭环；观测采用“网格化图像 + 向量信息”融合，动作采用离散化航向与攻击选择，并结合平台对战调度与回放机制进行行为验证。

- **不建议写的点（容易夸大/埋坑）**
  - “多智能体协同学习 / 通信 / QMIX / MADDPG” —— 本仓库当前示例是 **参数共享 + 独立决策** 的 DQN 形式，并非协同 RL。
  - “分布式并行训练/大规模并行采样” —— 代码未体现。
  - “显著提升胜率 XX%” —— 除非你能复现实验并给出评测协议与结果表。

## 2. 简历 Bullet ↔ 代码证据（建议你面试时就按这个顺序讲）

下面给的是“证据链”，每一条简历描述都能在对应文件中找到。

### 2.1 环境与接口（你真的用到了 RL-API）
- 环境入口与 RL-API：`environment/interface.py` 的 `Environment`
  - `reset()` / `step()` / `get_obs()` / `get_reward()` / `get_done()`
- 文档结构导览：`doc/tutorial.md`

### 2.2 观测空间（图像 + info 向量）
- 图像观测构造：`obs_construct/simple/construct.py`
  - 地图 1000×1000，经 `img_obs_reduce_ratio=10` 下采样为 **100×100** 网格
  - 拼接通道：个体 3 通道（可见敌方 id / 可见敌方类型 / 友方位置）+ 全局 2 通道（被动探测/可见敌方融合）= **5 通道**
  - 向量信息（fighter）：`fighter_data` 维度为 **6**（航向、弹药、最近可见敌距离/ID/方位等）

### 2.3 动作空间（离散化航向×攻击选择）
- 训练/示例中的动作编码思想：
  - 航向离散：`COURSE_NUM = 16`
  - 攻击索引：`ATTACK_IND_NUM = (DETECTOR_NUM+FIGHTER_NUM)*2 + 1`（长/短导弹目标 + no-op）
  - 离散动作总数：`ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM = 16*(2*10+1)=336`
- 典型动作解码逻辑可参考：`main.py`
  - `course = 360/COURSE_NUM * (action // ATTACK_IND_NUM)`
  - `attack = action % ATTACK_IND_NUM`

> 注意：示例里通常只让 DQN 控制 “航向 + 攻击选择”，而雷达频点/干扰频点用固定或随机策略（脚本控制），这是为了把学习难度收敛到核心对抗决策。

### 2.4 DQN 实现（target network + epsilon-greedy + checkpoint）
- DQN 网络与训练逻辑（信息维度为 6 的版本）：`dqn.py`
  - CNN 编码 5 通道 100×100 图像；`info_fc` 使用 6 维向量；融合后输出 Q(s,a)
  - `choose_action()`：epsilon-greedy
  - `learn()`：周期性 `target_net` 同步，保存 checkpoint 到 `model/simple/`

## 3. 仓库里几个“容易踩坑但必须知道”的地方（帮你避免面试翻车）

1) **同名 DQN 有多个版本，info 维度不一致**
- `dqn.py`（根目录）里 `info_fc` 输入是 **6 维**，与 `obs_construct/simple/construct.py` 的 fighter `info`（6 维）一致。
- `train/simple/dqn.py` 与 `agent/simple/dqn.py` 里 `info_fc` 输入是 **3 维**，与当前 `obs_construct/simple/construct.py` **不一致**。

建议：简历与讲述优先对齐 `main.py + dqn.py + obs_construct/simple/construct.py` 这条链路。

2) **训练保存的模型文件名与对战 agent 默认加载不一致**
- `dqn.py` 的 `learn()` 会保存 `model/simple/model_XXXXXXXXX.pkl`
- `agent/simple/agent.py` / `agent/simple/dqn.py` 默认加载的是 `model/simple/model.pkl`

如果你未来要把训练出来的模型接入 `fight_mp.py` 调用的 `simple` agent，需要：
- 手动拷贝/重命名 checkpoint 为 `model/simple/model.pkl`，或
- 修改 agent 的加载路径逻辑

3) **Windows 运行时强依赖“工作目录在 MaCA 根目录 + PYTHONPATH”**
- 运行脚本时请确保工作目录是 `MaCA-master/`
- `environment/` 需要在 `PYTHONPATH`

## 4. 为了让“理解水平”与简历匹配，你需要补哪些知识（最小清单）

### 4.1 强化学习 / DQN（必学）
- MDP：状态/动作/奖励/转移，episode return
- Q-learning 与 Bellman 最优方程
- DQN 三件套：
  - target network（稳定训练）
  - experience replay（打破相关性；本仓库示例更像“收集一段轨迹后整体更新”）
  - epsilon-greedy（探索/利用）
- 训练稳定性：奖励尺度、梯度爆炸/消失、学习率/折扣因子影响

### 4.2 深度学习工程（必学）
- PyTorch 基础：Tensor shape、`nn.Module`、forward、loss、optimizer
- CNN 输入维度（N,C,H,W）与观测构造的对应关系
- GPU 基础：`cuda()`、`map_location`、显存与 batch 的关系

### 4.3 多智能体（了解即可，别夸大）
- 参数共享 vs 独立网络
- 部分可观测性与信用分配（为什么多智能体比单智能体难）

### 4.4 评测与复现（强烈建议补齐）
- 评测协议：对手是谁（fix_rule / selfrule）、跑多少局、统计 win rate / 平均步数 / 伤亡比
- 可复现：固定 seed、记录配置、保存模型与曲线（哪怕是 CSV）

## 5. 重点阅读顺序（建议 60–90 分钟能过一遍）

1. `doc/tutorial.md`（平台架构与模块边界）
2. `environment/interface.py`（RL-API 与 obs_construct 动态加载）
3. `obs_construct/simple/construct.py`（你到底喂给网络什么）
4. `configuration/reward.py`（奖励来源与尺度）
5. `dqn.py`（网络结构 + 训练机制）
6. `main.py`（训练循环、动作编码/解码、与环境交互）
7. `fight_mp.py`（对战调度、胜率统计、日志回放入口）

---

如果你希望我帮你把“简历里的项目描述”进一步硬化（加入可复现指标），下一步就是：跑通 100 局对战评测并保存一张结果表（win%/步数/伤亡比）。我可以再给你一份最小评测脚本与记录模板。