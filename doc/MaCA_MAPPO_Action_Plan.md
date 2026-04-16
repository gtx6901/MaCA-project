MaCA 多智能体对抗项目最终技术报告

版本：Final v1.0（行动纲领）

一、执行摘要

本报告综合 MAPPO 与深度 Q 学习两条技术路线的研究成果，提出面向 MaCA 多智能体空战项目的最优工程方案。结论如下：

主线算法选择：MAPPO
当前仓库已完成向 MAPPO 的迁移，应继续深化而非回退。
MAPPO 在 cooperative MARL 中表现稳定，适合复杂对抗环境。
关键瓶颈
序列训练不足（伪 recurrent PPO）。
多智能体 credit assignment 过度团队化。
观测缺乏动力学与时间记忆。
机动与攻击决策未耦合。
工程目标
目标一：稳定击败固定策略（fix_rule）。
目标二：显著提升训练效率。
核心策略
构建“目标中心的 MAPPO”架构。
引入序列训练、因子化 credit、实体建模和条件化动作头。
将深度 Q 学习作为备选技术路线。
二、当前系统评估
2.1 已确认架构
主训练框架：Lightweight MAPPO。
Actor：共享参数 + GRU + 双离散动作头。
Critic：集中式价值网络。
执行范式：CTDE（集中训练、分散执行）。
2.2 核心问题
问题	描述	优先级
伪序列训练	RNN未按时间展开训练	P0
信用分配模糊	所有智能体共享团队优势	P1
观测缺乏动力学	无速度、角速度、目标记忆	P1
动作结构不合理	course与attack并列输出	P1
奖励设计偏差	proxy指标与胜负脱节	P2

这些问题是策略难以击败 fix_rule 的根本原因。

三、文献综述与技术启示
3.1 多智能体强化学习核心论文
方法	论文	启示
MAPPO	Yu et al., 2022	PPO可作为强MARL基线
COMA	Foerster et al., 2018	解决信用分配问题
FACMAC	Peng et al., 2021	因子化集中式critic
HAPPO	Zhao et al., 2021	多智能体稳定更新
QMIX	Rashid et al., 2018	值分解方法
VDN	Sunehag et al., 2017	协作任务值分解
DRQN	Hausknecht & Stone, 2015	处理部分可观测性
R2D2	Kapturowski et al., 2019	序列经验回放
3.2 空战与对抗强化学习
方法	贡献
LSTM-PPO空战研究	提升目标分配与追击能力
GraphZero-PPO	使用图结构建模多机关系
Transformer空战策略	强化长时序建模
Curriculum RL for Air Combat	自动课程学习
Coupled Reward for Air Combat	改善奖励设计

这些研究表明，持续接敌能力取决于时序建模、目标选择和信用分配机制。

四、算法路线决策
4.1 Actor-Critic vs Value-Based
维度	MAPPO	QMIX/DQN
稳定性	高	中
样本效率	中	高
连续决策能力	强	弱
部分可观测性	强（RNN）	需额外设计
工程复杂度	中	高
适配空战任务	优	良
结论
主线：MAPPO
备选：QMIX 或 Dueling DQN
五、目标架构设计
5.1 目标中心 MAPPO
Local Observation
        │
   Entity Encoder
        │
      GRU
        │
 ┌──────────────┬──────────────┬──────────────┐
 Maneuver Head  Target Head     Fire Head

Critic结构：

Global State → GRU → Centralized Critic
六、工程改造路线
6.1 P0（必须完成）
1. Recurrent PPO 序列训练

文件：

scripts/train_mappo_maca.py

任务：

实现 sequence-chunk PPO。
支持 burn-in 与 hidden state 继承。

参数建议：

rollout_len = 64
chunk_len = 16
burn_in = 8
2. 修复配置与支持策略开关

确保实验可复现。

6.2 P1（核心优化）
3. Credit Assignment 改进

文件：

marl_env/mappo_env.py

新增：

agent_aux_reward
4. 动力学观测增强

新增特征：

range_rate
bearing_rate
heading_diff
last_seen_age
5. 条件化动作头

文件：

marl_env/mappo_model.py

结构：

course_logits = course_head(h)
attack_logits = attack_head(torch.cat([h, course_embed], dim=-1))
6.3 P2（效率优化）
6. Elite Self-Imitation Warm Start

新增脚本：

collect_teacher_maca.py
pretrain_bc_maca.py
6.4 P3（长期研究）
图神经网络协同策略。
Transformer时序建模。
QMIX备选算法。
七、训练配置建议
参数	建议值
学习率	3e-4
PPO Epochs	5–8
Clip Ratio	0.1
Batch Size	≥ 8192
Value Normalization	启用
GAE λ	0.95
γ	0.99
Max Grad Norm	5.0
八、Codex 编程任务清单
P0
 实现 Recurrent PPO
 支持序列采样与burn-in
 修复配置保存逻辑
P1
 实现 agent-aware credit
 扩展局部观测
 实现条件化动作头
 统一合法动作mask
P2
 实现BC预训练
 构建精英轨迹回放
 优化训练吞吐
P3
 引入GNN协同策略
 实现QMIX备选路线
九、预期成果
阶段	目标
阶段1	能稳定接敌并存活
阶段2	打赢固定策略
阶段3	提升训练效率50%以上
阶段4	形成可发表的研究成果
十、最终结论
MAPPO 是当前最优主线。
最大瓶颈是序列训练与信用分配，而非算法选择。
通过结构优化与论文指导，可显著提升性能。
DQN/QMIX 可作为备选路线，但不应替代主线。
十一、最终行动纲领

立即执行（P0）：

实现 Recurrent MAPPO。

短期执行（P1）：

改进观测、信用分配和动作结构。

中期执行（P2）：

引入 BC 与精英模仿加速训练。

长期执行（P3）：

探索 GNN 与 QMIX。

cd /home/lehan/MaCA-master && conda run -n maca-py37-min env PYTHONPATH=.:./environment python scripts/train_mappo_maca.py --experiment maca_mappo_v2_fresh --train_dir train_dir/mappo --device gpu --seed 42 --num_envs 8 --num_workers 6 --rollout 64 --chunk_len 16 --burn_in 8 --num_mini_batches 4 --ppo_epochs 5   --learning_rate 3e-4 --gamma 0.99 --gae_lambda 0.95 --clip_ratio 0.1 --value_loss_coeff 0.5 --entropy_coeff 0.01 --max_grad_norm 5.0 --hidden_size 256         --team_adv_weight 1.0 --aux_adv_weight 0.5 --maca_opponent fix_rule --maca_delta_course_action True --maca_adaptive_support_policy True --maca_friendly_attrition_penalty 200.0 --maca_enemy_attrition_reward 100.0 --maca_contact_reward 0.1 --maca_progress_reward_scale 0.002 --maca_attack_window_reward 0.1 --maca_agent_aux_reward_scale 0.0 --save_every_sec 600 --log_every_sec 30 --train_for_env_steps 20000000 2>&1 | tee train_dir/mappo/maca_mappo_v2_fresh.log