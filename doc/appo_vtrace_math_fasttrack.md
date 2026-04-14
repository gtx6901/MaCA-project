# APPO + V-trace 从零到工程（重写版）

适用对象：你当前水平（学到蒙特卡洛，想完整看懂本项目）  
目标：不跳步搞懂“公式从哪来、怎么更新、在代码里对应什么”

---

## 0. 先回答你最关心的三个问题

### Q1. “用 $\delta_t$ 做一步更新”到底怎么更？

先定义 TD 误差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

如果是表格法（每个状态一个值）：

$$
V(s_t) \leftarrow V(s_t) + \alpha \delta_t
$$

如果是神经网络价值函数 $V_w(s)$（参数是 $w$）：

$$
w \leftarrow w + \alpha \delta_t \nabla_w V_w(s_t)
$$

这就是“用 $\delta_t$ 一步更新”的完整数学含义。

### Q2. 策略梯度定理怎么来的？基线怎么来的？Actor-Critic 是啥？

文档后面第 4-6 节完整推导，结论先给你：

$$
\nabla_\theta J(\theta)=E[\nabla_\theta \log \pi_\theta(a_t|s_t)\,Q^\pi(s_t,a_t)]
$$

再减一个只依赖状态的基线 $b(s_t)$ 不改变期望梯度：

$$
E[\nabla_\theta \log \pi_\theta(a_t|s_t)\,b(s_t)] = 0
$$

所以可写成优势形式：

$$
\nabla_\theta J(\theta)=E[\nabla_\theta \log \pi_\theta(a_t|s_t)\,A_t],\ A_t=Q_t-V_t
$$

Actor-Critic：Actor 学策略 $\pi_\theta$，Critic 学价值 $V_w$，Critic 给 Actor 提供 $A_t$。

### Q3. PPO/APPO 是啥，全称和名字什么意思？

- PPO：**Proximal Policy Optimization**  
  - Proximal = “近端/别走太远”
  - 核心：每次更新限制新旧策略差异，避免一步走崩
- APPO：**Asynchronous PPO**  
  - A = Asynchronous（异步采样/更新）
  - 异步会带来 off-policy 偏差，所以常配 V-trace 修正

---

## 1. 你现在的项目数学地图（先建立全局）

本项目主线（代码事实）：

- 算法：APPO + V-trace
- 结构：共享策略 + LSTM（10 红机共享参数）
- 动作：离散 336 + action mask
- 训练：异步采样，有 policy lag

必须掌握 6 条主线：

1. 值函数与 Bellman  
2. TD 更新  
3. 策略梯度  
4. 基线与优势  
5. PPO clip 目标  
6. V-trace off-policy 修正

---

## 2. 从 MC 到 TD：为什么要 $\delta_t$

你已知 MC 回报：

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
$$

MC 问题：方差大，必须等回合结束。

TD(0) 用 bootstrapping：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

解释：  
- 如果 $\delta_t > 0$：当前 $V(s_t)$ 估小了  
- 如果 $\delta_t < 0$：当前 $V(s_t)$ 估大了  

所以用它做一步纠偏更新（见第 0 节 Q1）。

---

## 3. 神经网络版 Critic 更新（你工程实际用的）

常见损失：

$$
L_v(w)=\frac{1}{2}\delta_t^2
$$

对 $w$ 求梯度并做梯度下降，等价得到：

$$
w \leftarrow w + \alpha \delta_t \nabla_w V_w(s_t)
$$

这条式子很重要：你日志里 `vl`（value loss）爆高时，说明 $V$ 的目标尺度或方差出问题，更新会很不稳。

---

## 4. 策略梯度定理：从目标函数推到更新公式（不跳步）

目标函数：

$$
J(\theta)=E_{\tau\sim\pi_\theta}[R(\tau)]
$$

写成轨迹求和：

$$
J(\theta)=\sum_\tau p_\theta(\tau)R(\tau)
$$

求导：

$$
\nabla_\theta J(\theta)=\sum_\tau \nabla_\theta p_\theta(\tau)R(\tau)
$$

用恒等式 $\nabla p = p\nabla \log p$：

$$
\nabla_\theta J(\theta)=\sum_\tau p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)R(\tau)
$$

即：

$$
\nabla_\theta J(\theta)=E_{\tau\sim\pi_\theta}[\nabla_\theta\log p_\theta(\tau)R(\tau)]
$$

轨迹概率中与 $\theta$ 相关的只有策略项，所以：

$$
\log p_\theta(\tau)=const+\sum_t \log \pi_\theta(a_t|s_t)
$$

代入后可得策略梯度定理：

$$
\nabla_\theta J(\theta)=E[\nabla_\theta\log\pi_\theta(a_t|s_t)\,Q^\pi(s_t,a_t)]
$$

---

## 5. 常用基线怎么来？为什么不改变期望

加基线后的项：

$$
E[\nabla_\theta\log\pi_\theta(a_t|s_t)\,b(s_t)]
$$

对固定状态 $s_t$ 看动作期望：

$$
\sum_a \pi_\theta(a|s_t)\nabla_\theta\log\pi_\theta(a|s_t)b(s_t)
=
b(s_t)\sum_a \nabla_\theta \pi_\theta(a|s_t)
=
b(s_t)\nabla_\theta \sum_a \pi_\theta(a|s_t)
=0
$$

所以可以安全减去基线降方差。常用 $b(s_t)=V(s_t)$，得到：

$$
A_t=Q_t-V_t
$$

$$
\nabla_\theta J(\theta)=E[\nabla_\theta\log\pi_\theta(a_t|s_t)\,A_t]
$$

---

## 6. Actor-Critic 到底是什么

- Actor：参数 $\theta$，输出策略 $\pi_\theta(a|s)$
- Critic：参数 $w$，输出价值 $V_w(s)$（或 $Q_w$）

Actor 更新（提高高优势动作概率）：

$$
\theta \leftarrow \theta + \alpha \nabla_\theta \log\pi_\theta(a_t|s_t)\,A_t
$$

Critic 更新（拟合价值）：

$$
w \leftarrow w + \beta \delta_t \nabla_w V_w(s_t)
$$

你可以记成：Actor 管“做什么”，Critic 管“做得值不值”。

---

## 7. PPO 是什么，为什么叫这个名字

全称：**Proximal Policy Optimization**。  
“Proximal” 就是每次更新保持“近端”，别离旧策略太远。

定义概率比：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
$$

PPO-Clip 目标：

$$
L_{clip}=E[\min(r_tA_t,\ clip(r_t,1-\epsilon,1+\epsilon)A_t)]
$$

直觉：  
- $A_t>0$：希望增大该动作概率，但别增太猛  
- $A_t<0$：希望减小该动作概率，但别减太猛

---

## 8. APPO 的 A 是什么

A = **Asynchronous**。  

APPO = 异步版 PPO：多个 actor 并行采样，learner 端持续训练。吞吐高，但样本来自“稍旧策略”，形成 off-policy 偏差（policy lag）。

这正是 V-trace 要解决的问题。

---

## 9. V-trace：你这套方案稳定性的关键

行为策略（采样时）：$\mu$  
目标策略（更新时）：$\pi$

重要性比率截断：

$$
\rho_t=\min(\rho_{max},\frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}),\;
c_t=\min(c_{max},\frac{\pi(a_t|s_t)}{\mu(a_t|s_t)})
$$

TD 残差修正：

$$
\delta_t^V=\rho_t(r_t+\gamma V(s_{t+1})-V(s_t))
$$

V-trace 目标：

$$
v_s=V(s)+\sum_{t=s}^{s+n-1}\gamma^{t-s}(\prod_{i=s}^{t-1}c_i)\delta_t^V
$$

Actor 常用优势近似：

$$
A_t\approx \rho_t(r_t+\gamma v_{t+1}-V(s_t))
$$

一句话：V-trace 在“纠偏”和“控方差”之间做折中。

---

## 10. 回到你工程：为什么会出现“有机会但不开火”

你看到的模式（`attack_opportunity` 高、`fire_action_frac` 低）通常来自：

1. 奖励设计让“保命/拖局”更稳  
2. 开火回报稀疏且延迟，信用分配困难  
3. Critic 不稳导致优势估计噪声大（`vl` 尖峰就是信号）  
4. 异步 + 探索参数不当

所以这是“目标函数与优化稳定性问题”，不只是“训练时间不够”。

---

## 11. 你一周内该怎么学（按这份文档）

Day 1-2：第 2-6 节，手推 TD 与策略梯度  
Day 3：第 7 节，搞懂 PPO clip  
Day 4：第 8-9 节，搞懂 APPO 与 V-trace  
Day 5：把第 10 节对应到你日志指标  
Day 6-7：独立写一份“为什么这轮失败、下一轮怎么改”的数学审计

---

## 12. 最小背诵版（必须会）

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
w \leftarrow w + \alpha \delta_t \nabla_w V_w(s_t)
$$

$$
\nabla_\theta J(\theta)=E[\nabla_\theta\log\pi_\theta(a_t|s_t)\,A_t]
$$

$$
L_{clip}=E[\min(r_tA_t,\ clip(r_t,1-\epsilon,1+\epsilon)A_t)]
$$

$$
v_s=V(s)+\sum_{t=s}^{s+n-1}\gamma^{t-s}(\prod_{i=s}^{t-1}c_i)\delta_t^V
$$
