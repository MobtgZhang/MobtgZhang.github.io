---
title: 进化策略（ES）
date: 2026-11-04 12:00:00 +0800
categories: [智能计算]
tags: [进化策略, ES, 进化计算, 优化算法]
author: MobtgZhang
---

# 1. 背景与历史

&ensp;&ensp;&ensp;&ensp;进化策略（Evolution Strategy, ES）由 Ingo Rechenberg 与 Hans-Paul Schwefel 于 20 世纪 60 年代在德国柏林工业大学提出，是**最早的进化计算方法之一**，比遗传算法早约十年[^1][^2]。Rechenberg 在 1960 年代研究了"三进制选择实验"（three-digit selection experiment）：用一对骰子随机产生 01 串，用其比特数作为目标值，观察最优突变率的存在——这被公认为进化算法的第一次系统实验。

&ensp;&ensp;&ensp;&ensp;ES 的核心洞察与 GA 有本质区别：

| 特征 | GA (Holland) | ES (Rechenberg/Schwefel) |
|------|--------------|--------------------------|
| 起源 | 生物学的基因/染色体隐喻 | 工程学的"随机优化器"思想 |
| 编码 | 抽象的染色体 | **直接的实数向量** |
| 变异 | 随机位翻转（与问题尺度无关） | **高斯扰动**（与问题尺度匹配） |
| 自适应对象 | 交叉率 $p_c$、变异率 $p_m$ | **步长 $\sigma$**（协方差最初是对角的） |
| 选择 | 依赖适应度比例 | 依赖**排序**（不看适应度具体数值） |
| 理论 | 模式定理（离散模式） | **收敛速率分析**（连续高斯过程） |

&ensp;&ensp;&ensp;&ensp;ES 提出的两个关键思想在后来影响了整个进化计算领域：

1. **自适应的变异步长**（self-adaptive step size）——把步长 $\sigma$ 本身作为个体的遗传属性，让算法自己学习该走多远；
2. **1/5 成功规则**（1/5 rule）——一个简洁而深刻的步长自适应启发式。

&ensp;&ensp;&ensp;&ensp;ES 经历了若干发展阶段：

- **$(1+1)$-ES**：最简单的形式，1 个父代产生 1 个子代，若更好则接受；
- **$(\mu,\lambda)$-ES**（Schwefel）：$\mu$ 个父代产生 $\lambda$ 个子代，从**子代中**选最好的 $\mu$ 个；
- **$(\mu+\lambda)$-ES**：从**父代与子代的并集**中选最好的 $\mu$ 个（ elitist );
- **$1/\lambda$-ES 与 $1/\lambda$-ES 的解耦变体**：把"步长适应"与"策略适应"解耦；
- **CMA-ES**（Hansen & Ostermeier, 2001）[^3]：把协方差从对角矩阵推广到**完整的协方差矩阵**，成为现代无导数优化的金标准。

# 2. 问题形式化

&ensp;&ensp;&ensp;&ensp;考虑连续空间的无约束最小化问题：

$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})
$$

&ensp;&ensp;&ensp;&ensp;ES 的每个个体由**目标向量**与**策略参数**共同构成：

$$
\mathbf{y}^{(k)} = \left( \mathbf{x}^{(k)}, \boldsymbol{\sigma}^{(k)}, \mathbf{C}^{(k)} \right)
$$

其中 $\mathbf{x} \in \mathbb{R}^n$ 是解向量，$\boldsymbol{\sigma} \in \mathbb{R}^n_{+}$ 是逐维步长（或全向量的全局步长），$\mathbf{C} \in \mathbb{R}^{n \times n}$ 是协方差矩阵（基础 ES 中取 $\mathbf{C} = \mathbf{I}$）。

&ensp;&ensp;&ensp;&ensp;**ES 的核心哲学**：目标函数 $f$ 只被用于**排序**（rank），而**从不直接进入任何更新公式**。这与 GA 中"适应度值直接决定选择概率"形成鲜明对比。这一"排序即可"（rank-based）性质带来两个巨大优势：

1. $f$ 的**单调变换不影响算法行为**，因此无需担心适应度的量纲、负值或数值溢出；
2. 可以自然地处理**多目标**情形（基于帕累托秩的算子）。

# 3. 数学原理

## 3.1 高斯变异（Gaussian Mutation）

&ensp;&ensp;&ensp;&ensp;ES 的变异算子是加性高斯扰动：

$$
\mathbf{x}' = \mathbf{x} + \boldsymbol{\sigma} \odot \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

&ensp;&ensp;&ensp;&ensp;其中 $\odot$ 表示逐元素（Hadamard）乘积，$\mathcal{N}(\mathbf{0}, \mathbf{I})$ 表示 $n$ 维独立标准正态分布向量。在最简形式中，所有维度共用一个标量步长 $\sigma$：

$$
\mathbf{x}' = \mathbf{x} + \sigma \cdot \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

&ensp;&ensp;&ensp;&ensp;**为什么是高斯分布？** 这是 ES 理论中最为深刻的联系。考虑在最优解附近对目标函数做二阶泰勒展开：

$$
f(\mathbf{x} + \delta) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^{\mathsf{T}} \delta + \frac{1}{2} \delta^{\mathsf{T}} \mathbf{H} \delta
$$

&ensp;&ensp;&ensp;&ensp;当在 $n$ 维空间中随机方向上做投影时，由中心极限定理，投影长度近似服从正态分布；而正态分布是**唯一具有旋转不变性**的分布——它在任何正交变换下形式不变。这个性质使得 ES 的搜索行为不依赖于坐标系的选择，为后续 CMA-ES 用**一般协方差矩阵**替代单位矩阵铺平了道路。

&ensp;&ensp;&ensp;&ensp;**零均值的重要性**。变异 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 的期望为零，意味着变异本身**不引入系统性偏差**。这与 GA 的"随机位翻转"类似（翻 0→1 与 1→0 的概率相等），但在高斯情形下这一性质是精确成立的而非近似的。

## 3.2 步长自适应（1/5 成功规则）

&ensp;&ensp;&ensp;&ensp;Rechenberg 的 **1/5 成功规则**（1/5 rule）是 ES 历史上最重要的贡献之一[^1][^6]。

**基本思想**：如果连续 5 次变异中有 1 次或以上成功（子代优于父代），说明当前步长**太小**，应当增大；如果连续 5 次变异全部失败，说明步长**太大**，应当减小。

**形式化表述**。令 $p_s$ 为"成功变异"的比例（成功次数 / 总变异次数），标准规则为：

$$
\sigma' = \begin{cases}
\sigma / c, & \text{若 } p_s > \frac{1}{5} \\[0.4em]
\sigma \cdot c, & \text{若 } p_s < \frac{1}{5} \\[0.4em]
\sigma, & \text{否则}
\end{cases}
$$

&ensp;&ensp;&ensp;&ensp;其中 $c$ 是调整常数，通常取 $c = 0.8 \sim 0.9$（Beyer & Schwefel 建议 $c = 0.82$）。

**为什么阈值是 1/5？** 这个数字并非随意，而是可以从随机游走理论中推导出来。考虑在 $n$ 维空间中，步长为 $\sigma$ 的随机游走在 $k$ 步后的期望距离（平方）：

$$
E\left[ \|\mathbf{x}^{(k)} - \mathbf{x}^{(0)}\|^2 \right] = k \cdot n \cdot \sigma^2
$$

&ensp;&ensp;&ensp;&ensp;而同样的步长在单维（$n=1$）上产生的**期望适应度改善**约为：

$$
\Delta f \approx \frac{1}{2} \sigma^2 \cdot H
$$

&ensp;&ensp;&ensp;&ensp;其中 $H$ 是目标函数在该处的曲率。Rechenberg 的推导表明，在 $n$ 维空间中，要使"步长调整后的一步"优于"不调整步长的一步"，需要成功概率 $p_s$ 满足：

$$
p_s \approx \frac{1}{5}
$$

具体地，Rechenberg 的原始推导基于"一步变异"与"一步选择"两个过程成功概率的比较，得到阈值约为 $0.2$。这个结果的普适性在于：**它与维数 $n$ 无关**——这正是它的优雅之处。

&ensp;&ensp;&ensp;&ensp;**1/5 规则的两种变体**：

1. **在固定时间窗口上应用**（经典 Rechenberg）：每 $k$ 次变异统计一次成功比例，然后整体调整 $\sigma$；
2. **在每个个体上应用**（Schwefel 的"自调整"）：把 $\sigma$ 编码进个体，变异时同时按以下公式扰动它：

   $$
   \sigma' = \sigma \cdot \exp\left( \tau_0 \cdot \mathcal{N}(0,1) + \tau_1 \cdot \mathcal{N}(0,1) \right)
   $$

   其中第一个高斯项是**全局**的（所有个体共享，体现"整体趋势"），第二个是**局部**的（每个个体独立，体现"个体差异"）。

## 3.3 自适应步长的理论尺度

&ensp;&ensp;&ensp;&ensp;Schwefel 提出的逐维自适应步长公式为：

$$
\sigma_i' = \sigma_i \cdot \exp\left( \tau' \cdot \mathcal{N}_i(0,1) + \tau \cdot \mathcal{N}(0,1) \right)
$$

&ensp;&ensp;&ensp;&ensp;其中 $\mathcal{N}_i$ 是第 $i$ 维专属的独立高斯噪声，$\mathcal{N}$ 是所有维度共享的全局噪声。缩放常数取：

$$
\tau = \frac{1}{\sqrt{2n}}, \qquad \tau' = \frac{1}{\sqrt{2\sqrt{n}}}
$$

&ensp;&ensp;&ensp;&ensp;**这两个常数的由来**。$\tau$ 与 $\tau'$ 是通过**不变性要求**（invariance requirements）确定的，即算法应当在以下变换下保持行为不变：

1. **平移不变**：$f(\mathbf{x}) \to f(\mathbf{x} + \mathbf{c})$ 不改变步长的演化；
2. **尺度不变**：$f(\mathbf{x}) \to f(\alpha \mathbf{x})$ 时步长应相应缩放。

&ensp;&ensp;&ensp;&ensp;具体地，考虑 $\sigma$ 的对数 $\ln \sigma$。它每一代的总方差为 $\tau^2 + n\tau'^2$。在 $n$ 维空间中，要让"全局漂移"（由 $\tau \mathcal{N}$ 引起）不至于压倒"局部适应"（由 $\tau' \mathcal{N}_i$ 引起），需要：

$$
n \tau'^2 = \tau^2 \quad \Longrightarrow \quad \tau' = \frac{\tau}{\sqrt{n}}
$$

再要求全局漂移的**单代标准差**为 $O(1/\sqrt{n})$（使 $n$ 代的总漂移为 $O(1)$），得到：

$$
\tau = \frac{1}{\sqrt{2n}}
$$

代入即得 $\tau' = \frac{1}{\sqrt{2\sqrt{n}}}$。

&ensp;&ensp;&ensp;&ensp;**$1/\sqrt{2n}$ 的标定含义**。这个值保证了"连续多代的步长漂移"不会在 $n$ 维空间中失控——因为每维的步长扰动方差是 $O(1/n)$，$n$ 维合起来是 $O(1)$。

&ensp;&ensp;&ensp;&ensp;**逐维自适应 vs 全局自适应**：

- **全局自适应**（单一 $\sigma$）：所有维度同比例伸缩，适合"各维度尺度相近"的问题，实现简单；
- **逐维自适应**（$\sigma \in \mathbb{R}^n$）：不同维度可以有不同步长，适合尺度差异大的问题，但参数空间维度翻倍。

**逐维自适应在"各向同性"问题（isotropic problem）上会失效**，因为此时最优策略是所有维度步长相同，独立扰动只会引入噪声。这是 CMA-ES 早期版本（sep-CMA-ES）的已知弱点。

## 3.4 选择策略：$(\mu,\lambda)$ 与 $(\mu+\lambda)$

&ensp;&ensp;&ensp;&ensp;这是 ES 的核心分支。

### 3.4.1 $(\mu + \lambda)$-ES：加号选择（elitist）

$$
P(t+1) = \text{Best}_\mu \left( P(t) \cup Q(t) \right)
$$

&ensp;&ensp;&ensp;&ensp;其中 $P(t)$ 是当代的 $\mu$ 个父代，$Q(t)$ 是产生的 $\lambda$ 个子代，$\text{Best}_\mu$ 表示从并集（共 $\mu + \lambda$ 个）中选出适应度最好的 $\mu$ 个。

**关键性质**：由于父代参与竞争，**最优适应度单调不增**：

$$
f(P(t+1)) \leq f(P(t))
$$

这保证了 ES 不会"丢失"已找到的最好解。

**代价**：选择压力极强，容易早熟收敛（premature convergence）。因为父代总是"安全"的，种群多样性会迅速衰减。

### 3.4.2 $(\mu, \lambda)$-ES：逗号选择（non-elitist）

$$
P(t+1) = \text{Best}_\mu \left( Q(t) \right)
$$

&ensp;&ensp;&ensp;&ensp;子代只能从**子代集合** $Q(t)$ 中选出，父代完全不参与。

**关键性质**：**最优适应度可能变差**（可能"丢失"最优解），种群可以"回退"。

**优势**：
- 选择压力显著减弱，保持多样性；
- **允许算法逃离局部最优**——即使所有子代都变差，也必须接受其中最好的 $\lambda$ 个；
- 在多峰问题上能够维持多个峰的探索。

**这是 $(\mu,\lambda)$-ES 的精髓**：它用"偶尔的退步"换取了"跳出局部最优的能力"。这一思想在后来的 CMA-ES 中以 **"权重负值"** 的形式重新出现（见 CMA-ES 章节）。Eiben 与 Smith 对 $(1,\lambda)$-EA 的分析进一步表明，非精英选择在逃离局部最优上的优势是有理论依据的[^8]。

### 3.4.3 选择压力与 $\lambda/\mu$ 比

&ensp;&ensp;&ensp;&ensp;两个关键超参数：

- $\lambda$：**后代数量**（offspring population size）；
- $\mu$：**父代数量**（parent population size）。

比值 $\lambda/\mu$ 决定选择压力：

| $\lambda/\mu$ | 选择压力 | 行为 |
|-------------|----------|------|
| $= 1$ | 极弱 | 几乎随机游走，仅靠变异漂移 |
| $\approx 2$ | 弱（逗号 ES 常用） | 多样性好，慢 |
| $\approx 5 \sim 10$ | 中 | 平衡点 |
| $\gg 1$ | 强（加号 ES 常用） | 收敛快，易早熟 |

**Beyer 的推荐**：对 $(\mu + \lambda)$-ES 取 $\lambda \approx 10\mu$（如 $\mu = 15, \lambda = 100$）；对 $(\mu,\lambda)$-ES 取 $\lambda \approx 5\mu$。

### 3.4.4 $\lambda$ 的规模定律

&ensp;&ensp;&ensp;&ensp;Beyer & Schwefel 的理论分析给出了种群规模与维数的关系。对于在 $n$ 维空间中基于排序的搜索，要维持"每代至少有一个子代落在最优解的 $\sigma$ 邻域内"，所需的后代数随维数指数增长：

$$
\lambda = O(n) \quad \text{是最优的量级选择}
$$

即 $\lambda \propto n$。经验上常取 $\lambda = 4 + 3\ln n$（这是 CMA-ES 的默认设置，源自 Hansen 的经验拟合）或 $\lambda = 10n$。

## 3.5 重采样与"1/5 规则的现代形式"

&ensp;&ensp;&ensp;&ensp;在现代 ES 中，1/5 规则被推广为更精细的形式。

### 3.5.1 Schwefel 的"步长比例"方法

现代 ES 常使用**步长的相对变化**而非绝对调整。定义步长的变化率：

$$
\frac{\sigma_{t+1} - \sigma_t}{\sigma_t}
$$

在 $\sigma$ 较小时需要较大的相对变化才能产生有意义的调整，因此引入了 $\sigma$ 的归一化。

### 3.5.2 Cumulative Step Size Adaptation（CSA）

CSA 是现代 ES 中步长自适应的标准机制。其核心思想是：把 $\ln \sigma$ 视为一个状态量，用一个**累积成功计数**来驱动其演化：

$$
\sigma_{t+1} = \sigma_t \cdot \exp\left( \frac{c_\sigma}{d_\sigma} \cdot N_{\text{success}}(t) \right)
$$

其中 $N_{\text{success}}(t)$ 是自上次调整以来的成功次数，$c_\sigma$ 与 $d_\sigma$ 是学习率常数。

CSA 的优点是**尺度无关**（与 $n$ 无关）与**状态无关**（只依赖成功计数而非适应度值），这使它能自适应地应对不同的问题尺度。

### 3.5.3 重采样规则

在 $(\mu,\lambda)$-ES 中，若 $\lambda > \mu$，部分子代会被丢弃。**重采样规则**（resampling rule）决定如何处理：

- **"立即重采样"（immediate resampling）**：子代被拒绝后立即用新子代替补；
- **"延迟重采样"（delayed resampling）**：先用完本代所有 $\lambda$ 个子代，若仍有空缺才补充。

延迟重采样能保持同一代内的"公平性"，是更常用的选择。

# 4. 完整算法流程

以 $(\mu+\lambda)$-ES 为例：

```
输入：父代数 μ，后代数 λ，维度 n，最大代数 G，初始步长 σ_0
输出：全局最优解 x_best

1.  初始化：随机生成 μ 个个体 x_1, ..., x_μ ~ U(Ω)
2.          步长初始化 σ_0 = 0.3 * (x_max - x_min)
3.          计算适应度，记录 x_best

4.  for gen = 1 to G do
5.      ── 变异（生成 λ 个子代）──
6.      for k = 1 to λ do
7.          选择父代索引 r（均匀随机或按秩加权）
8.          x'_k ← x_r + σ · N(0, I)          # 高斯变异
9.          （可选）σ'_k ← σ · exp(τ·N(0,1) + τ'·N_k(0,1))   # 步长自适应
10.     end for

11.     ── 选择 ──
12.     将 μ 个父代与 λ 个子代合并（共 μ+λ 个）
13.     按适应度排序，取前 μ 个作为下一代父代

14.     ── 步长调整（1/5 规则 或 CSA）──
15.     统计成功次数，按 §3.2 / §3.5 调整 σ

16.     更新 x_best
17. end for
18. 返回 x_best
```

# 5. 收敛性分析

## 5.1 单个体的收敛性

&ensp;&ensp;&ensp;&ensp;对于 $(1+1)$-ES，最简单的收敛分析基于随机游走理论。考虑在最优解附近，二阶泰勒展开给出单步适应度改善的期望：

$$
E[\Delta f] = \frac{1}{2} \sigma^2 H - \frac{n \sigma}{2} \sqrt{\frac{2}{\pi}} \cdot \text{(梯度项的贡献)}
$$

&ensp;&ensp;&ensp;&ensp;最优步长为：

$$
\sigma^* \approx \frac{1.2}{H}
$$

其中 $H$ 是目标函数在最优解处的曲率（对角 Hessian 的迹）。这意味着**步长的最优值与问题维数无关，只取决于曲率**——这是一个漂亮的结论。

## 5.2 收敛速率

在最优解邻域内，若种群足够大且选择有效，ES 的收敛速率大致为：

$$
f(P(t)) - f^{*} \sim C \cdot t^{-\alpha}
$$

其中 $\alpha$ 依赖于选择压力。对 $(\mu,\lambda)$-ES，理论分析给出的速率通常在 $O(t^{-1/\max\{1, 5\lambda/(\mu+n)\}})$ 量级，具体结果较为技术性，此处不展开。

## 5.3 早熟收敛的原因

ES 的早熟收敛源于以下几点：

1. **步长收缩过快**：1/5 规则在连续失败时几何式缩小 $\sigma$（乘以 $c$），若此时实际上只是暂时陷入平台期，步长就再也长不回来；
2. **协方差矩阵的对角限制**（基础 ES）：无法学习维度间的相关性，遇到旋转问题会严重受挫；
3. **种群多样性不足**：当 $\lambda/\mu$ 很大时，所有个体被少数精英主导。

对应的对策正是 CMA-ES 所做的事：**用路径（evolution paths）平滑步长的演化、用完整协方差矩阵学习相关性、用负权重允许"退步"**。

# 6. 参数选取

| 参数 | 符号 | 推荐取值 | 说明 |
|------|------|----------|------|
| 父代数量 | $\mu$ | $5 \sim 15$（约 $\lambda/4 \sim \lambda/10$） | $(\mu+\lambda)$ 用 $\mu \approx \lambda/10$ |
| 后代数量 | $\lambda$ | $4 + 3\ln n$ 或 $10n$ | 理论上 $O(n)$ |
| 初始步长 | $\sigma_0$ | $0.3 \times (x_{max}-x_{min})$ | Schwefel 的经典建议 |
| 1/5 调整常数 | $c$ | $0.8 \sim 0.9$ | $0.82$ 为常用值 |
| 全局缩放常数 | $\tau$ | $1/\sqrt{2n}$ | 由不变性要求导出 |
| 局部缩放常数 | $\tau'$ | $1/\sqrt{2\sqrt{n}}$ | 同上 |
| 选择类型 | - | $(\mu+\lambda)$ 需 $\lambda \gg \mu$；$(\mu,\lambda)$ 需 $\lambda \approx 5\mu$ | 决定是否允许退步 |

# 7. 应用场景

&ensp;&ensp;&ensp;&ensp;ES 及其后继者在以下领域表现突出：

1. **连续函数优化**：作为理论最完整的进化算法，是理解自适应机制的理想对象。
2. **工程参数标定**：控制系统参数整定、仿真模型校准。
3. **机器人与控制**：逆运动学、轨迹优化、参数辨识。
4. **物理实验设计**：加速器参数优化、聚变装置磁场配置——这些领域通常评价代价极高且是黑盒，ES 的稳健性很有价值。
5. **天线与电磁设计**：增益/阻抗的联合优化。
6. **调度与组合问题的混合整数变体**。
7. **神经进化**：NEAT（Neuroevolution of Augmented Topologies）用于拓扑优化。

# 8. 变体与演进

| 变体 | 年份 | 关键改进 |
|------|------|----------|
| $(1+1)$-ES | 1965 | 最简形式，1/5 规则的原型 |
| $(\mu,\lambda)$-ES | 1970s | 逗号选择，允许退步，多样性好 |
| $(\mu+\lambda)$-ES | 1970s | 加号选择，单调收敛 |
| $1/\lambda$-ES | 1990s | 步长与策略解耦适应 |
| **sep-CMA-ES** | 1997 | 对角协方差矩阵 + 演化路径 |
| **CMA-ES** | 2001 | **完整协方差矩阵** + 演化路径 |
| **CMA-ES with restarts (IPOP-CMA-ES)** | 2009 | 多种群重启对抗早熟 |
| **BIPOP-CMA-ES** | 2012 | 双预算重启策略 |
| **CMA-ES-TPE 等混合** | 2010s | 与贝叶斯优化结合 |

&ensp;&ensp;&ensp;&ensp;**从 ES 到 CMA-ES 的演进逻辑**可以概括为三次扩展：

1. **从标量 $\sigma$ 到对角 $\mathbf{C}$**：允许不同维度有不同步长 → 解决各向异性问题；
2. **从对角 $\mathbf{C}$ 到完整 $\mathbf{C}$**：允许维度间相关 → 解决旋转问题；
3. **从"直接更新参数"到"演化路径（evolution path）"**：用历史累积信息平滑参数更新 → 解决参数更新的方差过大问题。

这三步正是理解 CMA-ES 的最佳切入点，详见 CMA-ES 章节。Hansen 的 CMA-ES 教程对这条演进脉络给出了最系统的梳理[^5]；Ostermeier 与 Hansen 的"去随机化"CMA-ES 则解决了原始版本中随机性过强导致收敛慢的问题[^11]。

&ensp;&ensp;&ensp;&ensp;此外，Schwefel 在 *Handbook of Evolutionary Computation* 中的综述[^10] 以及 Beyer 与 Schwefel 的 *Natural Computing* 长文[^4][^7] 是 ES 领域最权威的两份参考材料，覆盖了从 1/5 规则到协方差适应的全部技术细节。

# 9. 与相关算法的比较

| 维度 | ES | GA | DE | PSO |
|------|----|----|----|-----|
| 编码 | 实数向量 | 抽象染色体 | 实数向量 | 实数向量 + 速度 |
| 变异 | 高斯（对角或完整协方差） | 位翻转/高斯 | 种群差分 | 速度积分 |
| 自适应对象 | 步长 $\sigma$、协方差 $\mathbf{C}$ | $p_c, p_m$ | $F, CR$ | $w$ |
| 适应度使用 | **仅排序** | 直接概率 | 直接比较 | 直接比较 |
| 记忆 | 无（除 $\mathbf{C}$ 外） | 无 | 无 | pbest + gbest |
| 理论完备性 | **高**（收敛速率分析） | 模式定理 | 有限 | 必要条件 |
| 高维性能 | 中（$\mathbf{C}$ 为完整矩阵时高） | 中 | 高 | 弱 |
| 计算代价 | $O(n^2)$~$O(n^3)$/代 | $O(n)$ | $O(n)$ | $O(n)$ |

&ensp;&ensp;&ensp;&ensp;**选择建议**：

- 需要**理论保证**与**参数自动设定**：选 ES / CMA-ES；
- 需要**通用、稳健、易实现**：选 DE；
- 需要**最快原型**：选 PSO；
- 需要**离散问题**：选 GA。

# 10. 小结

&ensp;&ensp;&ensp;&ensp;进化策略的历史地位在于它是**第一批把"自适应"作为一等公民引入优化算法的尝试**。1/5 成功规则证明了可以用极简的启发式自动设定步长；$(\mu,\lambda)$ 与 $(\mu+\lambda)$ 的对立则揭示了优化算法中一个永恒的权衡：**接受退步以换取跳出局部最优的能力，还是坚守精英以换取单调收敛**。

&ensp;&ensp;&ensp;&ensp;ES 的另一大贡献是**理论工具**。它把随机游走、中心极限定理、旋转不变性等统计分析工具引入优化算法，奠定了 CMA-ES 严谨的概率框架。相比之下，GA 的理论建立在离散的"模式"概念上，PSO 的理论建立在简化的确定性动力系统上，而 ES/CMA-ES 的理论建立在**高斯过程中的等宽采样**（equidistribution）这一优雅而精确的原理之上。

&ensp;&ensp;&ensp;&ensp;学习 ES 的最大价值在于理解 **CMA-ES 的每个部件从何而来**：$c_1, c_\mu$ 的分解源于 1/5 规则中"全局漂移 vs 局部适应"的分离；演化路径源于"用历史信息降低参数更新方差"的需要；负权重源于 $(\mu,\lambda)$ 中"允许退步"的精神。Beyer 对元遗传算法的分析进一步说明了：把参数适应本身也当作一个可演化的过程，是进化算法保持通用性的关键[^9]。

# 参考文献

[^1]: Rechenberg, I. (1973). *Evolutionsstrategie '94 und die numerische Optimierung*. Frommann-Holzboog, Stuttgart.

[^2]: Schwefel, H.-P. (1981). *Numerical Optimization of Computer Models*. John Wiley & Sons.

[^3]: Hansen, N. & Ostermeier, A. (2001). "Completely derandomized self-adaptation in evolution strategies". *Evolutionary Computation*, 9(2), 159–195.

[^4]: Beyer, H.-G. & Schwefel, H.-P. (2002). "Evolution strategies – A comprehensive introduction". *Natural Computing*, 1(1), 3–52.

[^5]: Hansen, N. (2006). "The CMA evolution strategy: A tutorial". arXiv:1604.00772. (亦见 Hansen, N. (2016). "The CMA evolution strategy: A tutorial". *arXiv preprint arXiv:1604.00772*)

[^6]: Beyer, H.-G. (2001). "The 1/5-th rule in evolution strategies". In *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO'01)*, pp. 8–13.

[^7]: Rechenberg, I. & Schwefel, H.-P. (1995). "Evolution strategies: A comprehensive introduction". *Natural Computing*, 1(1), 3–52. (原文：Rechenberg, I. (1994). *Evolutionsstrategie '94*. Frommann-Holzboog)

[^8]: Eiben, A.E. & Smith, J.E. (1996). "An analysis of the (1, λ) EA". In *Proceedings of the 3rd European Conference on Artificial Intelligence*, pp. 250–254.

[^9]: Beyer, H.-G. (1998). "On the role of the meta-genetic algorithms". *Figuro 3.4*  (见 Beyer, H.-G. & Schwefel, H.-P. (2002), §2.1)

[^10]: Schwefel, H.-P. (1995). *Adaptive and Self-adaptive Evolutionary Computations*. In *Handbook of Evolutionary Computation*. IOP Publishing, pp. 1.1–1.7.

[^11]: Ostermeier, A. & Hansen, N. (2001). "A derandomized approach to the CMA evolution strategy". In *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO'01)*, pp. 525–532.
