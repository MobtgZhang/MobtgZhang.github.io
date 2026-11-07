---
title: 黎曼猜想发现过程（二）：数学原理与解析框架
categories: [数学原理]
tags: [黎曼猜想, 解析数论, 复分析, 数论]
author: MobtgZhang
toc: true
comments: true
math: true
description: 从 ζ 函数的解析延拓出发，完整推导函数方程、Hadamard 乘积展开、显式公式，并解释零点密度估计这条主线为何难以推进。
---

> 本文是《[黎曼猜想发现过程（一）](/wiki/riemann-hypothesis-1/)》的数学续篇。
> 上一篇讲的是**人**与**历史**，这一篇讲的是**结构**与**证明**。
>
> 全文的符号约定：
> $$s = \sigma + it \in \mathbb{C}$$，$$\zeta$$ 表示 Riemann $$\zeta$$ 函数，
> $$\rho = \beta + i\gamma$$ 表示其非平凡零点。

---

## 0. 问题陈述

**黎曼猜想（RH）**：Riemann $$\zeta$$ 函数的所有非平凡零点均位于临界线

$$
\Re(s) = \frac{1}{2}
$$

上。

等价地，记

$$
\xi(s) = \frac{1}{2}s(s-1)\pi^{-s/2}\Gamma\!\left(\frac{s}{2}\right)\zeta(s)
$$

则 RH 等价于断言 $$\Xi(t) = \xi\!\left(\frac{1}{2}+it\right)$$ 的所有零点都是实数。

**为什么这个"零点位置"问题如此难？** 因为显式公式表明：

$$
\text{素数分布的涨落} \;\longleftrightarrow\; \text{零点的位置}
$$

这是一个**全局性的**对应关系。单个零点的位置影响极小，但**全部零点的集体位置**决定了我们对素数分布的认知精度。要证明 RH，等价于要证明一个由无穷多个点组成的集合恰好落在一条直线上——而我们只有有限的手段去检验无穷。

下面把这个框架一层层搭起来。

---

## 1. Euler 乘积：解析数论的起点

### 1.1 绝对收敛区域

对 $$\Re(s) > 1$$，级数

$$
\zeta(s) = \sum_{n=1}^{\infty} n^{-s}
$$

绝对收敛，且由 Euler 乘积公式

$$
\zeta(s) = \prod_{p}\left(1 - p^{-s}\right)^{-1}
$$

与之等价。

### 1.2 取对数展开

在 $$\Re(s) > 1$$ 中，对乘积取对数并展开几何级数，双重求和可交换（绝对收敛保证）：

$$
\log\zeta(s) = \sum_{p}\sum_{k \ge 1}\frac{p^{-ks}}{k}
$$

于是

$$
\frac{\log\zeta(s)}{s} = \sum_{n=1}^{\infty}\frac{\Lambda(n)}{n^{s}}
$$

其中 $$\Lambda$$ 是 von Mangoldt 函数：

$$
\Lambda(n) = \begin{cases} \log p, & n = p^k \\ 0, & \text{其他} \end{cases}
$$

> **这是整个理论最关键的一步**：一个"关于所有整数求和"的函数，被改写成了"关于素数（及其幂）求和"的函数。素数的乘法结构被编码进了 $$\Lambda$$。

### 1.3 Perron 反演：回到计数函数

由 Mellin 反演，若 $$F(s) = \sum_{n\ge1} a_n n^{-s}$$（$$\Re(s) > \sigma_a > 0$$），则

$$
\sum_{n \le x} a_n = \frac{1}{2\pi i}\int_{c-i\infty}^{c+i\infty}\frac{F(s)x^{s}}{s}\,ds, \qquad c > \sigma_a
$$

取 $$a_n = \Lambda(n)$$、$$F(s) = -\log\zeta(s)/s$$，得到

$$
\psi(x) := \sum_{n \le x}\Lambda(n) = -\frac{1}{2\pi i}\int_{c-i\infty}^{c+i\infty}\frac{\log\zeta(s)}{s}x^{s}\,ds
$$

**核心思想**：$$x^s = e^{s\log x}$$，积分沿竖线时 $$x^{it}$$ 以频率 $$\log x$$ 振荡，因此积分可以用**留数定理**转化为"竖线左侧各奇点贡献之和"。

这就把一个**离散的计数问题**转成了**解析的奇点问题**。

### 1.4 为什么需要解析延拓

上面的积分要求把竖线从 $$c > 1$$ 往左推。推到哪儿为止，取决于 $$\log\zeta(s)$$ 在左侧的解析结构——而这**必须**先有解析延拓。

> **这就是为什么 1859 年的那篇论文必须从"让 $$\zeta$$ 定义在整个复平面上"开始。**

---

## 2. 解析延拓与函数方程

### 2.1 第一步：积分表示

分部积分（对 $$\Re(s) > 1$$）给出

$$
\zeta(s) = \frac{1}{\Gamma(s)}\int_0^\infty \frac{x^{s-1}}{e^x - 1}\,dx
$$

**关键观察**：被积函数在 $$x=0$$ 附近可展为 $$x^{-1} - \frac12 + \frac{x}{12} - \cdots$$，所以积分在 $$\Re(s) > 0$$ 收敛。

因此：

$$
\zeta(s) = \frac{1}{\Gamma(s)}\int_0^\infty\frac{x^{s-1}}{e^x-1}\,dx, \qquad \Re(s) > 0
$$

这个式子把解析延拓的问题变成了一个**关于积分的估计**问题。

### 2.2 第二步：围道积分与留数

令

$$
I(s) = \int_0^\infty\frac{(-x)^{s-1}}{e^x-1}\,dx
$$

取** Hankel 围道**：沿正实轴向上、绕无穷远、沿负实轴向下（其中 $$-x$$ 取主值分支）。函数 $$1/(e^x-1)$$ 在

$$
x = \pm 2\pi i n, \qquad n = 1, 2, 3, \dots
$$

处有极点。

**围道技巧的关键**：由 $$e^x - 1$$ 的周期性，

$$
\frac{1}{e^x - 1} + \frac{1}{e^{-x}-1} = -1
$$

这个恒等式让沿正负实轴的两段积分**互为补数**，从而整个围道积分等于正实轴积分的一个倍数：

$$
\int_H\frac{(-x)^{s-1}}{e^x-1}\,dx = \left(1 - e^{i\pi s}\right)\int_0^\infty\frac{x^{s-1}}{e^x-1}\,dx
$$

同时，因为围道左侧的函数在无穷远处指数衰减，正实轴的贡献可以忽略。于是留数定理给出

$$
\oint_H\frac{(-x)^{s-1}}{e^x-1}\,dx = 2\pi i \sum_{n=1}^{\infty}\operatorname{Res}_{x=2\pi i n}\frac{(-x)^{s-1}}{e^x-1}
$$

在极点 $$x = 2\pi i n$$ 处，$$e^x-1 \approx e^{2\pi i n}(x - 2\pi i n) = x - 2\pi i n$$，故

$$
\operatorname{Res} = \frac{(-2\pi i n)^{s-1}}{1} = (2\pi n)^{s-1}e^{i\pi(s-1)}
$$

代入并整理，结合 $$\Gamma(s)\Gamma(1-s) = \pi/\sin(\pi s)$$ 与 $$2^{s}\pi^{s-1}\sin\frac{\pi s}{2}\Gamma(1-s) = \frac{\pi^s}{\Gamma(s)\sin\frac{\pi s}{2}}$$ 的关系，得到**函数方程**：

$$
\boxed{\;\zeta(s) = 2^{s}\,\pi^{s-1}\,\sin\frac{\pi s}{2}\,\Gamma(1-s)\,\zeta(1-s)\;}
$$

> **注意一个细节**：这里出现的留数来自 $$1/(e^x-1)$$ 的极点 $$2\pi i n$$，**不是**素数幂。这是一个常见的误解——函数方程的推导里并没有用到 Euler 乘积。

### 2.3 函数方程的三重对称

把函数方程写成对称形式

$$
\zeta(s) = \chi(s)\,\zeta(1-s), \qquad \chi(s) = 2^{s}\pi^{s-1}\sin\frac{\pi s}{2}\Gamma(1-s)
$$

立刻得到：

**（a）实轴对称**：$$\chi(\bar{s}) = \overline{\chi(s)}$$，故

$$
\zeta(\bar{s}) = \overline{\zeta(s)}
$$

**（b）临界线反射**：令 $$s = \frac{1}{2}+it$$，则 $$1-s = \bar{s}$$，结合 (a)：

$$
\zeta\!\left(\frac{1}{2}+it\right) = \chi\!\left(\frac{1}{2}+it\right)\overline{\zeta\!\left(\frac{1}{2}+it\right)}
$$

由于 $$\lvert\chi\!\left(\frac{1}{2}+it\right)\rvert = 1$$，这说明临界线上的函数值与一个**单位复数**相乘——这正是构造实值函数

$$
Z(t) = e^{i\theta(t)}\zeta\!\left(\frac{1}{2}+it\right)
$$

的动机，其中 $$\theta(t)$$ 选取为使 $$\chi\!\left(\frac{1}{2}+it\right) = e^{-2i\theta(t)}$$。

**（c）零点对称**：由 (a)(b)，若 $$\rho$$ 是零点，则 $$1-\rho$$、$$\bar\rho$$、$$1-\bar\rho$$ 都是零点。

### 2.4 平凡零点

由函数方程右端的零点：

- $$\sin\frac{\pi s}{2} = 0$$ 于 $$s = 2k$$（偶整数）——但这些点的零点被 $$\Gamma(1-s) = \Gamma(1-2k) = \infty$$ 抵消；
- 真正的来源是 $$\Gamma(1-s)$$ 在 $$s = 2, 4, 6, \dots$$ 处的极点。

综合可知：**$$\zeta(s)$$ 在 $$s = -2, -4, -6, \dots$$ 处有零点**，称为**平凡零点**。

另一方面，$$\zeta(s)$$ 在 $$s=1$$ 处有唯一极点（留数 1）。

### 2.5 临界带

**定理（临界带）**：除平凡零点外，$$\zeta(s)$$ 的所有零点都在 $$0 < \Re(s) < 1$$ 内。

**证明要点**：设 $$\rho = \beta + i\gamma$$ 是零点，$$\beta > 1$$。由 Euler 乘积

$$
\zeta(\beta) = \prod_p\left(1-p^{-\beta}\right)^{-1}
$$

收敛（因为 $$\sum_p p^{-\beta} < \infty$$），故 $$\zeta(\beta) \ne 0$$，矛盾。同理 $$\beta < 0$$ 时由函数方程与 $$\zeta(1-\beta) \ne 0$$ 导出矛盾。$$\beta = 0$$ 或 $$1$$ 的情形需要更精细的论证（利用 $$\Gamma$$ 与三角函数的联合零点分析）。

---

## 3. 完备函数 $$\xi(s)$$ 与 Hadamard 乘积

### 3.1 定义与性质

定义

$$
\xi(s) = \frac{1}{2}s(s-1)\pi^{-s/2}\Gamma\!\left(\frac{s}{2}\right)\zeta(s)
$$

则：

| 性质 | 说明 |
| :--- | :--- |
| 全纯性 | 抵消了 $$s=1$$ 的极点与 $$s=0,1$$ 的三角函数零点 |
| 函数方程 | $$\xi(s) = \xi(1-s)$$（完美对称） |
| 零点 | 与 $$\zeta$$ 的非平凡零点**完全一致** |
| 阶 | 1（极大型），且 $$\xi(\bar{s}) = \overline{\xi(s)}$$ |
| 实值性 | 当 $$s\in\mathbb{R}$$ 时 $$\xi(s) \in \mathbb{R}$$ |

### 3.2 Hadamard 乘积展开

**定理（Hadamard, 1893）**：存在常数 $$A, B$$ 使得

$$
\xi(s) = e^{A+Bs}\prod_{\rho}\left(1 - \frac{s}{\rho}\right)e^{s/\rho}
$$

乘积对所有非平凡零点 $$\rho$$（按重数）取。

**推导的三个关键步骤**：

**（i）Jensen 定理与零点密度**。由 Jensen 公式，对 $$\lvert s\rvert = R$$：

$$
\int_0^R \frac{n(r)}{r}\,dr = \frac{1}{2\pi}\int_0^{2\pi}\log\left\lvert\xi\left(Re^{i\theta}\right)\right\rvert d\theta - \log\lvert\xi(0)\rvert
$$

其中 $$n(r)$$ 是半径为 $$r$$ 的圆内零点数。由于 $$\xi$$ 是阶 1 的整函数，

$$
n(R) = O(R\log R)
$$

这个估计**不够好**——它使得无穷乘积 $$e^{s/\rho}$$ 的补偿因子**必须**保留（否则对数项发散）。

**（ii）收敛性的精确化**。把乘积按 $$\lvert\rho\rvert \le R$$ 截断：

$$
P_R(s) = \prod_{\lvert\rho\rvert \le R}\left(1 - \frac{s}{\rho}\right)e^{s/\rho}
$$

利用 $$\sum_\rho \frac{1}{\lvert\rho\rvert^2} < \infty$$（由 $$n(R)=O(R\log R)$$ 推得），可以证明 $$P_R(s)$$ 在紧集上一致收敛到非零整函数。

**（iii）识别因子**。整函数 $$f(s) = \xi(s)/P(s)$$ 无零点，故 $$f(s) = e^{g(s)}$$。由阶数估计，$$g$$ 是一次多项式：

$$
g(s) = A + Bs
$$

**由此得到 Euler 乘积的推广形式**：

$$
\xi(s) = \xi(0)\prod_\rho\left(1 - \frac{s}{\rho}\right)
$$

（此处已吸收常数因子并利用了 $$\prod$$ 的对称性。）

### 3.3 为什么这一节最要命

**Hadamard 乘积把"零点的位置"编码成了"函数的整体行为"**。反过来，**任何关于 $$\xi(s)$$ 增长率的估计都会转化为对零点数量的约束**。

这就是下一节的主线。

---

## 4. 零点计数与 Riemann–von Mangoldt 公式

### 4.1 计数函数的定义

记 $$N(T)$$ 为满足 $$0 < \gamma \le T$$ 的非平凡零点数（按重数计），并约定

$$
N_0(T) = \#\{\rho : 0 < \gamma \le T,\ \Re\rho = \tfrac12\}
$$

**定理（Riemann–von Mangoldt, 1895）**：

$$
N(T) = \frac{T}{2\pi}\log\frac{T}{2\pi} - \frac{T}{2\pi} + \frac{7}{8} + S(T) + O\!\left(\frac{1}{T}\right)
$$

其中

$$
S(T) = \frac{1}{\pi}\arg\zeta\!\left(\frac{1}{2}+iT\right)
$$

是 Riemann–Siegel 函数（更准确地说，$$S(T)$$ 定义为从 $$T$$ 到无穷的适当路径上 $$z^{-1/2}\log z + \frac{z^{1/2}}{2(1-z)} - \log\sqrt{2\pi}$$ 的辐角变化）。

### 4.2 关键推论

**（a）零点无穷多**：由 $$N(T) \sim \frac{T}{2\pi}\log T \to \infty$$。

**（b）零点密度（渐近密度）**：在区间 $$[T, T+\Delta T]$$ 内的零点数约为

$$
\frac{\log T}{2\pi}\Delta T
$$

即零点密度随高度**对数增长**。

**（c）RH 成立的必要条件**：RH 若成立，则

$$
N_0(T) = N(T)
$$

**（d）RH 的一个等价表述**：

$$
\boxed{\;\text{RH} \iff N(\sigma, T) = 0 \quad \text{对所有 } \sigma > \tfrac12 \text{ 与 } T > 0\;}
$$

其中

$$
N(\sigma,T) := \#\{\rho : \Re\rho \ge \sigma,\ \lvert\Im\rho\rvert \le T\}
$$

---

## 5. 显式公式：零点到素数的桥梁

### 5.1 von Mangoldt 公式

**定理**：

$$
\psi(x) = x - \sum_{\rho}\frac{x^{\rho}}{\rho} - \log(2\pi) - \frac{1}{2}\log\left(1 - x^{-2}\right)
$$

**这个公式的每一项都对应一个物理对象**：

| 项 | 含义 |
| :--- | :--- |
| $$x$$ | 主项：素数的平均行为 |
| $$-\sum_\rho x^\rho/\rho$$ | **零点贡献**：所有振荡的来源 |
| $$-\log 2\pi$$ | 常数项 |
| $$-\frac12\log(1-x^{-2})$$ | 来自平凡零点的修正项 |

**核心洞察**：$$x^\rho = x^\beta e^{i\gamma\log x}$$。实部 $$\beta$$ 决定振荡的**衰减率**，虚部 $$\gamma$$ 决定振荡的**频率**。

- 若 RH 成立（$$\beta = 1/2$$），振荡幅度为 $$x^{1/2}$$。
- 若有反例（$$\beta > 1/2$$），该振荡**衰减更慢**，会**压制**其他振荡。

**这解释了为什么 Bombieri 说 RH 失败会"造成灾难性的混乱"**：一个偏离的零点会主导整个误差项。

### 5.2 Riemann 的原始显式公式

黎曼自己的版本（用于 $$J(x) = \sum_{p^k\le x}1/k$$）是

$$
J(x) = \frac{1}{2\pi i}\int_{a-i\infty}^{a+i\infty}\frac{x^{s}}{s}\cdot\frac{\zeta'(s)}{\zeta(s)}\,ds - \sum_{n=1}^{\infty}\frac{\pi^{2n}x^{2n}}{(2n)!(2n-1)}, \qquad a>1
$$

注意 $$\frac{\zeta'}{\zeta}(s)$$ 的极点位置：每个零点 $$\rho$$ 贡献一个留数 $$\frac{x^\rho}{\rho}$$。**这就是"零点 → 素数"贡献的严格来源。**

### 5.3 与对数积分的关系

由 $$\operatorname{Li}(x) = \int_2^x \frac{dt}{\log t}$$，素数计数误差

$$
\pi(x) - \operatorname{Li}(x)
$$

的**主导项**恰好由零点的虚部给出。Gauss 在 1796 年写信给天文学家 Hencke 时，就已用数值比较过 $$\pi(x)$$ 与 $$\operatorname{Li}(x)$$ 到 $$x = 3\times10^6$$，并注意到 $$\pi(x)$$ **系统性地低于** $$\operatorname{Li}(x)$$。

黎曼在 1859 年把这个数值观察提升为了理论：**这个系统性偏差，正是由零点的集体位置决定的。**

---

## 6. 为什么 $$\Re(s) = 1/2$$ 是"自然的"

这是本篇最重要的**结构性**洞察，也是理解 RH 难度的关键。

### 6.1 对称性只给出"中心"，不给出"对齐"

回顾 2.3 节：零点关于实轴与临界线都对称。这保证了零点成四元组

$$
\{\rho,\ \bar\rho,\ 1-\rho,\ 1-\bar\rho\}
$$

分布。

但**这完全不能推出**它们都在 $$\Re(s) = 1/2$$ 上。一个位于 $$\beta + i\gamma$$（$$\beta \ne 1/2$$）的零点，会与它的镜像 $$\beta - i\gamma$$ 等一起落在直线两侧。

> **黎曼猜想的困难正在于此**：对称性告诉我们零点的**中心**在哪，但猜想要求它们**对齐**在一条直线上。而"对齐"是一个远比"对称"强得多的条件。

### 6.2 Euler 乘积的 Ramanujan 求和：为什么是 $$1/2$$

一个更深刻的角度。考虑 Ramanujan 对素数计数的求和：

$$
R(x) = \sum_{n \le x}\frac{\Lambda(n)}{n} - \log\log x
$$

展开 Euler 乘积的对数：

$$
R(x) = \sum_{n=1}^{\infty}\frac{\mu(n)}{n}\log\left(\frac{\zeta(nx)}{\zeta(n)}\right)
$$

对 $$n \ge 2$$，当 $$x \to \infty$$ 时 $$\log\zeta(nx) \to 0$$，而 $$\log\zeta(n) = -\log n + \text{小量}$$，故

$$
R(x) = \gamma + \sum_{n=2}^{\infty}\frac{\mu(n)}{n}\log\frac{n}{\zeta(n)} + O\!\left(\frac{1}{x}\right)
$$

**有意思的是**：这个常数展开中，$$n$$ 的求和范围恰好是 $$n \ge 2$$，而 $$n = 1$$ 项被 $$\zeta(1)$$ 的极点"切掉"。**临界线 $$\Re(s) = 1/2$$ 的位置，本质上是"极点的倒数平方根"**——这与 Weierstrass 判别式（判别式是曲线的 $$-1$$ 次幂测度）的精神一致：**临界值出现在幂次分界处**。

### 6.3 随机零点矩阵：为什么 $\frac12$ 是"最可能"的位置

一个纯现象学的论证。取 $$N(T)$$ 个零点的实部 $$\beta_j$$，考虑

$$
S = \sum_{j=1}^{N(T)} (2\beta_j - 1)^2
$$

若 RH 成立则 $$S = 0$$。Montgomery 与 Dyson 的**对数关联**计算表明，$$S$$ 的分布尾部按

$$
\Pr(S > \lambda) \sim e^{-\frac{\pi^2\lambda}{4\gamma}}
$$

衰减（$$\gamma = \log(T/2\pi)$$ 为"密度参数"）。这与指数分布的尾巴吻合——正是"独立随机点落在直线上"的特征。

**这解释了为什么 RH"非常可能"**：如果零点的实部近似独立地随机分布在 $$(0,1)$$ 上，那么全部落在中点附近的概率是 0；能全部落在中点附近，只可能是因为存在**额外的结构性约束**。

**而这个"额外的结构性约束"究竟是什么，正是 RH 的未解之处。**

---

## 7. 主线：零点密度估计

### 7.1 问题设定

核心的未解问题可表述为：证明

$$
N(\sigma, T) = 0 \qquad \text{对所有 } \sigma > \tfrac12
$$

历史上，人们转而研究更弱的**零点密度估计**：控制 $$N(\sigma,T)$$ 随 $$T$$ 的增长速度。

**定义（密度假设 DH）**：若存在常数 $$C$$ 使得

$$
N(\sigma, T) \ll T^{C(1-\sigma)+\varepsilon}
$$

对所有 $$\frac12 \le \sigma \le 1$$、$$T \ge 1$$、$$\varepsilon > 0$$ 成立，则称 DH 成立。

**重要事实**：$$N\!\left(\frac12, T\right) \asymp T\log T$$，因此 DH 的最优指数是

$$
C = \frac{T\log T}{T(1-\frac12)} = 2
$$

而 RH 等价于 DH 取到最优指数 $$C=2$$ **并且**在 $$\sigma > 1/2$$ 处严格为零。

### 7.2 已知的层级

$$
\text{RH} \;\Longrightarrow\; \text{Lindelöf 假设 (LH)} \;\Longrightarrow\; \text{密度假设 (DH)}
$$

其中 LH 断言

$$
\zeta\!\left(\frac{1}{2}+it\right) \ll_\varepsilon t^{\varepsilon}
$$

| 指数 $C$ | 年份 | 作者 | 备注 |
| :--- | :--- | :--- | :--- |
| $$3/2$$（在 $$\sigma > 3/4$$） | 1940 | Ingham | |
| $$12/5 = 2.4$$ | 1955 | Huxley | **长期无人能破** |
| $$30/13 \approx 2.3077$$ | 2024 | Guth–Maynard | 当前记录 |

**Guth–Maynard 的定理**（2024）：

$$
N(\sigma, T) \ll T^{\frac{30(1-\sigma)}{13}+o(1)}
$$

其核心技术是 Dirichlet 多项式的**大值估计**：若 $$(b_n)_{n \in [N,2N]}$$ 满足 $$\lvert b_n\rvert \le 1$$，且在 $$R$$ 个 1-分离点 $$t_r \in [0,T]$$ 上有

$$
\left\lvert\sum_{N \le n \le 2N} b_n n^{it_r}\right\rvert \ge V
$$

则

$$
R \le T^{o(1)}\left(N^{2}V^{-2} + N^{18/5}V^{-4} + T N^{12/5}V^{-4}\right)
$$

### 7.3 为什么"逼近 2"这么难

把 $$C \to 2$$ 看成一系列逐步逼近：

| 阶段 | 需排除的情形 | 手段 |
| :--- | :--- | :--- |
| 零自由区 | 单个零点落在极端位置 | Euler 乘积 + 三角和估计 |
| 密度估计 | 中等偏离的零点**数量** | 大值估计 + 均值定理 |
| RH | **任何**偏离，哪怕只有一个 | ？ |

**关键困难在最后一步**：密度估计控制的是**数量**，而 RH 要求的是**一个也没有**。即使 $$N(\sigma,T) \ll T^{2.3(1-\sigma)}$$，对于任意小的固定 $$\sigma - \tfrac12 > 0$$，这个上界仍然可以是无穷大。

> 换句话说：密度估计这条路线**在原理上就无法直接证明 RH**。它的价值在于给出**无条件**的近似结果（素数在短区间中的分布等），而非证明猜想本身。

### 7.4 零点自由区：另一条线

与密度估计平行的是**零点自由区**（zero-free region）：证明某个区域

$$
\sigma \ge 1 - c(\log t)^{-2/3}(\log\log t)^{-1/3}
$$

内无零点（**Vinogradov–Korobov 区域**，1958）。

这个区域自 1958 年以来**几乎纹丝未动**——而陶哲轩指出，它恰恰是"排除**单个**极其糟糕的违反"的关键工具，因为 RH 的失败必然导致一个零点非常靠近实轴。

**两条主线都卡在同一处**：Guth–Maynard 推进了密度估计，却没能推动零点自由区；而零点自由区才是直接攻击 RH 的工具。

---

## 8. 临界线上的零点：已知的最强结果

### 8.1 Hardy (1914)

对实值函数

$$
Z(t) = e^{i\theta(t)}\zeta\!\left(\frac{1}{2}+it\right), \qquad
\theta(t) = \operatorname{Im}\log\Gamma\!\left(\frac{1}{4}+\frac{it}{2}\right) - \frac{t}{2}\log\pi
$$

分析 $$Z(t)$$ 的**符号变化**。由于 $$Z(t)$$ 是实值的连续函数，在临界线上变号处必有零点。Hardy 证明这样的变号点无穷多。

### 8.2 Hardy–Littlewood (1921)

给出定量下界 $$N_0(T) \gg T$$。由于 $$N(T) \sim \frac{T}{2\pi}\log T$$，这只说明"比例趋于 0"——远不够。

### 8.3 Selberg (1942)：平滑化因子

**这是方法论上的转折点。**

Selberg 的核心想法：把 $$ζ(s)$$ 乘上一个**平滑化因子**

$$
M(s) = \sum_{n \le \xi}\frac{\alpha_n\left(1 - \frac{\log n}{\log \xi}\right)}{n^{s}},
\qquad
\sum_{n=1}^{\infty}\frac{\alpha_n}{n^{s}} = \frac{1}{\sqrt{\zeta(s)}}
$$

**直觉**：$$M(s)$$ 抵消了 $$\zeta(s)$$ 的**大值**，使得 $$\lvert\zeta(s)M(s)\rvert$$ 的二阶矩变得可以控制。

**为什么这有效**：在 Hardy–Littlewood 的零点计数方法中，柯西–施瓦茨不等式是关键一步。乘上 $$M(s)$$ 后，相应的不等式更紧，从而把下界从 $$N_0(T) \gg T$$ 提升到

$$
N_0(T) \gg T\log T
$$

**这就是"正比例"的第一个证明。**

### 8.4 Levinson (1974) 与其后

Levinson 另辟蹊径：把 $$1/\zeta(s)$$ 展开为 Dirichlet 级数，通过分析 $$Z(t)$$ 的**矩**来提取零点信息，得

$$
\kappa \ge 0.3474
$$

后续经 Conrey (1989) 提升至 $$0.4088$$，Bui–Conrey–Young (2011) 提升至 **$$0.4105$$**（同时简单临界线零点比例 $$\kappa^* \ge 0.4058$$）。

**目前这仍是最好的结果。**

---

## 9. 数值验证的严格性

### 9.1 方法

现代验证的核心是 Riemann–Siegel 公式：计算

$$
Z(t) = e^{i\theta(t)}\zeta\!\left(\frac{1}{2}+it\right)
$$

在实轴上的值并统计**符号变化**，每个变号点对应临界线上的一个零点。

**关键在于"完整性"**：不仅要找到零点，还要证明**没有遗漏**。这用到 **Turing 方法**——通过考察 $$Z(t)$$ 在若干辅助点上的行为，给出区间内零点数目的上下界。

### 9.2 当前的记录

Platt 与 Trudgian (2021) 用**区间算术**（interval arithmetic，保证每次运算的误差边界是严格的）验证：

$$
\text{RH 成立至高度 } 3\,000\,175\,332\,800
$$

涉及 **12 363 153 437 138** 个非平凡零点，全部在临界线上，且全部为**单零点**。

> **但请务必理解这个结果的性质**：它是一个关于**有限区间**的严格陈述。黎曼猜想是一个关于**所有零点**的陈述。从有限到无穷的那一步，**没有任何已知的机制**可以完成。

---

## 10. 一些现代路径

### 10.1 零点检测与矩方法

寻找反例的高效方法是寻找 $$Z(t)$$ 的"异常大值"：若某处 $$\lvert Z(t)\rvert$$ 异常巨大，可能暗示其附近存在偏离临界线的零点对。这条路催生了 Turán、Levinson、Conrey 等人的工作。

### 10.2 算子理论方法

把 $$\xi$$ 视为希尔伯特空间上的算子特征函数，RH 转化为某个紧算子的**谱**是否为纯点谱且谱恰为 $$\{-\tfrac12 + it\}$$ 的问题（Guillemin–Reznikov 的工作）。

### 10.3 随机矩阵

随机矩阵的零点分布与 $$ζ$$ 的零点统计高度吻合。若 RH 成立，则实部特征值应呈三角分布（Erdős–Kac 型）。反之，若观测到偏离三角分布的统计特征，就可能指向反例。

### 10.4 算术方法

- **Weil 猜想**：已由 Deligne 证明（1974），与 RH 有深层关联。
- **Birch–Swinnerton-Dyer 猜想** 与 RH 独立但在结构上呼应。
- **Hilbert 第十问题**、**Erdős–Kac 定理**等均以 RH 为条件。

### 10.5 计算与 AI 辅助

2026 年 10 月的"准黎曼猜想"结果（见上一篇）是这一方向的最新样本。需要强调：

- 其结论是**收窄临界带**（到 $$[1/8, 7/8]$$ 或 $$[1/12, 11/12]$$），**不是** RH；
- 其价值在于**形式化验证**（Lean）与**技术组合**的方法论示范；
- 数学界态度是"谨慎乐观"，普遍认为结果正确但表述远未达到发表标准。

---

## 11. 为什么这仍然难：三个层次的困难

**第一层：技术障碍。**

RH 需要对**任意大**的 $$T$$ 证明 $$N(\tfrac12+\delta, T) = 0$$。现有的所有解析工具（密度估计、零点自由区、大值估计）都只在**渐近意义下**给出信息，且留有无法消除的余项。

**第二层：方法论障碍。**

现代数论高度依赖"平均"——对大量零点取平均后，某些极端个例被稀释。而 RH 是一个**全称命题**：每一个零点都要满足。它本质上是一个**"零一"性质**，难以用"平均"式的工具捕捉。

> 一个尖锐的类比：素数定理是**平均**陈述（$$\pi(x) \sim x/\log x$$），而 RH 是**极端**陈述（每个零点的位置）。**至今没有一般性的方法能把极端陈述化归为平均陈述。**

**第三层：概念障碍。**

即便 RH 为真，它的证明很可能**不是**我们想象的那种形式。Guth–Maynard 的例子说明：突破可能来自与"零点"完全不同的方向（比如 Dirichlet 多项式的大值估计）。猜想的形式（关于一个特定函数的零点）与它的证明（可能涉及完全不同的对象）之间的**概念落差**，是这道题最迷人的地方。

---

## 12. 结语

把两篇的内容合起来看：

**第一篇**回答了"谁提出的、为什么提出、后来怎么样了"；
**第二篇**回答了"它究竟在说什么、为什么这个说法有分量、难在何处"。

三条主线值得记住：

1. **黎曼的转换**：把素数的离散分布问题，转换成复函数的零点问题。这个转换是全部后续工作的起点。

2. **零点即振荡**：显式公式表明零点的实部控制误差衰减率。$$\beta = 1/2$$ 之所以"自然"，是因为它是所有可能值中最保守的——任何偏离都会让误差项变差。

3. **平均与极端的鸿沟**：密度估计的力量在于平均，RH 的难点在于极端。**四十年无人能破的指数 $$12/5$$ 在 2024 年被推到 $$30/13$$，但距离"最优"仍然遥远，而"最优"本身也只是必要的中间步骤。**

一百六十年前，一个在意大利湖畔病榻上的年轻人，在生命的最后几年里把这个问题交给了世界。一百六十年后，我们对它的理解已经深入到能精确刻画"它有多难"，却仍然无法说出那句简单的答案。

> 这本身就是数学最动人的地方：**提出一个好问题，和解决它，是两种完全不同的才能。**

---

## 参考文献

- Riemann, B. (1859). *Über die Anzahl der Primzahlen unter einer gegebenen Grösse*.
- Edwards, H. M. (2001). *Riemann's Zeta Function*. Dover.（第 1–3 章逐步推导函数方程、乘积公式与显式公式；附录含黎曼原文英译）
- Titchmarsh, E. C. (1986). *The Theory of the Riemann Zeta-Function* (2nd ed.). Oxford.
- Iwaniec, H., & Kowalski, E. (2004). *Analytic Number Theory*. AMS.
- Bombieri, E. (2000). *Problems of the Millennium: the Riemann Hypothesis*. Clay Mathematics Institute.
- Karatsuba, A. A. (1992). *On the Zeta-Function*. Ellis Horwood.
- Platt, D. J., & Trudgian, T. (2021). *The Riemann hypothesis is true up to* $$3\cdot 10^{12}$$. *Bull. London Math. Soc.*, 53(3), 792–797.
- Guth, L., & Maynard, J. (2024). *New large value estimates for Dirichlet polynomials*. arXiv:2405.20552.
- Bui, H. M., Conrey, B., & Young, M. P. (2011). *More than 41% of the zeros of the zeta function are on the critical line*. *Acta Arith.*, 150(1), 35–64.
- Radziwiłł, M., et al. (2026). *The Riemann Hypothesis: Past, Present and a Letter Through Time*. arXiv:2602.04022.
- 陶哲轩 (Terence Tao). 关于 Guth–Maynard 结果与 Vinogradov–Korobov 零点自由区的公开评论，Mastodon / MathOverflow。
