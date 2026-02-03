## 1. Linear Regression (线性回归)

### 1.1 Model Representation

给定训练集 $\{(x^{(i)}, y^{(i)}); i=1,\dots,n\}$，其中 $x \in \mathbb{R}^d$。

- **Hypothesis**: $h_\theta(x) = \theta^T x$ (假设 $x_0=1$)。

- **Cost Function (Ordinary Least Squares)**:

  $$J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2 = \frac{1}{2} \|X\theta - \vec{y}\|^2$$

### 1.2 The Normal Equations (Derivations)

为了求 $J(\theta)$ 的解析解，我们利用矩阵微积分求梯度并令其为 0。

**推导过程**：

$$\begin{align*} J(\theta) &= \frac{1}{2} (X\theta - \vec{y})^T (X\theta - \vec{y}) \\ &= \frac{1}{2} (\theta^T X^T X \theta - 2(X^T \vec{y})^T \theta + \vec{y}^T \vec{y}) \end{align*}$$

对 $\theta$ 求导：

$$\nabla_\theta J(\theta) = \frac{1}{2} (2X^T X \theta - 2X^T \vec{y}) = X^T X \theta - X^T \vec{y}$$

令 $\nabla_\theta J(\theta) = 0$：

$$X^T X \theta = X^T \vec{y} \implies \theta = (X^T X)^{-1} X^T \vec{y}$$

### 1.3 Probabilistic Interpretation

假设 $y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$，其中噪声 $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$ IID  **Log Likelihood**:

$$\ell(\theta) = \sum_{i=1}^n \log p(y^{(i)}|x^{(i)}; \theta) = -\frac{1}{2\sigma^2} \sum_{i=1}^n (y^{(i)} - \theta^T x^{(i)})^2 + C$$

最大化 $\ell(\theta)$ 等价于最小化平方误差项。这证明了 Least Squares 是高斯噪声假设下的 MLE。

## 2. Generative Learning Algorithms (生成学习算法)

生成式模型对联合概率 $p(x, y) = p(x|y)p(y)$ 建模，通过贝叶斯公式求 $p(y|x)$。

### 2.1 Gaussian Discriminant Analysis (GDA)

适用于连续值特征 $x \in \mathbb{R}^n$。 **模型假设**：

- $y \sim \text{Bernoulli}(\phi)$
- $x|y=0 \sim \mathcal{N}(\mu_0, \Sigma)$
- $x|y=1 \sim \mathcal{N}(\mu_1, \Sigma)$

**Log Likelihood**:

$$\begin{aligned} \ell(\phi, \mu_0, \mu_1, \Sigma) &= \log \prod_{i=1}^n p(x^{(i)}, y^{(i)}) \\ &= \log \prod_{i=1}^n p(x^{(i)}|y^{(i)}) p(y^{(i)}) \\ &= \sum_{i=1}^n \left[ \log p(x^{(i)}|y^{(i)}) + \log p(y^{(i)}) \right] \end{aligned}$$

**最大似然估计 (MLE)** 结果直观且符合统计直觉：

- $\phi = \frac{1}{n} \sum_{i=1}^n 1\{y^{(i)}=1\}$
- $\mu_k = \frac{\sum 1\{y^{(i)}=k\} x^{(i)}}{\sum 1\{y^{(i)}=k\}}$ (类均值向量)
- $\Sigma = \frac{1}{n} \sum_{i=1}^n (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T$ (类内散度矩阵)

**GDA vs Logistic Regression**: GDA 的后验分布 $p(y=1|x)$ 可以推导为 Sigmoid 形式：

$$p(y=1|x) = \frac{1}{1 + e^{-\theta^T x}}$$

其中 $\theta$ 是 $\phi, \mu_0, \mu_1, \Sigma$ 的函数。

- **Generative (GDA)**: 假设 $x|y$ 是高斯分布。如果假设成立，GDA 比 Logistic Regression **收敛更快 (more data efficient)**。
- **Discriminative (Logistic)**: 直接拟合 $p(y|x)$，不关心 $x$ 的分布。如果 $x|y$ 不是高斯分布（例如 Poisson），GDA 效果会变差，而 Logistic Regression 依然表现良好（**更鲁棒**）。

### 2.2 Naive Bayes (朴素贝叶斯)

核心假设：**Given** $y$**, features** $x_j$ **are conditionally independent**. 适用于文本分类等高维离散数据。根据特征建模方式不同，分为两种模型：

#### A. Multivariate Bernoulli Event Model

- **特征定义**: $x \in \{0, 1\}^d$。$x_j=1$ 表示词典中第 $j$ 个词在文档中出现（不计次数）。

- **Likelihood**:

  $$p(x|y) = \prod_{j=1}^d p(x_j|y)^{x_j} (1 - p(x_j|y))^{1-x_j}$$

- **适用场景**: 短文本，关注“词是否出现”。

#### B. Multinomial Event Model

- **特征定义**: $x$ 为文档中的词序列 $\{w_1, w_2, \dots, w_m\}$，或者理解为词频向量。

- **Likelihood**:

  $$p(x|y) = \prod_{k=1}^m p(w_k|y)$$

  即假设生成文档是多次独立的掷骰子试验，每次从词典中选一个词。

- **参数**: $\phi_{k|y=1} = p(\text{word}=k | y=1)$。

- **MLE with Laplace Smoothing**:

  $$\phi_{k|y=1} = \frac{\sum_{i=1}^n 1\{y^{(i)}=1\} \cdot (\text{count of word } k \text{ in doc } i) + 1}{\sum_{i=1}^n 1\{y^{(i)}=1\} \cdot (\text{total words in doc } i) + |V|}$$

## 3. Support Vector Machines (SVM)

SVM 的核心是最大化**几何间隔 (Geometric Margin)**，并通过**对偶理论 (Duality)** 引入核函数 (Kernels)。

### 3.1 Margins (间隔)

- **Functional Margin**: $\hat{\gamma}^{(i)} = y^{(i)}(w^T x^{(i)} + b)$。
  - 如果 $y^{(i)}=1$，我们希望 $w^T x + b \gg 0$。
  - 问题：缩放 $(w, b) \to (2w, 2b)$ 会导致 $\hat{\gamma}$ 翻倍，但超平面不变。
- **Geometric Margin**: $\gamma^{(i)} = \frac{\hat{\gamma}^{(i)}}{\|w\|}$。
  - 这是点到超平面的欧几里得距离（标准化后的函数间隔）。
  - 具有尺度不变性 (Scale Invariant)。

### 3.2 The Primal Problem (原始问题)

我们的目标是最大化最小几何间隔 $\gamma$：

$$\max_{\gamma, w, b} \gamma \quad \text{s.t.} \quad \frac{y^{(i)}(w^T x^{(i)} + b)}{\|w\|} \geq \gamma, \forall i$$

利用尺度不变性，令函数间隔 $\hat{\gamma}=1$，这就强制了 $\|w\| = 1/\gamma$。 问题转化为**最小化** $\|w\|^2$：

$$\begin{aligned} \min_{w, b} \quad & \frac{1}{2} \|w\|^2 \\ \text{s.t.} \quad & y^{(i)}(w^T x^{(i)} + b) \geq 1, \quad i=1,\dots,n \end{aligned}$$

这是一个**凸二次规划 (Convex Quadratic Programming)** 问题。

### 3.3 Lagrangian Duality (详细推导)

为了求解约束优化问题，构建 **Lagrangian**:

$$\mathcal{L}(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^n \alpha_i [y^{(i)}(w^T x^{(i)} + b) - 1]$$

其中 $\alpha_i \geq 0$ 是拉格朗日乘子。 原始问题等价于：$\min_{w, b} \max_{\alpha: \alpha_i \geq 0} \mathcal{L}(w, b, \alpha)$。 对偶问题 (Dual) 是交换 min 和 max：$\max_{\alpha: \alpha_i \geq 0} \min_{w, b} \mathcal{L}(w, b, \alpha)$。

**Step 1: Minimize** $\mathcal{L}$ **w.r.t** $w$ **and** $b$ 对 $w$ 求偏导：

$$\nabla_w \mathcal{L} = w - \sum_{i=1}^n \alpha_i y^{(i)} x^{(i)} = 0 \implies w = \sum_{i=1}^n \alpha_i y^{(i)} x^{(i)}$$

对 $b$ 求偏导：

$$\nabla_b \mathcal{L} = - \sum_{i=1}^n \alpha_i y^{(i)} = 0 \implies \sum_{i=1}^n \alpha_i y^{(i)} = 0$$

**Step 2: Substitute back into** $\mathcal{L}$ 将 $w$ 代回 Lagrangian：

$$\begin{aligned} \mathcal{L} &= \frac{1}{2} \| \sum_{i} \alpha_i y^{(i)} x^{(i)} \|^2 - \sum_{i} \alpha_i y^{(i)} (\sum_{j} \alpha_j y^{(j)} x^{(j)})^T x^{(i)} - \sum_{i} \alpha_i y^{(i)} b + \sum_{i} \alpha_i \\ &= \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y^{(i)} y^{(j)} \langle x^{(i)}, x^{(j)} \rangle - \sum_{i,j} \alpha_i \alpha_j y^{(i)} y^{(j)} \langle x^{(i)}, x^{(j)} \rangle - 0 + \sum_{i} \alpha_i \\ &= \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j=1}^n y^{(i)} y^{(j)} \alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle \end{aligned}$$

**Step 3: The Dual Problem**

$$\begin{aligned} \max_\alpha \quad & W(\alpha) = \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j=1}^n y^{(i)}y^{(j)}\alpha_i\alpha_j \langle x^{(i)}, x^{(j)} \rangle \\ \text{s.t.} \quad & \alpha_i \geq 0, \quad \sum_{i=1}^n \alpha_i y^{(i)} = 0 \end{aligned}$$

此问题依然是 QP 问题，可以利用 **SMO (Sequential Minimal Optimization)** 算法高效求解。SMO 的核心思想是每次只选取两个 $\alpha_i, \alpha_j$ 进行优化，固定其他变量，从而获得解析解。

### 3.4 KKT Conditions TODO

由于原问题是凸的且满足 Slater 条件，强对偶性 (Strong Duality) 成立。最优解 $(w^*, b^*, \alpha^*)$ 必须满足 KKT 条件：

1. **Stationarity**: $\nabla_w \mathcal{L} = 0, \nabla_b \mathcal{L} = 0$

2. **Primal Feasibility**: $y^{(i)}(w^T x^{(i)} + b) \geq 1$

3. **Dual Feasibility**: $\alpha_i \geq 0$

4. **Complementary Slackness (互补松弛性)**:

   $$\alpha_i [y^{(i)}(w^T x^{(i)} + b) - 1] = 0$$

**物理意义**：

- 如果 $\alpha_i > 0$，则必须有 $y^{(i)}(w^T x^{(i)} + b) = 1$。这些点正好落在最大间隔边界上，称为 **Support Vectors (支持向量)**。
- 对于非支持向量，函数间隔 $>1$，必然有 $\alpha_i = 0$。
- 这就是 SVM 的**稀疏性 (Sparsity)**：模型参数 $w$ 仅由少数支持向量决定。

### 3.5 Kernels TODO

在 Dual Form 中，数据仅以内积 $\langle x^{(i)}, x^{(j)} \rangle$ 形式出现。 定义映射 $\phi: \mathbb{R}^d \to \mathbb{R}^D$ (通常 $D \gg d$)。 定义 Kernel Function $K(x, z) = \phi(x)^T \phi(z)$。 我们在高维空间寻找线性超平面，相当于在低维空间寻找非线性边界。

**常见核函数**：

- **Linear Kernel**: $K(x, z) = x^T z$
- **Polynomial Kernel**: $K(x, z) = (x^T z + c)^p$
- **RBF (Gaussian) Kernel**: $K(x, z) = \exp(-\frac{\|x-z\|^2}{2\sigma^2})$
  - 对应无限维特征空间。
  - $\sigma$ 越小，模型越复杂（易过拟合）；$\sigma$ 越大，决策边界越平滑。

**Mercer's Theorem**: 函数 $K$ 是有效核函数的充要条件是：对于任意数据集，其对应的 Gram Matrix $G$ ($G_{ij} = K(x^{(i)}, x^{(j)})$) 是半正定 (Positive Semi-Definite) 的。

### 3.6 Regularization & Soft Margin TODO

如果数据线性不可分，原始约束会导致无解。我们引入松弛变量 (Slack Variables) $\xi_i \geq 0$：

$$\begin{aligned} \min_{w, b, \xi} \quad & \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i \\ \text{s.t.} \quad & y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i \\ & \xi_i \geq 0 \end{aligned}$$

- $C$ **的作用**：权衡间隔宽度 ($\|w\|^2$) 和误分类惩罚 ($\sum \xi_i$)。

  - $C$ 很大：对误分类惩罚重 $\to$ Hard Margin (Low Bias, High Variance)。
  - $C$ 很小：允许更多误分类 $\to$ Wider Margin (High Bias, Low Variance)。

- **Soft Margin Dual**: 推导过程类似，唯一的变化是约束条件变为：

  $$0 \leq \alpha_i \leq C$$

  这限制了单个样本对模型的影响力上限（通过 $\alpha_i$），增强了鲁棒性。

## 1. Bias-Variance Tradeoff (偏差-方差权衡)

泛化误差 (Generalization Error) 的分解是理解模型复杂度与性能关系的核心。

### 1.1 Problem Setup

假设数据生成分布为：

$$y = f(x) + \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, \sigma^2)$$

我们通过算法学习到一个假设函数 $\hat{f}(x; D)$（依赖于训练集 $D$）。

我们要评估在测试点 $x$ 处的期望平方误差 (Expected Squared Error)（对所有可能的训练集 $D$ 和噪声 $\epsilon$ 取期望）：

$$\text{Err}(x) = E_{D, \epsilon} \left[ (y - \hat{f}(x; D))^2 \right]$$

### 1.2 Decomposition Derivation (详细推导)

为了简化符号，令 $\hat{f} = \hat{f}(x; D)$，$f = f(x)$，$E[\cdot]$ 表示对 $D$ 和 $\epsilon$ 的期望。

注意 $y$ 包含噪声 $\epsilon$，而 $f$ 是确定性的。

$$\begin{aligned} \text{Err}(x) &= E \left[ (f + \epsilon - \hat{f})^2 \right] \\ &= E \left[ (f - \hat{f})^2 \right] + E[\epsilon^2] + 2 E[(f - \hat{f})\epsilon] \end{aligned}$$

- **噪声项**: $E[\epsilon^2] = \sigma^2$ (Irreducible Error)。
- **交叉项**: 由于 $\epsilon$ 独立于 $\hat{f}$ 且 $E[\epsilon]=0$，故 $2 E[(f - \hat{f})] E[\epsilon] = 0$。

剩下项 $E[(f - \hat{f})^2]$ 衡量估计值与真实函数的偏差。引入 $\hat{f}$ 的期望 $E[\hat{f}]$（即无限次训练得到的平均模型）：

$$\begin{aligned} E[(f - \hat{f})^2] &= E \left[ (f - E[\hat{f}] + E[\hat{f}] - \hat{f})^2 \right] \\ &= E \left[ (f - E[\hat{f}])^2 \right] + E \left[ (E[\hat{f}] - \hat{f})^2 \right] + 2 E \left[ (f - E[\hat{f}])(E[\hat{f}] - \hat{f}) \right] \end{aligned}$$

- **Bias 项**: 第一项 $(f - E[\hat{f}])^2$ 是常数（关于 $D$），定义为 $(\text{Bias}[\hat{f}(x)])^2$。
- **Variance 项**: 第二项 $E \left[ (\hat{f} - E[\hat{f}])^2 \right]$ 正是方差定义 $\text{Var}(\hat{f}(x))$。
- **交叉项消去**: 注意 $f - E[\hat{f}]$ 是常数，而 $E[E[\hat{f}] - \hat{f}] = E[\hat{f}] - E[\hat{f}] = 0$。

**最终分解**:

$$\text{Err}(x) = \underbrace{(E[\hat{f}(x)] - f(x))^2}_{\text{Bias}^2} + \underbrace{E[(\hat{f}(x) - E[\hat{f}(x)])^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}$$

### 1.3 Interpretation

- **Bias (偏差)**: 模型假设的期望输出与真实值的差异。高偏差意味着**欠拟合 (Underfitting)**（模型太简单，如用线性模型拟合非线性数据）。
- **Variance (方差)**: 模型输出对训练集 $D$ 变化的敏感度。高方差意味着**过拟合 (Overfitting)**（模型太复杂，记住了噪声）。
- **Tradeoff**: 随着模型复杂度增加，Bias $\downarrow$ 但 Variance $\uparrow$。最优模型位于两者之和最小处（U形曲线底部）。
  - *注*: 现代深度学习中出现了 "Double Descent" 现象，即在参数量极大时泛化误差再次下降，但这超出了经典统计学习理论范畴。

## 2. Learning Theory TODO

本节重点讨论在**有限假设集 (Finite Hypothesis Class)** 下的学习理论保证。我们将通过 **PAC (Probably Approximately Correct)** 框架来回答：*我们需要多少数据才能保证学习到的模型是可靠的？*

### 2.1 Mathematical Setup (形式化设定)

- **Assumption**: 训练样本 $S = \{(x^{(i)}, y^{(i)})\}_{i=1}^m$ 是从分布 $\mathcal{D}$ 中 **独立同分布 (i.i.d.)** 采样的。

- **Hypothesis Class**: $\mathcal{H}$ 是一个有限的假设集合，大小为 $k = |\mathcal{H}|$。

- True Risk (Generalization Error) $\epsilon(h)$: 假设 $h$ 在未见数据上的期望误差。

  

  $$\epsilon(h) = P_{(x,y) \sim \mathcal{D}}(h(x) \neq y)$$

- Empirical Risk (Training Error) $\hat{\epsilon}(h)$: 假设 $h$ 在训练集 $S$ 上的平均误差。

  

  $$\hat{\epsilon}(h) = \frac{1}{m} \sum_{i=1}^m 1\{h(x^{(i)}) \neq y^{(i)}\}$$

- ERM (Empirical Risk Minimizer): 学习算法选择在训练集上表现最好的假设：

  

  $$\hat{h} = \arg\min_{h \in \mathcal{H}} \hat{\epsilon}(h)$$

- Best Hypothesis: 在 $\mathcal{H}$ 中真实误差最小的假设：

  

  $$h^* = \arg\min_{h \in \mathcal{H}} \epsilon(h)$$

### 2.2 The Two Main Tools (两大理论基石)

为了证明 ERM 的有效性，我们需要两个概率不等式工具。

1. The Union Bound (Boole's Inequality):

   对于任意 $k$ 个事件 $A_1, \dots, A_k$（无论是否独立）：

   

   $$P(A_1 \cup \dots \cup A_k) \le \sum_{i=1}^k P(A_i)$$

2. Hoeffding's Inequality (Concentration Inequality):

   令 $Z_1, \dots, Z_m$ 为 $m$ 个 i.i.d. 的伯努利随机变量，参数为 $\phi$（即 $E[Z_i] = \phi$）。令 $\hat{\phi} = \frac{1}{m}\sum_{i=1}^m Z_i$ 为经验均值。对于任意 $\gamma > 0$：

   

   $$P(|\hat{\phi} - \phi| > \gamma) \le 2 \exp(-2\gamma^2 m)$$

   

   物理意义：随着样本量 $m$ 增加，经验均值 $\hat{\phi}$ 偏离真实均值 $\phi$ 的概率呈指数级下降。

### 2.3 Uniform Convergence (一致收敛)

**Core Problem**: 我们不能简单地对 $\hat{h}$ 应用 Hoeffding 不等式，因为 $\hat{h}$ 是依赖于数据的（data-dependent）。为了保证 $\hat{h}$ 的泛化能力，我们需要保证**所有** $h \in \mathcal{H}$ 的经验误差都接近真实误差。

**Theorem (Uniform Convergence):**

对于有限假设集 $\mathcal{H}$，至少以 $1-\delta$ 的概率，对于所有的 $h \in \mathcal{H}$，我们有：



$$|\hat{\epsilon}(h) - \epsilon(h)| \le \gamma$$



只要样本量满足 $m \ge \frac{1}{2\gamma^2} \log \frac{2k}{\delta}$。

**Proof Strategy**:

1. 单一假设: 对任意固定的 $h_j$，应用 Hoeffding 不等式：

   

   $$P(|\hat{\epsilon}(h_j) - \epsilon(h_j)| > \gamma) \le 2 \exp(-2\gamma^2 m)$$

2. **所有假设 (Union Bound)**: 我们希望 *不存在* 任何一个 $h$ 违反上述界限。

   $$\begin{aligned} P(\exists h \in \mathcal{H}, |\hat{\epsilon}(h) - \epsilon(h)| > \gamma) &= P(\bigcup_{j=1}^k \{ |\hat{\epsilon}(h_j) - \epsilon(h_j)| > \gamma \}) \\ &\le \sum_{j=1}^k 2 \exp(-2\gamma^2 m) \\ &= 2k \exp(-2\gamma^2 m) \end{aligned}$$

3. **置信度**: 令右侧概率等于 $\delta$，解出 $m$。

### 2.4 Generalization Bound for ERM (ERM 的误差界限)

这是该章节最重要的推导：**为什么一致收敛能保证 ERM 找到好模型？**

假设一致收敛成立（即 $|\hat{\epsilon}(h) - \epsilon(h)| \le \gamma, \forall h$），我们来推导 $\hat{h}$ 的真实泛化误差 $\epsilon(\hat{h})$ 与最优模型 $h^*$ 的泛化误差 $\epsilon(h^*)$ 之间的关系。

**Derivation**:

$$\begin{aligned} \epsilon(\hat{h}) &\le \hat{\epsilon}(\hat{h}) + \gamma  & (\text{Uniform Convergence on } \hat{h}) \\ &\le \hat{\epsilon}(h^*) + \gamma & (\text{By definition of ERM, } \hat{\epsilon}(\hat{h}) \le \hat{\epsilon}(h^*)) \\ &\le (\epsilon(h^*) + \gamma) + \gamma & (\text{Uniform Convergence on } h^*) \\ &= \epsilon(h^*) + 2\gamma \end{aligned}$$

Conclusion:



$$\epsilon(\hat{h}) \le \epsilon(h^*) + 2\sqrt{\frac{1}{2m} \log \frac{2k}{\delta}}$$



这个结论非常有力量：

- 只要样本量 $m$ 足够大，ERM 选出的模型 $\hat{h}$ 的表现几乎和全知全能选出的最优模型 $h^*$ 一样好（只差 $2\gamma$）。
- **Sample Complexity**: 为了保证 $\epsilon(\hat{h}) \le \epsilon(h^*) + \epsilon'$（即 $2\gamma = \epsilon'$），我们需要 $m = O(\frac{1}{\epsilon'^2} \log k)$。

## 3. Infinite Hypothesis Classes (VC Dimension)

当 $\mathcal{H}$ 是无限集（如线性分类器参数为连续实数）时，$\log |\mathcal{H}|$ 无意义。我们需要 **VC Dimension** 来衡量复杂度。

### 3.1 Shattering (打散)

给定集合 $S = \{x^{(1)}, \dots, x^{(d)}\}$，如果对于所有可能的 $2^d$ 种标签组合，$\mathcal{H}$ 中都存在一个假设能将它们正确分类（即 $\hat{\epsilon}(h)=0$），则称 $\mathcal{H}$ **Shatter (打散)** 了 $S$。

### 3.2 VC Dimension Definition

**Definition**: $VC(\mathcal{H})$ 是 $\mathcal{H}$ 能 shatter 的最大数据集的大小。

$$VC(\mathcal{H}) = \max \{ d : \exists S \text{ s.t. } |S|=d \text{ and } \mathcal{H} \text{ shatters } S \}$$

- 注意：只要存在**一个**大小为 $d$ 的集合能被 shatter 即可，不需要所有大小为 $d$ 的集合都能被 shatter。
- 如果 $\mathcal{H}$ 能 shatter 任意大的集合，则 $VC(\mathcal{H}) = \infty$。

**Examples**:

- **Linear Classifiers in 2D**: $VC(\mathcal{H}) = 3$。任意3个非共线点可被 shatter，但4个点（如XOR结构）不能。
- **Linear Classifiers in** $\mathbb{R}^n$: $VC(\mathcal{H}) = n + 1$。

### 3.3 Sample Complexity with VC Dimension

Vapnik 和 Chervonenkis 证明了对于无限假设集，一致收敛依然成立，只要 $m$ 满足：

$$m = O \left( \frac{VC(\mathcal{H})}{\gamma^2} \log \frac{1}{\gamma} + \frac{1}{\gamma^2} \log \frac{1}{\delta} \right)$$

(更简化的经验法则是 $N \approx 10 \cdot VC(\mathcal{H})$)。

Fundamental Theorem of Statistical Learning:

一个二分类任务是 PAC Learnable 的，当且仅当 $\mathcal{H}$ 的 VC 维是有限的。

### 3.4 Structural Risk Minimization (SRM)

由于 $VC(\mathcal{H})$ 越高，需要的 $m$ 越大（Variance 越大），但同时训练误差 $\hat{\epsilon}(h)$ 可能越小（Bias 越小）。

SRM 准则：

$$\hat{h} = \arg\min_{h \in \mathcal{H}} \left( \hat{\epsilon}(h) + \text{ComplexityPenalty}(VC(\mathcal{H}), m) \right)$$

这对应于正则化 (Regularization) 的理论基础。

## 4. Model Selection (模型选择)

### 4.1 Cross-Validation (交叉验证)

- **Hold-out CV**: 分割 $70\%$ 训练，$30\%$ 验证。缺点是浪费了 $30\%$ 数据。
- **k-Fold CV**: 将数据分 $k$ 份，轮流做验证。
  - 计算量是 Hold-out 的 $k$ 倍。
  - Variance 较小，Bias 较小（因为每次用了 $(k-1)/k$ 的数据）。
- **LOOCV (Leave-One-Out)**: $k=m$。无偏但方差大，且计算昂贵。

### 4.2 Feature Selection (特征选择)

当 $n \gg m$ 时（VC 维过高），需要减少特征。

- **Forward Search (Wrapper Method)**:

  1. 初始化 $F = \emptyset$。
  2. 每次尝试加入一个特征 $i \notin F$，训练并评估性能。
  3. 选择提升最大的特征加入 $F$。

  - 计算复杂度高，需调用 $O(n^2)$ 次学习算法。

- **Filter Method**:

  - 使用某种评分（如 Mutual Information, Correlation）预先筛选特征。
  - 计算快，但忽略了特征组合效果。



## 1. Clustering: K-means (聚类：K-均值)

### 1.1 Problem Definition

给定数据集 $\{x^{(1)}, \dots, x^{(m)}\}$，无标签。我们希望将数据划分到 $k$ 个簇中。

- **参数**:

  - $c^{(i)} \in \{1, \dots, k\}$: 样本 $x^{(i)}$ 的簇分配。
  - $\mu_j \in \mathbb{R}^n$: 第 $j$ 个簇的质心 (Centroid)。

- **Objective Function (Distortion)**: 目标是最小化所有点到其对应质心的距离平方和：

  $$J(c, \mu) = \sum_{i=1}^m \| x^{(i)} - \mu_{c^{(i)}} \|^2$$

### 1.2 Coordinate Descent View (坐标下降视角)

K-means 算法并非使用梯度下降，而是**坐标下降 (Coordinate Descent)**。它交替优化两组变量 $c$ 和 $\mu$。

**Step 1: Minimize** $J$ **w.r.t** $c$ **(Cluster Assignment)** 固定 $\mu$，对每个 $i$ 独立优化 $c^{(i)}$：

$$\min_{c^{(i)}} \| x^{(i)} - \mu_{c^{(i)}} \|^2$$

显然，最优解是把 $x^{(i)}$ 分配给距离最近的质心：

$$c^{(i)} := \arg\min_{j \in \{1,\dots,k\}} \| x^{(i)} - \mu_j \|^2$$

**Step 2: Minimize** $J$ **w.r.t** $\mu$ **(Move Centroids)** 固定 $c$，对每个簇中心 $\mu_j$ 独立优化。$J$ 中只包含涉及 $\mu_j$ 的项：

$$J_{\mu_j} = \sum_{i: c^{(i)}=j} \| x^{(i)} - \mu_j \|^2$$

这是一个凸二次函数。对 $\mu_j$ 求梯度并令其为 0：

$$\nabla_{\mu_j} J = \sum_{i: c^{(i)}=j} 2(\mu_j - x^{(i)}) = 0$$

$$\Rightarrow \sum_{i: c^{(i)}=j} \mu_j = \sum_{i: c^{(i)}=j} x^{(i)}$$

设 $n_j$ 为簇 $j$ 中的样本数，则 $n_j \mu_j = \sum_{c^{(i)}=j} x^{(i)}$。

$$\mu_j := \frac{1}{n_j} \sum_{i: c^{(i)}=j} x^{(i)}$$

**物理意义**: 新的质心是簇内所有点的几何中心（均值）。

### 1.3 Convergence (收敛性)

- **单调性**: 每一步更新（无论是 $c$ 还是 $\mu$）都严格保证 $J(c, \mu)$ 不增加（Non-increasing）。
- **有界性**: $J(c, \mu) \ge 0$ 有下界。
- **结论**: 算法必然收敛。
- **局部最优**: 由于 $J$ 是非凸函数 (Non-convex)，K-means 只能保证收敛到局部最优。
  - **解决方法**: 多次随机初始化运行 K-means，取最终 Distortion 最小的那次结果。

## 2. The EM Algorithm (期望最大化算法)

EM 是处理含有**隐变量 (Latent Variables)** 的概率模型参数估计的核心算法。

### 2.1 The Challenge: Intractability

设模型参数为 $\theta$，观测数据 $x$，隐变量 $z$。 目标是最大化对数似然：

$$\ell(\theta) = \sum_{i=1}^m \log p(x^{(i)}; \theta) = \sum_{i=1}^m \log \sum_{z^{(i)}} p(x^{(i)}, z^{(i)}; \theta)$$

**难点**: $\log$ 函数在求和符号外面 ($\log \sum$)，导致梯度计算极其复杂（各参数耦合在一起）。

### 2.2 Jensen's Inequality (理论基石)

设 $f$ 为**凹函数 (Concave function)**（如 $\log$），$X$ 为随机变量。则：

$$f(E[X]) \ge E[f(X)]$$

- **Intuition**: 割线在函数图像下方。
- **Equality Condition (取等条件)**: 当且仅当 $X$ 为常数 (with probability 1)，即随机变量的方差为 0。

### 2.3 Deriving the ELBO (Evidence Lower Bound)

为了解决 $\log \sum$，我们引入一个任意分布 $Q_i(z^{(i)})$（满足 $\sum_z Q_i(z) = 1$）。 利用 Jensen 不等式构建下界：

$$\begin{aligned} \ell(\theta) &= \sum_{i=1}^m \log \sum_{z^{(i)}} Q_i(z^{(i)}) \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} \\ &\ge \sum_{i=1}^m \underbrace{\sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}}_{\text{ELBO}(\theta, Q_i)} \quad (\text{Jensen: } \log E \ge E \log) \end{aligned}$$

### 2.4 The EM Strategy (Why it works?)

EM 算法通过不断提升这个下界 (ELBO) 来间接提升 $\ell(\theta)$。

**E-Step (Expectation): Tighten the Bound** 固定 $\theta$，调节 $Q_i$ 使得下界贴合当前的似然值（即 Jensen 取等号）。 Jensen 取等号的条件是随机变量为常数：

$$\frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} = c \quad (\text{constant w.r.t } z)$$

意味着 $Q_i(z^{(i)}) \propto p(x^{(i)}, z^{(i)}; \theta)$。 由于 $\sum_z Q_i(z) = 1$，归一化后必然得到：

$$Q_i(z^{(i)}) = \frac{p(x^{(i)}, z^{(i)}; \theta)}{\sum_z p(x^{(i)}, z; \theta)} = p(z^{(i)} | x^{(i)}; \theta)$$

**物理意义**: E-step 计算的是在当前参数下，隐变量的**后验概率 (Posterior Probability)**。

**M-Step (Maximization): Maximize the Bound** 固定 $Q_i$，调整 $\theta$ 使 ELBO 最大化。

$$\theta^{(t+1)} := \arg\max_\theta \sum_{i=1}^m \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}$$

这等价于最大化**期望完整对数似然 (Expected Complete Log-Likelihood)**：

$$\arg\max_\theta \sum_{i=1}^m E_{z \sim Q_i} [\log p(x^{(i)}, z; \theta)]$$

**Convergence Proof**:

$$\ell(\theta^{(t)}) \underset{\text{E-step}}{=} \text{ELBO}(\theta^{(t)}, Q^{(t)}) \underset{\text{M-step}}{\le} \text{ELBO}(\theta^{(t+1)}, Q^{(t)}) \underset{\text{Jensen}}{\le} \ell(\theta^{(t+1)})$$

每一步都保证似然函数非递减。

## 3. Mixture of Gaussians (GMM, 高斯混合模型)

GMM 是 EM 算法最经典的实战案例。

### 3.1 Model Setup

- **Latent Variable**: $z^{(i)} \sim \text{Multinomial}(\phi)$, where $\phi_j = p(z^{(i)}=j)$, $\sum \phi_j=1$.
- **Observable**: $x^{(i)} | z^{(i)}=j \sim \mathcal{N}(\mu_j, \Sigma_j)$.
- **Joint Prob**: $p(x, z) = p(z)p(x|z) = \phi_{z} \cdot \mathcal{N}(x | \mu_z, \Sigma_z)$.

### 3.2 EM Derivation for GMM

**E-Step**: 计算隐变量的后验概率（通常称为 **Responsibility** $w_j^{(i)}$）：

$$w_j^{(i)} = Q_i(z^{(i)}=j) = p(z^{(i)}=j | x^{(i)}; \theta) = \frac{\phi_j \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)}{\sum_{k=1}^K \phi_k \mathcal{N}(x^{(i)}; \mu_k, \Sigma_k)}$$

这实际上是一个 Softmax 操作，衡量第 $j$ 个高斯分量对样本 $i$ 的解释程度。

**M-Step (Detailed Derivation)**: 我们需要最大化期望对数似然：

$$\mathcal{L}(\theta) = \sum_{i=1}^m \sum_{j=1}^K w_j^{(i)} \log p(x^{(i)}, z^{(i)}=j; \theta)$$

$$= \sum_{i=1}^m \sum_{j=1}^K w_j^{(i)} \left( \log \phi_j + \log \frac{1}{(2\pi)^{n/2}|\Sigma_j|^{1/2}} - \frac{1}{2}(x^{(i)}-\mu_j)^T \Sigma_j^{-1} (x^{(i)}-\mu_j) \right)$$

1. **Update** $\mu_j$: 对 $\mu_j$ 求偏导（只关注包含 $\mu_j$ 的项）：

   $$\nabla_{\mu_j} \mathcal{L} = \sum_{i=1}^m w_j^{(i)} \nabla_{\mu_j} \left[ -\frac{1}{2}(x^{(i)}-\mu_j)^T \Sigma_j^{-1} (x^{(i)}-\mu_j) \right]$$

   利用矩阵求导 $\nabla_x (x-b)^T A (x-b) = 2A(x-b)$：

   $$= \sum_{i=1}^m w_j^{(i)} \Sigma_j^{-1} (x^{(i)} - \mu_j) = 0$$

   消去 $\Sigma_j^{-1}$，得 $\sum_i w_j^{(i)} x^{(i)} = (\sum_i w_j^{(i)}) \mu_j$。

   $$\mu_j = \frac{\sum_{i=1}^m w_j^{(i)} x^{(i)}}{\sum_{i=1}^m w_j^{(i)}}$$

2. **Update** $\phi_j$: 包含 $\phi_j$ 的项是 $\sum_i \sum_j w_j^{(i)} \log \phi_j$。 约束条件是 $\sum_{j=1}^K \phi_j = 1$。 构造 Lagrange 函数 $\mathcal{J}(\phi) = \sum_i \sum_j w_j^{(i)} \log \phi_j - \lambda (\sum_j \phi_j - 1)$。 对 $\phi_j$ 求导：$\frac{\sum_i w_j^{(i)}}{\phi_j} - \lambda = 0 \implies \phi_j = \frac{\sum_i w_j^{(i)}}{\lambda}$。 代入约束 $\sum \phi_j = 1$ 可得 $\lambda = m$。

   $$\phi_j = \frac{1}{m} \sum_{i=1}^m w_j^{(i)}$$

3. **Update** $\Sigma_j$: 同样求导令为 0，得到加权协方差矩阵：

   $$\Sigma_j = \frac{\sum_{i=1}^m w_j^{(i)} (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\sum_{i=1}^m w_j^{(i)}}$$

## 





## Factor Analysis (因子分析) - Detailed Derivation

当样本量 $m$ 远小于特征维度 $n$ ($m \ll n$) 时，直接拟合多元高斯分布会导致协方差矩阵奇异。FA 假设数据由低维隐变量生成，并捕捉了特征间的相关性。

### 4.1 Generative Model (生成模型)

$$\begin{aligned} z &\sim \mathcal{N}(0, I) \quad (z \in \mathbb{R}^k, k < n) \\ \epsilon &\sim \mathcal{N}(0, \Psi) \quad (\Psi \text{ is diagonal}, \epsilon \in \mathbb{R}^n) \\ x &= \Lambda z + \mu + \epsilon \end{aligned}$$

- $\Lambda \in \mathbb{R}^{n \times k}$: 因子载荷矩阵 (Factor Loading Matrix)。
- $\Psi$: 噪声协方差矩阵 (对角阵)。这意味着给定 $z$ 后，特征 $x_i$ 之间是条件独立的。

### 4.2 Joint Distribution Derivation (联合分布推导)

为了进行 E-step，我们需要求 $p(z|x)$，这首先需要写出 $(z, x)$ 的联合高斯分布。

$$\begin{pmatrix} z \\ x \end{pmatrix} \sim \mathcal{N}(\mu_{joint}, \Sigma_{joint})$$

1. **Expectation (Mean)**:

   - $E[z] = 0$

   - $E[x] = E[\Lambda z + \mu + \epsilon] = \Lambda E[z] + \mu + E[\epsilon] = \mu$

     $$\mu_{joint} = \begin{pmatrix} 0 \\ \mu \end{pmatrix}$$

2. **Covariance (Block Matrix)**:

   $$\Sigma_{joint} = \begin{pmatrix} \Sigma_{zz} & \Sigma_{zx} \\ \Sigma_{xz} & \Sigma_{xx} \end{pmatrix}$$

   - $\Sigma_{zz} = Cov(z) = I$

   - $\Sigma_{xx} = Cov(x) = E[(x-\mu)(x-\mu)^T] = E[(\Lambda z + \epsilon)(\Lambda z + \epsilon)^T]$

     $$= \Lambda E[zz^T] \Lambda^T + E[\epsilon \epsilon^T] \quad (\text{Cross terms are 0})$$

     $$= \Lambda I \Lambda^T + \Psi = \Lambda \Lambda^T + \Psi$$

   - $\Sigma_{zx} = E[(z - E[z])(x - E[x])^T] = E[z (\Lambda z + \epsilon)^T] = E[zz^T]\Lambda^T + E[z\epsilon^T]$

     $$= I \Lambda^T + 0 = \Lambda^T$$

   **Result**:

   $$\begin{pmatrix} z \\ x \end{pmatrix} \sim \mathcal{N} \left( \begin{pmatrix} 0 \\ \mu \end{pmatrix}, \begin{pmatrix} I & \Lambda^T \\ \Lambda & \Lambda\Lambda^T + \Psi \end{pmatrix} \right)$$

### 4.3 EM Algorithm for FA

**E-Step: Inference (推断)** 我们需要计算 $Q_i(z^{(i)}) = p(z^{(i)} | x^{(i)})$。根据高斯条件分布公式：

$$\mu_{z|x} = \mu_z + \Sigma_{zx}\Sigma_{xx}^{-1}(x - \mu_x)$$

$$\Sigma_{z|x} = \Sigma_{zz} - \Sigma_{zx}\Sigma_{xx}^{-1}\Sigma_{xz}$$

代入 FA 的参数：

- **Mean**: $\mu_{z^{(i)}|x^{(i)}} = \Lambda^T (\Lambda \Lambda^T + \Psi)^{-1} (x^{(i)} - \mu)$
- **Covariance**: $\Sigma_{z^{(i)}|x^{(i)}} = I - \Lambda^T (\Lambda \Lambda^T + \Psi)^{-1} \Lambda$

为了 M-step 方便，我们定义以下期望值（注意 $E[zz^T] \ne E[z]E[z]^T$）：

- $E[z^{(i)}|x^{(i)}] = \mu_{z^{(i)}|x^{(i)}}$
- $E[z^{(i)}z^{(i)T}|x^{(i)}] = \mu_{z^{(i)}|x^{(i)}}\mu_{z^{(i)}|x^{(i)}}^T + \Sigma_{z^{(i)}|x^{(i)}}$

**M-Step: Parameter Learning (参数学习)** 最大化期望对数似然 (Expected Log-Likelihood)。 完整数据的对数似然：

$$\ell_{complete} = \sum_{i=1}^m \left( \log p(x^{(i)}|z^{(i)}) + \log p(z^{(i)}) \right)$$

由于 $p(z)$ 不含参数 $\Lambda, \Psi$（或是常数），我们只关注 $\log p(x|z)$：

$$\log p(x^{(i)}|z^{(i)}) \propto -\frac{1}{2} \log |\Psi| - \frac{1}{2} (x^{(i)} - \mu - \Lambda z^{(i)})^T \Psi^{-1} (x^{(i)} - \mu - \Lambda z^{(i)})$$

对该式取期望 $E_{z|x}[\cdot]$ 并对 $\Lambda$ 求导（利用迹技巧 $\text{tr}$）：

1. **Update** $\Lambda$: 类似最小二乘法的推导，最优解为：

   $$\Lambda_{new} = \left( \sum_{i=1}^m (x^{(i)} - \mu) E[z^{(i)}|x^{(i)}]^T \right) \left( \sum_{i=1}^m E[z^{(i)}z^{(i)T}|x^{(i)}] \right)^{-1}$$

   *直观理解*：这就好比输入是 $z$，输出是 $x-\mu$ 的线性回归系数 $(X^T Z)(Z^T Z)^{-1}$。

2. **Update** $\Psi$: $\Psi_{new}$ 是对角矩阵，其对角元素由下式给出：

   $$\Phi = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu)(x^{(i)} - \mu)^T - \Lambda_{new} E[z^{(i)}|x^{(i)}](x^{(i)} - \mu)^T$$

   取对角线部分：$\Psi_{new} = \text{diag}(\Phi)$。

## 5. Principal Component Analysis (PCA, 主成分分析)

PCA 可以看作是 FA 的一种特殊极限情况，也可以从几何角度推导。

### 5.1 Connection to Factor Analysis (Probabilistic PCA)

如果我们在 FA 中限制噪声协方差 $\Psi = \sigma^2 I$（各向同性噪声），并令 $\sigma^2 \to 0$：

- FA 退化为 PCA。
- $p(z|x)$ 的均值会收敛到 $x$ 在主子空间上的正交投影。
- 这被称为 **Probabilistic PCA (PPCA)**。

### 5.2 Maximum Variance Formulation (最大方差视角)

寻找单位向量 $u$，使得投影方差 $u^T \Sigma u$ 最大。 构造 Lagrangian $\mathcal{L} = u^T \Sigma u - \lambda(u^T u - 1)$，导数为 0 意味着 $u$ 是 $\Sigma$ 的特征向量。

### 5.3 Minimum Reconstruction Error Formulation (最小重构误差视角) - Detailed Proof

这是 PCA 最直观的几何解释。

**Setup**: 有一组标准正交基 $\{u_1, \dots, u_n\}$。 任何样本 $x^{(i)}$ 可以精确表示为：$x^{(i)} = \sum_{j=1}^n (x^{(i)T} u_j) u_j$。 我们想用前 $k$ 个基向量来近似重构 $\hat{x}^{(i)}$：

$$\hat{x}^{(i)} = \sum_{j=1}^k z_j u_j \approx x^{(i)}$$

为了最小化误差，显然系数 $z_j$ 应该是投影长度 $x^{(i)T} u_j$。 对于剩下的 $n-k$ 个维度，我们只能用 0 来近似（或者用均值，假设已中心化）。

**Objective**: 最小化重构误差平方和

$$\sum_{i=1}^m \| x^{(i)} - \hat{x}^{(i)} \|^2 = \sum_{i=1}^m \left\| \sum_{j=k+1}^n (x^{(i)T} u_j) u_j \right\|^2$$

利用勾股定理（或基的正交性）：

$$= \sum_{i=1}^m \sum_{j=k+1}^n (x^{(i)T} u_j)^2$$

**Transformation**: 总能量（总方差，Total Variance）是固定的：$\sum_{i=1}^m \|x^{(i)}\|^2 = \text{const}$。

$$\sum_{i=1}^m \|x^{(i)}\|^2 = \sum_{i=1}^m \sum_{j=1}^n (x^{(i)T} u_j)^2$$

因此，**最小化**尾部 $n-k$ 个维度的投影平方和（丢失的信息），等价于**最大化**前 $k$ 个维度的投影平方和（保留的信息）：

$$\min \sum_{j=k+1}^n u_j^T \Sigma u_j \iff \max \sum_{j=1}^k u_j^T \Sigma u_j$$

**Conclusion**: 为了最大化 $\sum_{j=1}^k u_j^T \Sigma u_j$，我们应该选择 $\Sigma$ 对应的**最大的** $k$ **个特征值**的特征向量作为 $\{u_1, \dots, u_k\}$。

### 5.4 SVD Implementation (奇异值分解)

$$X = U S V^T$$

- $X \in \mathbb{R}^{m \times n}$ (已中心化)。
- $V$ 的列是 $X^T X$ (即协方差矩阵) 的特征向量 $\to$ **主成分方向**。
- $U S$ 的列是 $X X^T$ 的特征向量 $\to$ **主成分得分 (Scores)** (即降维后的坐标)。





## 1. Markov Decision Processes (MDPs)

### 1.1 Formal Definition

MDP 是强化学习的数学框架，由五元组 $(S, A, \{P_{sa}\}, \gamma, R)$ 定义：

- **State Space** $S$: 状态空间。

- **Action Space** $A$: 动作空间。

- **Transition Probabilities** $P_{sa}$: 状态转移分布。

  $$P(s_{t+1} = s' | s_t = s, a_t = a) = P_{sa}(s')$$

  - **Markov Property (马尔可夫性质)**: 下一状态仅取决于当前状态和动作，与历史无关： $P(s_{t+1} | s_t, a_t, s_{t-1}, \dots) = P(s_{t+1} | s_t, a_t)$。

- **Reward Function** $R: S \times A \mapsto \mathbb{R}$: 奖励函数 $R(s, a)$（有时简写为 $R(s)$）。有界：$|R(s, a)| \le R_{max}$。

- **Discount Factor** $\gamma \in [0, 1)$: 折扣因子，确保无限步回报收敛。

### 1.2 The Objective

寻找策略 $\pi: S \to A$，最大化期望累积折扣回报 (Expected Cumulative Discounted Reward)：

$$\max_\pi E \left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \bigg| s_0, \pi \right]$$

## 2. Value Functions & Bellman Equations (核心理论)

我们需要区分 **State Value Function (**$V$**)** 和 **Action-Value Function (**$Q$**)**。

### 2.1 State Value Function $V^\pi$

定义：从状态 $s$ 出发，遵循策略 $\pi$ 的期望回报。

$$V^\pi(s) = E [ G_t | s_t = s, \pi ] = R(s) + \gamma \sum_{s'} P_{s\pi(s)}(s') V^\pi(s')$$

(这是 **Bellman Expectation Equation** for $V$)

### 2.2 Action-Value Function $Q^\pi$ (Important for Q-Learning)

定义：从状态 $s$ 出发，**先执行动作** $a$，之后遵循策略 $\pi$ 的期望回报。

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P_{sa}(s') V^\pi(s')$$

利用 $V^\pi(s') = Q^\pi(s', \pi(s'))$，可得 $Q$ 的递归形式：

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P_{sa}(s') Q^\pi(s', \pi(s'))$$

### 2.3 Optimal Value Functions ($V^*$ and $Q^*$)

最优策略 $\pi^*$ 同时最大化所有状态的 $V$ 和 $Q$。

$$V^*(s) = \max_\pi V^\pi(s), \quad Q^*(s, a) = \max_\pi Q^\pi(s, a)$$

**两者关系**:

$$V^*(s) = \max_{a \in A} Q^*(s, a)$$

$$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P_{sa}(s') V^*(s')$$

### 2.4 Bellman Optimality Equations (贝尔曼最优方程)

将上述关系代入，得到核心非线性方程：

1. **For** $V^*$:

   $$V^*(s) = \max_{a} \left( R(s, a) + \gamma \sum_{s'} P_{sa}(s') V^*(s') \right)$$

2. **For** $Q^*$:

   $$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P_{sa}(s') \max_{a'} Q^*(s', a')$$

## 3. Planning & Control (Known Model)

已知 $P_{sa}$ 和 $R$ 时的求解方法。

### 3.1 Discrete State: Value Iteration (值迭代)

直接迭代 Bellman Optimality Operator $\mathcal{T}$。

$$V_{k+1}(s) := \max_a \left( R(s, a) + \gamma \sum_{s'} P_{sa}(s') V_k(s') \right)$$

- **Convergence**: $\mathcal{T}$ 是 $\gamma$-Contraction Mapping（压缩映射），根据 Banach Fixed Point Theorem，必然收敛到唯一不动点 $V^*$。

  $$\| \mathcal{T}V - \mathcal{T}U \|_\infty \le \gamma \| V - U \|_\infty$$

### 3.2 Discrete State: Policy Iteration (策略迭代)

1. **Policy Evaluation**: 解线性方程组求 $V^{\pi_k}$。
2. **Policy Improvement**: $\pi_{k+1}(s) := \arg\max_a \sum_{s'} P_{sa}(s') V^{\pi_k}(s')$。

- **特点**: 收敛步数少，但单步计算量大（矩阵求逆 $O(|S|^3)$）。

### 3.3 Continuous State: Linear Quadratic Regulator (LQR)

当状态 $s \in \mathbb{R}^n$，动作 $a \in \mathbb{R}^d$，且动力学线性、代价函数二次时：

- **Dynamics**: $s_{t+1} = A s_t + B a_t + w_t$ (其中 $w_t \sim \mathcal{N}(0, \Sigma)$)
- **Cost Function**: $J = \sum_{t=0}^T (s_t^T Q s_t + a_t^T R a_t)$ (注意这里 $R$ 是正定矩阵，非 Reward)

**Solution (Riccati Equation)**: 假设 $V_t(s_t) = s_t^T P_t s_t$（二次型价值函数）。 通过倒推法 (Dynamic Programming)，可推导出矩阵 $P_t$ 的迭代公式（即 **Discrete-time Riccati Equation**）：

1. 初始化 $P_T = Q$。

2. 逆序迭代 $t = T-1, \dots, 0$:

   $$P_t = Q + A^T P_{t+1} A - A^T P_{t+1} B (R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A$$

3. **最优策略**是线性的：

   $$a_t^* = - \underbrace{(R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A}_{K_t} s_t = - K_t s_t$$

## 4. Reinforcement Learning (Unknown Model)

未知 $P_{sa}$ 和 $R$，需从交互数据中学习。

### 4.1 Model-Based RL (先学模型，再规划)

1. **Learn Model**: 收集数据 $(s, a, s', r)$，训练监督学习模型：
   - $s' \approx \hat{f}(s, a)$ (或估计概率表 $\hat{P}_{sa}$)
   - $r \approx \hat{R}(s, a)$
2. **Plan**: 在学习到的模型上运行 VI 或 LQR。

- **优点**: 样本效率高 (Sample Efficient)。
- **缺点**: 模型偏差 (Model Bias) 可能导致学出错误策略。

### 4.2 Model-Free RL: Q-Learning (直接学价值)

不估计 $P_{sa}$，直接逼近 $Q^*$。 基于 Bellman Optimality Equation 的随机近似 (Stochastic Approximation)。

**Update Rule**: 观测到转移 $(s, a, r, s')$ 后：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ \underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{Target}} - Q(s, a) \right]$$

- $\alpha$: 学习率。
- **Off-policy**: 目标值使用 $\max_{a'}$ (贪婪策略)，而数据可能来自 $\epsilon$-greedy (探索策略)。

## 5. Continuous State Spaces (Generalization)

当状态无限时，无法用表格存 $V$ 或 $Q$，需使用 **Function Approximation (函数近似)**。

$$V^*(s) \approx \theta^T \phi(s) \quad \text{or} \quad Q^*(s, a) \approx \theta^T \phi(s, a)$$

### 5.1 Fitted Value Iteration (FVI)

针对连续状态的模型化方法（假设有 Simulator）。

1. 采样有限状态集 $S_{sample} = \{s^{(1)}, \dots, s^{(m)}\}$。
2. **Supervised Step**: 计算目标值 $y^{(i)} = \max_a (R(s^{(i)}) + \gamma E_{s' \sim P(\cdot|s^{(i)}, a)}[V_\theta(s')])$。 更新 $\theta$ 以最小化回归损失 $\sum_i (y^{(i)} - \theta^T \phi(s^{(i)}))^2$。
3. 重复直至收敛。

### 5.2 Fitted Q-Iteration (FQI) & DQN

如果是 Model-Free 且连续状态，则拟合 $Q$ 函数。 收集数据集 $D = \{(s_i, a_i, r_i, s'_i)\}$。

1. 计算目标: $y_i = r_i + \gamma \max_{a'} Q_\theta(s'_i, a')$。
2. 回归更新: $\theta \leftarrow \arg\min_\theta \sum (y_i - Q_\theta(s_i, a_i))^2$。

- **Connection to Deep RL**: 如果使用深度神经网络代替线性函数 $\theta^T \phi(s)$，并引入 Replay Buffer 和 Target Network，这就是 **DQN (Deep Q-Network)**。

## 6. Summary of Key Algorithms

| Setting      | Model Known? | Discrete State? | Algorithm                        |
| ------------ | ------------ | --------------- | -------------------------------- |
| **Planning** | Yes          | Yes             | Value / Policy Iteration         |
| **Control**  | Yes          | No (Continuous) | LQR (Linear) / iLQR (Non-linear) |
| **RL**       | No           | Yes             | Q-Learning / SARSA               |
| **RL**       | No           | No (Continuous) | Fitted Q-Iteration / DQN         |

## 1. Convex Analysis Foundations (凸分析基础)

### 1.1 Convex Sets & Separating Hyperplanes (凸集与分离定理)

- **Definition**: 集合 $C$ 是凸的，若 $\forall x, y \in C, \theta \in [0, 1] \implies \theta x + (1-\theta)y \in C$。
- **Key Operations Preserving Convexity**:
  - **Intersection (交集)**: $\cap_{i \in I} C_i$ 仍为凸集。
  - **Affine Image/Pre-image (仿射变换)**: $f(x) = Ax+b$。
  - **Perspective Function (透视函数)**: $P(z, t) = z/t$ ($t > 0$)。若 $C$ 为凸，其透视集也为凸。
- **Separating Hyperplane Theorem (分离超平面定理)**:
  - 若 $C, D$ 为不相交凸集，存在 $a \neq 0, b$ 使得 $a^T x \le b \le a^T y$ 对所有 $x \in C, y \in D$ 成立。
  - *Strict Separation*: 需要集合闭且紧 (Closed & Compact)。
  - **核心意义**: 它是凸优化几何直觉的核心，也是证明 Strong Duality 的基础。

### 1.2 Convex Functions: Equivalent Definitions (凸函数的多重刻画)

对于定义域为凸的函数 $f: \text{dom}(f) \to \mathbb{R}$：

1. **Zeroth Order (Jensen's Inequality)**:

   $$f(\theta x + (1-\theta)y) \le \theta f(x) + (1-\theta)f(y)$$

2. **First Order (Global Underestimator)**:

   $$f(y) \ge f(x) + \langle \nabla f(x), y-x \rangle$$

   - *Intuition*: 一阶泰勒展开永远位于函数图像下方。这是分析 Gradient Descent 下降性质的关键。

3. **Second Order (Hessian)**:

   $$\nabla^2 f(x) \succeq 0 \quad (\text{Positive Semidefinite})$$

4. **Epigraph (上方图)**: $\text{epi}(f) = \{(x, t) \mid f(x) \le t\}$ 是凸集。

## 2. Optimality Conditions: From Geometry to Algebra (最优性条件：从几何到代数)

**核心逻辑**: Geometric Optimality (Variational Inequality) $\to$ Normal Cone $\to$ KKT Conditions $\to$ Duality.

### 2.1 Geometric Optimality (几何最优性)

考虑约束优化 $\min_{x \in C} f(x)$。

- **Variational Inequality (变分不等式)**: $x^*$ 是最优解当且仅当梯度方向与所有可行方向的夹角为钝角（即梯度指向可行域“外部”）：

  $$\langle \nabla f(x^*), y - x^* \rangle \ge 0, \quad \forall y \in C$$

### 2.2 Normal Cone & Deriving Lagrangian (法锥与拉格朗日导出)

- **Normal Cone (法锥) 定义**:

  $$N_C(x) = \{ g \mid \langle g, y-x \rangle \le 0, \forall y \in C \}$$

  上述变分不等式等价于：

  $$-\nabla f(x^*) \in N_C(x^*)$$

  *(直觉：负梯度必须落在法锥内，即由于约束限制，无法继续沿负梯度方向下降)*

- **Deriving KKT (导出 KKT)**: 设 $C = \{x \mid h_i(x) \le 0\}$。若满足 **Slater's Condition**，法锥可由约束梯度的锥组合生成：

  $$N_C(x^*) = \left\{ \sum \lambda_i \nabla h_i(x^*) \mid \lambda_i \ge 0, \lambda_i h_i(x^*) = 0 \right\}$$

  代入最优性条件 $-\nabla f(x^*) \in N_C(x^*)$，即存在 $\lambda \ge 0$ 使得：

  $$-\nabla f(x^*) = \sum \lambda_i \nabla h_i(x^*) \implies \nabla f(x^*) + \sum \lambda_i \nabla h_i(x^*) = 0$$

  这正是 **Lagrangian** $L(x, \lambda) = f(x) + \sum \lambda_i h_i(x)$ 的平稳性条件 (Stationarity)。

### 2.3 Duality (对偶理论)

- **Lagrangian**: $L(x, \lambda, \nu) = f_0(x) + \sum \lambda_i f_i(x) + \sum \nu_i h_i(x)$。
- **Dual Function**: $g(\lambda, \nu) = \inf_x L(x, \lambda, \nu)$。
  - *性质*: $g$ 永远是凹函数 (Concave)，无论原问题凸否。
- **Weak Duality**: $d^* \le p^*$。
- **Strong Duality**: $d^* = p^*$。
  - *条件*: 对于凸问题，若存在严格可行点 (Slater's Condition)，则强对偶成立。

## 3. Regularity Conditions: Analysis Tools (正则性条件：收敛分析基石)

分析收敛速率 (Convergence Rates) 的核心不等式。

### 3.1 $\beta$-Smoothness (平滑性 / Lipschitz Gradient)

**Definition**: $\|\nabla f(x) - \nabla f(y)\| \le \beta \|x - y\|$。 意味着曲率 (Hessian) 有上界 $\beta I$。

**Quadratic Upper Bound (Descent Lemma)**:

$$f(y) \le f(x) + \langle \nabla f(x), y-x \rangle + \frac{\beta}{2} \|y-x\|^2$$

- **Derivation**: 利用 $f(y) - f(x) = \int_0^1 \langle \nabla f(x + \tau(y-x)), y-x \rangle d\tau$ 并放缩。
- **Significance**: 只要步长 $\eta \le 1/\beta$，GD 保证下降。

### 3.2 $\alpha$-Strong Convexity (强凸性)

**Definition**: $f(x) - \frac{\alpha}{2}\|x\|^2$ 是凸函数 ($\alpha > 0$)。 意味着曲率有下界 $\alpha I$。

**Quadratic Lower Bound**:

$$f(y) \ge f(x) + \langle \nabla f(x), y-x \rangle + \frac{\alpha}{2} \|y-x\|^2$$

**PL Condition (Polyak-Lojasiewicz)**:

$$\frac{1}{2} \|\nabla f(x)\|^2 \ge \alpha (f(x) - f^*)$$

- *Note*: PL 条件比强凸弱，但也足以证明线性收敛 (Linear Convergence)。

### 3.3 Condition Number (条件数)

$$\kappa = \frac{\beta}{\alpha} \ge 1$$

- **Geometry**: 决定了等高线的“离心率”。High $\kappa$ (峡谷状) $\implies$ 梯度方向与最优解方向偏差大 $\implies$ GD 震荡 (Zig-zag)。

## 4. First-Order Methods (一阶方法)

### 4.1 Gradient Descent (GD)

Update: $x_{t+1} = x_t - \eta \nabla f(x_t)$.

| Function Class           | Convergence Rate                   | Iterations Needed            |
| ------------------------ | ---------------------------------- | ---------------------------- |
| Convex, $\beta$-smooth   | $O(1/T)$ (Sublinear)               | $O(\beta/\epsilon)$          |
| $\alpha$-Strongly Convex | $O((1 - \alpha/\beta)^T)$ (Linear) | $O(\kappa \log(1/\epsilon))$ |

**Proof Sketch (Linear Rate)**:

1. 由 Descent Lemma: $f(x_{t+1}) \le f(x_t) - \frac{1}{2\beta}\|\nabla f(x_t)\|^2$.
2. 由 PL Condition: $\|\nabla f(x_t)\|^2 \ge 2\alpha(f(x_t) - f^*)$.
3. 结合: $f(x_{t+1}) - f^* \le (1 - \frac{\alpha}{\beta})(f(x_t) - f^*)$.

### 4.2 Nesterov Acceleration (加速梯度法)

- **Motivation**: GD 的 $O(1/T)$ 不是最优的，一阶方法理论下界是 $O(1/T^2)$。

- **Scheme (Momentum)**:

  $$\begin{aligned} y_{t+1} &= x_t - \frac{1}{\beta} \nabla f(x_t) \\ x_{t+1} &= y_{t+1} + \mu_t (y_{t+1} - y_t) \end{aligned}$$

- **Rate**:

  - Convex: $O(1/T^2)$ (Optimal).
  - Strongly Convex: $O((1 - \sqrt{1/\kappa})^T)$。依赖 $\sqrt{\kappa}$ 而非 $\kappa$，在病态问题上提升巨大。

### 4.3 Subgradient Method (非光滑优化)

- **Subgradient**: $g \in \partial f(x) \iff f(y) \ge f(x) + \langle g, y-x \rangle$。
- **Update**: $x_{t+1} = x_t - \eta_t g_t$.
- **Rate**: $O(1/\sqrt{T})$。
  - *Note*: 必须使用递减步长，且因为不是 Descent Method，需记录 $f_{best}$。

## 5. Proximal Methods (近端梯度法)

针对复合优化问题 $\min F(x) = f(x) + h(x)$，其中 $f$ 光滑，$h$ 非光滑但简单（如 L1 Norm）。

### 5.1 Proximal Operator

$$\text{prox}_h(v) = \arg\min_x \left( h(x) + \frac{1}{2}\|x - v\|^2 \right)$$

- **Example**: 若 $h(x) = \lambda \|x\|_1$ (Lasso)，Prox 即为 **Soft Thresholding**。

### 5.2 ISTA (Proximal Gradient Descent)

Update:

$$x_{t+1} = \text{prox}_{\eta h} (x_t - \eta \nabla f(x_t))$$

- **Interpretation**: 先沿 $f$ 的梯度走一步，再通过 Prox 算子处理 $h$（隐式梯度步）。
- **Rate**: 恢复了光滑 GD 的 $O(1/T)$ 速率，远快于次梯度的 $O(1/\sqrt{T})$。

## 6. Second-Order Methods (二阶方法)

### 6.1 Newton's Method (牛顿法)

Update: $x_{t+1} = x_t - [\nabla^2 f(x_t)]^{-1} \nabla f(x_t)$.

- **Affine Invariance (仿射不变性)**: 算法路径不受坐标系线性变换影响（不像 GD）。
- **Quadratic Convergence (二次收敛)**: 近似解附近误差平方级衰减 $\|x_{t+1} - x^*\| \le C \|x_t - x^*\|^2$。

### 6.2 Self-Concordance (自协调性)

为了不依赖未知常数 ($\alpha, \beta$) 证明牛顿法的全局收敛性。

- **Definition**: $|f'''(x)[h,h,h]| \le 2 (f''(x)[h,h])^{3/2}$。
  - *直觉*: 三阶导数（曲率变化）被二阶导数（曲率）控制。
- **Standard Barrier**: Log-barrier $F(x) = -\sum \log(b_i - a_i^T x)$ 是自协调的。

## 7. Interior Point Methods (内点法)

解决约束问题 $\min c^T x \text{ s.t. } Ax \le b$。

### 7.1 Log-Barrier Method

将约束转化为目标函数中的势垒：

$$\min_x t c^T x + \phi(x), \quad \phi(x) = -\sum \log(b_i - a_i^T x)$$

### 7.2 Path Following (路径追踪)

- **Central Path**: 不同 $t$ 下的最优解轨迹 $x^*(t)$。
- **Algorithm**:
  1. 用 Newton Method 求当前 $t$ 的 $x^*(t)$。
  2. 增加 $t \leftarrow \mu t$。
  3. 利用上一步解作为 Warm Start。
- **Complexity**: $O(\sqrt{m} \log(1/\epsilon))$ 次牛顿迭代。
  - $\sqrt{m}$ 来源于 Barrier 的自协调参数。

## Cheat Sheet: Important Inequalities for Proofs

1. **Cauchy-Schwarz**: $|\langle x, y \rangle| \le \|x\| \|y\|$.
2. **Young's Inequality**: $xy \le \frac{x^2}{2} + \frac{y^2}{2}$.
3. **Convexity (First Order)**: $f(y) \ge f(x) + \nabla f(x)^T(y-x)$.
4. **Smoothness (Quadratic Upper)**: $f(y) \le f(x) + \nabla f(x)^T(y-x) + \frac{\beta}{2}\|y-x\|^2$.
5. **Strong Convexity (Quadratic Lower)**: $f(y) \ge f(x) + \nabla f(x)^T(y-x) + \frac{\alpha}{2}\|y-x\|^2$.
6. **Co-coercivity**: $\langle \nabla f(x) - \nabla f(y), x - y \rangle \ge \frac{1}{\beta} \|\nabla f(x) - \nabla f(y)\|^2$.



## 1. Constrained Optimization (受限优化)

### 1.1 标准形式 (Standard Form)

考虑一般化的凸优化问题 (Convex Optimization Problem)：

$$\begin{aligned} \min_{x \in \mathcal{D}} \quad & f_0(x) \\ \text{s.t.} \quad & f_i(x) \le 0, \quad i = 1, \dots, m \\ & h_j(x) = 0, \quad j = 1, \dots, p \end{aligned}$$

其中 $f_0, f_i$ 是凸函数，$h_j$ 是仿射函数 ($h_j(x) = a_j^T x - b_j$)。 定义域 $\mathcal{D} = (\cap \textbf{dom} f_i) \cap (\cap \textbf{dom} h_j)$。

### 1.2 KKT Conditions (Karush-Kuhn-Tucker)

对于凸优化问题，KKT 条件是**强对偶 (Strong Duality)** 成立时的最优解的充要条件（前提是满足 Slater 条件）。

设 $x^*$ 为原问题最优解，$(\lambda^*, \nu^*)$ 为对偶问题最优解。

1. **Stationarity (平稳性)**: 拉格朗日函数的梯度为 0。

   $$\nabla f_0(x^*) + \sum_{i=1}^m \lambda_i^* \nabla f_i(x^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(x^*) = 0$$

   *直观理解*: 目标函数的负梯度 $-\nabla f_0$ 必须处于约束梯度的锥组合 (Conic Combination) 中。

2. **Primal Feasibility (原问题可行性)**:

   $$f_i(x^*) \le 0, \quad h_j(x^*) = 0$$

3. **Dual Feasibility (对偶可行性)**:

   $$\lambda_i^* \ge 0$$

4. **Complementary Slackness (互补松弛性)**:

   $$\lambda_i^* f_i(x^*) = 0, \quad \forall i$$

   这意味着若 $\lambda_i^* > 0$，则 $f_i(x^*) = 0$（约束处于激活状态/紧绷）。

## 2. Duality Theory (对偶理论)

对偶理论提供了原问题的下界，并揭示了问题的深层结构。

### 2.1 Lagrange Duality (拉格朗日对偶)

**Lagrangian**:

$$L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x)$$

**Dual Function**:

$$g(\lambda, \nu) = \inf_{x \in \mathcal{D}} L(x, \lambda, \nu)$$

- **性质**: $g(\lambda, \nu)$ 是凹函数 (Concave)，因为它是关于 $(\lambda, \nu)$ 的仿射函数的逐点下确界。这无论原问题是否为凸都成立。
- **Weak Duality**: $g(\lambda, \nu) \le p^*$ 对任意可行 $\lambda \ge 0$ 成立。
- **Strong Duality**: $p^* = d^*$。
  - **Slater's Condition**: 对于凸问题，如果存在 $x \in \textbf{relint}(\mathcal{D})$ 使得 $f_i(x) < 0$ (严格可行)，则强对偶成立。

### 2.2 Fenchel Duality (Fenchel 对偶)

Fenchel 对偶是 Lagrange 对偶在特定形式问题上的推广，涉及共轭函数。

**Conjugate Function (共轭函数)**: 对于函数 $f(x)$，其共轭函数定义为：

$$f^*(y) = \sup_{x \in \textbf{dom} f} (y^T x - f(x))$$

- $f^*$ 总是凸函数（它是仿射函数的逐点上确界）。
- **Fenchel-Young Inequality**: $f(x) + f^*(y) \ge x^T y$。

**Fenchel Duality Problem**: 原问题形式：

$$\min_x f(x) + g(Ax)$$

其中 $f, g$ 为凸函数。 **Dual Problem**:

$$\max_\nu -f^*(-A^T \nu) - g^*(\nu)$$

- **Derivation Hint**: 引入变量 $y=Ax$，原问题变为 $\min f(x) + g(y)$ s.t. $y=Ax$。 写出 Lagrangian $L(x, y, \nu) = f(x) + g(y) + \nu^T(Ax - y)$。 求 $\inf_{x,y} L$ 即可推导出上述对偶形式。

## 3. Linear Programming (LP, 线性规划)

线性规划是凸优化的一个特例，也是学习对偶理论的最佳案例。

### 3.1 Standard Form LP

$$\begin{aligned} \text{(Primal)} \quad \min_x \quad & c^T x \\ \text{s.t.} \quad & Ax = b \\ & x \ge 0 \end{aligned}$$

### 3.2 Dual LP Derivation

Lagrangian:

$$L(x, \nu, \lambda) = c^T x + \nu^T (Ax - b) - \lambda^T x = (c + A^T \nu - \lambda)^T x - b^T \nu$$

Dual function $g(\nu, \lambda) = \inf_x L$。 为了使下确界有限（不是 $-\infty$），必须有 $x$ 的系数为 0：

$$c + A^T \nu - \lambda = 0 \implies A^T \nu + \lambda = c$$

由于 $\lambda \ge 0$，这等价于 $A^T \nu \le c$。 代回目标函数得到对偶问题：

$$\begin{aligned} \text{(Dual)} \quad \max_\nu \quad & -b^T \nu \\ \text{s.t.} \quad & A^T \nu \le c \end{aligned}$$

*(注：有时对偶变量* $\nu$ *符号取反，目标变为* $\max b^T y$*，取决于* $Ax=b$ *的 Lagrange 乘子定义符号)*

### 3.3 Strong Duality in LP

对于 LP，只要原问题有可行解，强对偶即成立（除了极少数病态情况）。 **互补松弛性** 在单纯形法 (Simplex) 和内点法中至关重要：

$$x_j (c_j - a_j^T \nu) = 0, \quad \forall j$$

即如果 $x_j > 0$，则对偶约束 $a_j^T \nu \le c_j$ 必须取等号。

## 4. Stochastic Gradient Descent (SGD, 随机梯度下降)

SGD 是大规模机器学习（如深度学习）的核心优化算法。

### 4.1 Motivation & Algorithm

假设目标函数是大量样本损失的平均：

$$J(\theta) = \frac{1}{m} \sum_{i=1}^m J_i(\theta)$$

如果 $m$ 很大，计算全梯度 $\nabla J(\theta)$ 极其昂贵。

**SGD Update Rule**: 每次迭代随机采样一个样本 $i$（或一个小批量 Mini-batch），仅利用该样本更新参数：

$$\theta := \theta - \eta \nabla J_i(\theta)$$

- **Unbiased Estimator**: $E_i [\nabla J_i(\theta)] = \nabla J(\theta)$。即 SGD 的梯度是真实梯度的无偏估计。

### 4.2 Convergence & Issues

- **Noise**: 由于每次只用一个样本，更新方向会有剧烈震荡 (High Variance)。目标函数值不会单调下降。
- **Step Size** $\eta$: 为了保证收敛，步长必须随时间衰减（如 $\eta_k \propto 1/k$ 或 $\eta_k \propto 1/\sqrt{k}$）。这满足 Robbins-Monro 条件： $\sum \eta_k = \infty, \quad \sum \eta_k^2 < \infty$。
- **Comparison with Batch GD**:
  - **Batch GD**: 每步计算量 $O(m)$，收敛平稳，线性收敛速率。
  - **SGD**: 每步计算量 $O(1)$，收敛震荡，次线性收敛速率，但在处理大数据时，通常在最初几个 epoch 就能取得极好的效果。

### 4.3 Mini-batch SGD

折中方案：每次采样 $b$ 个样本（Batch size $b \approx 32 \sim 512$）。

- 减少了梯度的方差，使收敛更稳定。
- 可以利用矩阵运算并行化加速。





