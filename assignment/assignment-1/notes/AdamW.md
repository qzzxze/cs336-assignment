---

### **AdamW 算法简介**

AdamW 是 Adam 的改进版，主要区别在于 **权重衰减（weight decay）方式**。

* Adam 在实现 L2 正则化时，会把正则化项合并到梯度中，这在实践中会导致正则化不准确。
* AdamW 将 **权重衰减直接应用到参数更新** 上，而不是通过梯度。

---

### **符号**

* $\theta_t$：第 t 步的参数
* $g_t$：梯度 $\nabla_\theta L(\theta_t)$
* $m_t$：一阶矩（动量）
* $v_t$：二阶矩（平方梯度的指数平均）
* $\beta_1, \beta_2$：动量衰减系数
* $\epsilon$：平滑项，防止除零
* $\eta$：学习率
* $\lambda$：权重衰减系数

---

### **AdamW 更新步骤**

**1️⃣ 初始化**

$$
m_0 = 0, \quad v_0 = 0, \quad t = 0
$$

---

**2️⃣ 对每个时间步 t 执行：**

1. **计算梯度**

$$
g_t = \nabla_\theta L(\theta_{t-1})
$$

2. **更新一阶矩估计（动量）**

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

3. **更新二阶矩估计（平方梯度）**

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

4. **偏差修正（bias correction）**

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

5. **参数更新（AdamW 核心）**

$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_{t-1}
$$

✅ 这里注意，最后的 $- \eta \lambda \theta_{t-1}$ 是 **权重衰减直接作用于参数**，而不是通过梯度。

---

### **总结关键点**

1. AdamW 保留 Adam 的动量与自适应学习率机制。
2. **权重衰减与梯度解耦**，更加符合正则化意图。
3. 在实践中常比 Adam 更稳定，尤其在训练大模型时效果更好。

---

[附：从SGD到AdamW，优化器详解](https://zhuanlan.zhihu.com/p/1928857996655588384)
