import torch


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        print(f"🤖 AdamW 机器人已启动！学习率={lr}, 权重衰减={weight_decay}")

    def step(self):
        self.t += 1
        print(f"--- 第 {self.t} 次更新开始 ---")

        beta1, beta2 = self.betas
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data
            print(f"  参数 {i}: 梯度大小 = {grad.abs().mean().item():.6f}")  # 打印平均梯度，看看训练状态

            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad

            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)

            denominator = v_hat.sqrt() + self.eps

            update = self.lr * m_hat / denominator

            param.data = param.data - update

            # 📊 打印一些信息，观察更新过程
            print(f"    更新量大小: {update.abs().mean().item():.6f}")
            print(f"    参数更新后大小: {param.data.abs().mean().item():.6f}")

        print(f"--- 第 {self.t} 次更新结束 ---\n")

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
        print("梯度已清零")


x = torch.tensor([10.0], requires_grad=True)
print(f"初始值：x = {x.item()}")

optimizer = AdamW(params=[x], lr=0.01, weight_decay=0.0)

for step in range(20):
    loss = (x - 3) ** 2

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if step % 5 == 0:
        print(f"步骤 {step}: x = {x.item():.4f}, 损失 = {loss.item():.4f}")

print(f"训练结束！最终x ~~ {x.item():.4f}")
