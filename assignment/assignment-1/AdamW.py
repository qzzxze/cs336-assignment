import torch


class AdamW:
    def __init__(self, params, lr=0.01, eps=1e-8, weight_decay=1e-2, betas=(0.9, 0.999)):
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.betas = betas

        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        print(f"--- 第 {self.t} 次更新开始 ---")
        beta1, beta2 = self.betas

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data
            print(f"  参数 {i}: 梯度大小 = {grad.abs().mean().item():.6f}")  # 打印平均梯度，看看训练状态

            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * grad ** 2

            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)

            update = self.lr * (m_hat / (v_hat.sqrt_() + self.eps))
            param.data = param.data - update - (self.lr * self.weight_decay * param.data)

            # 📊 打印一些信息，观察更新过程
            print(f"    更新量大小: {update.abs().mean().item():.6f}")
            print(f"    参数更新后大小: {param.data.abs().mean().item():.6f}")

        print(f"--- 第 {self.t} 次更新结束 ---\n")

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()



x = torch.tensor([10.0], requires_grad=True)

print(f"初始值：x = {x.item()}")

optimizer = AdamW(params=[x], lr=0.1)

for epoch in range(1000):

    loss = (x - 3) ** 2

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 5 == 0:
        print(f"步骤 {epoch}: x = {x.item():.4f}, 损失 = {loss.item():.4f}")

    print(f"训练结束！最终x ~~ {x.item():.4f}")
