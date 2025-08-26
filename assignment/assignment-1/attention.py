import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    多头因果注意力机制（带 Dropout）
    就像一场“只准看过去”的舞会，每个 token 只能关注它之前的伙伴。
    """

    def __init__(self, d_in, d_out, context_length, dropout, head_nums, qkv_bias=False):
        super().__init__()
        self.d_in = d_in          # 输入维度：每个 token 的向量长度
        self.d_out = d_out        # 输出维度：每个头输出的总维度
        self.head_nums = head_nums  # 多少个“注意力小分队”（head）
        self.context_length = context_length  # 最大上下文长度（用于掩码）
        self.head_dim = d_out // head_nums   # 每个小分队的“智力容量”

        # 每个小分队都有自己的“观察方式”（线性投影）
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # Query：我想看谁？
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)  # Key：我有什么可被看到的？
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # Value：我的真实信息是什么？

        # 预先制作一张“禁止偷看未来”的黑名单（因果掩码）
        # 上三角为 1，表示“不能看”
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask)  # 永久保存，不参与梯度更新

        # 最后的“总结报告”投影层：把所有小分队的信息汇总成最终输出
        self.proj = nn.Linear(d_out, d_out)

        # 注意力权重随机“失忆”机制（Dropout），防止过度关注某个 token
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播：让每个 token 学会“从过去中提取重点”
        x: (batch_size, num_tokens, d_in)
        """
        batch_size, token_nums, d_in = x.shape

        # Step 1: 每个 token 都生成自己的“提问”（Query）、“名片”（Key）和“真实内容”（Value）
        queries = self.W_query(x)  # 我想关注什么？
        keys    = self.W_key(x)    # 我有什么可以被关注的？
        values  = self.W_value(x)  # 我的真实信息是什么？

        # Step 2: 把大团队拆分成多个“注意力小分队”（多头机制）
        # 从 (b, n, d_out) → (b, n, num_heads, head_dim)
        queries = queries.view(batch_size, token_nums, self.head_nums, self.head_dim)
        keys    = keys.view(batch_size, token_nums, self.head_nums, self.head_dim)
        values  = values.view(batch_size, token_nums, self.head_nums, self.head_dim)

        # Step 3: 调整顺序，让“小分队”成为主维度，方便并行计算
        # (b, n, h, d) → (b, h, n, d)
        queries = queries.transpose(1, 2)
        keys    = keys.transpose(1, 2)
        values  = values.transpose(1, 2)

        # Step 4: 计算“谁该关注谁”的热度图（注意力分数）
        # 每个 query 和所有 key 打分：Q @ K^T
        attention_scores = queries @ keys.transpose(2, 3)  # (b, h, n, n)

        # Step 5: 拿出“黑名单”，禁止任何 token 偷看未来！
        # 只保留左下角的合法区域（因果掩码）
        mask_bool = self.mask.bool()[:token_nums, :token_nums]  # 动态截取当前长度
        attention_scores.masked_fill_(mask_bool, -torch.inf)    # 偷看者分数归 -∞

        # Step 6: 把热度分转成“注意力概率”（softmax），并缩放防止爆炸
        # 除以 sqrt(d_k) 是为了让 softmax 更稳定
        attention_weight = torch.softmax(
            attention_scores / (self.head_dim ** 0.5), dim=-1
        )

        # Step 7: 随机让一些注意力连接“失忆”（Dropout），防止过拟合
        attention_weight = self.dropout(attention_weight)

        # Step 8: 按注意力权重，对所有 value 进行“加权求和” → 得到每个 token 的新表示
        # 就像根据重要性重新组合信息
        context_vec = (attention_weight @ values)  # (b, h, n, d)
        context_vec = context_vec.transpose(1, 2)  # 换回 (b, n, h, d)

        # Step 9: 把所有小分队的信息“拼接”起来
        # (b, n, h, d) → (b, n, d_out)
        context_vec = context_vec.contiguous().view(batch_size, token_nums, self.d_out)

        # Step 10: 最后的“总结报告”：用 proj 层整合信息，输出最终结果
        return self.proj(context_vec)


# 🎉 使用示例
if __name__ == "__main__":
    mha = MultiHeadAttention(
        d_in=768,           # 输入维度（如 BERT 的 hidden size）
        d_out=768,          # 输出维度
        context_length=1024, # 最大上下文长度
        dropout=0.1,         # 注意力 dropout 概率
        head_nums=12,        # 12 个注意力头（如 BERT-base）
        qkv_bias=False
    )

    # 模拟一批 2 个样本，每个有 5 个 token
    x = torch.randn(2, 5, 768)
    output = mha(x)
    print(output.shape)  # 应该是 torch.Size([2, 5, 768])