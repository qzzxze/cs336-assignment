import torch
import tiktoken
from GPTModel import GPTModel
import torch.nn.functional as F

# 加载模型配置（必须和训练时一致）
GPT_CONFIG_MAC = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 32,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False
}

GPT_CONFIG_A100 = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8, device='cpu'):
    """生成文本函数"""
    model.eval()

    # 编码输入
    input_ids = tokenizer.encode(prompt)
    generated = torch.tensor(input_ids).unsqueeze(0).to(device)

    print(f"输入: '{prompt}'")

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(generated)
            next_token_logits = logits[0, -1, :] / temperature

            # 采样下一个token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

    # 解码生成的文本
    return tokenizer.decode(generated[0].cpu().tolist())


if __name__ == '__main__':
    # 加载训练好的模型
    device = torch.device('cpu')
    model = GPTModel(GPT_CONFIG_A100)

    # 加载保存的权重
    # 加载保存的权重
    checkpoint = torch.load('best_model.pt', map_location=device)

    # <<< 新增：处理 DataParallel 保存的权重 >>>
    state_dict = checkpoint['model_state_dict']

    # 如果权重名以 'module.' 开头，就去掉
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉 'module.' 这7个字符
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)

    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 生成文本
    prompts = [
        "Once upon a time"
    ]

    for prompt in prompts:
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=128,
            temperature=0.8,
            device=device
        )
        print(f"生成: {generated_text}")
        print("-" * 50)