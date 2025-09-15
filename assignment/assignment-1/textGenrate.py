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
    model = GPTModel(GPT_CONFIG_MAC)

    # 加载保存的权重
    checkpoint = torch.load('best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # 初始化分词器
    tokenizer = tiktoken.encoding_for_model("gpt2")

    # 生成文本
    prompts = [
        "Once upon a time",
        "The little cat",
        "In a magical forest"
    ]

    for prompt in prompts:
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=30,
            temperature=0.8,
            device=device
        )
        print(f"生成: {generated_text}")
        print("-" * 50)