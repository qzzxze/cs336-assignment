import tiktoken
import torch
import tqdm
from GPTModel import GPTModel
from DatasetLoader import DatasetLoader
from Loss import CrossEntropyLoss
from torch.utils.data import DataLoader

GPT_CONFIG_MAC = {
    "vocab_size": 50257,
    "context_length": 128,  # 减半：256→128
    "emb_dim": 32,  # 大幅减小：64→32
    "n_heads": 2,  # 减半：4→2
    "n_layers": 2,  # 减半：4→2
    "drop_rate": 0.0,
    "qkv_bias": False
}

GPT_CONFIG_1M = {
    "vocab_size": 50257,  # TinyStories使用更小的词汇表，约8k tokens
    "context_length": 256,  # TinyStories故事较短，256长度足够
    "emb_dim": 64,  # 更小的嵌入维度
    "n_heads": 4,  # 较少的注意力头
    "n_layers": 4,  # 较少的层数
    "drop_rate": 0.0,  # TinyStories通常不需要dropout
    "qkv_bias": False
}


class GPTTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.step = 0
        self.best_loss = float('inf')

    def train_step(self, input_ids, target_ids):
        """单步训练"""
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        logits = self.model(input_ids)
        loss = self.criterion(logits, target_ids)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def validate(self):
        """验证"""
        self.model.eval()
        count = 0
        total_loss = 0

        with torch.no_grad():
            for input_ids, target_ids in self.val_loader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                logits = self.model(input_ids)
                loss = self.criterion(logits, target_ids)

                total_loss += loss.item()
                count += 1

        self.model.train()
        return total_loss / count

    def save_model(self, filename):
        """保存模型"""
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'step': self.step
            },
            filename
        )
        print(f"模型已保存: {filename}")

    def train(self, max_steps=10000, eval_every=500):
        """训练主循环"""
        print(f"开始训练，目标步数: {max_steps}")

        pbar = tqdm.tqdm(total=max_steps, desc="训练进度")

        while self.step < max_steps:
            for input_ids, target_ids in self.train_loader:
                loss = self.train_step(input_ids, target_ids)
                self.step += 1

                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss:.4f}', 'step': self.step})

                if self.step % eval_every == 0:
                    val_loss = self.validate()
                    print(f"步数 {self.step}: 验证损失 {val_loss:.4f}")

                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_model("best_model.pt")
                if self.step >= max_steps:
                    break
            if self.step >= max_steps:
                break
        pbar.close()
        print("训练完成!")


if __name__ == '__main__':
    model = GPTModel(GPT_CONFIG_MAC)

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = DatasetLoader(
        file_path="/Users/zhangdebiao/pycharmProjects/cs336-assignment/data/TinyStoriesV2-GPT4-train.txt",
        tokenizer=tokenizer,
        max_length=128
    )

    val_dataset = DatasetLoader(
        file_path="/Users/zhangdebiao/pycharmProjects/cs336-assignment/data/TinyStoriesV2-GPT4-valid.txt",
        tokenizer=tokenizer,
        max_length=128
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01
    )
    criterion = CrossEntropyLoss()
    trainer = GPTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion
    )

    trainer.train(max_steps=1000, eval_every=100)
