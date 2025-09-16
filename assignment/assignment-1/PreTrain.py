import os
import tiktoken
import torch
import tqdm
from GPTModel import GPTModel
from DatasetLoader import DatasetLoader
from Loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import math
from torch.optim.lr_scheduler import LambdaLR

GPT_CONFIG_MAC = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 32,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
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


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1
):
    """
    创建一个带 warmup 的余弦退火学习率调度器
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class GPTTrainer:
    def __init__(
        self, model, train_loader, val_loader, optimizer, criterion, scheduler=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler  # 新增：支持调度器

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 数据并行
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行数据并行")
            self.model = torch.nn.DataParallel(self.model)

        self.step = 0
        self.best_loss = float("inf")

    def train_step(self, input_ids, target_ids):
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        logits = self.model(input_ids)
        loss = self.criterion(logits, target_ids)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # <<< 新增：更新学习率 >>>
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    def validate(self):
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
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "step": self.step,
                "best_loss": self.best_loss,
            },
            filename,
        )
        print(f"模型已保存: {filename}")

    def train(self, max_steps=10000, eval_every=500):
        print(f"开始训练，目标步数: {max_steps}")
        pbar = tqdm.tqdm(total=max_steps, desc="训练进度")

        while self.step < max_steps:
            for input_ids, target_ids in self.train_loader:
                loss = self.train_step(input_ids, target_ids)
                self.step += 1

                # <<< 新增：获取当前学习率用于显示 >>>
                current_lr = self.optimizer.param_groups[0]["lr"]

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "step": self.step,
                        "lr": f"{current_lr:.2e}",
                    }
                )

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


if __name__ == "__main__":
    model = GPTModel(GPT_CONFIG_A100)
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = DatasetLoader(
        file_path="/home/hjy/pythonPoject/cs336-assignment/data/TinyStoriesV2-GPT4-train.txt",
        tokenizer=tokenizer,
        max_length=256,
    )

    val_dataset = DatasetLoader(
        file_path="/home/hjy/pythonPoject/cs336-assignment/data/TinyStoriesV2-GPT4-valid.txt",
        tokenizer=tokenizer,
        max_length=256,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=min(16, os.cpu_count() or 8),
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=min(16, os.cpu_count() or 8),
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    num_training_steps = 10000
    num_warmup_steps = 1000  # 10% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=0.1,  # 最终学习率 = 3e-4 * 0.1 = 3e-5
    )

    criterion = CrossEntropyLoss()

    trainer = GPTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
    )

    trainer.train(max_steps=10000, eval_every=2000)
