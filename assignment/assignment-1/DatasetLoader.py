import torch

from torch.utils.data import Dataset


class DatasetLoader(Dataset):

    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        print(f"正在加载数据：{file_path}")
        self._load_and_process_data(file_path)
        print(f"数据加载完成，共{len(self.data)}个样本")

    def _load_and_process_data(self, file_path):

        with open(file=file_path, mode='r', encoding='utf-8') as f:
            text = f.read()

        stories = text.strip().split('\n\n')
        print(f"找到{len(stories)}个故事")

        stories = stories[:1000]
        print(f"验证模式：只处理前{len(stories)}个故事")

        for story_idx, story in enumerate(stories):
            story = story.strip()
            if len(story) == 0:
                continue

            tokens = self.tokenizer.encode(story, allowed_special={"<|endoftext|>"})

            if len(tokens) > self.max_length:
                stride = self.max_length // 2
                for i in range(0, len(tokens) - self.max_length + 1, stride):
                    chunk = tokens[i: i + self.max_length]
                    self.data.append(chunk)
            else:
                padded = tokens + [self.tokenizer.eot_token] * (self.max_length - len(tokens))
                self.data.append(padded)

            if story_idx % 1000 == 0:
                print(f"处理进度：{story_idx}/{len(stories)}个故事")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sequence = torch.tensor(self.data[item], dtype=torch.long)
        input_idx = sequence[:-1]
        output_idx = sequence[1:]
        return input_idx, output_idx
