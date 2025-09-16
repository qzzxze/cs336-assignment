import os
import pickle
import torch

from torch.utils.data import Dataset


class DatasetLoader(Dataset):

    def __init__(
        self,
        file_path,
        tokenizer,
        max_length=512,
        cache_dir="/home/hjy/pythonPoject/cs336-assignment/cache",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.cache_dir = cache_dir

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 生成缓存文件名（基于文件路径和参数）
        cache_filename = self._get_cache_filename(file_path, max_length)
        cache_path = os.path.join(cache_dir, cache_filename)

        # 检查缓存是否存在且比原文件新
        if self._should_use_cache(cache_path, file_path):
            print(f"加载缓存数据: {cache_path}")
            self.data = self._load_cache(cache_path)
        else:
            print(f"处理原始数据: {file_path}")
            self._load_and_process_data(file_path)
            self._save_cache(cache_path)
        print(f"数据加载完成，共{len(self.data)}个样本")

    def _get_cache_filename(self, file_path, max_length):
        """生成缓存文件名"""
        import hashlib

        # 基于文件路径和参数生成唯一标识
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return f"dataset_{file_hash}_{max_length}.pkl"

    def _should_use_cache(self, cache_path, original_path):
        """检查是否应该使用缓存"""
        if not os.path.exists(cache_path):
            return False

        # 检查缓存是否比原文件新
        cache_time = os.path.getmtime(cache_path)
        original_time = os.path.getmtime(original_path)

        return cache_time > original_time

    def _load_cache(self, cache_path):
        """加载缓存数据"""
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def _save_cache(self, cache_path):
        """保存缓存数据"""
        with open(cache_path, "wb") as f:
            pickle.dump(self.data, f)
        print(f"数据已缓存: {cache_path}")

    def _load_and_process_data(self, file_path):

        with open(file=file_path, mode="r", encoding="utf-8") as f:
            text = f.read()

        stories = text.strip().split("\n\n")
        print(f"找到{len(stories)}个故事")

        # stories = stories[:1000]
        # print(f"验证模式：只处理前{len(stories)}个故事")

        for story_idx, story in enumerate(stories):
            story = story.strip()
            if len(story) == 0:
                continue

            tokens = self.tokenizer.encode(story, allowed_special={"<|endoftext|>"})

            if len(tokens) > self.max_length:
                stride = self.max_length // 2
                for i in range(0, len(tokens) - self.max_length + 1, stride):
                    chunk = tokens[i : i + self.max_length]
                    self.data.append(chunk)
            else:
                padded = tokens + [self.tokenizer.eot_token] * (
                    self.max_length - len(tokens)
                )
                self.data.append(padded)

            if story_idx % 100000 == 0:
                print(f"处理进度：{story_idx}/{len(stories)}个故事")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sequence = torch.tensor(self.data[item], dtype=torch.long)
        input_idx = sequence[:-1]
        output_idx = sequence[1:]
        return input_idx, output_idx
