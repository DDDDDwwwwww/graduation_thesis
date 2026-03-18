from __future__ import annotations

"""训练数据集封装。"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ValueDataset(Dataset):
    """把 JSONL 样本映射为 `(x, y)` 的 PyTorch Dataset。"""

    def __init__(self, samples, encoder):
        """保存原始样本与编码器引用。"""
        self.samples = list(samples)
        self.encoder = encoder

    def __len__(self):
        """返回样本总数。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """按索引读取一条样本并编码为张量。"""
        sample = self.samples[idx]
        facts = sample["state_facts"]
        x = self.encoder.encode_facts(
            facts,
            role=sample.get("acting_role"),
            ply_index=sample.get("ply_index", 0),
            terminal=sample.get("terminal", False),
        )
        y = float(sample["value_target"])
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)

    @staticmethod
    def load_jsonl(path: str | Path):
        """从 JSONL 文件加载全部样本。"""
        rows = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
