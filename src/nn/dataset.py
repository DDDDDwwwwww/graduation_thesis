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
        if hasattr(self.encoder, "encode_facts"):
            x = self.encoder.encode_facts(
                facts,
                role=sample.get("acting_role"),
                ply_index=sample.get("ply_index", 0),
                terminal=sample.get("terminal", False),
            )
        else:
            x = self.encoder.encode(
                facts,
                game=None,
                role=sample.get("acting_role"),
                ply_index=sample.get("ply_index", 0),
                terminal=sample.get("terminal", False),
            )
        y = float(sample["value_target"])
        if isinstance(x, dict):
            return x, torch.tensor([y], dtype=torch.float32)
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)

    @staticmethod
    def collate_board_tokens(batch):
        """拼接可变长度 token 样本，输出 transformer 兼容 batch。"""
        xs, ys = zip(*batch)
        batch_size = len(xs)
        max_t = max(int(len(x["tile_content_ids"])) for x in xs)

        pos_is_xy = bool(xs[0]["tile_positions"].ndim == 2)
        tile_content_ids = torch.zeros((batch_size, max_t), dtype=torch.long)
        tile_positions = (
            torch.zeros((batch_size, max_t, 2), dtype=torch.long)
            if pos_is_xy
            else torch.zeros((batch_size, max_t), dtype=torch.long)
        )
        mask = torch.zeros((batch_size, max_t), dtype=torch.bool)

        has_global = "global_features" in xs[0] and len(xs[0]["global_features"]) > 0
        global_features = None
        if has_global:
            g_dim = int(len(xs[0]["global_features"]))
            global_features = torch.zeros((batch_size, g_dim), dtype=torch.float32)

        for i, x in enumerate(xs):
            t = int(len(x["tile_content_ids"]))
            tile_content_ids[i, :t] = torch.as_tensor(x["tile_content_ids"], dtype=torch.long)
            tile_positions[i, :t] = torch.as_tensor(x["tile_positions"], dtype=torch.long)
            valid = torch.ones((t,), dtype=torch.bool)
            if "mask" in x and len(x["mask"]) == t:
                valid = torch.as_tensor(x["mask"], dtype=torch.bool)
            mask[i, :t] = valid
            if has_global:
                global_features[i] = torch.as_tensor(x["global_features"], dtype=torch.float32)

        y = torch.stack(list(ys), dim=0)
        batch_x = {
            "tile_content_ids": tile_content_ids,
            "tile_positions": tile_positions,
            "mask": mask,
        }
        if has_global:
            batch_x["global_features"] = global_features
        return batch_x, y

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
