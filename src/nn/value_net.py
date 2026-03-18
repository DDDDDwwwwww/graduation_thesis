from __future__ import annotations

"""价值网络模型定义。"""

import torch
from torch import nn


class MLPValueNet(nn.Module):
    """MLP 价值网络，输出范围约束在 [-1, 1]。"""

    def __init__(self, input_dim: int, hidden_dims=(256, 128), dropout=0.1):
        """按给定隐藏层配置构建多层感知机。"""
        super().__init__()
        layers = []
        in_dim = int(input_dim)
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, int(hidden)))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden)
        # 最后一层输出标量价值。
        layers.append(nn.Linear(in_dim, 1))
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Tanh()

    def forward(self, x):
        """前向推理：返回形状 `[B, 1]` 的价值预测。"""
        return self.output(self.backbone(x))
