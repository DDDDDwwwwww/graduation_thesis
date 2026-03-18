from __future__ import annotations

"""价值网络推理工具。

负责从磁盘加载模型与编码器，并提供单样本价值预测接口。
"""

from pathlib import Path

import numpy as np
import torch

from encoding.fact_vector_encoder import FactVectorEncoder
from encoding.vocab import FactVocabulary

from .value_net import MLPValueNet


def load_value_artifacts(model_path, vocab_path, encoder_config_path=None, device="cpu"):
    """加载模型 checkpoint、词汇表和编码器配置。"""
    checkpoint = torch.load(model_path, map_location=device)
    if checkpoint.get("model_type") != "mlp":
        raise ValueError(f"Unsupported model_type: {checkpoint.get('model_type')}")

    vocab = FactVocabulary.load(vocab_path)
    if encoder_config_path is not None and Path(encoder_config_path).exists():
        encoder = FactVectorEncoder.load(encoder_config_path, vocab=vocab)
    else:
        # 兜底：若没有编码器配置，则使用最小配置编码器。
        encoder = FactVectorEncoder(vocab=vocab, roles=[], include_role=False, include_turn_features=False)

    model = MLPValueNet(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=tuple(checkpoint.get("hidden_dims", [256, 128])),
        dropout=float(checkpoint.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, encoder, vocab


def predict_value(model, x: np.ndarray, device="cpu") -> float:
    """对单条输入向量做前向推理并返回浮点值。"""
    with torch.no_grad():
        t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        return float(model(t).item())
