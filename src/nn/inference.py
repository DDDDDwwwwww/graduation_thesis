from __future__ import annotations

"""价值网络推理工具。

负责从磁盘加载模型与编码器，并提供单样本价值预测接口。
"""

from pathlib import Path

import numpy as np
import torch

from encoding.board_token_encoder import BoardTokenEncoder
from encoding.fact_vector_encoder import FactVectorEncoder
from encoding.vocab import FactVocabulary

from .value_net import MLPValueNet, TransformerValueNet


def load_value_artifacts(model_path, vocab_path=None, encoder_config_path=None, device="cpu"):
    """加载模型 checkpoint、词汇表和编码器配置。"""
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get("model_type", "mlp")

    if model_type == "mlp":
        if vocab_path is None:
            raise ValueError("vocab_path is required for mlp model.")
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
        extra = vocab
    elif model_type == "transformer":
        if encoder_config_path is None or not Path(encoder_config_path).exists():
            raise ValueError("encoder_config_path is required for transformer model.")
        encoder = BoardTokenEncoder.load(encoder_config_path)
        model = TransformerValueNet(
            num_tokens=int(checkpoint.get("num_tokens", encoder.num_tokens)),
            d_model=int(checkpoint.get("d_model", 128)),
            n_heads=int(checkpoint.get("n_heads", 4)),
            n_layers=int(checkpoint.get("n_layers", 3)),
            dim_feedforward=int(checkpoint.get("dim_feedforward", 256)),
            dropout=float(checkpoint.get("dropout", 0.1)),
            position_encoding=str(checkpoint.get("position_encoding", "sinusoidal")),
            use_global_features=bool(checkpoint.get("use_global_features", True)),
            max_positions=int(checkpoint.get("max_positions", 4096)),
        )
        # 兼容训练时按 batch 首次创建的 global_proj 参数。
        if "global_proj.weight" in checkpoint["state_dict"]:
            w = checkpoint["state_dict"]["global_proj.weight"]
            model.global_proj = torch.nn.Linear(int(w.shape[1]), int(w.shape[0]))
        extra = None
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, encoder, extra


def predict_value(model, x, device="cpu") -> float:
    """对单条输入向量做前向推理并返回浮点值。"""
    with torch.no_grad():
        if isinstance(x, dict):
            batch = {}
            for k, v in x.items():
                t = torch.as_tensor(v, device=device)
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                elif t.dim() == 2 and k == "tile_positions" and t.size(-1) == 2:
                    t = t.unsqueeze(0)
                batch[k] = t
            return float(model(batch).item())
        t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        return float(model(t).item())
