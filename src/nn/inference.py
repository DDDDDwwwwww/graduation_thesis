from __future__ import annotations

"""Inference helpers for loading value model artifacts."""

import json
from pathlib import Path

import numpy as np
import torch

from encoding.board_token_encoder import BoardTokenEncoder
from encoding.board_token_mlp_encoder import BoardTokenMLPEncoder
from encoding.vocab import FactVocabulary

from .value_net import MLPValueNet, TransformerValueNet


def _read_encoder_type(encoder_config_path: str | Path | None) -> str | None:
    if encoder_config_path is None:
        return None
    p = Path(encoder_config_path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    et = payload.get("encoder_type")
    if et:
        return str(et)
    if "board_token" in payload and payload.get("encoder_type") is None:
        return "board_token_mlp"
    if "content_to_id" in payload and "x_to_idx" in payload:
        return "board_token"
    if "include_role" in payload and "roles" in payload:
        return "fact_vector"
    return None


def load_value_artifacts(model_path, vocab_path=None, encoder_config_path=None, device="cpu"):
    """Load model checkpoint and corresponding encoder artifacts."""
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get("model_type", "mlp")

    if model_type == "mlp":
        encoder_type = _read_encoder_type(encoder_config_path)

        if encoder_type == "board_token_mlp":
            if encoder_config_path is None or not Path(encoder_config_path).exists():
                raise ValueError("encoder_config_path is required for board_token_mlp + mlp model.")
            encoder = BoardTokenMLPEncoder.load(encoder_config_path)
            extra = None
        else:
            if vocab_path is None:
                raise ValueError("vocab_path is required for fact_vector + mlp model.")
            vocab = FactVocabulary.load(vocab_path)
            if encoder_config_path is not None and Path(encoder_config_path).exists():
                encoder = BoardTokenMLPEncoder.load(encoder_config_path, vocab=vocab)
            else:
                encoder = BoardTokenMLPEncoder(vocab=vocab, roles=[], include_role=False, include_turn_features=False)
            extra = vocab

        model = MLPValueNet(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dims=tuple(checkpoint.get("hidden_dims", [256, 128])),
            dropout=float(checkpoint.get("dropout", 0.1)),
        )

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
    """Run single-sample inference and return scalar value."""
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
