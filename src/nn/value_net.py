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


class TransformerValueNet(nn.Module):
    """基于 token 序列的 Transformer 价值网络。"""

    def __init__(
        self,
        num_tokens: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        position_encoding: str = "sinusoidal",
        use_global_features: bool = True,
        max_positions: int = 4096,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.position_encoding = str(position_encoding).lower()
        self.use_global_features = bool(use_global_features)
        self.max_positions = int(max_positions)
        if self.position_encoding not in {"sinusoidal", "learned"}:
            raise ValueError("position_encoding must be 'sinusoidal' or 'learned'")

        self.content_emb = nn.Embedding(self.num_tokens, self.d_model)
        if self.position_encoding == "learned":
            self.pos_emb = nn.Embedding(self.max_positions, self.d_model)
        else:
            self.pos_emb = None
            self.register_buffer("_sin_cache", torch.empty(0, self.d_model), persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
        self.global_proj: nn.Linear | None = None
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def _ensure_batch(self, t: torch.Tensor, target_dims: int):
        if t.dim() == target_dims - 1:
            return t.unsqueeze(0)
        return t

    def _positions_to_ids(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.dim() == 2:
            return positions.long()
        if positions.dim() != 3 or positions.size(-1) != 2:
            raise ValueError(f"Unsupported tile_positions shape: {tuple(positions.shape)}")
        # [B, T, 2] -> [B, T]
        x = positions[..., 0].long()
        y = positions[..., 1].long()
        x_shift = x - x.min(dim=1, keepdim=True).values
        y_shift = y - y.min(dim=1, keepdim=True).values
        width = y_shift.max(dim=1, keepdim=True).values + 1
        return x_shift * width + y_shift

    def _get_sinusoidal(self, max_pos: int, device: torch.device) -> torch.Tensor:
        cache = self._sin_cache
        if cache.numel() == 0 or cache.size(0) < max_pos:
            positions = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32)
                * (-torch.log(torch.tensor(10000.0)) / self.d_model)
            )
            table = torch.zeros(max_pos, self.d_model, dtype=torch.float32)
            table[:, 0::2] = torch.sin(positions * div)
            table[:, 1::2] = torch.cos(positions * div)
            self._sin_cache = table
            cache = table
        return cache.to(device)

    def _apply_positional_encoding(self, content_x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        if self.position_encoding == "learned":
            clipped = position_ids.clamp(min=0, max=self.max_positions - 1)
            return content_x + self.pos_emb(clipped)
        max_pos = int(position_ids.max().item()) + 1 if position_ids.numel() else 1
        table = self._get_sinusoidal(max_pos=max_pos, device=content_x.device)
        return content_x + table[position_ids]

    def _masked_mean_pool(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        # mask: True 表示有效 token。
        m = mask.to(dtype=x.dtype).unsqueeze(-1)
        summed = (x * m).sum(dim=1)
        denom = m.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, batch):
        """输入编码 token batch，输出 `[B,1]` 价值。"""
        tile_content_ids = batch["tile_content_ids"]
        tile_positions = batch["tile_positions"]
        mask = batch.get("mask")
        global_features = batch.get("global_features")

        tile_content_ids = self._ensure_batch(tile_content_ids, target_dims=2).long()
        if tile_positions.dim() == 1:
            tile_positions = tile_positions.unsqueeze(0)
        elif tile_positions.dim() == 2:
            # [T,2] (single xy sample) -> [1,T,2], [B,T] 保持不变。
            if tile_positions.size(-1) == 2 and tile_content_ids.size(0) == 1:
                tile_positions = tile_positions.unsqueeze(0)
        elif tile_positions.dim() != 3:
            raise ValueError(f"Unsupported tile_positions shape: {tuple(tile_positions.shape)}")
        if mask is not None:
            mask = self._ensure_batch(mask, target_dims=2).bool()

        content_x = self.content_emb(tile_content_ids)
        pos_ids = self._positions_to_ids(tile_positions)
        x = self._apply_positional_encoding(content_x, pos_ids)

        # src_key_padding_mask: True 表示需要被忽略。
        key_padding_mask = None if mask is None else (~mask)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled = self._masked_mean_pool(x, mask=mask)

        if self.use_global_features and global_features is not None:
            global_features = self._ensure_batch(global_features, target_dims=2).to(dtype=pooled.dtype)
            if self.global_proj is None:
                self.global_proj = nn.Linear(global_features.size(-1), self.d_model).to(pooled.device)
            pooled = pooled + self.global_proj(global_features)

        return self.value_head(pooled)
