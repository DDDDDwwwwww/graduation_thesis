from __future__ import annotations

"""事实向量编码器。

把 GGP 符号状态编码成固定长度的稠密向量，供 MLP 价值网络输入。
"""

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .vocab import FactVocabulary


class FactVectorEncoder:
    """将符号事实编码为固定维度向量。"""

    def __init__(
        self,
        vocab: FactVocabulary,
        roles: Iterable[str] | None = None,
        include_role: bool = True,
        include_turn_features: bool = True,
    ):
        """初始化编码配置。

        Args:
            vocab: fact 词汇表。
            roles: 角色列表，用于角色 one-hot 附加特征。
            include_role: 是否附加角色特征。
            include_turn_features: 是否附加回合/终局特征。
        """
        self.vocab = vocab
        self.include_role = bool(include_role)
        self.include_turn_features = bool(include_turn_features)
        self.roles = sorted(str(r) for r in (roles or []))
        self.role_to_index = {r: i for i, r in enumerate(self.roles)}

    @property
    def role_feature_dim(self) -> int:
        """角色特征维度（含一个 unknown 槽位）。"""
        if not self.include_role:
            return 0
        # +1 slot for unknown role at inference time.
        return len(self.roles) + 1

    @property
    def turn_feature_dim(self) -> int:
        """回合特征维度。"""
        return 2 if self.include_turn_features else 0

    @property
    def input_dim(self) -> int:
        """模型输入总维度。"""
        return self.vocab.size + self.role_feature_dim + self.turn_feature_dim

    def encode_facts(
        self,
        facts: Iterable[str],
        role: str | None = None,
        ply_index: int = 0,
        terminal: bool = False,
    ) -> np.ndarray:
        """直接从 fact 列表编码。"""
        x = np.zeros(self.input_dim, dtype=np.float32)

        # 事实部分采用 multi-hot 编码（出现即置 1）。
        for fact in facts:
            idx = self.vocab.encode_fact(str(fact))
            x[idx] = 1.0

        cursor = self.vocab.size

        # 角色 one-hot：当前 acting role 对应位置置 1。
        if self.include_role:
            role_idx = self.role_to_index.get(str(role), len(self.roles))
            x[cursor + role_idx] = 1.0
            cursor += self.role_feature_dim

        # 回合信息：平滑后的 ply + 终局标志。
        if self.include_turn_features:
            # Smooth bounded turn feature and explicit terminal flag.
            x[cursor] = float(np.tanh(max(0.0, float(ply_index)) / 50.0))
            x[cursor + 1] = 1.0 if terminal else 0.0

        return x

    def encode(
        self,
        state,
        game,
        role: str | None = None,
        ply_index: int = 0,
        terminal: bool = False,
    ) -> np.ndarray:
        """从状态对象编码；支持传入 state facts 或状态对象。"""
        if isinstance(state, list) and (not state or isinstance(state[0], str)):
            facts = [str(f) for f in state]
        else:
            facts = game.get_state_facts_as_strings(state)
        return self.encode_facts(facts, role=role, ply_index=ply_index, terminal=terminal)

    def save(self, path: str | Path) -> None:
        """保存编码器配置（不包含词汇表本体）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "include_role": self.include_role,
            "include_turn_features": self.include_turn_features,
            "roles": self.roles,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, vocab: FactVocabulary) -> "FactVectorEncoder":
        """从配置文件和词汇表恢复编码器。"""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            vocab=vocab,
            roles=payload.get("roles", []),
            include_role=payload.get("include_role", True),
            include_turn_features=payload.get("include_turn_features", True),
        )
