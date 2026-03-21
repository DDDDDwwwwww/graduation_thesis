from __future__ import annotations

"""棋盘张量编码器。

将类似 `cell(x,y,content)` 的事实编码为 `[C, H, W]` 张量。
当无法稳定识别棋盘结构时，回退到 `FactVectorEncoder`。
"""

import re

import numpy as np

from .fact_vector_encoder import FactVectorEncoder
from .vocab import FactVocabulary


_FACT_RE = re.compile(r"^\s*([a-zA-Z_][\w]*)\((.*)\)\s*$")


def _natural_key(text: str):
    """数字优先的稳定排序键。"""
    s = str(text).strip()
    return (0, int(s)) if s.isdigit() else (1, s)


def _split_args(payload: str) -> list[str]:
    """按逗号切分一层参数。"""
    out = []
    buf = []
    depth = 0
    for ch in payload:
        if ch == "(":
            depth += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            token = "".join(buf).strip()
            if token:
                out.append(token)
            buf = []
            continue
        buf.append(ch)
    token = "".join(buf).strip()
    if token:
        out.append(token)
    return out


class BoardTensorEncoder:
    """棋盘类状态编码器。

    默认识别 `cell(x,y,content)`，也可通过 schema 指定谓词与参数位次。
    """

    def __init__(self, schema=None, include_player_plane=True, include_turn_features=True):
        self.schema = dict(schema or {})
        self.include_player_plane = bool(include_player_plane)
        self.include_turn_features = bool(include_turn_features)

        self.predicate = str(self.schema.get("predicate", "cell"))
        self.x_index = int(self.schema.get("x_index", 0))
        self.y_index = int(self.schema.get("y_index", 1))
        self.content_index = int(self.schema.get("content_index", 2))

        self.x_values = [str(x) for x in self.schema.get("x_values", [])]
        self.y_values = [str(y) for y in self.schema.get("y_values", [])]
        self.content_values = [str(c) for c in self.schema.get("content_values", [])]

        self.x_to_idx: dict[str, int] = {}
        self.y_to_idx: dict[str, int] = {}
        self.content_to_channel: dict[str, int] = {"<UNK>": 0}
        self.roles: list[str] = []
        self.role_to_index: dict[str, int] = {}
        self.is_fitted = False
        self._fallback_encoder: FactVectorEncoder | None = None

    @property
    def board_shape(self) -> tuple[int, int]:
        return (len(self.y_to_idx), len(self.x_to_idx))

    @property
    def channel_dim(self) -> int:
        base = len(self.content_to_channel)
        extra = 0
        if self.include_player_plane:
            extra += 1
        if self.include_turn_features:
            extra += 2
        return base + extra

    def _parse_board_fact(self, fact: str):
        m = _FACT_RE.match(str(fact))
        if not m:
            return None
        predicate = m.group(1)
        if predicate != self.predicate:
            return None
        args = _split_args(m.group(2))
        need = max(self.x_index, self.y_index, self.content_index)
        if len(args) <= need:
            return None
        return (
            str(args[self.x_index]).strip(),
            str(args[self.y_index]).strip(),
            str(args[self.content_index]).strip(),
        )

    def _iter_fact_lists(self, samples):
        for sample in samples:
            if isinstance(sample, dict):
                facts = sample.get("state_facts", [])
            else:
                facts = sample
            yield [str(f) for f in facts]

    def fit(self, samples):
        """拟合棋盘坐标与内容词表，并构建回退编码器。"""
        samples = list(samples)
        all_fact_lists = list(self._iter_fact_lists(samples))

        x_seen = set(self.x_values)
        y_seen = set(self.y_values)
        content_seen = set(self.content_values)
        roles_seen = set()

        for sample in samples:
            if isinstance(sample, dict) and sample.get("acting_role") is not None:
                roles_seen.add(str(sample["acting_role"]))

        for facts in all_fact_lists:
            for fact in facts:
                parsed = self._parse_board_fact(fact)
                if parsed is None:
                    continue
                x, y, content = parsed
                x_seen.add(x)
                y_seen.add(y)
                content_seen.add(content)

        # 词表/映射确定性构建。
        self.x_to_idx = {x: i for i, x in enumerate(sorted(x_seen, key=_natural_key))}
        self.y_to_idx = {y: i for i, y in enumerate(sorted(y_seen, key=_natural_key))}
        self.content_to_channel = {"<UNK>": 0}
        for content in sorted(content_seen, key=_natural_key):
            if content == "<UNK>":
                continue
            self.content_to_channel[content] = len(self.content_to_channel)
        self.roles = sorted(roles_seen)
        self.role_to_index = {r: i for i, r in enumerate(self.roles)}

        # 构建回退编码器，确保无法识别棋盘时仍可输出。
        fallback_vocab = FactVocabulary.fit(all_fact_lists)
        self._fallback_encoder = FactVectorEncoder(
            vocab=fallback_vocab,
            roles=self.roles,
            include_role=True,
            include_turn_features=self.include_turn_features,
        )
        self.is_fitted = bool(self.x_to_idx and self.y_to_idx and len(self.content_to_channel) > 1)
        return self

    def _encode_role_scalar(self, role: str | None) -> float:
        if not self.roles:
            return 0.0
        idx = self.role_to_index.get(str(role), -1)
        if idx < 0:
            return 0.0
        if len(self.roles) == 1:
            return 1.0
        return -1.0 + 2.0 * (idx / (len(self.roles) - 1))

    def encode(self, state, game, role=None, ply_index: int = 0, terminal: bool = False) -> np.ndarray:
        """编码状态为 `[C, H, W]`，若失败则回退为 fact 向量。"""
        if isinstance(state, list) and (not state or isinstance(state[0], str)):
            facts = [str(f) for f in state]
        else:
            facts = game.get_state_facts_as_strings(state)

        if not self.is_fitted or not self.x_to_idx or not self.y_to_idx:
            if self._fallback_encoder is None:
                raise RuntimeError("BoardTensorEncoder must be fitted before encode().")
            return self._fallback_encoder.encode_facts(
                facts,
                role=role,
                ply_index=ply_index,
                terminal=terminal,
            )

        h, w = self.board_shape
        base_channels = len(self.content_to_channel)
        tensor = np.zeros((self.channel_dim, h, w), dtype=np.float32)
        board_hits = 0

        for fact in facts:
            parsed = self._parse_board_fact(fact)
            if parsed is None:
                continue
            x, y, content = parsed
            if x not in self.x_to_idx or y not in self.y_to_idx:
                continue
            xi = self.x_to_idx[x]
            yi = self.y_to_idx[y]
            ci = self.content_to_channel.get(content, 0)
            tensor[ci, yi, xi] = 1.0
            board_hits += 1

        # 自动回退：当前状态无任何棋盘事实时使用 fact 向量编码。
        if board_hits == 0 and self._fallback_encoder is not None:
            return self._fallback_encoder.encode_facts(
                facts,
                role=role,
                ply_index=ply_index,
                terminal=terminal,
            )

        cursor = base_channels
        if self.include_player_plane:
            tensor[cursor, :, :] = self._encode_role_scalar(role)
            cursor += 1
        if self.include_turn_features:
            tensor[cursor, :, :] = float(np.tanh(max(0.0, float(ply_index)) / 50.0))
            tensor[cursor + 1, :, :] = 1.0 if terminal else 0.0

        return tensor
