from __future__ import annotations

"""棋盘 token 编码器。

将 `cell(x,y,content)` 类事实编码为可变长 token 序列，供 Transformer 使用。
"""

import json
import re
from pathlib import Path

import numpy as np


_FACT_RE = re.compile(r"^\s*([a-zA-Z_][\w]*)\((.*)\)\s*$")


def _natural_key(text: str):
    s = str(text).strip()
    return (0, int(s)) if s.isdigit() else (1, s)


def _split_args(payload: str) -> list[str]:
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


class BoardTokenEncoder:
    """棋盘 token 编码器。"""

    def __init__(
        self,
        content_vocab=None,
        position_mode="index",
        include_player_feature=True,
        include_turn_features=True,
        schema=None,
    ):
        if position_mode not in {"index", "xy"}:
            raise ValueError("position_mode must be 'index' or 'xy'")
        self.position_mode = str(position_mode)
        self.include_player_feature = bool(include_player_feature)
        self.include_turn_features = bool(include_turn_features)
        self.schema = dict(schema or {})

        self.predicate = str(self.schema.get("predicate", "cell"))
        self.x_index = int(self.schema.get("x_index", 0))
        self.y_index = int(self.schema.get("y_index", 1))
        self.content_index = int(self.schema.get("content_index", 2))

        self.x_to_idx: dict[str, int] = {}
        self.y_to_idx: dict[str, int] = {}
        self.pos_to_id: dict[tuple[int, int], int] = {}
        self.content_to_id: dict[str, int] = {"<UNK>": 0}
        self.roles: list[str] = []
        self.role_to_index: dict[str, int] = {}
        self.is_fitted = False

        if content_vocab:
            items = list(content_vocab)
            self.content_to_id = {"<UNK>": 0}
            for c in items:
                c = str(c)
                if c == "<UNK>":
                    continue
                self.content_to_id[c] = len(self.content_to_id)

    @property
    def num_tokens(self) -> int:
        return len(self.content_to_id)

    @property
    def global_feature_dim(self) -> int:
        dim = 0
        if self.include_player_feature:
            dim += 1
        if self.include_turn_features:
            dim += 2
        return dim

    def _parse_board_fact(self, fact: str):
        m = _FACT_RE.match(str(fact))
        if not m:
            return None
        if m.group(1) != self.predicate:
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

    def fit(self, samples):
        samples = list(samples)
        x_seen = set()
        y_seen = set()
        content_seen = set()
        roles_seen = set()

        for sample in samples:
            facts = sample["state_facts"] if isinstance(sample, dict) else sample
            if isinstance(sample, dict) and sample.get("acting_role") is not None:
                roles_seen.add(str(sample["acting_role"]))
            for fact in facts:
                parsed = self._parse_board_fact(str(fact))
                if parsed is None:
                    continue
                x, y, content = parsed
                x_seen.add(x)
                y_seen.add(y)
                content_seen.add(content)

        self.x_to_idx = {x: i for i, x in enumerate(sorted(x_seen, key=_natural_key))}
        self.y_to_idx = {y: i for i, y in enumerate(sorted(y_seen, key=_natural_key))}

        self.pos_to_id = {}
        pid = 0
        for y in sorted(self.y_to_idx, key=_natural_key):
            for x in sorted(self.x_to_idx, key=_natural_key):
                self.pos_to_id[(self.x_to_idx[x], self.y_to_idx[y])] = pid
                pid += 1

        if self.num_tokens <= 1:
            self.content_to_id = {"<UNK>": 0}
            for content in sorted(content_seen, key=_natural_key):
                if content == "<UNK>":
                    continue
                self.content_to_id[content] = len(self.content_to_id)

        self.roles = sorted(roles_seen)
        self.role_to_index = {r: i for i, r in enumerate(self.roles)}
        self.is_fitted = bool(self.x_to_idx and self.y_to_idx and self.num_tokens > 1)
        return self

    def _role_scalar(self, role: str | None) -> float:
        if not self.roles:
            return 0.0
        idx = self.role_to_index.get(str(role), -1)
        if idx < 0:
            return 0.0
        if len(self.roles) == 1:
            return 1.0
        return -1.0 + 2.0 * (idx / (len(self.roles) - 1))

    def encode_facts(self, facts, role=None, ply_index: int = 0, terminal: bool = False) -> dict:
        tokens = []
        for fact in facts:
            parsed = self._parse_board_fact(str(fact))
            if parsed is None:
                continue
            x, y, content = parsed
            if x not in self.x_to_idx or y not in self.y_to_idx:
                continue
            xi = self.x_to_idx[x]
            yi = self.y_to_idx[y]
            cid = self.content_to_id.get(content, 0)
            if self.position_mode == "xy":
                pos = (xi, yi)
            else:
                pos = self.pos_to_id.get((xi, yi), 0)
            tokens.append((yi, xi, cid, pos))

        # 保证确定性顺序。
        tokens.sort(key=lambda t: (t[0], t[1]))

        if not tokens:
            tile_content_ids = np.asarray([0], dtype=np.int64)
            if self.position_mode == "xy":
                tile_positions = np.asarray([[0, 0]], dtype=np.int64)
            else:
                tile_positions = np.asarray([0], dtype=np.int64)
            mask = np.asarray([True], dtype=np.bool_)
        else:
            tile_content_ids = np.asarray([t[2] for t in tokens], dtype=np.int64)
            if self.position_mode == "xy":
                tile_positions = np.asarray([t[3] for t in tokens], dtype=np.int64)
            else:
                tile_positions = np.asarray([t[3] for t in tokens], dtype=np.int64)
            mask = np.ones((len(tokens),), dtype=np.bool_)

        globals_ = []
        if self.include_player_feature:
            globals_.append(self._role_scalar(role))
        if self.include_turn_features:
            globals_.append(float(np.tanh(max(0.0, float(ply_index)) / 50.0)))
            globals_.append(1.0 if terminal else 0.0)
        global_features = np.asarray(globals_, dtype=np.float32)

        return {
            "tile_content_ids": tile_content_ids,
            "tile_positions": tile_positions,
            "global_features": global_features,
            "mask": mask,
        }

    def encode(self, state, game, role=None, ply_index: int = 0, terminal: bool = False) -> dict:
        if isinstance(state, list) and (not state or isinstance(state[0], str)):
            facts = [str(f) for f in state]
        else:
            facts = game.get_state_facts_as_strings(state)
        return self.encode_facts(facts, role=role, ply_index=ply_index, terminal=terminal)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "position_mode": self.position_mode,
            "include_player_feature": self.include_player_feature,
            "include_turn_features": self.include_turn_features,
            "schema": self.schema,
            "x_to_idx": self.x_to_idx,
            "y_to_idx": self.y_to_idx,
            "content_to_id": self.content_to_id,
            "roles": self.roles,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BoardTokenEncoder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        enc = cls(
            position_mode=payload.get("position_mode", "index"),
            include_player_feature=payload.get("include_player_feature", True),
            include_turn_features=payload.get("include_turn_features", True),
            schema=payload.get("schema", {}),
        )
        enc.x_to_idx = {str(k): int(v) for k, v in payload.get("x_to_idx", {}).items()}
        enc.y_to_idx = {str(k): int(v) for k, v in payload.get("y_to_idx", {}).items()}
        enc.content_to_id = {str(k): int(v) for k, v in payload.get("content_to_id", {}).items()}
        if "<UNK>" not in enc.content_to_id:
            enc.content_to_id = {"<UNK>": 0, **enc.content_to_id}
        enc.roles = [str(r) for r in payload.get("roles", [])]
        enc.role_to_index = {r: i for i, r in enumerate(enc.roles)}

        enc.pos_to_id = {}
        pid = 0
        for y in sorted(enc.y_to_idx, key=_natural_key):
            for x in sorted(enc.x_to_idx, key=_natural_key):
                enc.pos_to_id[(enc.x_to_idx[x], enc.y_to_idx[y])] = pid
                pid += 1
        enc.is_fitted = bool(enc.x_to_idx and enc.y_to_idx and enc.num_tokens > 1)
        return enc
