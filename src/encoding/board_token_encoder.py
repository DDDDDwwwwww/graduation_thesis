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
        global_feature_set="legacy",
        schema=None,
    ):
        if position_mode not in {"index", "xy"}:
            raise ValueError("position_mode must be 'index' or 'xy'")
        self.position_mode = str(position_mode)
        self.include_player_feature = bool(include_player_feature)
        self.include_turn_features = bool(include_turn_features)
        self.global_feature_set = str(global_feature_set).lower()
        if self.global_feature_set not in {"legacy", "basic10", "none"}:
            raise ValueError("global_feature_set must be one of: legacy, basic10, none")
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
        if self.global_feature_set == "none":
            return 0
        if self.global_feature_set == "basic10":
            return 10
        dim = 0
        if self.include_player_feature:
            dim += 1
        if self.include_turn_features:
            dim += 2
        return dim

    def _is_empty_content(self, content: str) -> bool:
        token = str(content).strip().lower()
        return token in {"", "b", "blank", "empty", "e", "nil", "none", "0"}

    def _estimate_player_piece_counts(self, role: str | None, content_counts: dict[str, int]) -> tuple[int, int]:
        if not content_counts:
            return 0, 0
        sorted_items = sorted(content_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        if role is None:
            current = int(sorted_items[0][1]) if sorted_items else 0
            opponent = int(sorted_items[1][1]) if len(sorted_items) > 1 else max(0, sum(content_counts.values()) - current)
            return current, opponent

        role_l = str(role).strip().lower()
        role_hints = {role_l}
        role_hints.add(role_l.replace("player", "").replace("role", "").strip())
        for ch in role_l:
            if ch.isalpha():
                role_hints.add(ch)
                break
        role_hints = {h for h in role_hints if h}

        current = 0
        total = sum(int(v) for v in content_counts.values())
        for token, count in content_counts.items():
            token_l = str(token).strip().lower()
            if any((hint == token_l) or (hint in token_l) for hint in role_hints):
                current += int(count)
        opponent = max(0, int(total) - int(current))
        if current == 0 and sorted_items:
            current = int(sorted_items[0][1])
            opponent = int(sorted_items[1][1]) if len(sorted_items) > 1 else max(0, int(total) - int(current))
        return int(current), int(opponent)

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
            tokens.append((yi, xi, cid, pos, content))

        # 保证确定性顺序。
        tokens.sort(key=lambda t: (t[0], t[1]))

        if not tokens:
            tile_content_ids = np.asarray([0], dtype=np.int64)
            if self.position_mode == "xy":
                tile_positions = np.asarray([[0, 0]], dtype=np.int64)
            else:
                tile_positions = np.asarray([0], dtype=np.int64)
            mask = np.asarray([True], dtype=np.bool_)
            token_contents: list[str] = []
        else:
            tile_content_ids = np.asarray([t[2] for t in tokens], dtype=np.int64)
            if self.position_mode == "xy":
                tile_positions = np.asarray([t[3] for t in tokens], dtype=np.int64)
            else:
                tile_positions = np.asarray([t[3] for t in tokens], dtype=np.int64)
            mask = np.ones((len(tokens),), dtype=np.bool_)
            token_contents = [str(t[4]) for t in tokens]

        globals_ = []
        if self.global_feature_set == "legacy":
            if self.include_player_feature:
                globals_.append(self._role_scalar(role))
            if self.include_turn_features:
                globals_.append(float(np.tanh(max(0.0, float(ply_index)) / 50.0)))
                globals_.append(1.0 if terminal else 0.0)
        elif self.global_feature_set == "basic10":
            total_cells = int(len(self.pos_to_id)) if self.pos_to_id else max(1, int(len(tile_content_ids)))
            occupied_contents = [c for c in token_contents if not self._is_empty_content(c)]
            occupied = int(len(occupied_contents))
            occupied_ratio = float(occupied / max(1, total_cells))
            empty_ratio = float(1.0 - occupied_ratio)

            content_counts: dict[str, int] = {}
            for c in occupied_contents:
                content_counts[c] = int(content_counts.get(c, 0)) + 1
            current_cnt, opp_cnt = self._estimate_player_piece_counts(role=role, content_counts=content_counts)
            current_ratio = float(current_cnt / max(1, total_cells))
            opp_ratio = float(opp_cnt / max(1, total_cells))
            piece_diff_ratio = float((current_cnt - opp_cnt) / max(1, total_cells))
            uniq_ratio = float(len(content_counts) / max(1, self.num_tokens - 1))
            max_content_ratio = (
                float(max(content_counts.values()) / max(1, occupied))
                if content_counts
                else 0.0
            )

            globals_ = [
                self._role_scalar(role),
                float(np.tanh(max(0.0, float(ply_index)) / 50.0)),
                1.0 if terminal else 0.0,
                occupied_ratio,
                empty_ratio,
                current_ratio,
                opp_ratio,
                piece_diff_ratio,
                uniq_ratio,
                max_content_ratio,
            ]
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
            "global_feature_set": self.global_feature_set,
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
            global_feature_set=payload.get("global_feature_set", "legacy"),
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
