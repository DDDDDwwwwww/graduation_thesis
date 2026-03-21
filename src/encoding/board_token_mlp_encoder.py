from __future__ import annotations

"""Board token encoder variant for fixed-length MLP inputs.

This encoder reuses BoardTokenEncoder parsing, then pools tile token ids into a
fixed-size content histogram and appends optional global features.
"""

import json
from pathlib import Path

import numpy as np

from .board_token_encoder import BoardTokenEncoder


class BoardTokenMLPEncoder:
    """Encode board-token states into fixed vectors for MLP value models."""

    def __init__(
        self,
        board_encoder: BoardTokenEncoder,
        normalize_counts: bool = True,
    ):
        self.board_encoder = board_encoder
        self.normalize_counts = bool(normalize_counts)

    @property
    def input_dim(self) -> int:
        return int(self.board_encoder.num_tokens + self.board_encoder.global_feature_dim)

    def encode_facts(
        self,
        facts,
        role: str | None = None,
        ply_index: int = 0,
        terminal: bool = False,
    ) -> np.ndarray:
        token_payload = self.board_encoder.encode_facts(
            facts,
            role=role,
            ply_index=ply_index,
            terminal=terminal,
        )
        token_ids = np.asarray(token_payload["tile_content_ids"], dtype=np.int64)
        hist = np.zeros((self.board_encoder.num_tokens,), dtype=np.float32)
        if token_ids.size > 0:
            for tid in token_ids.tolist():
                if 0 <= int(tid) < hist.shape[0]:
                    hist[int(tid)] += 1.0
            if self.normalize_counts:
                hist /= float(token_ids.size)

        global_features = np.asarray(token_payload.get("global_features", []), dtype=np.float32)
        if global_features.size == 0:
            return hist
        return np.concatenate([hist, global_features], axis=0)

    def encode(self, state, game, role: str | None = None, ply_index: int = 0, terminal: bool = False) -> np.ndarray:
        if isinstance(state, list) and (not state or isinstance(state[0], str)):
            facts = [str(f) for f in state]
        else:
            facts = game.get_state_facts_as_strings(state)
        return self.encode_facts(facts, role=role, ply_index=ply_index, terminal=terminal)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "encoder_type": "board_token_mlp",
            "normalize_counts": self.normalize_counts,
            "board_token": {
                "position_mode": self.board_encoder.position_mode,
                "include_player_feature": self.board_encoder.include_player_feature,
                "include_turn_features": self.board_encoder.include_turn_features,
                "schema": self.board_encoder.schema,
                "x_to_idx": self.board_encoder.x_to_idx,
                "y_to_idx": self.board_encoder.y_to_idx,
                "content_to_id": self.board_encoder.content_to_id,
                "roles": self.board_encoder.roles,
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BoardTokenMLPEncoder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        board_payload = payload.get("board_token", {})
        board_encoder = BoardTokenEncoder(
            position_mode=board_payload.get("position_mode", "index"),
            include_player_feature=board_payload.get("include_player_feature", True),
            include_turn_features=board_payload.get("include_turn_features", True),
            schema=board_payload.get("schema", {}),
        )
        board_encoder.x_to_idx = {str(k): int(v) for k, v in board_payload.get("x_to_idx", {}).items()}
        board_encoder.y_to_idx = {str(k): int(v) for k, v in board_payload.get("y_to_idx", {}).items()}
        board_encoder.content_to_id = {str(k): int(v) for k, v in board_payload.get("content_to_id", {}).items()}
        if "<UNK>" not in board_encoder.content_to_id:
            board_encoder.content_to_id = {"<UNK>": 0, **board_encoder.content_to_id}
        board_encoder.roles = [str(r) for r in board_payload.get("roles", [])]
        board_encoder.role_to_index = {r: i for i, r in enumerate(board_encoder.roles)}

        board_encoder.pos_to_id = {}
        pid = 0
        for y in sorted(board_encoder.y_to_idx, key=lambda v: (0, int(v)) if str(v).isdigit() else (1, str(v))):
            for x in sorted(board_encoder.x_to_idx, key=lambda v: (0, int(v)) if str(v).isdigit() else (1, str(v))):
                board_encoder.pos_to_id[(board_encoder.x_to_idx[str(x)], board_encoder.y_to_idx[str(y)])] = pid
                pid += 1
        board_encoder.is_fitted = bool(board_encoder.x_to_idx and board_encoder.y_to_idx and board_encoder.num_tokens > 1)
        return cls(
            board_encoder=board_encoder,
            normalize_counts=payload.get("normalize_counts", True),
        )

    @classmethod
    def fit(
        cls,
        samples,
        position_mode: str = "index",
        include_player_feature: bool = True,
        include_turn_features: bool = True,
        normalize_counts: bool = True,
    ) -> "BoardTokenMLPEncoder":
        board_encoder = BoardTokenEncoder(
            position_mode=position_mode,
            include_player_feature=include_player_feature,
            include_turn_features=include_turn_features,
        ).fit(samples)
        return cls(board_encoder=board_encoder, normalize_counts=normalize_counts)
