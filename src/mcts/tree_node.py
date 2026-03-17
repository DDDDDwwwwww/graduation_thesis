from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TreeNode:
    """Reusable MCTS node container."""

    state_key: str
    parent: "TreeNode | None" = None
    action_from_parent: Any | None = None
    children: dict = field(default_factory=dict)
    untried_actions: list = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    mean_value: float = 0.0
    player_to_move: str | None = None
    is_terminal: bool = False
    state: Any = None
    action_stats: dict = field(default_factory=dict)

    def update_value(self, value: float) -> None:
        self.visits += 1
        self.total_value += float(value)
        self.mean_value = self.total_value / self.visits
