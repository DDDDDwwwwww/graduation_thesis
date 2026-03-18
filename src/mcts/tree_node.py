from __future__ import annotations

"""MCTS 树节点数据结构定义。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TreeNode:
    """可复用的 MCTS 节点容器。

    说明：
    - `state_key` 用于快速比较状态是否相同。
    - `children` 以“联合动作键”索引子节点。
    - `action_stats` 记录每个角色、每个动作的访问和价值统计。
    """

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
        """更新节点访问次数与累计/平均价值。"""
        self.visits += 1
        self.total_value += float(value)
        self.mean_value = self.total_value / self.visits
