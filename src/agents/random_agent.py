from __future__ import annotations

"""随机基线智能体。

该智能体不进行搜索或估值，仅从合法动作中均匀随机采样。
"""

import random

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """均匀随机动作的基线智能体。"""

    def __init__(self, name: str, role: str, seed=None):
        """创建独立随机数发生器，保证可复现实验。"""
        super().__init__(name, role)
        self._rng = random.Random(seed)

    def select_action(self, game, state, legal_actions, time_limit=None):
        """从合法动作中随机返回一个动作。"""
        if not legal_actions:
            return None
        return self._rng.choice(legal_actions)
