from __future__ import annotations

import random

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Uniformly random baseline agent."""

    def __init__(self, name: str, role: str, seed=None):
        super().__init__(name, role)
        if seed is not None:
            random.seed(seed)

    def select_action(self, game, state, legal_actions, time_limit=None):
        if not legal_actions:
            return None
        return random.choice(legal_actions)
