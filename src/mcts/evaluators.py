from __future__ import annotations

"""叶子评估器抽象与实现。

该模块把“如何给叶子状态打分”从 MCTS 主循环中解耦出来，便于替换评估方式。
"""

import random
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .rollout import rollout_scores


class LeafEvaluator:
    """叶子评估器抽象基类。"""

    def evaluate(self, game, state, role, time_limit=None) -> float:
        raise NotImplementedError

    def evaluate_for_roles(self, game, state, roles, budget_end=None) -> dict:
        raise NotImplementedError


class RandomRolloutEvaluator(LeafEvaluator):
    """随机 rollout 评估器。"""

    def __init__(self, depth_limit=200, rng=None):
        self.depth_limit = max(1, int(depth_limit))
        self.rng = rng or random

    def evaluate_for_roles(self, game, state, roles, budget_end=None):
        """返回给定角色集合的估值字典。"""
        if game.is_terminal(state):
            return {role: float(game.get_goal(state, role)) for role in roles}

        return rollout_scores(
            game,
            state,
            roles,
            move_selector=lambda _role, legal_moves: self.rng.choice(legal_moves),
            depth_limit=self.depth_limit,
            budget_end=budget_end,
        )

    def evaluate(self, game, state, role, time_limit=None) -> float:
        return float(self.evaluate_for_roles(game, state, [role]).get(role, 0.0))


class HeuristicRolloutEvaluator(LeafEvaluator):
    """带启发式动作采样器的 rollout 评估器。"""

    def __init__(self, move_sampler, depth_limit=200):
        self.move_sampler = move_sampler
        self.depth_limit = max(1, int(depth_limit))

    def evaluate_for_roles(self, game, state, roles, budget_end=None):
        if game.is_terminal(state):
            return {role: float(game.get_goal(state, role)) for role in roles}

        return rollout_scores(
            game,
            state,
            roles,
            move_selector=lambda role, legal_moves: self.move_sampler(role, legal_moves),
            depth_limit=self.depth_limit,
            budget_end=budget_end,
        )

    def evaluate(self, game, state, role, time_limit=None) -> float:
        return float(self.evaluate_for_roles(game, state, [role]).get(role, 0.0))


class ValueNetworkEvaluator(LeafEvaluator):
    """神经价值评估器：非终局用网络，终局用真实分值。"""

    def __init__(self, value_model, encoder, device="cpu"):
        if torch is None:
            raise ImportError("PyTorch is required for ValueNetworkEvaluator.")
        self.value_model = value_model
        self.encoder = encoder
        self.device = device
        self.value_model.eval()

    def evaluate_for_roles(self, game, state, roles, budget_end=None):
        """对多个角色分别编码并推理，返回 role->value。"""
        if game.is_terminal(state):
            return {role: float(game.get_goal(state, role)) for role in roles}

        values = {}
        with torch.no_grad():
            for role in roles:
                x = self.encoder.encode(state, game, role=role)
                t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
                values[role] = float(self.value_model(t).item())
        return values

    def evaluate(self, game, state, role, time_limit=None) -> float:
        return float(self.evaluate_for_roles(game, state, [role]).get(role, 0.0))
