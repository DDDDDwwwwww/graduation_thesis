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

    def _to_model_input(self, encoded):
        """把编码结果转换为模型可直接前向的 batch 输入。"""
        if isinstance(encoded, dict):
            batch = {}
            for key, value in encoded.items():
                t = torch.as_tensor(value, device=self.device)
                if key in {"tile_content_ids", "tile_positions", "mask"}:
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    elif key == "tile_positions" and t.dim() == 2 and t.size(-1) == 2:
                        # 单样本 xy 坐标: [T,2] -> [1,T,2]
                        t = t.unsqueeze(0)
                elif key == "global_features":
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    t = t.to(dtype=torch.float32)
                batch[key] = t
            return batch
        return torch.tensor(encoded, dtype=torch.float32, device=self.device).unsqueeze(0)

    def evaluate_for_roles(self, game, state, roles, budget_end=None):
        """对多个角色分别编码并推理，返回 role->value。"""
        if game.is_terminal(state):
            return {role: float(game.get_goal(state, role)) for role in roles}

        values = {}
        with torch.no_grad():
            for role in roles:
                encoded = self.encoder.encode(state, game, role=role)
                model_input = self._to_model_input(encoded)
                values[role] = float(self.value_model(model_input).item())
        return values

    def evaluate(self, game, state, role, time_limit=None) -> float:
        return float(self.evaluate_for_roles(game, state, [role]).get(role, 0.0))
