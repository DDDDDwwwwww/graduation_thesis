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


class SelectiveValueEvaluator(LeafEvaluator):
    """选择性神经评估器：按条件决定用神经、cheap fallback 或二者混合。"""

    def __init__(
        self,
        value_evaluator: ValueNetworkEvaluator,
        fallback_evaluator: LeafEvaluator,
        alpha: float = 1.0,
        max_neural_evals_per_move: int | None = None,
        legal_move_threshold: int | None = None,
    ):
        self.value_evaluator = value_evaluator
        self.fallback_evaluator = fallback_evaluator
        self.alpha = min(1.0, max(0.0, float(alpha)))
        self.max_neural_evals_per_move = (
            None
            if max_neural_evals_per_move is None
            else max(0, int(max_neural_evals_per_move))
        )
        self.legal_move_threshold = (
            None if legal_move_threshold is None else max(1, int(legal_move_threshold))
        )
        self._stats = self._new_stats()

    def _new_stats(self) -> dict[str, int]:
        return {
            "eval_calls_total": 0,
            "eval_calls_neural": 0,
            "eval_calls_fallback": 0,
            "eval_calls_mixed": 0,
        }

    def reset_for_new_move(self) -> None:
        self._stats = self._new_stats()

    def get_stats(self) -> dict[str, int]:
        return dict(self._stats)

    def _should_use_neural(self, game, state, roles) -> bool:
        if self.max_neural_evals_per_move is not None:
            if self._stats["eval_calls_neural"] >= self.max_neural_evals_per_move:
                return False
        if self.legal_move_threshold is not None:
            total_legal = sum(len(game.get_legal_moves(state, role)) for role in roles)
            if total_legal > self.legal_move_threshold:
                return False
        return True

    def evaluate_for_roles(self, game, state, roles, budget_end=None):
        if game.is_terminal(state):
            return {role: float(game.get_goal(state, role)) for role in roles}

        self._stats["eval_calls_total"] += 1
        use_neural = self._should_use_neural(game, state, roles)

        fallback_values = None
        if not use_neural:
            self._stats["eval_calls_fallback"] += 1
            return self.fallback_evaluator.evaluate_for_roles(game, state, roles, budget_end=budget_end)

        neural_values = self.value_evaluator.evaluate_for_roles(game, state, roles, budget_end=budget_end)
        self._stats["eval_calls_neural"] += 1
        if self.alpha >= 1.0:
            return neural_values

        fallback_values = self.fallback_evaluator.evaluate_for_roles(game, state, roles, budget_end=budget_end)
        self._stats["eval_calls_fallback"] += 1
        self._stats["eval_calls_mixed"] += 1
        return {
            role: (1.0 - self.alpha) * float(fallback_values.get(role, 0.0))
            + self.alpha * float(neural_values.get(role, 0.0))
            for role in roles
        }

    def evaluate(self, game, state, role, time_limit=None) -> float:
        return float(self.evaluate_for_roles(game, state, [role]).get(role, 0.0))
