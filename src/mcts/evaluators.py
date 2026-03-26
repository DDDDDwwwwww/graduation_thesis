from __future__ import annotations

"""叶子评估器抽象与实现。

该模块把“如何给叶子状态打分”从 MCTS 主循环中解耦出来，便于替换评估方式。
"""

import random
import time
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .rollout import rollout_scores


class LeafEvaluator:
    """叶子评估器抽象基类。"""

    def evaluate(self, game, state, role, time_limit=None, **kwargs) -> float:
        raise NotImplementedError

    def evaluate_for_roles(self, game, state, roles, budget_end=None, **kwargs) -> dict:
        raise NotImplementedError


class RandomRolloutEvaluator(LeafEvaluator):
    """随机 rollout 评估器。"""

    def __init__(self, depth_limit=200, rng=None):
        self.depth_limit = max(1, int(depth_limit))
        self.rng = rng or random

    def evaluate_for_roles(self, game, state, roles, budget_end=None, **kwargs):
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

    def evaluate(self, game, state, role, time_limit=None, **kwargs) -> float:
        return float(self.evaluate_for_roles(game, state, [role]).get(role, 0.0))


class HeuristicRolloutEvaluator(LeafEvaluator):
    """带启发式动作采样器的 rollout 评估器。"""

    def __init__(self, move_sampler, depth_limit=200):
        self.move_sampler = move_sampler
        self.depth_limit = max(1, int(depth_limit))

    def evaluate_for_roles(self, game, state, roles, budget_end=None, **kwargs):
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

    def evaluate(self, game, state, role, time_limit=None, **kwargs) -> float:
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

    def _extract_value_scalar(self, model_output) -> float:
        if isinstance(model_output, dict):
            return float(model_output["value"].item())
        return float(model_output.item())

    def evaluate_for_roles(self, game, state, roles, budget_end=None, **kwargs):
        """对多个角色分别编码并推理，返回 role->value。"""
        if game.is_terminal(state):
            return {role: float(game.get_goal(state, role)) for role in roles}

        values = {}
        with torch.no_grad():
            for role in roles:
                encoded = self.encoder.encode(state, game, role=role)
                model_input = self._to_model_input(encoded)
                values[role] = self._extract_value_scalar(self.value_model(model_input))
        return values

    def evaluate(self, game, state, role, time_limit=None, **kwargs) -> float:
        return float(self.evaluate_for_roles(game, state, [role]).get(role, 0.0))


class TwoStageValueEvaluator(LeafEvaluator):
    """Fast-slow value evaluator with uncertainty/visit/budget gating."""

    def __init__(
        self,
        fast_value_model,
        fast_encoder,
        slow_value_model,
        slow_encoder,
        device="cpu",
        uncertainty_type="variance_head",
        gate_type="combined",
        tau=0.15,
        visit_threshold=4,
        slow_budget_per_move=16,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for TwoStageValueEvaluator.")
        self.fast_value_model = fast_value_model.to(device)
        self.fast_encoder = fast_encoder
        self.slow_value_model = slow_value_model.to(device)
        self.slow_encoder = slow_encoder
        self.device = device
        self.uncertainty_type = str(uncertainty_type)
        self.gate_type = str(gate_type)
        self.tau = float(tau)
        self.visit_threshold = int(visit_threshold)
        self.slow_budget_per_move = int(slow_budget_per_move)

        self.fast_value_model.eval()
        self.slow_value_model.eval()
        self._move_stats = {}
        self._totals = {}
        self.start_move()

    def start_move(self):
        self._move_stats = {
            "leaf_evaluations": 0,
            "fast_calls": 0,
            "slow_calls": 0,
            "fast_time_sec": 0.0,
            "slow_time_sec": 0.0,
            "slow_budget_used": 0,
        }

    def _to_model_input(self, encoded):
        if isinstance(encoded, dict):
            batch = {}
            for key, value in encoded.items():
                t = torch.as_tensor(value, device=self.device)
                if key in {"tile_content_ids", "tile_positions", "mask"}:
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    elif key == "tile_positions" and t.dim() == 2 and t.size(-1) == 2:
                        t = t.unsqueeze(0)
                elif key == "global_features":
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    t = t.to(dtype=torch.float32)
                batch[key] = t
            return batch
        return torch.tensor(encoded, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _fast_forward(self, state, game, role):
        encoded = self.fast_encoder.encode(state, game, role=role)
        model_input = self._to_model_input(encoded)
        t0 = time.perf_counter()
        out = self.fast_value_model(model_input)
        self._move_stats["fast_time_sec"] += time.perf_counter() - t0
        self._move_stats["fast_calls"] += 1

        if isinstance(out, dict):
            v_fast = float(out["value"].item())
            variance = out.get("variance")
            variance_value = None if variance is None else float(variance.item())
        else:
            v_fast = float(out.item())
            variance_value = None

        if self.uncertainty_type == "margin":
            u_fast = max(0.0, 1.0 - abs(v_fast))
        elif self.uncertainty_type == "variance_head" and variance_value is not None:
            u_fast = variance_value
        else:
            u_fast = max(0.0, 1.0 - abs(v_fast))
        return v_fast, u_fast

    def _slow_forward(self, state, game, role):
        encoded = self.slow_encoder.encode(state, game, role=role)
        model_input = self._to_model_input(encoded)
        t0 = time.perf_counter()
        out = self.slow_value_model(model_input)
        self._move_stats["slow_time_sec"] += time.perf_counter() - t0
        self._move_stats["slow_calls"] += 1
        self._move_stats["slow_budget_used"] += 1
        if isinstance(out, dict):
            return float(out["value"].item())
        return float(out.item())

    def _need_slow(self, u_fast: float, node_visit_count: int) -> bool:
        uncertainty_ok = u_fast > self.tau
        visit_ok = int(node_visit_count) >= self.visit_threshold
        budget_ok = self._move_stats["slow_budget_used"] < self.slow_budget_per_move

        if self.gate_type == "uncertainty":
            return uncertainty_ok
        if self.gate_type == "visit":
            return visit_ok
        if self.gate_type == "budget":
            return budget_ok
        if self.gate_type == "uncertainty_visit":
            return uncertainty_ok and visit_ok
        if self.gate_type == "none":
            return True
        return uncertainty_ok and visit_ok and budget_ok

    def evaluate_for_roles(self, game, state, roles, budget_end=None, **kwargs):
        if game.is_terminal(state):
            return {role: float(game.get_goal(state, role)) for role in roles}

        node_visit_count = int(kwargs.get("node_visit_count", 0))
        values = {}
        with torch.no_grad():
            for role in roles:
                self._move_stats["leaf_evaluations"] += 1
                v_fast, u_fast = self._fast_forward(state, game, role)
                use_slow = self._need_slow(u_fast=u_fast, node_visit_count=node_visit_count)
                values[role] = self._slow_forward(state, game, role) if use_slow else v_fast
        return values

    def evaluate(self, game, state, role, time_limit=None, **kwargs) -> float:
        return float(self.evaluate_for_roles(game, state, [role], **kwargs).get(role, 0.0))

    def get_move_stats(self) -> dict:
        leaf = max(1, int(self._move_stats["leaf_evaluations"]))
        slow_calls = int(self._move_stats["slow_calls"])
        fast_calls = int(self._move_stats["fast_calls"])
        out = dict(self._move_stats)
        out["slow_call_ratio"] = (slow_calls / max(1, fast_calls))
        out["avg_fast_time_sec"] = float(self._move_stats["fast_time_sec"]) / leaf
        out["avg_slow_time_sec"] = float(self._move_stats["slow_time_sec"]) / max(1, slow_calls)
        return out
