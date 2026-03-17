from __future__ import annotations

import random

from .rollout import rollout_scores


class LeafEvaluator:
    """Abstract leaf evaluator."""

    def evaluate(self, game, state, role, time_limit=None) -> float:
        raise NotImplementedError

    def evaluate_for_roles(self, game, state, roles, budget_end=None) -> dict:
        raise NotImplementedError


class RandomRolloutEvaluator(LeafEvaluator):
    def __init__(self, depth_limit=200, rng=None):
        self.depth_limit = max(1, int(depth_limit))
        self.rng = rng or random

    def evaluate_for_roles(self, game, state, roles, budget_end=None):
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
