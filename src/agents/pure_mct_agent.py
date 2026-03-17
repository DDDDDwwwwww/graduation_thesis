from __future__ import annotations

from ._mcts_core import _MCTSCoreAgent


class PureMCTAgent(_MCTSCoreAgent):
    """Pure UCT + random rollout baseline."""

    def __init__(
        self,
        name,
        role,
        iterations=400,
        exploration_constant=40.0,
        discount_factor=0.99,
        temperature=10.0,
        simulation_limit=None,
        rollout_depth_limit=200,
        fallback_legal_threshold=None,
        seed=None,
    ):
        super().__init__(
            name=name,
            role=role,
            iterations=iterations,
            exploration_constant=exploration_constant,
            discount_factor=discount_factor,
            temperature=temperature,
            simulation_limit=simulation_limit,
            rollout_depth_limit=rollout_depth_limit,
            fallback_legal_threshold=fallback_legal_threshold,
            seed=seed,
            use_history_sampling=False,
            update_history_on_backprop=False,
        )
