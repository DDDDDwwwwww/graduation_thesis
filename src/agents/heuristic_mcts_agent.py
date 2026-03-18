from __future__ import annotations

"""启发式 MCTS 基线智能体。

特征：UCT 选子 + 历史统计引导的 rollout / 动作采样。
"""

from ._mcts_core import _MCTSCoreAgent


class HeuristicMCTSAgent(_MCTSCoreAgent):
    """带历史先验的 MCTS 基线实现。"""

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
        # 与 PureMCT 的主要差异在于两个 history 相关开关。
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
            use_history_sampling=True,
            update_history_on_backprop=True,
        )
