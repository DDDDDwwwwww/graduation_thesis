from __future__ import annotations

"""神经价值 MCTS 智能体。

在统一 MCTS 核心流程中，把叶子评估器替换为价值网络评估器。
"""

from mcts.evaluators import (
    RandomRolloutEvaluator,
    SelectiveValueEvaluator,
    ValueNetworkEvaluator,
)

from ._mcts_core import _MCTSCoreAgent


class NeuralValueMCTSAgent(_MCTSCoreAgent):
    """使用价值网络做叶子评估的 MCTS 智能体。"""

    def __init__(
        self,
        name,
        role,
        value_model,
        encoder,
        iterations=200,
        exploration_c=1.4,
        discount_factor=0.99,
        device="cpu",
        evaluator_mode="value",
        seed=None,
        fallback_legal_threshold=None,
        selective_max_neural_evals_per_move=None,
        selective_legal_move_threshold=None,
        selective_alpha=1.0,
        selective_rollout_depth_limit=64,
    ):
        """初始化神经 MCTS 参数与评估器。"""
        super().__init__(
            name=name,
            role=role,
            iterations=iterations,
            exploration_constant=exploration_c,
            discount_factor=discount_factor,
            fallback_legal_threshold=fallback_legal_threshold,
            seed=seed,
            use_history_sampling=False,
            update_history_on_backprop=False,
        )
        self.evaluator_mode = evaluator_mode
        if self.evaluator_mode not in {"value", "selective"}:
            raise ValueError(f"Unsupported evaluator_mode: {self.evaluator_mode}")
        self._last_search_stats = {}
        value_eval = ValueNetworkEvaluator(value_model=value_model.to(device), encoder=encoder, device=device)
        if self.evaluator_mode == "value":
            self.evaluator = value_eval
        else:
            self.evaluator = SelectiveValueEvaluator(
                value_evaluator=value_eval,
                fallback_evaluator=RandomRolloutEvaluator(
                    depth_limit=selective_rollout_depth_limit,
                    rng=self._rng,
                ),
                alpha=selective_alpha,
                max_neural_evals_per_move=selective_max_neural_evals_per_move,
                legal_move_threshold=selective_legal_move_threshold,
            )

    @classmethod
    def from_artifacts(
        cls,
        name,
        role,
        model_path,
        vocab_path=None,
        encoder_config_path=None,
        iterations=200,
        exploration_c=1.4,
        discount_factor=0.99,
        device="cpu",
        evaluator_mode="value",
        seed=None,
        fallback_legal_threshold=None,
        selective_max_neural_evals_per_move=None,
        selective_legal_move_threshold=None,
        selective_alpha=1.0,
        selective_rollout_depth_limit=64,
    ):
        """从训练产物快速构造神经 MCTS 智能体。"""
        from nn.inference import load_value_artifacts

        try:
            value_model, encoder, _ = load_value_artifacts(
                model_path=model_path,
                vocab_path=vocab_path,
                encoder_config_path=encoder_config_path,
                device=device,
            )
        except ValueError as exc:
            msg = str(exc)
            if "vocab_path is required for mlp model" in msg:
                raise ValueError(
                    "Loading MLP artifacts requires vocab_path. "
                    "For transformer checkpoints, vocab_path can be omitted."
                ) from exc
            raise
        return cls(
            name=name,
            role=role,
            value_model=value_model,
            encoder=encoder,
            iterations=iterations,
            exploration_c=exploration_c,
            discount_factor=discount_factor,
            device=device,
            evaluator_mode=evaluator_mode,
            seed=seed,
            fallback_legal_threshold=fallback_legal_threshold,
            selective_max_neural_evals_per_move=selective_max_neural_evals_per_move,
            selective_legal_move_threshold=selective_legal_move_threshold,
            selective_alpha=selective_alpha,
            selective_rollout_depth_limit=selective_rollout_depth_limit,
        )

    def select_action(self, game_machine, state, legal_actions, time_limit=None):
        reset_fn = getattr(self.evaluator, "reset_for_new_move", None)
        if callable(reset_fn):
            reset_fn()
        action = super().select_action(game_machine, state, legal_actions, time_limit=time_limit)
        stats_fn = getattr(self.evaluator, "get_stats", None)
        if callable(stats_fn):
            self._last_search_stats = stats_fn()
        else:
            self._last_search_stats = {}
        return action

    def get_last_search_stats(self):
        return dict(self._last_search_stats)
