from __future__ import annotations

"""神经价值 MCTS 智能体。

在统一 MCTS 核心流程中，把叶子评估器替换为价值网络评估器。
"""

from mcts.evaluators import ValueNetworkEvaluator

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
        # 当前里程碑仅实现纯 value 模式，不混入 rollout。
        if self.evaluator_mode != "value":
            raise ValueError(f"Unsupported evaluator_mode: {self.evaluator_mode}")
        self.evaluator = ValueNetworkEvaluator(value_model=value_model.to(device), encoder=encoder, device=device)

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
        )
