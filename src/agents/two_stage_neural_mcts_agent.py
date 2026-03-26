from __future__ import annotations

"""Two-stage neural MCTS agent with fast-slow value gating."""

from mcts.evaluators import TwoStageValueEvaluator

from ._mcts_core import _MCTSCoreAgent


class TwoStageNeuralMCTSAgent(_MCTSCoreAgent):
    def __init__(
        self,
        name,
        role,
        fast_value_model,
        fast_encoder,
        slow_value_model,
        slow_encoder,
        iterations=200,
        exploration_c=1.4,
        discount_factor=0.99,
        device="cpu",
        uncertainty_type="variance_head",
        gate_type="combined",
        tau=0.15,
        visit_threshold=4,
        slow_budget_per_move=16,
        seed=None,
        fallback_legal_threshold=None,
    ):
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
        self.evaluator = TwoStageValueEvaluator(
            fast_value_model=fast_value_model,
            fast_encoder=fast_encoder,
            slow_value_model=slow_value_model,
            slow_encoder=slow_encoder,
            device=device,
            uncertainty_type=uncertainty_type,
            gate_type=gate_type,
            tau=tau,
            visit_threshold=visit_threshold,
            slow_budget_per_move=slow_budget_per_move,
        )
        self._last_decision_diagnostics = {}

    @classmethod
    def from_artifacts(
        cls,
        name,
        role,
        fast_model_path,
        fast_encoder_config_path,
        slow_model_path,
        slow_encoder_config_path,
        fast_vocab_path=None,
        slow_vocab_path=None,
        iterations=200,
        exploration_c=1.4,
        discount_factor=0.99,
        device="cpu",
        uncertainty_type="variance_head",
        gate_type="combined",
        tau=0.15,
        visit_threshold=4,
        slow_budget_per_move=16,
        seed=None,
        fallback_legal_threshold=None,
    ):
        from nn.inference import load_value_artifacts

        fast_model, fast_encoder, _ = load_value_artifacts(
            model_path=fast_model_path,
            vocab_path=fast_vocab_path,
            encoder_config_path=fast_encoder_config_path,
            device=device,
        )
        slow_model, slow_encoder, _ = load_value_artifacts(
            model_path=slow_model_path,
            vocab_path=slow_vocab_path,
            encoder_config_path=slow_encoder_config_path,
            device=device,
        )
        return cls(
            name=name,
            role=role,
            fast_value_model=fast_model,
            fast_encoder=fast_encoder,
            slow_value_model=slow_model,
            slow_encoder=slow_encoder,
            iterations=iterations,
            exploration_c=exploration_c,
            discount_factor=discount_factor,
            device=device,
            uncertainty_type=uncertainty_type,
            gate_type=gate_type,
            tau=tau,
            visit_threshold=visit_threshold,
            slow_budget_per_move=slow_budget_per_move,
            seed=seed,
            fallback_legal_threshold=fallback_legal_threshold,
        )

    def select_action(self, game, state, legal_actions, time_limit=None):
        action = super().select_action(game, state, legal_actions, time_limit=time_limit)
        self._last_decision_diagnostics = self.evaluator.get_move_stats()
        return action

    def consume_last_decision_diagnostics(self) -> dict:
        out = dict(self._last_decision_diagnostics)
        self._last_decision_diagnostics = {}
        return out
