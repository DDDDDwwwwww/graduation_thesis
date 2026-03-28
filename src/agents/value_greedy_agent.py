from __future__ import annotations

"""价值贪心智能体。

该智能体不做树搜索，只做“一步前瞻”：
枚举当前动作 -> 计算后继状态 -> 用价值网络打分 -> 选分最高动作。
"""

import random

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .base_agent import BaseAgent


class ValueGreedyAgent(BaseAgent):
    """使用价值网络进行一步贪心决策的智能体。"""

    def __init__(
        self,
        name,
        role,
        value_model,
        encoder,
        device="cpu",
        seed=None,
        debug=False,
    ):
        """初始化推理所需模型与编码器。"""
        if torch is None:
            raise ImportError("PyTorch is required for ValueGreedyAgent.")
        super().__init__(name, role)
        self.value_model = value_model.to(device)
        self.encoder = encoder
        self.device = device
        self.debug = bool(debug)
        self._rng = random.Random(seed)
        self.value_model.eval()

    @classmethod
    def from_artifacts(
        cls,
        name,
        role,
        model_path,
        vocab_path=None,
        encoder_config_path=None,
        device="cpu",
        seed=None,
        debug=False,
    ):
        """从训练产物（模型/词汇表/编码器配置）快速构造智能体。"""
        from nn.inference import load_value_artifacts

        value_model, encoder, _ = load_value_artifacts(
            model_path=model_path,
            vocab_path=vocab_path,
            encoder_config_path=encoder_config_path,
            device=device,
        )
        return cls(
            name=name,
            role=role,
            value_model=value_model,
            encoder=encoder,
            device=device,
            seed=seed,
            debug=debug,
        )

    def _build_joint_move(self, game, action):
        """构造联合动作：本方用给定动作，其余角色默认 `noop`。"""
        roles = [str(r) for r in game.get_roles()]
        joint = {r: "noop" for r in roles}
        joint[self.role] = action
        return joint

    def _score_action(self, game, state, action):
        """计算某个候选动作的后继状态价值。"""
        joint = self._build_joint_move(game, action)
        next_state = game.get_next_state(state, joint)
        x = self.encoder.encode(next_state, game, role=self.role)
        model_input = self._to_model_input(x)
        with torch.no_grad():
            return float(self.value_model(model_input).item())

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

    def select_action(self, game, state, legal_actions, time_limit=None):
        """遍历全部合法动作并返回价值最高者。"""
        if not legal_actions:
            return None
        if len(legal_actions) == 1:
            return legal_actions[0]

        scored = []
        for action in legal_actions:
            try:
                scored.append((self._score_action(game, state, action), action))
            except Exception:
                continue

        if not scored:
            return self._rng.choice(legal_actions)

        # 固定排序规则：先按分数降序，再按动作字符串升序，提升确定性。
        scored.sort(key=lambda item: (-item[0], str(item[1])))
        best_score = scored[0][0]
        ties = [a for s, a in scored if s == best_score]
        chosen = ties[0] if len(ties) == 1 else self._rng.choice(ties)

        # 调试模式打印每个动作的预测分值，便于分析模型行为。
        if self.debug:
            print(f"[ValueGreedy][{self.role}] action_scores=" + ", ".join(f"{a}:{s:.4f}" for s, a in scored))
        return chosen
