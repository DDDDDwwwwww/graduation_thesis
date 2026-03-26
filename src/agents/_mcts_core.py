from __future__ import annotations

"""可复用的 MCTS 主循环实现。

PureMCT / HeuristicMCTS / NeuralValueMCTS 都可复用这套“选择-扩展-评估-回传”流程。
"""

import random
import time

from mcts.evaluators import HeuristicRolloutEvaluator, RandomRolloutEvaluator
from mcts.selectors import gibbs_sample, select_by_uct
from mcts.tree_node import TreeNode

from .base_agent import BaseAgent


class _MCTSCoreAgent(BaseAgent):
    """MCTS 核心模板类。

    该类通过参数控制策略差异：
    - `use_history_sampling=False`：更接近纯 UCT。
    - `use_history_sampling=True`：使用历史统计做 softmax 采样。
    - `self.evaluator` 可在子类里替换为神经网络评估器。
    """

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
        use_history_sampling=True,
        update_history_on_backprop=True,
    ):
        super().__init__(name, role)

        if simulation_limit is not None:
            iterations = simulation_limit

        # 每个智能体使用独立 RNG，避免全局随机状态互相污染。
        self._rng = random.Random(seed)
        self.iterations = max(1, int(iterations))
        self.exploration_constant = float(exploration_constant)
        self.discount_factor = float(discount_factor)
        self.temperature = max(1e-6, float(temperature))
        self.rollout_depth_limit = max(1, int(rollout_depth_limit))
        self.fallback_legal_threshold = (
            None
            if fallback_legal_threshold is None
            else max(1, int(fallback_legal_threshold))
        )
        self.use_history_sampling = bool(use_history_sampling)
        self.update_history_on_backprop = bool(update_history_on_backprop)

        self.root = None
        self.history_stats = {}

        # 评估器默认按是否启用历史采样决定 rollout 形态；神经 MCTS 会覆盖它。
        if self.use_history_sampling:
            self.evaluator = HeuristicRolloutEvaluator(
                move_sampler=self._sample_move_for_rollout,
                depth_limit=self.rollout_depth_limit,
            )
        else:
            self.evaluator = RandomRolloutEvaluator(
                depth_limit=self.rollout_depth_limit,
                rng=self._rng,
            )

    def select_move(self, game_machine, state, time_limit=None):
        """兼容接口：自动获取合法动作后调用 `select_action`。"""
        legal_moves = game_machine.get_legal_moves(state, self.role)
        return self.select_action(game_machine, state, legal_moves, time_limit=time_limit)

    def select_action(self, game_machine, state, legal_actions, time_limit=None):
        """MCTS 决策入口。"""
        legal_moves = legal_actions
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        # 若合法动作数量特别大，可直接退化到策略采样，避免搜索开销过高。
        if (
            self.fallback_legal_threshold is not None
            and len(legal_moves) > self.fallback_legal_threshold
        ):
            move_keys = [self._move_key(m) for m in legal_moves]
            chosen_key = self._sample_by_policy(self.role, move_keys)
            action = self._action_from_key(legal_moves, chosen_key)
            return action if action is not None else self._rng.choice(legal_moves)

        roles = [str(r) for r in game_machine.get_roles()]
        if self.role not in roles:
            raise ValueError(
                f"Agent role '{self.role}' is not defined in game roles: {roles}"
            )

        for role in roles:
            self.history_stats.setdefault(role, {})

        if hasattr(self.evaluator, "start_move"):
            self.evaluator.start_move()

        root = self._prepare_root(state)
        budget_end = None
        if time_limit is not None:
            budget_end = time.time() + max(0.0, float(time_limit))

        # 固定迭代预算 + 可选时间预算并行约束。
        for _ in range(self.iterations):
            if budget_end is not None and time.time() >= budget_end:
                break
            if not self._run_single_iteration(game_machine, root, roles, budget_end):
                break

        # 默认按“访问次数优先，其次均值”从根节点选动作。
        best_key = self._best_action_from_node(root, self.role, legal_moves)
        if best_key is None:
            return self._rng.choice(legal_moves)

        action = self._action_from_key(legal_moves, best_key)
        if action is None:
            return self._rng.choice(legal_moves)
        return action

    def _run_single_iteration(self, game_machine, root, roles, budget_end=None):
        """执行一次完整迭代：选择/扩展 -> 叶评估 -> 反向传播。"""
        node = root
        path = []

        while True:
            if budget_end is not None and time.time() >= budget_end:
                return False

            # 到达终局时直接停止向下并交给评估器返回真实分值。
            if game_machine.is_terminal(node.state):
                node.is_terminal = True
                leaf_node = node
                break

            chosen_actions = self._select_joint_actions(game_machine, node, roles)
            if not chosen_actions:
                leaf_node = node
                break

            joint_key = self._joint_key(chosen_actions, roles)
            path.append((node, chosen_actions))

            # 若该联合动作分支已存在则继续向下走，否则创建新子节点。
            if joint_key in node.children:
                node = node.children[joint_key]
                continue

            moves = {r: chosen_actions[r]["obj"] for r in chosen_actions}
            next_state = game_machine.get_next_state(node.state, moves)
            child = TreeNode(
                state_key=self._state_key(next_state),
                parent=node,
                action_from_parent=joint_key,
                state=next_state,
            )
            node.children[joint_key] = child
            leaf_node = child
            break

        # 评估器统一返回“各角色价值”字典。
        terminal_values = self.evaluator.evaluate_for_roles(
            game_machine,
            leaf_node.state,
            roles,
            budget_end=budget_end,
            node_visit_count=int(getattr(leaf_node, "visits", 0)),
        )
        self._backpropagate(path, leaf_node, terminal_values)
        return True

    def _select_joint_actions(self, game_machine, node, roles):
        """为每个角色各选一个动作，组合成联合动作。"""
        chosen = {}

        for role in roles:
            legal_moves = game_machine.get_legal_moves(node.state, role)
            if not legal_moves:
                continue

            move_keys = [self._move_key(m) for m in legal_moves]
            role_stats = node.action_stats.setdefault(role, {})
            unexplored = [
                key for key in move_keys if key not in role_stats or role_stats[key]["visits"] == 0
            ]

            # 先扩展未探索动作；全探索后使用 UCT。
            if unexplored:
                chosen_key = self._sample_by_policy(role, unexplored)
            else:
                chosen_key = select_by_uct(
                    node,
                    role,
                    move_keys,
                    exploration_constant=self.exploration_constant,
                )

            chosen_obj = self._action_from_key(legal_moves, chosen_key)
            if chosen_obj is None:
                chosen_obj = self._rng.choice(legal_moves)
                chosen_key = self._move_key(chosen_obj)

            chosen[role] = {"key": chosen_key, "obj": chosen_obj}

        return chosen

    def _backpropagate(self, path, leaf_node, terminal_values):
        """沿路径回传价值，同时更新节点统计与可选历史统计。"""
        q_values = dict(terminal_values)

        for node, chosen_actions in reversed(path):
            # 折扣可模拟“越早收益越重要”的偏好。
            for role in q_values:
                q_values[role] *= self.discount_factor

            node.visits += 1
            node.total_value += float(q_values.get(self.role, 0.0))
            node.mean_value = node.total_value / node.visits

            for role, payload in chosen_actions.items():
                move_key = payload["key"]
                reward = float(q_values.get(role, 0.0))

                role_stats = node.action_stats.setdefault(role, {})
                entry = role_stats.setdefault(move_key, {"visits": 0, "value_sum": 0.0})
                entry["visits"] += 1
                entry["value_sum"] += reward

                # 启发式模式下会更新全局历史统计，供后续 softmax 采样使用。
                if self.update_history_on_backprop:
                    hist_role = self.history_stats.setdefault(role, {})
                    hist_entry = hist_role.setdefault(move_key, {"visits": 0, "value_sum": 0.0})
                    hist_entry["visits"] += 1
                    hist_entry["value_sum"] += reward

        leaf_node.update_value(float(terminal_values.get(self.role, 0.0)))

    def _prepare_root(self, state):
        """根节点复用策略：尽量复用原树中可对齐的子树，减少重复搜索。"""
        key = self._state_key(state)

        if self.root is None:
            self.root = TreeNode(state_key=key, state=state)
            return self.root

        if self.root.state_key == key:
            self.root.state = state
            return self.root

        for child in self.root.children.values():
            if child.state_key == key:
                child.parent = None
                child.state = state
                self.root = child
                return self.root

        self.root = TreeNode(state_key=key, state=state)
        return self.root

    def _best_action_from_node(self, node, role, legal_moves):
        """在根节点动作中选择“访问次数最多、均值更高”的动作。"""
        role_stats = node.action_stats.get(role, {})
        if not role_stats:
            return None

        best = None
        for move in legal_moves:
            key = self._move_key(move)
            stat = role_stats.get(key)
            if not stat or stat["visits"] <= 0:
                continue

            avg = stat["value_sum"] / stat["visits"]
            candidate = (stat["visits"], avg, key)
            if best is None or candidate > best:
                best = candidate

        return None if best is None else best[2]

    def _sample_by_policy(self, role, move_keys):
        """动作采样策略：随机或基于历史 Q 的 Gibbs 采样。"""
        if not move_keys:
            return None

        if not self.use_history_sampling:
            return self._rng.choice(move_keys)

        role_hist = self.history_stats.setdefault(role, {})
        q_values = []
        for key in move_keys:
            stat = role_hist.get(key)
            if stat is None or stat["visits"] <= 0:
                q_values.append(100.0)
            else:
                q_values.append(stat["value_sum"] / stat["visits"])

        return gibbs_sample(
            move_keys,
            q_values,
            temperature=self.temperature,
            rng=self._rng,
        )

    def _sample_move_for_rollout(self, role, legal_moves):
        """rollout 期间的单步动作选择。"""
        move_keys = [self._move_key(m) for m in legal_moves]
        chosen_key = self._sample_by_policy(role, move_keys)
        chosen_obj = self._action_from_key(legal_moves, chosen_key)
        if chosen_obj is None:
            chosen_obj = self._rng.choice(legal_moves)
        return chosen_obj

    def _state_key(self, state):
        """将状态标准化为可哈希键，保证顺序无关。"""
        return tuple(sorted(str(fact) for fact in state))

    def _joint_key(self, chosen_actions, roles):
        return tuple(
            (role, chosen_actions[role]["key"])
            for role in roles
            if role in chosen_actions
        )

    def _move_key(self, move):
        return str(move)

    def _action_from_key(self, legal_moves, move_key):
        for move in legal_moves:
            if self._move_key(move) == move_key:
                return move
        return None
