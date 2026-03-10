import math
import random
import time


class Agent:
    """
    智能体基类接口。
    所有具体算法智能体都需要实现 select_move。
    """

    def __init__(self, name, role):
        self.name = name
        self.role = role

    def select_move(self, game_machine, state, time_limit=None):
        raise NotImplementedError("Subclasses must implement select_move")

    def __str__(self):
        return f"{self.name} ({self.role})"


class RandomAgent(Agent):
    """
    随机基线智能体。
    """

    def select_move(self, game_machine, state, time_limit=None):
        legal_moves = game_machine.get_legal_moves(state, self.role)
        if not legal_moves:
            return None
        return random.choice(legal_moves)


class MCTSAgent(Agent):
    """
    基于 UCT 的蒙特卡洛树搜索智能体。

    实现要点：
    1. 选择：UCT，未探索动作优先。
    2. 扩展：每次迭代只向树中新增首个新节点。
    3. 模拟：基于历史启发 Q_h 的默认策略（Gibbs 采样）。
    4. 回传：按折扣因子 gamma 逐层回传，并更新局部统计和全局历史统计。
    """

    class Node:
        """
        MCTS 树节点。

        属性说明：
        - state: 状态机返回的状态（fact 列表）
        - state_key: 状态的可哈希规范化表示
        - parent: 父节点
        - children: joint action -> child node
        - visits: 节点访问次数
        - action_stats: 每个角色在该节点上的动作统计
            action_stats[role][move_key] = {"visits": int, "value_sum": float}
        """

        def __init__(self, state, state_key, parent=None):
            self.state = state
            self.state_key = state_key
            self.parent = parent
            self.children = {}
            self.visits = 0
            self.action_stats = {}

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
        """
        参数说明：
        - iterations: 每次决策的最大迭代次数
        - exploration_constant: UCT 探索常数 C
        - discount_factor: 回传折扣因子 gamma
        - temperature: Gibbs 采样温度 tau
        - simulation_limit: 兼容旧参数名，若给定会覆盖 iterations
        - rollout_depth_limit: rollout 最大深度
        - seed: 随机种子
        """
        super().__init__(name, role)

        if simulation_limit is not None:
            iterations = simulation_limit

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

        if seed is not None:
            random.seed(seed)

        # 跨回合复用的搜索树根
        self.root = None

        # 全局历史启发统计：history_stats[role][move_key] = {"visits", "value_sum"}
        self.history_stats = {}

    def select_move(self, game_machine, state, time_limit=None):
        """
        与 game_runner 兼容的主入口。
        """
        return self.act(game_machine, state, time_limit=time_limit)

    def act(self, game_machine, game_state, time_limit=None):
        """
        执行 MCTS 搜索并返回当前角色动作。
        """
        state = game_state
        legal_moves = game_machine.get_legal_moves(state, self.role)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]
        if (
            self.fallback_legal_threshold is not None
            and len(legal_moves) > self.fallback_legal_threshold
        ):
            # Avoid expensive state-expansion when branching factor explodes.
            move_keys = [self._move_key(m) for m in legal_moves]
            chosen_key = self._sample_by_history(self.role, move_keys)
            action = self._action_from_key(legal_moves, chosen_key)
            return action if action is not None else random.choice(legal_moves)

        roles = [str(r) for r in game_machine.get_roles()]
        if self.role not in roles:
            raise ValueError(
                f"Agent role '{self.role}' is not defined in game roles: {roles}"
            )
        for role in roles:
            self.history_stats.setdefault(role, {})

        root = self._prepare_root(state)

        start_time = time.time()
        budget_end = None
        if time_limit is not None:
            budget_end = start_time + max(0.0, float(time_limit))

        for _ in range(self.iterations):
            if budget_end is not None and time.time() >= budget_end:
                break
            finished = self._run_single_iteration(game_machine, root, roles, budget_end=budget_end)
            if not finished:
                break

        best_key = self._best_action_from_node(root, self.role, legal_moves)
        if best_key is None:
            return random.choice(legal_moves)

        action = self._action_from_key(legal_moves, best_key)
        if action is None:
            return random.choice(legal_moves)
        return action

    def _run_single_iteration(self, game_machine, root, roles, budget_end=None):
        """
        执行一次 Selection -> Expansion -> Simulation -> Backpropagation。
        """
        node = root
        path = []

        # 1) Selection + 2) Expansion（仅扩展首个新节点）
        while True:
            if budget_end is not None and time.time() >= budget_end:
                return False
            if game_machine.is_terminal(node.state):
                leaf_node = node
                break

            chosen_actions = self._select_joint_actions(game_machine, node, roles)
            if not chosen_actions:
                # Defensive handling for malformed states where no role has legal moves.
                # Treat current node as leaf so iteration can still finish safely.
                leaf_node = node
                break
            joint_key = self._joint_key(chosen_actions, roles)
            path.append((node, chosen_actions))

            if joint_key in node.children:
                node = node.children[joint_key]
                continue

            # 扩展一个新子节点
            moves = {r: chosen_actions[r]["obj"] for r in chosen_actions}
            next_state = game_machine.get_next_state(node.state, moves)
            child = self.Node(next_state, self._state_key(next_state), parent=node)
            node.children[joint_key] = child
            leaf_node = child
            break

        # 3) Simulation
        terminal_values = self._simulate(game_machine, leaf_node.state, roles, budget_end=budget_end)

        # 4) Backpropagation
        self._backpropagate(path, leaf_node, terminal_values)
        return True

    def _select_joint_actions(self, game_machine, node, roles):
        """
        在当前节点为每个角色选择一个动作，组合成 joint action。

        规则：
        - 若有未探索动作：优先从未探索动作中按 Q_h 做 Gibbs 采样。
        - 否则：按 UCT 最大值选择。
        """
        chosen = {}

        for role in roles:
            legal_moves = game_machine.get_legal_moves(node.state, role)
            if not legal_moves:
                continue

            move_keys = [self._move_key(m) for m in legal_moves]
            role_stats = node.action_stats.setdefault(role, {})

            unexplored = [
                k
                for k in move_keys
                if k not in role_stats or role_stats[k]["visits"] == 0
            ]

            if unexplored:
                chosen_key = self._sample_by_history(role, unexplored)
            else:
                chosen_key = self._select_by_uct(node, role, move_keys)

            chosen_obj = self._action_from_key(legal_moves, chosen_key)
            if chosen_obj is None:
                chosen_obj = random.choice(legal_moves)
                chosen_key = self._move_key(chosen_obj)

            chosen[role] = {"key": chosen_key, "obj": chosen_obj}

        return chosen

    def _simulate(self, game_machine, state, roles, budget_end=None):
        """
        从叶节点状态进行 rollout。

        默认策略：
        - 按 Q_h 做 Gibbs 采样选择动作。
        - 若采样异常则回退随机。
        """
        current_state = state
        depth = 0

        while depth < self.rollout_depth_limit and not game_machine.is_terminal(current_state):
            if budget_end is not None and time.time() >= budget_end:
                break
            joint_moves = {}
            for role in roles:
                legal_moves = game_machine.get_legal_moves(current_state, role)
                if not legal_moves:
                    continue

                move_keys = [self._move_key(m) for m in legal_moves]
                chosen_key = self._sample_by_history(role, move_keys)
                chosen_obj = self._action_from_key(legal_moves, chosen_key)
                if chosen_obj is None:
                    chosen_obj = random.choice(legal_moves)
                joint_moves[role] = chosen_obj

            if not joint_moves:
                break

            current_state = game_machine.get_next_state(current_state, joint_moves)
            depth += 1

        values = {}
        for role in roles:
            values[role] = float(game_machine.get_goal(current_state, role))
        return values

    def _backpropagate(self, path, leaf_node, terminal_values):
        """
        从叶到根回传。

        回传规则：
        - 每层先做 q <- gamma * q
        - 更新父节点访问计数
        - 更新该层 joint action 中每个角色动作统计
        - 同步更新全局历史统计 Q_h
        """
        q_values = dict(terminal_values)

        for node, chosen_actions in reversed(path):
            for role in q_values:
                q_values[role] *= self.discount_factor

            node.visits += 1

            for role, payload in chosen_actions.items():
                move_key = payload["key"]
                reward = float(q_values.get(role, 0.0))

                role_stats = node.action_stats.setdefault(role, {})
                entry = role_stats.setdefault(move_key, {"visits": 0, "value_sum": 0.0})
                entry["visits"] += 1
                entry["value_sum"] += reward

                history_role = self.history_stats.setdefault(role, {})
                h_entry = history_role.setdefault(move_key, {"visits": 0, "value_sum": 0.0})
                h_entry["visits"] += 1
                h_entry["value_sum"] += reward

        # 叶节点也计访问，确保树统计闭合
        leaf_node.visits += 1

    def _prepare_root(self, state):
        """
        根据当前真实状态重定根：
        - 根一致：复用
        - 根子节点可匹配：提升该子节点为新根
        - 否则：新建根
        """
        key = self._state_key(state)

        if self.root is None:
            self.root = self.Node(state, key)
            return self.root

        if self.root.state_key == key:
            self.root.state = state
            return self.root

        for child in self.root.children.values():
            if child.state_key == key:
                child.parent = None
                self.root = child
                self.root.state = state
                return self.root

        self.root = self.Node(state, key)
        return self.root

    def _best_action_from_node(self, node, role, legal_moves):
        """
        在根节点按“访问次数优先，均值次之”选动作。
        """
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
            cand = (stat["visits"], avg, key)
            if best is None or cand > best:
                best = cand

        return None if best is None else best[2]

    def _select_by_uct(self, node, role, move_keys):
        """
        UCT 选择。
        """
        role_stats = node.action_stats.get(role, {})
        parent_visits = max(1, node.visits)

        best_key = None
        best_score = None

        for key in move_keys:
            stat = role_stats.get(key)
            if stat is None or stat["visits"] <= 0:
                return key

            avg = stat["value_sum"] / stat["visits"]
            bonus = self.exploration_constant * math.sqrt(
                math.log(parent_visits + 1e-10) / stat["visits"]
            )
            score = avg + bonus

            if best_score is None or score > best_score:
                best_score = score
                best_key = key

        return best_key

    def _sample_by_history(self, role, move_keys):
        """
        使用历史启发 Q_h 进行 Gibbs 采样。

        未见动作默认 Q_h = 100，用于偏置早期探索。
        """
        if not move_keys:
            return None

        role_hist = self.history_stats.setdefault(role, {})
        q_values = []

        for key in move_keys:
            stat = role_hist.get(key)
            if stat is None or stat["visits"] <= 0:
                q_values.append(100.0)
            else:
                q_values.append(stat["value_sum"] / stat["visits"])

        scaled = [q / self.temperature for q in q_values]
        max_scaled = max(scaled)
        weights = [math.exp(v - max_scaled) for v in scaled]
        total = sum(weights)

        if total <= 0:
            return random.choice(move_keys)

        rnd = random.random() * total
        acc = 0.0
        for key, w in zip(move_keys, weights):
            acc += w
            if acc >= rnd:
                return key

        return move_keys[-1]

    def _state_key(self, state):
        """
        规范化状态键，避免同一状态因列表顺序不同而重复建树。
        """
        return tuple(sorted(str(fact) for fact in state))

    def _joint_key(self, chosen_actions, roles):
        """
        将 joint action 规范化为可哈希键。
        """
        return tuple((role, chosen_actions[role]["key"]) for role in roles if role in chosen_actions)

    def _move_key(self, move):
        return str(move)

    def _action_from_key(self, legal_moves, move_key):
        for move in legal_moves:
            if self._move_key(move) == move_key:
                return move
        return None
