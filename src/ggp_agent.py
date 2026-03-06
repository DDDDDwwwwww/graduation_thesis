import random

class Agent:
    """
    智能体基类 (Interface)。
    所有的算法（Random, MCTS, AlphaZero）都必须继承此类并实现 select_move。
    """
    def __init__(self, name, role):
        self.name = name
        self.role = role

    def select_move(self, game_machine, state, time_limit=None):
        """
        核心接口：给定当前状态，返回一个合法动作。
        
        Args:
            game_machine: 你的 GameStateMachine 实例，用于查询合法动作或模拟。
            state: 当前的游戏状态 (fact list)。
            time_limit: (可选) 思考时间限制，单位秒。对于 MCTS 很重要。
        """
        raise NotImplementedError("Subclasses must implement select_move")

    def __str__(self):
        return f"{self.name} ({self.role})"


class RandomAgent(Agent):
    """
    基准智能体：随机选择动作。
    用于测试流程是否跑通，也是评估高级算法时的 Baseline。
    """
    def select_move(self, game_machine, state, time_limit=None):
        # 1. 获取当前角色的所有合法动作
        legal_moves = game_machine.get_legal_moves(state, self.role)
        
        if not legal_moves:
            return None # 理论上不应发生，除非规则有误
            
        # 2. 随机选一个
        return random.choice(legal_moves)


class MCTSAgent(Agent):
    """
    【预留接口】蒙特卡洛树搜索智能体。
    这是你下一阶段要填充的核心部分。
    """
    def __init__(self, name, role, simulation_limit=100):
        super().__init__(name, role)
        self.simulation_limit = simulation_limit
        # 这里将来可以初始化 MCTS 的树结构

    def select_move(self, game_machine, state, time_limit=None):
        legal_moves = game_machine.get_legal_moves(state, self.role)
        if len(legal_moves) == 1:
            return legal_moves[0] # 如果只有一步可选（如 forced noop），直接返回，省算力

        print(f"[{self.name}] Thinking using MCTS... (Not Implemented Yet)")
        
        # --- 算法接口预留位置 ---
        # TODO: 在第二阶段，我们将在这里实现：
        # 1. Selection
        # 2. Expansion
        # 3. Simulation (Rollout)
        # 4. Backpropagation
        # ----------------------
        
        # 目前暂时用随机代替，防止报错
        return random.choice(legal_moves)