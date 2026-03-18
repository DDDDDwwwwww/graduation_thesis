"""智能体兼容导出层。

历史代码通常使用 `from ggp_agent import ...` 导入智能体。
当前项目已把实现迁移到 `src/agents/`，这里保留重导出以兼容旧脚本。
"""

from agents.base_agent import BaseAgent
from agents.heuristic_mcts_agent import HeuristicMCTSAgent
from agents.neural_value_mcts_agent import NeuralValueMCTSAgent
from agents.pure_mct_agent import PureMCTAgent
from agents.random_agent import RandomAgent
from agents.value_greedy_agent import ValueGreedyAgent

# 兼容历史命名：旧代码可能仍引用 Agent / MCTSAgent。
Agent = BaseAgent
MCTSAgent = HeuristicMCTSAgent

__all__ = [
    "Agent",
    "BaseAgent",
    "RandomAgent",
    "PureMCTAgent",
    "HeuristicMCTSAgent",
    "ValueGreedyAgent",
    "NeuralValueMCTSAgent",
    "MCTSAgent",
]
