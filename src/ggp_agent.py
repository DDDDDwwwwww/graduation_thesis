"""Backward-compatible exports for agent classes.

Use `src/agents/` for new implementations. This module remains as a compatibility
shim for existing scripts importing from `ggp_agent`.
"""

from agents.base_agent import BaseAgent
from agents.heuristic_mcts_agent import HeuristicMCTSAgent
from agents.pure_mct_agent import PureMCTAgent
from agents.random_agent import RandomAgent

# Backward compatibility: keep historical base-class name.
Agent = BaseAgent

__all__ = [
    "Agent",
    "BaseAgent",
    "RandomAgent",
    "PureMCTAgent",
    "HeuristicMCTSAgent",
]
