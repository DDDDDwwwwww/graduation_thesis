from __future__ import annotations

"""智能体抽象基类。

该文件定义所有智能体应遵循的统一接口，`GameRunner` 和实验脚本都依赖这组方法。
"""


class BaseAgent:
    """所有智能体的公共接口。"""

    def __init__(self, name: str, role: str):
        """初始化智能体基础信息。

        Args:
            name: 智能体名称，用于日志显示。
            role: 智能体所扮演的角色名。
        """
        self.name = name
        self.role = role

    def meta_game(self, game, role, rules, startclock, playclock):
        """可选的开局前准备阶段（当前默认不做任何事情）。"""
        return None

    def select_action(self, game, state, legal_actions, time_limit=None):
        """根据当前状态与合法动作选择一个动作。子类必须实现。"""
        raise NotImplementedError("Subclasses must implement select_action")

    def select_move(self, game_machine, state, time_limit=None):
        """兼容旧接口：先取本角色合法动作，再调用 `select_action`。"""
        legal_moves = game_machine.get_legal_moves(state, self.role)
        return self.select_action(game_machine, state, legal_moves, time_limit=time_limit)

    def cleanup(self):
        """可选的清理阶段（例如释放模型、缓存等）。"""
        return None

    def __str__(self):
        """打印时显示“名称(角色)”格式，便于比赛日志阅读。"""
        return f"{self.name} ({self.role})"
