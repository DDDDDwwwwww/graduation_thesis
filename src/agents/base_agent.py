from __future__ import annotations


class BaseAgent:
    """Common agent interface used by the game runner and experiment scripts."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def meta_game(self, game, role, rules, startclock, playclock):
        return None

    def select_action(self, game, state, legal_actions, time_limit=None):
        raise NotImplementedError("Subclasses must implement select_action")

    def select_move(self, game_machine, state, time_limit=None):
        legal_moves = game_machine.get_legal_moves(state, self.role)
        return self.select_action(game_machine, state, legal_moves, time_limit=time_limit)

    def cleanup(self):
        return None

    def __str__(self):
        return f"{self.name} ({self.role})"
