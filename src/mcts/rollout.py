from __future__ import annotations


def rollout_scores(game_machine, start_state, roles, move_selector, depth_limit=200, budget_end=None):
    """Run a rollout and return terminal/leaf goal scores for all roles."""
    current_state = start_state
    depth = 0

    while depth < max(1, int(depth_limit)) and not game_machine.is_terminal(current_state):
        if budget_end is not None:
            import time

            if time.time() >= budget_end:
                break

        joint_moves = {}
        for role in roles:
            legal_moves = game_machine.get_legal_moves(current_state, role)
            if not legal_moves:
                continue
            joint_moves[role] = move_selector(role, legal_moves)

        if not joint_moves:
            break

        current_state = game_machine.get_next_state(current_state, joint_moves)
        depth += 1

    return {role: float(game_machine.get_goal(current_state, role)) for role in roles}
