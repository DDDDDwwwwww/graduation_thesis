from __future__ import annotations

import math
import random


def select_by_uct(node, role, move_keys, exploration_constant):
    """Choose an action key by UCT from node-local action statistics."""
    role_stats = node.action_stats.get(role, {})
    parent_visits = max(1, node.visits)

    best_key = None
    best_score = None

    for key in move_keys:
        stat = role_stats.get(key)
        if stat is None or stat["visits"] <= 0:
            return key

        avg = stat["value_sum"] / stat["visits"]
        bonus = exploration_constant * math.sqrt(
            math.log(parent_visits + 1e-10) / stat["visits"]
        )
        score = avg + bonus

        if best_score is None or score > best_score:
            best_score = score
            best_key = key

    return best_key


def gibbs_sample(move_keys, q_values, temperature, rng=None):
    """Sample an action key with softmax/Gibbs distribution over q_values."""
    if not move_keys:
        return None

    rng = rng or random
    tau = max(1e-6, float(temperature))

    scaled = [q / tau for q in q_values]
    max_scaled = max(scaled)
    weights = [math.exp(v - max_scaled) for v in scaled]
    total = sum(weights)

    if total <= 0:
        return rng.choice(move_keys)

    threshold = rng.random() * total
    running = 0.0
    for key, w in zip(move_keys, weights):
        running += w
        if running >= threshold:
            return key

    return move_keys[-1]
