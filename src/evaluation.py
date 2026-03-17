import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

from ggp_agent import HeuristicMCTSAgent, PureMCTAgent, RandomAgent
from ggp_statemachine import GameStateMachine


AGENT_TYPES = ("random", "pure_mct", "mcts")


@dataclass
class EvalConfig:
    games: List[str]
    repeats: int
    move_time_limit: float
    step_limit: int
    base_seed: int
    output_dir: str
    mcts_iterations: int
    mcts_rollout_depth: int
    mcts_exploration_constant: float
    mcts_fallback_legal_threshold: int
    cache_modes: List[str]


def create_agent(agent_type: str, role: str, config: EvalConfig):
    name = f"{agent_type}_{role}"
    if agent_type == "random":
        return RandomAgent(name, role)
    if agent_type == "pure_mct":
        return PureMCTAgent(
            name,
            role,
            iterations=config.mcts_iterations,
            rollout_depth_limit=config.mcts_rollout_depth,
            exploration_constant=config.mcts_exploration_constant,
            fallback_legal_threshold=config.mcts_fallback_legal_threshold,
        )
    if agent_type == "mcts":
        return HeuristicMCTSAgent(
            name,
            role,
            iterations=config.mcts_iterations,
            rollout_depth_limit=config.mcts_rollout_depth,
            exploration_constant=config.mcts_exploration_constant,
            fallback_legal_threshold=config.mcts_fallback_legal_threshold,
        )
    raise ValueError(f"Unsupported agent type: {agent_type}")


def rotate_lineup(lineup: Tuple[str, ...], shift: int) -> Tuple[str, ...]:
    n = len(lineup)
    if n == 0:
        return lineup
    k = shift % n
    return lineup[k:] + lineup[:k]


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def normalize_cache_modes(raw_modes: List[str]) -> List[str]:
    # 允许用户用 both 一次性展开为 A/B 两组。
    expanded = []
    for mode in raw_modes:
        if mode == "both":
            expanded.extend(["enabled", "disabled"])
        else:
            expanded.append(mode)
    # 去重并保持顺序，避免重复跑相同组别。
    return list(dict.fromkeys(expanded))


def evaluate_single_match(
    game_file: str,
    roles: List[str],
    lineup: Tuple[str, ...],
    seed: int,
    match_id: int,
    repeat_id: int,
    cache_mode: str,
    config: EvalConfig,
):
    random.seed(seed)
    # A/B 对照开关：enabled=使用缓存，disabled=禁用缓存。
    game = GameStateMachine(game_file, cache_enabled=(cache_mode == "enabled"))
    game.clear_caches()
    game.reset_perf_stats()

    agents: Dict[str, object] = {}
    for role, agent_type in zip(roles, lineup):
        agents[role] = create_agent(agent_type, role, config)

    decision_time_sum = {role: 0.0 for role in roles}
    decision_time_max = {role: 0.0 for role in roles}
    decision_count = {role: 0 for role in roles}
    legal_calls_by_role = {role: 0 for role in roles}
    legal_hits_by_role = {role: 0 for role in roles}
    next_calls_by_role = {role: 0 for role in roles}
    next_hits_by_role = {role: 0 for role in roles}

    state = game.get_initial_state()
    steps = 0
    failed = 0
    error_msg = ""
    wall_start = time.time()

    try:
        while not game.is_terminal(state):
            if steps >= config.step_limit:
                raise RuntimeError(f"Step limit exceeded: {config.step_limit}")

            steps += 1
            moves = {}
            for role in roles:
                before = game.get_perf_stats()
                t0 = time.time()
                move = agents[role].select_move(
                    game, state, time_limit=config.move_time_limit
                )
                dt = time.time() - t0
                after = game.get_perf_stats()

                moves[role] = move
                decision_count[role] += 1
                decision_time_sum[role] += dt
                decision_time_max[role] = max(decision_time_max[role], dt)

                legal_calls_by_role[role] += after["legal_calls"] - before["legal_calls"]
                legal_hits_by_role[role] += (
                    after["legal_cache_hits"] - before["legal_cache_hits"]
                )
                next_calls_by_role[role] += after["next_calls"] - before["next_calls"]
                next_hits_by_role[role] += after["next_cache_hits"] - before["next_cache_hits"]

            state = game.get_next_state(state, moves)

        scores = {role: int(game.get_goal(state, role)) for role in roles}
    except Exception as exc:
        failed = 1
        error_msg = str(exc)
        scores = {role: 0 for role in roles}

    wall_clock_sec = time.time() - wall_start
    max_score = max(scores.values()) if scores else 0
    winners = [role for role, score in scores.items() if score == max_score]
    is_draw = 1 if (len(winners) != 1 and not failed) else 0

    rows = []
    for role in roles:
        agent_type = lineup[roles.index(role)]
        dcount = decision_count[role]
        legal_calls = legal_calls_by_role[role]
        next_calls = next_calls_by_role[role]
        is_win = 1 if (role in winners and not failed and not is_draw) else 0
        is_loss = 1 if (not failed and not is_draw and not is_win) else 0
        row = {
            "match_id": match_id,
            "repeat_id": repeat_id,
            "game_file": game_file,
            "game_name": os.path.splitext(os.path.basename(game_file))[0],
            "cache_mode": cache_mode,
            "seed": seed,
            "role": role,
            "agent_type": agent_type,
            "score": scores[role],
            "is_winner": is_win,
            "is_draw": is_draw,
            "is_loss": is_loss,
            "steps": steps,
            "wall_clock_sec": round(wall_clock_sec, 6),
            "decision_count": dcount,
            "decision_total_sec": round(decision_time_sum[role], 6),
            "decision_avg_sec": round(safe_div(decision_time_sum[role], dcount), 6),
            "decision_max_sec": round(decision_time_max[role], 6),
            "legal_calls": legal_calls,
            "legal_cache_hits": legal_hits_by_role[role],
            "legal_cache_hit_rate": round(
                safe_div(legal_hits_by_role[role], legal_calls), 6
            ),
            "next_calls": next_calls,
            "next_cache_hits": next_hits_by_role[role],
            "next_cache_hit_rate": round(
                safe_div(next_hits_by_role[role], next_calls), 6
            ),
            "failed": failed,
            "error_msg": error_msg,
        }
        rows.append(row)

    return rows


def generate_lineups(role_count: int) -> List[Tuple[str, ...]]:
    # Keep it simple: evaluate every role-assignment combination.
    return list(product(AGENT_TYPES, repeat=role_count))


def aggregate_summary(detail_rows: List[dict]) -> List[dict]:
    grouped = {}
    for row in detail_rows:
        key = (row["game_name"], row["cache_mode"], row["agent_type"])
        bucket = grouped.setdefault(
            key,
            {
                "game_name": row["game_name"],
                "cache_mode": row["cache_mode"],
                "agent_type": row["agent_type"],
                "appearances": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "failed_matches": 0,
                "score_sum": 0.0,
                "steps_sum": 0.0,
                "wall_sum": 0.0,
                "decision_total_sum": 0.0,
                "decision_avg_sum": 0.0,
                "decision_max_sum": 0.0,
                "legal_calls_sum": 0.0,
                "legal_hits_sum": 0.0,
                "next_calls_sum": 0.0,
                "next_hits_sum": 0.0,
            },
        )
        bucket["appearances"] += 1
        bucket["wins"] += row["is_winner"]
        bucket["draws"] += row["is_draw"]
        bucket["losses"] += row["is_loss"]
        bucket["failed_matches"] += row["failed"]
        bucket["score_sum"] += row["score"]
        bucket["steps_sum"] += row["steps"]
        bucket["wall_sum"] += row["wall_clock_sec"]
        bucket["decision_total_sum"] += row["decision_total_sec"]
        bucket["decision_avg_sum"] += row["decision_avg_sec"]
        bucket["decision_max_sum"] += row["decision_max_sec"]
        bucket["legal_calls_sum"] += row["legal_calls"]
        bucket["legal_hits_sum"] += row["legal_cache_hits"]
        bucket["next_calls_sum"] += row["next_calls"]
        bucket["next_hits_sum"] += row["next_cache_hits"]

    summary = []
    for (_, _, _), b in grouped.items():
        appearances = b["appearances"]
        losses = b["losses"]
        summary.append(
            {
                "game_name": b["game_name"],
                "cache_mode": b["cache_mode"],
                "agent_type": b["agent_type"],
                "appearances": appearances,
                "wins": b["wins"],
                "draws": b["draws"],
                "losses": losses,
                "win_rate": round(safe_div(b["wins"], appearances), 6),
                "draw_rate": round(safe_div(b["draws"], appearances), 6),
                "loss_rate": round(safe_div(losses, appearances), 6),
                "failed_rate": round(safe_div(b["failed_matches"], appearances), 6),
                "avg_score": round(safe_div(b["score_sum"], appearances), 6),
                "avg_steps": round(safe_div(b["steps_sum"], appearances), 6),
                "avg_wall_clock_sec": round(safe_div(b["wall_sum"], appearances), 6),
                "avg_decision_total_sec": round(
                    safe_div(b["decision_total_sum"], appearances), 6
                ),
                "avg_decision_avg_sec": round(
                    safe_div(b["decision_avg_sum"], appearances), 6
                ),
                "avg_decision_max_sec": round(
                    safe_div(b["decision_max_sum"], appearances), 6
                ),
                "avg_legal_calls": round(
                    safe_div(b["legal_calls_sum"], appearances), 6
                ),
                "avg_legal_cache_hit_rate": round(
                    safe_div(b["legal_hits_sum"], b["legal_calls_sum"]), 6
                ),
                "avg_next_calls": round(
                    safe_div(b["next_calls_sum"], appearances), 6
                ),
                "avg_next_cache_hit_rate": round(
                    safe_div(b["next_hits_sum"], b["next_calls_sum"]), 6
                ),
            }
        )

    summary.sort(key=lambda x: (x["game_name"], x["cache_mode"], x["agent_type"]))
    return summary


def write_csv(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(config: EvalConfig):
    details = []
    match_id = 0
    scenario_id = 0

    for game_file in config.games:
        probe = GameStateMachine(game_file)
        roles = [str(r) for r in probe.get_roles()]
        role_count = len(roles)
        lineups = generate_lineups(role_count)

        for lineup in lineups:
            for repeat_idx in range(config.repeats):
                rotated = rotate_lineup(lineup, repeat_idx)
                # 同一场景使用同一随机种子，确保 enabled/disabled 可直接配对比较。
                seed = config.base_seed + scenario_id
                for cache_mode in config.cache_modes:
                    match_rows = evaluate_single_match(
                        game_file=game_file,
                        roles=roles,
                        lineup=rotated,
                        seed=seed,
                        match_id=match_id,
                        repeat_id=repeat_idx,
                        cache_mode=cache_mode,
                        config=config,
                    )
                    details.extend(match_rows)
                    match_id += 1
                scenario_id += 1

    summary = aggregate_summary(details)
    detail_csv = os.path.join(config.output_dir, "match_details.csv")
    summary_csv = os.path.join(config.output_dir, "summary.csv")
    write_csv(detail_csv, details)
    write_csv(summary_csv, summary)
    return detail_csv, summary_csv, len(details)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Random / PureMCT / MCTS agents and export CSV reports."
    )
    parser.add_argument(
        "--games",
        nargs="+",
        default=["games/ticTacToe.kif", "games/connectFour.kif"],
        help="Game .kif files",
    )
    parser.add_argument("--repeats", type=int, default=6, help="Repeat count per lineup")
    parser.add_argument(
        "--move-time-limit",
        type=float,
        default=1.0,
        help="Decision time budget per move (seconds)",
    )
    parser.add_argument(
        "--step-limit", type=int, default=400, help="Safety cap for max steps per match"
    )
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--output-dir", type=str, default="logs/eval", help="Output folder for CSV"
    )
    parser.add_argument("--mcts-iterations", type=int, default=60)
    parser.add_argument("--mcts-rollout-depth", type=int, default=80)
    parser.add_argument("--mcts-exploration-constant", type=float, default=20.0)
    parser.add_argument("--mcts-fallback-legal-threshold", type=int, default=180)
    parser.add_argument(
        "--cache-modes",
        nargs="+",
        choices=["enabled", "disabled", "both"],
        default=["both"],
        help="Cache mode(s) for A/B evaluation. Use both to run enabled+disabled.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = EvalConfig(
        games=args.games,
        repeats=max(1, args.repeats),
        move_time_limit=max(0.0, args.move_time_limit),
        step_limit=max(1, args.step_limit),
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        mcts_iterations=max(1, args.mcts_iterations),
        mcts_rollout_depth=max(1, args.mcts_rollout_depth),
        mcts_exploration_constant=args.mcts_exploration_constant,
        mcts_fallback_legal_threshold=max(1, args.mcts_fallback_legal_threshold),
        cache_modes=normalize_cache_modes(args.cache_modes),
    )

    detail_path, summary_path, row_count = run_evaluation(cfg)
    print(f"[OK] detail csv: {detail_path}")
    print(f"[OK] summary csv: {summary_path}")
    print(f"[OK] total detail rows: {row_count}")
