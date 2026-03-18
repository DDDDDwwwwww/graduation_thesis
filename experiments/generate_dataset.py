from __future__ import annotations

"""自对弈数据集生成脚本。

功能：
1. 使用 PureMCT 或 HeuristicMCTS 进行自对弈；
2. 采样中间状态并构造 value target；
3. 输出 JSONL 训练样本。
"""

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# 允许脚本在项目根目录直接运行。
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agents.heuristic_mcts_agent import HeuristicMCTSAgent
from agents.pure_mct_agent import PureMCTAgent
from ggp_statemachine import GameStateMachine


def outcome_to_value(score_for_role, score_for_opponent=None) -> float:
    """将终局分数映射到 [-1, 1] 价值标签。"""
    if score_for_opponent is not None:
        if score_for_role > score_for_opponent:
            return 1.0
        if score_for_role < score_for_opponent:
            return -1.0
        return 0.0
    return max(-1.0, min(1.0, (float(score_for_role) - 50.0) / 50.0))


def choose_agent_name(agent_mode: str, rng: random.Random, heuristic_ratio: float) -> str:
    """Resolve the actual self-play agent for one game."""
    if agent_mode == "mixed_heuristic_pure":
        return "heuristic_mcts" if rng.random() < heuristic_ratio else "pure_mct"
    return agent_mode


def build_self_play_agents(agent_name, roles, iterations, seed):
    """按配置创建一组自对弈智能体（每个角色一个实例）。"""
    agents = {}
    for i, role in enumerate(roles):
        role_seed = None if seed is None else seed + i * 9973
        if agent_name == "pure_mct":
            agents[role] = PureMCTAgent(
                name=f"pure_{role}",
                role=role,
                iterations=iterations,
                seed=role_seed,
            )
        elif agent_name == "heuristic_mcts":
            agents[role] = HeuristicMCTSAgent(
                name=f"heur_{role}",
                role=role,
                iterations=iterations,
                seed=role_seed,
            )
        else:
            raise ValueError(f"Unsupported agent: {agent_name}")
    return agents


def run_single_game(game, agents, game_name, match_id, playclock, rng, sampling_mode, sample_rate, source_agent):
    """运行一局并返回该局采样得到的训练样本列表。"""
    roles = [str(r) for r in game.get_roles()]
    state = game.get_initial_state()
    ply = 0
    pending = []

    while not game.is_terminal(state):
        facts = game.get_state_facts_as_strings(state)
        current_role = game.get_current_role(state)
        sample_roles = [current_role] if current_role in roles else roles
        sample_roles = [r for r in sample_roles if r is not None]

        # 在行动方视角（或所有角色视角）记录样本。
        for acting_role in sample_roles:
            keep = sampling_mode == "all_states" or rng.random() < sample_rate
            if not keep:
                continue
            pending.append(
                {
                    "game_name": game_name,
                    "match_id": match_id,
                    "source_agent": source_agent,
                    "state_facts": facts,
                    "acting_role": acting_role,
                    "ply_index": ply,
                    "terminal": False,
                }
            )

        # 收集每个角色本步动作，形成联合动作。
        joint_moves = {}
        for role in roles:
            legal = game.get_legal_moves(state, role)
            move = agents[role].select_action(game, state, legal, time_limit=playclock)
            joint_moves[role] = move

        state = game.get_next_state(state, joint_moves)
        ply += 1

    # 根据终局结果回填每条样本的 value_target。
    final_scores = {r: float(game.get_goal(state, r)) for r in roles}
    out = []
    for row in pending:
        role = row["acting_role"]
        if len(roles) == 2:
            opp = roles[0] if roles[1] == role else roles[1]
            value = outcome_to_value(final_scores[role], final_scores[opp])
        else:
            value = outcome_to_value(final_scores[role], None)
        sample = dict(row)
        sample["value_target"] = value
        out.append(sample)
    return out


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="Generate self-play JSONL value dataset.")
    parser.add_argument("--game", required=True, help="Path to .kif game file")
    parser.add_argument(
        "--agent",
        choices=["pure_mct", "heuristic_mcts", "mixed_heuristic_pure"],
        default="mixed_heuristic_pure",
    )
    parser.add_argument(
        "--heuristic-ratio",
        type=float,
        default=0.8,
        help="Used when --agent=mixed_heuristic_pure. Probability of selecting heuristic_mcts per game.",
    )
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--playclock", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-mode", choices=["all_states", "subsampled_states"], default="all_states")
    parser.add_argument("--sample-rate", type=float, default=0.4)
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()
    if not (0.0 <= args.heuristic_ratio <= 1.0):
        raise ValueError("--heuristic-ratio must be in [0, 1].")

    random.seed(args.seed)
    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    game = GameStateMachine(args.game)
    roles = [str(r) for r in game.get_roles()]
    game_name = Path(args.game).stem

    all_samples = []
    game_source_counts = {"heuristic_mcts": 0, "pure_mct": 0}
    for game_idx in range(args.n_games):
        game.clear_caches()
        selected_agent = choose_agent_name(
            agent_mode=args.agent,
            rng=rng,
            heuristic_ratio=args.heuristic_ratio,
        )
        game_source_counts[selected_agent] += 1
        agents = build_self_play_agents(
            agent_name=selected_agent,
            roles=roles,
            iterations=args.iterations,
            seed=args.seed + game_idx * 1009,
        )
        samples = run_single_game(
            game=game,
            agents=agents,
            game_name=game_name,
            match_id=game_idx,
            playclock=args.playclock,
            rng=rng,
            sampling_mode=args.sampling_mode,
            sample_rate=args.sample_rate,
            source_agent=selected_agent,
        )
        all_samples.extend(samples)
        # 按 10% 进度粒度打印日志，数据量小时也至少每局打印一次。
        if (game_idx + 1) % max(1, args.n_games // 10) == 0:
            print(f"[generate_dataset] progress={game_idx + 1}/{args.n_games}, samples={len(all_samples)}")

    with output_path.open("w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(
        "[generate_dataset] source_games "
        f"heuristic_mcts={game_source_counts['heuristic_mcts']}, "
        f"pure_mct={game_source_counts['pure_mct']}"
    )
    print(f"[generate_dataset] saved {len(all_samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
