from __future__ import annotations

"""Generate a mixed-game JSONL self-play dataset."""

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.generate_dataset import (  # noqa: E402
    build_self_play_agents,
    choose_agent_name,
    run_single_game,
)
from ggp_statemachine import GameStateMachine  # noqa: E402


def parse_game_counts(items: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid --game-count item: {item}. Expected format path:n_games")
        game, n_str = item.rsplit(":", 1)
        out.append((game, int(n_str)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mixed-game self-play JSONL dataset.")
    parser.add_argument(
        "--game-count",
        action="append",
        required=True,
        help="Use multiple times. Format: games/ticTacToe.kif:500",
    )
    parser.add_argument(
        "--agent",
        choices=["pure_mct", "heuristic_mcts", "mixed_heuristic_pure"],
        default="mixed_heuristic_pure",
    )
    parser.add_argument("--heuristic-ratio", type=float, default=0.8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--playclock", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-mode", choices=["all_states", "subsampled_states"], default="all_states")
    parser.add_argument("--sample-rate", type=float, default=0.4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    game_counts = parse_game_counts(args.game_count)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples = []
    by_game = {}

    for game_index, (game_file, n_games) in enumerate(game_counts):
        game = GameStateMachine(game_file)
        game_name = Path(game_file).stem
        roles = [str(r) for r in game.get_roles()]
        by_game.setdefault(game_name, 0)

        for i in range(n_games):
            selected_agent = choose_agent_name(
                agent_mode=args.agent,
                rng=rng,
                heuristic_ratio=args.heuristic_ratio,
            )
            agents = build_self_play_agents(
                agent_name=selected_agent,
                roles=roles,
                iterations=args.iterations,
                seed=args.seed + game_index * 100000 + i * 101,
            )
            samples = run_single_game(
                game=game,
                agents=agents,
                game_name=game_name,
                match_id=i,
                playclock=args.playclock,
                rng=rng,
                sampling_mode=args.sampling_mode,
                sample_rate=args.sample_rate,
                source_agent=selected_agent,
            )
            all_samples.extend(samples)
            by_game[game_name] += len(samples)

            if (i + 1) % max(1, n_games // 10) == 0:
                print(f"[generate_multigame_dataset] {game_name} progress={i + 1}/{n_games}, total_samples={len(all_samples)}")

    with output_path.open("w", encoding="utf-8") as f:
        for row in all_samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("[generate_multigame_dataset] sample_counts_by_game:", by_game)
    print(f"[generate_multigame_dataset] saved {len(all_samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
