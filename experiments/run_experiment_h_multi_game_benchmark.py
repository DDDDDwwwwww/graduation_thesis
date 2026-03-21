from __future__ import annotations

import argparse
from pathlib import Path

from experiment_utils import (
    collect_cross_game_mean,
    default_games,
    init_output_layout,
    load_artifact_map,
    run_match_grid,
    write_csv,
    write_json,
)


EXP_NAME = "H_multi_game_benchmark"


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment H: multi-game benchmark.")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--games", nargs="+", default=default_games(multi=True))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--playclock", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-all-neural-configs", action="store_true")
    parser.add_argument("--out-dir", default=f"outputs/experiments/{EXP_NAME}")
    args = parser.parse_args()

    artifacts = load_artifact_map(args.artifacts)

    pairs = [
        ("random", "random"),
        ("pure_mct", "random"),
        ("heuristic_mcts", "pure_mct"),
        ("value_greedy:token_transformer", "random"),
        ("neural_mcts:token_transformer", "pure_mct"),
        ("neural_mcts:token_transformer", "heuristic_mcts"),
    ]

    if args.include_all_neural_configs:
        pairs.extend(
            [
                ("value_greedy:fact_mlp", "random"),
                ("value_greedy:token_mlp", "random"),
                ("neural_mcts:fact_mlp", "pure_mct"),
                ("neural_mcts:token_mlp", "pure_mct"),
                ("neural_mcts:fact_mlp", "heuristic_mcts"),
                ("neural_mcts:token_mlp", "heuristic_mcts"),
            ]
        )

    raw_rows, summary_rows = run_match_grid(
        games=args.games,
        pairs=pairs,
        rounds=args.rounds,
        playclock=args.playclock,
        iterations=args.iterations,
        seed=args.seed,
        cache_enabled=True,
        artifacts=artifacts,
        device=args.device,
    )

    cross = collect_cross_game_mean(summary_rows, ["agent_a_key", "agent_b_key"])

    layout = init_output_layout(EXP_NAME, args.out_dir, args=args)
    write_json(layout["raw"] / "matches.json", raw_rows)
    write_json(layout["summary"] / "by_game.json", summary_rows)
    write_json(layout["summary"] / "cross_game.json", cross)
    write_csv(layout["summary"] / "by_game.csv", summary_rows)
    write_csv(layout["summary"] / "cross_game.csv", cross)

    print(f"[{EXP_NAME}] finished. output={layout['root']}")


if __name__ == "__main__":
    main()
