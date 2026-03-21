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


EXP_NAME = "C_search_budget_sensitivity"


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment C: search-iteration budget sensitivity.")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--games", nargs="+", default=default_games(multi=True))
    parser.add_argument("--iterations-list", type=int, nargs="+", default=[50, 100, 200, 500])
    parser.add_argument("--playclock", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-all-neural-configs", action="store_true")
    parser.add_argument("--out-dir", default=f"outputs/experiments/{EXP_NAME}")
    args = parser.parse_args()

    artifacts = load_artifact_map(args.artifacts)

    pairs = [
        ("pure_mct", "random"),
        ("heuristic_mcts", "random"),
        ("neural_mcts:token_transformer", "random"),
    ]
    if args.include_all_neural_configs:
        pairs.extend(
            [
                ("neural_mcts:fact_mlp", "random"),
                ("neural_mcts:token_mlp", "random"),
            ]
        )

    raw_rows = []
    summary_rows = []

    for i, iters in enumerate(args.iterations_list):
        r, s = run_match_grid(
            games=args.games,
            pairs=pairs,
            rounds=args.rounds,
            playclock=args.playclock,
            iterations=int(iters),
            seed=args.seed + i * 1000,
            cache_enabled=True,
            artifacts=artifacts,
            device=args.device,
        )
        for row in r:
            row["search_iterations"] = int(iters)
        for row in s:
            row["search_iterations"] = int(iters)
        raw_rows.extend(r)
        summary_rows.extend(s)

    cross_game = collect_cross_game_mean(summary_rows, ["search_iterations", "agent_a_key", "agent_b_key"])

    layout = init_output_layout(EXP_NAME, args.out_dir, args=args)
    write_json(layout["raw"] / "matches.json", raw_rows)
    write_json(layout["summary"] / "by_game.json", summary_rows)
    write_json(layout["summary"] / "cross_game.json", cross_game)
    write_csv(layout["summary"] / "by_game.csv", summary_rows)
    write_csv(layout["summary"] / "cross_game.csv", cross_game)

    print(f"[{EXP_NAME}] finished. output={layout['root']}")


if __name__ == "__main__":
    main()
