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


EXP_NAME = "B_time_budget_sensitivity"


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment B: decision-time budget sensitivity.")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--games", nargs="+", default=default_games(multi=True))
    parser.add_argument("--time-budgets", type=float, nargs="+", default=[0.2, 0.7, 1.5])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=180)
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

    for i, budget in enumerate(args.time_budgets):
        r, s = run_match_grid(
            games=args.games,
            pairs=pairs,
            rounds=args.rounds,
            playclock=float(budget),
            iterations=args.iterations,
            seed=args.seed + i * 1000,
            cache_enabled=True,
            artifacts=artifacts,
            device=args.device,
        )
        for row in r:
            row["time_budget"] = float(budget)
        for row in s:
            row["time_budget"] = float(budget)
        raw_rows.extend(r)
        summary_rows.extend(s)

    cross_game = collect_cross_game_mean(summary_rows, ["time_budget", "agent_a_key", "agent_b_key"])

    layout = init_output_layout(EXP_NAME, args.out_dir, args=args)
    write_json(layout["raw"] / "matches.json", raw_rows)
    write_json(layout["summary"] / "by_game.json", summary_rows)
    write_json(layout["summary"] / "cross_game.json", cross_game)
    write_csv(layout["summary"] / "by_game.csv", summary_rows)
    write_csv(layout["summary"] / "cross_game.csv", cross_game)

    print(f"[{EXP_NAME}] finished. output={layout['root']}")


if __name__ == "__main__":
    main()
