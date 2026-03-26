from __future__ import annotations

import argparse

from experiment_utils import (
    collect_cross_game_mean,
    default_games,
    init_output_layout,
    load_artifacts,
    run_match_grid,
    write_csv,
    write_json,
)


EXP_NAME = "TS_C_search_budget_sensitivity"


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage Experiment C: search budget sensitivity.")
    parser.add_argument("--artifacts", default="outputs4two_stage/artifacts/two_stage_artifacts.json")
    parser.add_argument("--games", nargs="+", default=default_games(multi=False))
    parser.add_argument("--iterations-list", type=int, nargs="+", default=[50, 120, 300])
    parser.add_argument("--playclock", type=float, default=0.7)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--visit-threshold", type=int, default=4)
    parser.add_argument("--slow-budget-per-move", type=int, default=16)
    parser.add_argument("--uncertainty-type", choices=["margin", "variance_head"], default="variance_head")
    parser.add_argument("--gate-type", default="combined")
    parser.add_argument("--out-dir", default=f"outputs4two_stage/experiments/{EXP_NAME}")
    args = parser.parse_args()

    artifacts = load_artifacts(args.artifacts)
    two_stage_kwargs = {
        "tau": args.tau,
        "visit_threshold": args.visit_threshold,
        "slow_budget_per_move": args.slow_budget_per_move,
        "uncertainty_type": args.uncertainty_type,
        "gate_type": args.gate_type,
    }
    pairs = [
        ("neural_mcts:token_transformer", "neural_mcts:token_mlp"),
        ("two_stage_neural_mcts", "neural_mcts:token_transformer"),
    ]
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
            artifacts=artifacts,
            device=args.device,
            two_stage_kwargs=two_stage_kwargs,
        )
        for row in r:
            row["search_iterations"] = int(iters)
        for row in s:
            row["search_iterations"] = int(iters)
        raw_rows.extend(r)
        summary_rows.extend(s)
    cross = collect_cross_game_mean(summary_rows, ["search_iterations", "agent_a_key", "agent_b_key"])
    layout = init_output_layout(EXP_NAME, args.out_dir, args=args)
    write_json(layout["raw"] / "matches.json", raw_rows)
    write_json(layout["summary"] / "by_game.json", summary_rows)
    write_json(layout["summary"] / "cross_game.json", cross)
    write_csv(layout["summary"] / "by_game.csv", summary_rows)
    write_csv(layout["summary"] / "cross_game.csv", cross)
    print(f"[{EXP_NAME}] finished. output={layout['root']}")


if __name__ == "__main__":
    main()
