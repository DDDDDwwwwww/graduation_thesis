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


EXP_NAME = "TS_A_main_benchmark"


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage Experiment A: main benchmark.")
    parser.add_argument("--artifacts", default="outputs4two_stage/artifacts/two_stage_artifacts.json")
    parser.add_argument("--games", nargs="+", default=default_games(multi=True))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--playclock", type=float, default=0.7)
    parser.add_argument("--iterations", type=int, default=120)
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
        ("random", "pure_mct"),
        ("neural_mcts:token_mlp", "pure_mct"),
        ("neural_mcts:token_transformer", "pure_mct"),
        ("two_stage_neural_mcts", "pure_mct"),
        ("two_stage_neural_mcts", "neural_mcts:token_mlp"),
        ("two_stage_neural_mcts", "neural_mcts:token_transformer"),
    ]
    raw_rows, summary_rows = run_match_grid(
        games=args.games,
        pairs=pairs,
        rounds=args.rounds,
        playclock=args.playclock,
        iterations=args.iterations,
        seed=args.seed,
        artifacts=artifacts,
        device=args.device,
        two_stage_kwargs=two_stage_kwargs,
    )
    cross_game = collect_cross_game_mean(summary_rows, ["agent_a_key", "agent_b_key"])

    layout = init_output_layout(EXP_NAME, args.out_dir, args=args)
    write_json(layout["raw"] / "matches.json", raw_rows)
    write_json(layout["summary"] / "by_game.json", summary_rows)
    write_json(layout["summary"] / "cross_game.json", cross_game)
    write_csv(layout["summary"] / "by_game.csv", summary_rows)
    write_csv(layout["summary"] / "cross_game.csv", cross_game)
    print(f"[{EXP_NAME}] finished. output={layout['root']}")


if __name__ == "__main__":
    main()
