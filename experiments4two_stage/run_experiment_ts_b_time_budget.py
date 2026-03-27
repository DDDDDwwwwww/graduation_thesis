from __future__ import annotations

import argparse
from pathlib import Path

from experiment_utils import (
    collect_cross_game_mean,
    default_games,
    init_output_layout,
    load_artifacts,
    log_line,
    run_match_grid,
    set_log_file,
    write_csv,
    write_json,
)


EXP_NAME = "TS_B_time_budget_sensitivity"


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage Experiment B: time budget sensitivity.")
    parser.add_argument("--artifacts", default="outputs4two_stage/artifacts/two_stage_artifacts.json")
    parser.add_argument("--games", nargs="+", default=default_games(multi=False))
    parser.add_argument("--time-budgets", type=float, nargs="+", default=[0.2, 0.7, 1.5])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--visit-threshold", type=int, default=4)
    parser.add_argument("--slow-budget-per-move", type=int, default=16)
    parser.add_argument("--uncertainty-type", choices=["margin", "variance_head"], default="variance_head")
    parser.add_argument("--gate-type", default="combined")
    parser.add_argument("--out-dir", default=f"outputs4two_stage/experiments/{EXP_NAME}")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()
    log_path = Path(args.log_file) if args.log_file else (Path(args.out_dir) / "meta" / "run.log")
    set_log_file(log_path, mode="w")
    log_line(
        EXP_NAME,
        (
            f"start artifacts={args.artifacts} device={args.device} games={len(args.games)} "
            f"time_budgets={list(args.time_budgets)} rounds={args.rounds} iterations={args.iterations}"
        ),
    )

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
    for i, budget in enumerate(args.time_budgets):
        log_line(EXP_NAME, f"time_budget-start {i + 1}/{len(args.time_budgets)} budget={float(budget)}")
        r, s = run_match_grid(
            games=args.games,
            pairs=pairs,
            rounds=args.rounds,
            playclock=float(budget),
            iterations=args.iterations,
            seed=args.seed + i * 1000,
            artifacts=artifacts,
            device=args.device,
            two_stage_kwargs=two_stage_kwargs,
        )
        for row in r:
            row["time_budget"] = float(budget)
        for row in s:
            row["time_budget"] = float(budget)
        raw_rows.extend(r)
        summary_rows.extend(s)
        log_line(
            EXP_NAME,
            (
                f"time_budget-end {i + 1}/{len(args.time_budgets)} budget={float(budget)} "
                f"raw_rows_total={len(raw_rows)} summary_rows_total={len(summary_rows)}"
            ),
        )
    cross = collect_cross_game_mean(summary_rows, ["time_budget", "agent_a_key", "agent_b_key"])
    layout = init_output_layout(EXP_NAME, args.out_dir, args=args)
    write_json(layout["raw"] / "matches.json", raw_rows)
    write_json(layout["summary"] / "by_game.json", summary_rows)
    write_json(layout["summary"] / "cross_game.json", cross)
    write_csv(layout["summary"] / "by_game.csv", summary_rows)
    write_csv(layout["summary"] / "cross_game.csv", cross)
    log_line(EXP_NAME, f"finished output={layout['root']}")


if __name__ == "__main__":
    main()
