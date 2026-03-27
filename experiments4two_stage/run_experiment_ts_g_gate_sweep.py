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
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

EXP_NAME = "TS_G_gate_sweep"


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage Experiment G: gate parameter sweep.")
    parser.add_argument("--artifacts", default="outputs4two_stage/artifacts/two_stage_artifacts.json")
    parser.add_argument("--games", nargs="+", default=default_games(multi=False))
    parser.add_argument("--playclock", type=float, default=0.7)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--taus", nargs="+", type=float, default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--visit-thresholds", nargs="+", type=int, default=[0, 1, 2, 4])
    parser.add_argument("--slow-budgets", nargs="+", type=int, default=[4, 8, 16])
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
            f"rounds={args.rounds} playclock={args.playclock} iterations={args.iterations}"
        ),
    )

    artifacts = load_artifacts(args.artifacts)
    raw_rows = []
    summary_rows = []
    grid = [(tau, vt, sb) for tau in args.taus for vt in args.visit_thresholds for sb in args.slow_budgets]
    total = len(grid)
    for idx, (tau, visit_threshold, slow_budget) in enumerate(grid, start=1):
        log_line(
            EXP_NAME,
            (
                f"sweep-start {idx}/{total} tau={tau} "
                f"visit_threshold={visit_threshold} slow_budget={slow_budget}"
            ),
        )
        two_stage_kwargs = {
            "tau": tau,
            "visit_threshold": visit_threshold,
            "slow_budget_per_move": slow_budget,
            "uncertainty_type": args.uncertainty_type,
            "gate_type": args.gate_type,
        }
        pairs = [
            ("two_stage_neural_mcts", "neural_mcts:token_transformer"),
            ("two_stage_neural_mcts", "pure_mct"),
        ]
        r, s = run_match_grid(
            games=args.games,
            pairs=pairs,
            rounds=args.rounds,
            playclock=args.playclock,
            iterations=args.iterations,
            seed=args.seed + idx * 1000,
            artifacts=artifacts,
            device=args.device,
            two_stage_kwargs=two_stage_kwargs,
        )
        for row in r:
            row["tau"] = tau
            row["visit_threshold"] = visit_threshold
            row["slow_budget_per_move"] = slow_budget
            row["uncertainty_type"] = args.uncertainty_type
            row["gate_type"] = args.gate_type
        for row in s:
            row["tau"] = tau
            row["visit_threshold"] = visit_threshold
            row["slow_budget_per_move"] = slow_budget
            row["uncertainty_type"] = args.uncertainty_type
            row["gate_type"] = args.gate_type
        raw_rows.extend(r)
        summary_rows.extend(s)

    cross = collect_cross_game_mean(
        summary_rows,
        [
            "tau",
            "visit_threshold",
            "slow_budget_per_move",
            "uncertainty_type",
            "gate_type",
            "agent_a_key",
            "agent_b_key",
        ],
    )
    layout = init_output_layout(EXP_NAME, args.out_dir, args=args)
    write_json(layout["raw"] / "matches.json", raw_rows)
    write_json(layout["summary"] / "by_game.json", summary_rows)
    write_json(layout["summary"] / "cross_game.json", cross)
    write_csv(layout["summary"] / "by_game.csv", summary_rows)
    write_csv(layout["summary"] / "cross_game.csv", cross)
    log_line(EXP_NAME, f"finished output={layout['root']}")


if __name__ == "__main__":
    main()
