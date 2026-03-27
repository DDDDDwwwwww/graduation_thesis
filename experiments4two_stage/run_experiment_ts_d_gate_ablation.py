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


EXP_NAME = "TS_D_gate_ablation"


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage Experiment D: gate ablation.")
    parser.add_argument("--artifacts", default="outputs4two_stage/artifacts/two_stage_artifacts.json")
    parser.add_argument("--games", nargs="+", default=default_games(multi=False))
    parser.add_argument("--playclock", type=float, default=0.7)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--visit-threshold", type=int, default=4)
    parser.add_argument("--slow-budget-per-move", type=int, default=16)
    parser.add_argument("--uncertainty-type", choices=["margin", "variance_head"], default="variance_head")
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
    variants = [
        ("uncertainty", {"gate_type": "uncertainty"}),
        ("visit", {"gate_type": "visit"}),
        ("uncertainty_visit", {"gate_type": "uncertainty_visit"}),
        ("combined", {"gate_type": "combined"}),
    ]
    raw_rows = []
    summary_rows = []
    for i, (gate_name, extra) in enumerate(variants):
        log_line(EXP_NAME, f"gate_variant-start {i + 1}/{len(variants)} gate={gate_name}")
        two_stage_kwargs = {
            "tau": args.tau,
            "visit_threshold": args.visit_threshold,
            "slow_budget_per_move": args.slow_budget_per_move,
            "uncertainty_type": args.uncertainty_type,
            "gate_type": extra["gate_type"],
        }
        r, s = run_match_grid(
            games=args.games,
            pairs=[("two_stage_neural_mcts", "neural_mcts:token_transformer")],
            rounds=args.rounds,
            playclock=args.playclock,
            iterations=args.iterations,
            seed=args.seed + i * 1000,
            artifacts=artifacts,
            device=args.device,
            two_stage_kwargs=two_stage_kwargs,
        )
        for row in r:
            row["gate_variant"] = gate_name
        for row in s:
            row["gate_variant"] = gate_name
        raw_rows.extend(r)
        summary_rows.extend(s)
        log_line(
            EXP_NAME,
            (
                f"gate_variant-end {i + 1}/{len(variants)} gate={gate_name} "
                f"raw_rows_total={len(raw_rows)} summary_rows_total={len(summary_rows)}"
            ),
        )
    cross = collect_cross_game_mean(summary_rows, ["gate_variant", "agent_a_key", "agent_b_key"])
    layout = init_output_layout(EXP_NAME, args.out_dir, args=args)
    write_json(layout["raw"] / "matches.json", raw_rows)
    write_json(layout["summary"] / "by_game.json", summary_rows)
    write_json(layout["summary"] / "cross_game.json", cross)
    write_csv(layout["summary"] / "by_game.csv", summary_rows)
    write_csv(layout["summary"] / "cross_game.csv", cross)
    log_line(EXP_NAME, f"finished output={layout['root']}")


if __name__ == "__main__":
    main()
