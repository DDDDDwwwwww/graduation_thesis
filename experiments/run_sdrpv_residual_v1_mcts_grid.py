from __future__ import annotations

"""Residual_v1 MCTS integration small-grid runner.

Goals:
1) Increase smoke-match sample size.
2) Sweep a small set of integration hyper-parameters under fixed-sims and fixed-time.
"""

import argparse
from pathlib import Path

from experiment_utils import (
    ValueArtifact,
    init_output_layout,
    run_match_grid,
    write_csv,
    write_json,
)


def _default_integration_grid() -> list[dict]:
    return [
        {
            "tag": "value_full",
            "evaluator_mode": "value",
        },
        {
            "tag": "sel_cap40",
            "evaluator_mode": "selective",
            "selective_max_neural_evals_per_move": 40,
            "selective_alpha": 1.0,
        },
        {
            "tag": "sel_cap20",
            "evaluator_mode": "selective",
            "selective_max_neural_evals_per_move": 20,
            "selective_alpha": 1.0,
        },
        {
            "tag": "sel_cap20_alpha07",
            "evaluator_mode": "selective",
            "selective_max_neural_evals_per_move": 20,
            "selective_alpha": 0.7,
        },
    ]


def _extract_stage_row(summary_rows: list[dict]) -> dict:
    if not summary_rows:
        return {}
    return dict(summary_rows[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run residual_v1 integration hyper-parameter small grid.")
    parser.add_argument("--model-dir", required=True, help="Residual model dir with model.pt + encoder.json")
    parser.add_argument("--game", default="games/connectFour.kif")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fixed-sims", type=int, default=120)
    parser.add_argument("--fixed-sims-playclock", type=float, default=0.5)
    parser.add_argument("--fixed-time", type=float, default=0.5)
    parser.add_argument("--fixed-time-iters", type=int, default=120)
    parser.add_argument("--out-dir", default="outputs/experiments/SDRPV_residual_v1_mcts_grid")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / "model.pt"
    encoder_path = model_dir / "encoder.json"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"encoder not found: {encoder_path}")

    artifacts = {
        "token_transformer": ValueArtifact(
            model_path=str(model_path),
            encoder_config_path=str(encoder_path),
            vocab_path=None,
        )
    }
    games = [str(args.game)]
    pairs = [("neural_mcts:token_transformer", "pure_mct")]
    grid = _default_integration_grid()

    root_layout = init_output_layout("SDRPV_residual_v1_mcts_grid", args.out_dir, args=args)
    leaderboard_rows: list[dict] = []

    for idx, cfg in enumerate(grid):
        tag = str(cfg["tag"])
        cfg_dir = root_layout["root"] / f"{idx + 1:02d}_{tag}"
        cfg_layout = init_output_layout(
            "SDRPV_residual_v1_mcts_grid_cfg",
            cfg_dir,
            args=cfg,
        )

        agent_overrides = {"neural_mcts:token_transformer": dict(cfg)}
        agent_overrides["neural_mcts:token_transformer"].pop("tag", None)

        # Stage 1: fixed-sims
        fixed_sims_dir = cfg_layout["root"] / "fixed_sims"
        fixed_sims_layout = init_output_layout(
            "SDRPV_residual_v1_mcts_grid_fixed_sims",
            fixed_sims_dir,
            args={"fixed_sims": args.fixed_sims, "playclock": args.fixed_sims_playclock},
        )
        raw_sims, summary_sims = run_match_grid(
            games=games,
            pairs=pairs,
            rounds=args.rounds,
            playclock=float(args.fixed_sims_playclock),
            iterations=int(args.fixed_sims),
            seed=int(args.seed + idx * 10000),
            cache_enabled=True,
            artifacts=artifacts,
            device=args.device,
            agent_overrides=agent_overrides,
        )
        write_json(fixed_sims_layout["raw"] / "matches.json", raw_sims)
        write_json(fixed_sims_layout["summary"] / "by_game.json", summary_sims)
        write_csv(fixed_sims_layout["summary"] / "by_game.csv", summary_sims)

        # Stage 2: fixed-time
        fixed_time_dir = cfg_layout["root"] / "fixed_time"
        fixed_time_layout = init_output_layout(
            "SDRPV_residual_v1_mcts_grid_fixed_time",
            fixed_time_dir,
            args={"time_budget": args.fixed_time, "iterations_cap": args.fixed_time_iters},
        )
        raw_time, summary_time = run_match_grid(
            games=games,
            pairs=pairs,
            rounds=args.rounds,
            playclock=float(args.fixed_time),
            iterations=int(args.fixed_time_iters),
            seed=int(args.seed + 5000 + idx * 10000),
            cache_enabled=True,
            artifacts=artifacts,
            device=args.device,
            agent_overrides=agent_overrides,
        )
        write_json(fixed_time_layout["raw"] / "matches.json", raw_time)
        write_json(fixed_time_layout["summary"] / "by_game.json", summary_time)
        write_csv(fixed_time_layout["summary"] / "by_game.csv", summary_time)

        sims_row = _extract_stage_row(summary_sims)
        time_row = _extract_stage_row(summary_time)
        combined = {
            "config_tag": tag,
            "rounds": int(args.rounds),
            "game": Path(args.game).stem,
            "fixed_sims_win_rate_a": float(sims_row.get("win_rate_a", 0.0)),
            "fixed_sims_draw_rate": float(sims_row.get("draw_rate", 0.0)),
            "fixed_sims_avg_decision_sec_a": float(sims_row.get("avg_decision_sec_a", 0.0)),
            "fixed_sims_avg_eval_calls_neural_a": float(sims_row.get("avg_eval_calls_neural_a", 0.0)),
            "fixed_sims_avg_eval_calls_fallback_a": float(sims_row.get("avg_eval_calls_fallback_a", 0.0)),
            "fixed_time_win_rate_a": float(time_row.get("win_rate_a", 0.0)),
            "fixed_time_draw_rate": float(time_row.get("draw_rate", 0.0)),
            "fixed_time_avg_decision_sec_a": float(time_row.get("avg_decision_sec_a", 0.0)),
            "fixed_time_avg_eval_calls_neural_a": float(time_row.get("avg_eval_calls_neural_a", 0.0)),
            "fixed_time_avg_eval_calls_fallback_a": float(time_row.get("avg_eval_calls_fallback_a", 0.0)),
            "integration": dict(cfg),
        }
        leaderboard_rows.append(combined)
        write_json(cfg_layout["summary"] / "stage_summary.json", {"fixed_sims": sims_row, "fixed_time": time_row})
        write_json(cfg_layout["summary"] / "combined_row.json", combined)

    leaderboard_rows.sort(key=lambda r: (r["fixed_time_win_rate_a"], r["fixed_sims_win_rate_a"]), reverse=True)
    write_json(root_layout["summary"] / "leaderboard.json", leaderboard_rows)
    write_csv(root_layout["summary"] / "leaderboard.csv", leaderboard_rows)
    print(f"[residual_v1_mcts_grid] done. output={root_layout['root']}")


if __name__ == "__main__":
    main()
