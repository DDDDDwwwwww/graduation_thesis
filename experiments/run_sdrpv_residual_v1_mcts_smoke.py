from __future__ import annotations

"""Small-scale MCTS smoke matches for SDRPV residual_v1 artifacts.

Flow:
1) fixed-sims stage (fixed iterations, fixed playclock)
2) fixed-time stage (fixed time budget, fixed iterations cap)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run residual_v1 small-scale MCTS smoke matches.")
    parser.add_argument("--model-dir", required=True, help="Residual model dir with model.pt + encoder.json")
    parser.add_argument("--game", default="games/connectFour.kif")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fixed-sims", type=int, default=120, help="Iterations for fixed-sims stage.")
    parser.add_argument("--fixed-sims-playclock", type=float, default=0.5, help="Per-move time for fixed-sims stage.")
    parser.add_argument("--fixed-time", type=float, default=0.5, help="Per-move time for fixed-time stage.")
    parser.add_argument("--fixed-time-iters", type=int, default=120, help="Iterations cap for fixed-time stage.")
    parser.add_argument("--out-dir", default="outputs/experiments/SDRPV_residual_v1_mcts_smoke")
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

    root_layout = init_output_layout("SDRPV_residual_v1_mcts_smoke", args.out_dir, args=args)

    # Stage 1: fixed-sims
    fixed_sims_dir = root_layout["root"] / "fixed_sims"
    fixed_sims_layout = init_output_layout(
        "SDRPV_residual_v1_mcts_smoke_fixed_sims",
        fixed_sims_dir,
        args={"fixed_sims": args.fixed_sims, "playclock": args.fixed_sims_playclock},
    )
    raw_sims, summary_sims = run_match_grid(
        games=games,
        pairs=pairs,
        rounds=args.rounds,
        playclock=float(args.fixed_sims_playclock),
        iterations=int(args.fixed_sims),
        seed=int(args.seed),
        cache_enabled=True,
        artifacts=artifacts,
        device=args.device,
    )
    write_json(fixed_sims_layout["raw"] / "matches.json", raw_sims)
    write_json(fixed_sims_layout["summary"] / "by_game.json", summary_sims)
    write_csv(fixed_sims_layout["summary"] / "by_game.csv", summary_sims)

    # Stage 2: fixed-time
    fixed_time_dir = root_layout["root"] / "fixed_time"
    fixed_time_layout = init_output_layout(
        "SDRPV_residual_v1_mcts_smoke_fixed_time",
        fixed_time_dir,
        args={"time_budget": args.fixed_time, "iterations_cap": args.fixed_time_iters},
    )
    raw_time, summary_time = run_match_grid(
        games=games,
        pairs=pairs,
        rounds=args.rounds,
        playclock=float(args.fixed_time),
        iterations=int(args.fixed_time_iters),
        seed=int(args.seed + 1000),
        cache_enabled=True,
        artifacts=artifacts,
        device=args.device,
    )
    write_json(fixed_time_layout["raw"] / "matches.json", raw_time)
    write_json(fixed_time_layout["summary"] / "by_game.json", summary_time)
    write_csv(fixed_time_layout["summary"] / "by_game.csv", summary_time)

    stage_summary = {
        "fixed_sims": {
            "iterations": int(args.fixed_sims),
            "playclock": float(args.fixed_sims_playclock),
            "summary_rows": summary_sims,
        },
        "fixed_time": {
            "time_budget": float(args.fixed_time),
            "iterations_cap": int(args.fixed_time_iters),
            "summary_rows": summary_time,
        },
    }
    write_json(root_layout["summary"] / "stage_summary.json", stage_summary)

    print(f"[residual_v1_mcts_smoke] done. output={root_layout['root']}")


if __name__ == "__main__":
    main()

