from __future__ import annotations

import argparse
import json
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


EXP_NAME = "E_encoder_model_ablation"


def _collect_offline_metrics(artifacts_path: str) -> list[dict]:
    payload = json.loads(Path(artifacts_path).read_text(encoding="utf-8"))
    rows = []
    for key, item in payload.items():
        model_path = Path(item["model_path"]).resolve()
        metrics_path = model_path.parent / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "config": key,
                "model_path": str(model_path),
                "test_loss": float(m.get("test_loss", 0.0)),
                "test_sign_acc": float(m.get("test_sign_acc", 0.0)),
                "best_val_loss": float(m.get("best_val_loss", 0.0) or 0.0),
                "n_train": int(m.get("n_train", 0)),
                "n_val": int(m.get("n_val", 0)),
                "n_test": int(m.get("n_test", 0)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment E: encoder/model ablation.")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--games", nargs="+", default=default_games(multi=True))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--playclock", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=f"outputs/experiments/{EXP_NAME}")
    args = parser.parse_args()

    artifacts = load_artifact_map(args.artifacts)

    pairs = [
        ("value_greedy:fact_mlp", "random"),
        ("value_greedy:token_mlp", "random"),
        ("value_greedy:token_transformer", "random"),
        ("neural_mcts:fact_mlp", "pure_mct"),
        ("neural_mcts:token_mlp", "pure_mct"),
        ("neural_mcts:token_transformer", "pure_mct"),
        ("neural_mcts:fact_mlp", "heuristic_mcts"),
        ("neural_mcts:token_mlp", "heuristic_mcts"),
        ("neural_mcts:token_transformer", "heuristic_mcts"),
    ]

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
    offline_rows = _collect_offline_metrics(args.artifacts)

    layout = init_output_layout(EXP_NAME, args.out_dir, args=args)
    write_json(layout["metrics"] / "offline_metrics.json", offline_rows)
    write_csv(layout["metrics"] / "offline_metrics.csv", offline_rows)
    write_json(layout["raw"] / "matches.json", raw_rows)
    write_json(layout["summary"] / "by_game.json", summary_rows)
    write_json(layout["summary"] / "cross_game.json", cross)
    write_csv(layout["summary"] / "by_game.csv", summary_rows)
    write_csv(layout["summary"] / "cross_game.csv", cross)

    print(f"[{EXP_NAME}] finished. output={layout['root']}")


if __name__ == "__main__":
    main()
