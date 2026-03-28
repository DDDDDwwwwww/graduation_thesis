from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_utils import (
    ValueArtifact,
    collect_cross_game_mean,
    generate_dataset,
    init_output_layout,
    run_match_grid,
    train_value_model_artifact,
    write_csv,
    write_json,
)


EXP_NAME = "phase1_minimal_token_transformer_global"


def _artifact_for_dir(model_dir: Path) -> ValueArtifact:
    return ValueArtifact(
        model_path=str(model_dir / "model.pt"),
        encoder_config_path=str(model_dir / "encoder.json"),
        vocab_path=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 minimal runnable experiment for token_transformer_global.")
    parser.add_argument("--game", default="games/connectFour.kif")
    parser.add_argument("--dataset-sizes", type=int, nargs="+", default=[200, 500, 1000])
    parser.add_argument("--playclock", type=float, default=0.7)
    parser.add_argument("--selfplay-iterations", type=int, default=70)
    parser.add_argument("--eval-iterations", type=int, default=120)
    parser.add_argument("--eval-rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--global-hidden-dim", type=int, default=32)
    parser.add_argument("--out-dir", default=f"outputs/experiments/{EXP_NAME}")
    args = parser.parse_args()

    layout = init_output_layout(EXP_NAME, args.out_dir, args=args)
    data_dir = layout["datasets"]
    model_root = layout["models"]

    all_raw = []
    all_summary = []
    train_eval_rows = []
    artifact_manifest = {}

    for i, size in enumerate(args.dataset_sizes):
        dataset_path = data_dir / f"dataset_{size}.jsonl"
        generate_dataset(
            game=args.game,
            n_games=int(size),
            output_path=dataset_path,
            iterations=args.selfplay_iterations,
            playclock=args.playclock,
            seed=args.seed + i * 1000,
            agent="mixed_heuristic_pure",
            heuristic_ratio=0.8,
            sampling_mode="all_states",
        )

        cfg_key = "token_transformer_global"
        model_dir = model_root / f"size_{size}" / cfg_key
        train_value_model_artifact(
            dataset_path=dataset_path,
            output_dir=model_dir,
            encoder="board_token",
            model="transformer",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed + i * 1000 + 17,
            device=args.device,
            extra_args=[
                "--global-feature-set",
                "basic10",
                "--transformer-fusion-mode",
                "concat",
                "--global-hidden-dim",
                str(int(args.global_hidden_dim)),
            ],
        )
        artifact = _artifact_for_dir(model_dir)
        artifacts = {cfg_key: artifact}
        artifact_manifest[str(size)] = {
            cfg_key: {
                "model_path": artifact.model_path,
                "encoder_config_path": artifact.encoder_config_path,
                "vocab_path": artifact.vocab_path,
            }
        }

        metrics_path = model_dir / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            train_eval_rows.append(
                {
                    "dataset_games": int(size),
                    "config": cfg_key,
                    "test_loss": float(m.get("test_loss", 0.0)),
                    "test_sign_acc": float(m.get("test_sign_acc", 0.0)),
                    "best_val_loss": float(m.get("best_val_loss", 0.0) or 0.0),
                }
            )

        pairs = [("value_greedy:token_transformer_global", "random")]
        raw_rows, summary_rows = run_match_grid(
            games=[args.game],
            pairs=pairs,
            rounds=args.eval_rounds,
            playclock=args.playclock,
            iterations=args.eval_iterations,
            seed=args.seed + i * 2000,
            cache_enabled=True,
            artifacts=artifacts,
            device=args.device,
        )

        for row in raw_rows:
            row["dataset_games"] = int(size)
        for row in summary_rows:
            row["dataset_games"] = int(size)

        all_raw.extend(raw_rows)
        all_summary.extend(summary_rows)

    cross = collect_cross_game_mean(all_summary, ["dataset_games", "agent_a_key", "agent_b_key"])

    write_json(layout["artifacts"] / "artifact_manifest.json", artifact_manifest)
    write_json(layout["metrics"] / "training_metrics.json", train_eval_rows)
    write_csv(layout["metrics"] / "training_metrics.csv", train_eval_rows)
    write_json(layout["raw"] / "matches.json", all_raw)
    write_json(layout["summary"] / "by_game.json", all_summary)
    write_json(layout["summary"] / "cross_game.json", cross)
    write_csv(layout["summary"] / "by_game.csv", all_summary)
    write_csv(layout["summary"] / "cross_game.csv", cross)

    print(f"[{EXP_NAME}] finished. output={layout['root']}")


if __name__ == "__main__":
    main()
