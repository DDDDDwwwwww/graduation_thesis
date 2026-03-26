from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiment_utils import (
    ValueArtifact,
    collect_cross_game_mean,
    init_output_layout,
    run_match_grid,
    train_value_model_artifact,
    write_csv,
    write_json,
)


EXP_NAME = "I_cross_game_generalization"
CONFIGS = {
    "fact_mlp": ("fact_vector", "mlp"),
    "token_mlp": ("board_token", "mlp"),
    "token_transformer": ("board_token", "transformer"),
}
ROOT = Path(__file__).resolve().parents[1]


def _artifact_for_dir(model_dir: Path, key: str) -> ValueArtifact:
    vocab_path = model_dir / "vocab.json"
    return ValueArtifact(
        model_path=str(model_dir / "model.pt"),
        encoder_config_path=str(model_dir / "encoder.json"),
        vocab_path=str(vocab_path) if vocab_path.exists() and key == "fact_mlp" else None,
    )


def _generate_multigame_dataset(game_counts: list[tuple[str, int]], output_path: Path, seed: int, iterations: int, playclock: float):
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "generate_multigame_dataset.py"),
    ]
    for game, count in game_counts:
        cmd.extend(["--game-count", f"{game}:{int(count)}"])
    cmd.extend(
        [
            "--agent",
            "mixed_heuristic_pure",
            "--heuristic-ratio",
            "0.8",
            "--iterations",
            str(int(iterations)),
            "--playclock",
            str(float(playclock)),
            "--seed",
            str(int(seed)),
            "--sampling-mode",
            "all_states",
            "--output",
            str(output_path),
        ]
    )
    subprocess.run(cmd, check=True)


def _train_all_configs(dataset_path: Path, model_root: Path, seed: int, device: str, epochs: int, batch_size: int, learning_rate: float, weight_decay: float):
    artifacts = {}
    for i, (cfg_key, (encoder, model)) in enumerate(CONFIGS.items()):
        out_dir = model_root / cfg_key
        train_value_model_artifact(
            dataset_path=dataset_path,
            output_dir=out_dir,
            encoder=encoder,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed + i * 37,
            device=device,
        )
        artifacts[cfg_key] = _artifact_for_dir(out_dir, cfg_key)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment I: cross-game generalization.")
    parser.add_argument("--seen-games", nargs="+", default=["games/ticTacToe.kif", "games/connectFour.kif", "games/breakthrough.kif"])
    parser.add_argument("--unseen-game", default="games/maze.kif")
    parser.add_argument("--n-games-per-train-game", type=int, default=250)
    parser.add_argument("--selfplay-iterations", type=int, default=70)
    parser.add_argument("--eval-iterations", type=int, default=120)
    parser.add_argument("--eval-rounds", type=int, default=10)
    parser.add_argument("--playclock", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--out-dir", default=f"outputs/experiments/{EXP_NAME}")
    args = parser.parse_args()

    layout = init_output_layout(EXP_NAME, args.out_dir, args=args)
    datasets_dir = layout["datasets"]
    models_dir = layout["models"]

    unseen_game = args.unseen_game

    unseen_setting_dataset = datasets_dir / "train_unseen_setting.jsonl"
    _generate_multigame_dataset(
        game_counts=[(g, args.n_games_per_train_game) for g in args.seen_games],
        output_path=unseen_setting_dataset,
        seed=args.seed,
        iterations=args.selfplay_iterations,
        playclock=args.playclock,
    )

    seen_setting_dataset = datasets_dir / "train_seen_setting.jsonl"
    _generate_multigame_dataset(
        game_counts=[(g, args.n_games_per_train_game) for g in (list(args.seen_games) + [unseen_game])],
        output_path=seen_setting_dataset,
        seed=args.seed + 1000,
        iterations=args.selfplay_iterations,
        playclock=args.playclock,
    )

    unseen_artifacts = _train_all_configs(
        dataset_path=unseen_setting_dataset,
        model_root=models_dir / "unseen_setting",
        seed=args.seed + 2000,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    seen_artifacts = _train_all_configs(
        dataset_path=seen_setting_dataset,
        model_root=models_dir / "seen_setting",
        seed=args.seed + 3000,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    pairs = [
        ("value_greedy:fact_mlp", "random"),
        ("value_greedy:token_transformer", "random"),
        ("neural_mcts:fact_mlp", "random"),
        ("neural_mcts:token_mlp", "random"),
        ("neural_mcts:token_transformer", "random"),
    ]

    raw_rows = []
    summary_rows = []

    for offset, (setting, artifacts) in enumerate([
        ("unseen_game_eval", unseen_artifacts),
        ("seen_game_eval", seen_artifacts),
    ]):
        r, s = run_match_grid(
            games=[unseen_game],
            pairs=pairs,
            rounds=args.eval_rounds,
            playclock=args.playclock,
            iterations=args.eval_iterations,
            seed=args.seed + offset * 7000,
            cache_enabled=True,
            artifacts=artifacts,
            device=args.device,
        )
        for row in r:
            row["generalization_setting"] = setting
        for row in s:
            row["generalization_setting"] = setting
        raw_rows.extend(r)
        summary_rows.extend(s)

    cross = collect_cross_game_mean(summary_rows, ["generalization_setting", "agent_a_key", "agent_b_key"])

    manifest = {
        "unseen_setting": {
            k: {
                "model_path": v.model_path,
                "encoder_config_path": v.encoder_config_path,
                "vocab_path": v.vocab_path,
            }
            for k, v in unseen_artifacts.items()
        },
        "seen_setting": {
            k: {
                "model_path": v.model_path,
                "encoder_config_path": v.encoder_config_path,
                "vocab_path": v.vocab_path,
            }
            for k, v in seen_artifacts.items()
        },
    }

    write_json(layout["artifacts"] / "artifact_manifest.json", manifest)
    write_json(layout["raw"] / "matches.json", raw_rows)
    write_json(layout["summary"] / "by_game.json", summary_rows)
    write_json(layout["summary"] / "cross_game.json", cross)
    write_csv(layout["summary"] / "by_game.csv", summary_rows)
    write_csv(layout["summary"] / "cross_game.csv", cross)

    print(f"[{EXP_NAME}] finished. output={layout['root']}")


if __name__ == "__main__":
    main()
