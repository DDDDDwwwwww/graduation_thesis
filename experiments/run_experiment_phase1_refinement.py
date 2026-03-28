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


EXP_NAME = "phase1_refinement_token_global_ablation"


def _artifact_for_dir(model_dir: Path) -> ValueArtifact:
    return ValueArtifact(
        model_path=str(model_dir / "model.pt"),
        encoder_config_path=str(model_dir / "encoder.json"),
        vocab_path=None,
    )


def _build_variant_configs(
    variants: list[str],
    fusion_modes: list[str],
    global_feature_sets: list[str],
) -> list[dict]:
    out: list[dict] = []
    variants = [v.strip().lower() for v in variants]
    for variant in variants:
        if variant == "token_only":
            out.append(
                {
                    "key": "token_only",
                    "variant": "token_only",
                    "fusion_mode": "add",
                    "global_feature_set": "none",
                    "token_branch_mode": "normal",
                    "disable_global_features": True,
                }
            )
            continue
        if variant == "global_only":
            for feature_set in global_feature_sets:
                for fusion in fusion_modes:
                    out.append(
                        {
                            "key": f"global_only__{feature_set}__{fusion}",
                            "variant": "global_only",
                            "fusion_mode": fusion,
                            "global_feature_set": feature_set,
                            "token_branch_mode": "zero",
                            "disable_global_features": False,
                        }
                    )
            continue
        if variant == "token_global":
            for feature_set in global_feature_sets:
                for fusion in fusion_modes:
                    out.append(
                        {
                            "key": f"token_global__{feature_set}__{fusion}",
                            "variant": "token_global",
                            "fusion_mode": fusion,
                            "global_feature_set": feature_set,
                            "token_branch_mode": "normal",
                            "disable_global_features": False,
                        }
                    )
            continue
        raise ValueError(f"Unsupported variant: {variant}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase1.5 refinement: repeatable ablations for token/global fusion.")
    parser.add_argument("--game", default="games/connectFour.kif")
    parser.add_argument("--dataset-sizes", type=int, nargs="+", default=[200, 500, 1000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variants", nargs="+", default=["token_only", "global_only", "token_global"])
    parser.add_argument("--fusion-modes", nargs="+", default=["concat", "add"])
    parser.add_argument("--global-feature-sets", nargs="+", default=["basic10", "compact6"])
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
    parser.add_argument(
        "--reuse-datasets-dir",
        default=None,
        help="Reuse existing dataset_{size}.jsonl files from this directory; if set, skip dataset generation.",
    )
    parser.add_argument("--out-dir", default=f"outputs/experiments/{EXP_NAME}")
    args = parser.parse_args()

    if int(args.repeats) < 1:
        raise ValueError("--repeats must be >= 1")

    fusion_modes = [str(x).strip().lower() for x in args.fusion_modes]
    global_feature_sets = [str(x).strip().lower() for x in args.global_feature_sets]
    for f in fusion_modes:
        if f not in {"add", "concat"}:
            raise ValueError(f"Unsupported fusion mode: {f}")
    for s in global_feature_sets:
        if s not in {"basic10", "compact6"}:
            raise ValueError(f"Unsupported global feature set: {s}")

    cfgs = _build_variant_configs(
        variants=list(args.variants),
        fusion_modes=fusion_modes,
        global_feature_sets=global_feature_sets,
    )

    layout = init_output_layout(EXP_NAME, args.out_dir, args=args)
    data_dir = layout["datasets"]
    model_root = layout["models"]

    all_raw = []
    all_summary = []
    train_eval_rows = []
    artifact_manifest = {}
    reuse_root = Path(args.reuse_datasets_dir) if args.reuse_datasets_dir else None

    for i, size in enumerate(args.dataset_sizes):
        if reuse_root is not None:
            dataset_path = reuse_root / f"dataset_{size}.jsonl"
            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Missing reused dataset file: {dataset_path}. "
                    f"Expected name format: dataset_{size}.jsonl"
                )
            print(f"[{EXP_NAME}] reuse dataset size={size} path={dataset_path}", flush=True)
        else:
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

        for repeat_id in range(int(args.repeats)):
            rep_seed = int(args.seed) + i * 1000 + repeat_id * 10000
            artifact_manifest.setdefault(str(size), {})[f"repeat_{repeat_id}"] = {}
            artifacts: dict[str, ValueArtifact] = {}

            for cfg_idx, cfg in enumerate(cfgs):
                cfg_key = cfg["key"]
                model_dir = model_root / f"size_{size}" / f"repeat_{repeat_id}" / cfg_key
                extra_args = [
                    "--global-feature-set",
                    cfg["global_feature_set"],
                    "--transformer-fusion-mode",
                    cfg["fusion_mode"],
                    "--global-hidden-dim",
                    str(int(args.global_hidden_dim)),
                    "--token-branch-mode",
                    cfg["token_branch_mode"],
                ]
                if bool(cfg["disable_global_features"]):
                    extra_args.append("--disable-global-features")

                train_value_model_artifact(
                    dataset_path=dataset_path,
                    output_dir=model_dir,
                    encoder="board_token",
                    model="transformer",
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    seed=rep_seed + cfg_idx * 17,
                    device=args.device,
                    extra_args=extra_args,
                )

                art = _artifact_for_dir(model_dir)
                artifacts[cfg_key] = art
                artifact_manifest[str(size)][f"repeat_{repeat_id}"][cfg_key] = {
                    "model_path": art.model_path,
                    "encoder_config_path": art.encoder_config_path,
                    "vocab_path": art.vocab_path,
                    "variant": cfg["variant"],
                    "fusion_mode": cfg["fusion_mode"],
                    "global_feature_set": cfg["global_feature_set"],
                    "token_branch_mode": cfg["token_branch_mode"],
                    "disable_global_features": bool(cfg["disable_global_features"]),
                }

                metrics_path = model_dir / "metrics.json"
                if metrics_path.exists():
                    m = json.loads(metrics_path.read_text(encoding="utf-8"))
                    train_eval_rows.append(
                        {
                            "dataset_games": int(size),
                            "repeat_id": int(repeat_id),
                            "config": cfg_key,
                            "variant": cfg["variant"],
                            "fusion_mode": cfg["fusion_mode"],
                            "global_feature_set": cfg["global_feature_set"],
                            "token_branch_mode": cfg["token_branch_mode"],
                            "disable_global_features": bool(cfg["disable_global_features"]),
                            "test_loss": float(m.get("test_loss", 0.0)),
                            "test_sign_acc": float(m.get("test_sign_acc", 0.0)),
                            "best_val_loss": float(m.get("best_val_loss", 0.0) or 0.0),
                        }
                    )

            pairs = [(f"value_greedy:{cfg['key']}", "random") for cfg in cfgs]
            raw_rows, summary_rows = run_match_grid(
                games=[args.game],
                pairs=pairs,
                rounds=args.eval_rounds,
                playclock=args.playclock,
                iterations=args.eval_iterations,
                seed=rep_seed + 500000,
                cache_enabled=True,
                artifacts=artifacts,
                device=args.device,
            )

            cfg_index = {f"value_greedy:{cfg['key']}": cfg for cfg in cfgs}
            for row in raw_rows:
                row["dataset_games"] = int(size)
                row["repeat_id"] = int(repeat_id)
                meta = cfg_index.get(row.get("agent_a_key"), {})
                row["variant"] = meta.get("variant")
                row["fusion_mode"] = meta.get("fusion_mode")
                row["global_feature_set"] = meta.get("global_feature_set")
            for row in summary_rows:
                row["dataset_games"] = int(size)
                row["repeat_id"] = int(repeat_id)
                meta = cfg_index.get(row.get("agent_a_key"), {})
                row["variant"] = meta.get("variant")
                row["fusion_mode"] = meta.get("fusion_mode")
                row["global_feature_set"] = meta.get("global_feature_set")

            all_raw.extend(raw_rows)
            all_summary.extend(summary_rows)

    per_repeat = collect_cross_game_mean(
        all_summary,
        ["dataset_games", "repeat_id", "agent_a_key", "agent_b_key", "variant", "fusion_mode", "global_feature_set"],
    )
    across_repeats = collect_cross_game_mean(
        all_summary,
        ["dataset_games", "agent_a_key", "agent_b_key", "variant", "fusion_mode", "global_feature_set"],
    )

    write_json(layout["artifacts"] / "artifact_manifest.json", artifact_manifest)
    write_json(layout["metrics"] / "training_metrics.json", train_eval_rows)
    write_csv(layout["metrics"] / "training_metrics.csv", train_eval_rows)
    write_json(layout["raw"] / "matches.json", all_raw)
    write_json(layout["summary"] / "by_game.json", all_summary)
    write_json(layout["summary"] / "cross_game_per_repeat.json", per_repeat)
    write_json(layout["summary"] / "cross_game_repeat_mean.json", across_repeats)
    write_csv(layout["summary"] / "by_game.csv", all_summary)
    write_csv(layout["summary"] / "cross_game_per_repeat.csv", per_repeat)
    write_csv(layout["summary"] / "cross_game_repeat_mean.csv", across_repeats)

    print(f"[{EXP_NAME}] finished. output={layout['root']}")


if __name__ == "__main__":
    main()
