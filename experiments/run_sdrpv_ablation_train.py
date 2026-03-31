from __future__ import annotations

"""Train the core SDRPV ablation variants under a unified setup.

Variants included:
1) full_residual_teacher: residual target on q_t
2) no_residual_teacher: direct value prediction on q_t
3) no_teacher_residual: residual target on z
"""

import argparse
import sys
from pathlib import Path

from experiment_utils import init_output_layout, run_cmd, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def _base_train_args(args: argparse.Namespace) -> list[str]:
    out = [
        "--dataset",
        str(args.dataset),
        "--encoder",
        str(args.encoder),
        "--model",
        str(args.model),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--weight-decay",
        str(float(args.weight_decay)),
        "--seed",
        str(int(args.seed)),
        "--loss",
        str(args.loss),
        "--patience",
        str(int(args.patience)),
        "--position-mode",
        str(args.position_mode),
        "--d-model",
        str(int(args.d_model)),
        "--n-heads",
        str(int(args.n_heads)),
        "--n-layers",
        str(int(args.n_layers)),
        "--dim-feedforward",
        str(int(args.dim_feedforward)),
        "--position-encoding",
        str(args.position_encoding),
        "--max-positions",
        str(int(args.max_positions)),
        "--device",
        str(args.device),
    ]
    if args.game:
        out.extend(["--game", str(args.game)])
    if args.max_samples is not None:
        out.extend(["--max-samples", str(int(args.max_samples))])
    if args.disable_global_features:
        out.append("--disable-global-features")
    return out


def _run_variant(
    python_bin: str,
    script_name: str,
    variant_dir: Path,
    extra_args: list[str],
    common_args: list[str],
) -> None:
    cmd = [
        python_bin,
        str(ROOT / "experiments" / script_name),
        *common_args,
        *extra_args,
        "--output-dir",
        str(variant_dir),
    ]
    run_cmd(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SDRPV ablation variants.")
    parser.add_argument("--dataset", required=True, help="Input SDRPV JSONL path.")
    parser.add_argument("--game", default=None, help="Optional game_name filter.")
    parser.add_argument("--encoder", choices=["fact_vector", "board_token"], default="board_token")
    parser.add_argument("--model", choices=["mlp", "transformer"], default="transformer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--position-mode", choices=["index", "xy"], default="index")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--position-encoding", choices=["sinusoidal", "learned"], default="sinusoidal")
    parser.add_argument("--max-positions", type=int, default=4096)
    parser.add_argument("--disable-global-features", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--out-dir", default="outputs/experiments/SDRPV_ablation_train")
    args = parser.parse_args()

    layout = init_output_layout("SDRPV_ablation_train", args.out_dir, args=args)
    models_root = layout["models"]
    common_args = _base_train_args(args)

    variants = [
        {
            "tag": "01_full_residual_teacher",
            "script_name": "train_sdrpv_residual_v1.py",
            "extra_args": ["--target-field", "q_t", "--num-workers", "0"],
            "family": "full",
            "target_field": "q_t",
            "uses_residual": True,
        },
        {
            "tag": "02_ablation_no_residual_teacher",
            "script_name": "train_sdrpv_teacher_only.py",
            "extra_args": ["--target-field", "q_t"],
            "family": "ablation_no_residual",
            "target_field": "q_t",
            "uses_residual": False,
        },
        {
            "tag": "03_ablation_no_teacher_residual",
            "script_name": "train_sdrpv_residual_v1.py",
            "extra_args": ["--target-field", "z", "--num-workers", "0"],
            "family": "ablation_no_teacher",
            "target_field": "z",
            "uses_residual": True,
        },
    ]

    manifest_rows: list[dict] = []
    for variant in variants:
        variant_dir = models_root / str(variant["tag"])
        _run_variant(
            python_bin=str(args.python_bin),
            script_name=str(variant["script_name"]),
            variant_dir=variant_dir,
            extra_args=list(variant["extra_args"]),
            common_args=list(common_args),
        )
        manifest_rows.append(
            {
                "tag": str(variant["tag"]),
                "family": str(variant["family"]),
                "script_name": str(variant["script_name"]),
                "target_field": str(variant["target_field"]),
                "uses_residual": bool(variant["uses_residual"]),
                "model_dir": str(variant_dir),
                "model_path": str(variant_dir / "model.pt"),
                "encoder_config_path": str(variant_dir / "encoder.json"),
                "config_path": str(variant_dir / "config.json"),
            }
        )

    write_json(layout["summary"] / "model_manifest.json", manifest_rows)
    write_csv(layout["summary"] / "model_manifest.csv", manifest_rows)
    print(f"[sdrpv_ablation_train] done. output={layout['root']}")


if __name__ == "__main__":
    main()
