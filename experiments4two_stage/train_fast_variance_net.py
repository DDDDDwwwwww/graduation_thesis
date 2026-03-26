from __future__ import annotations

"""Train FastValueNet (value + variance) with optional distillation."""

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from encoding.board_token_mlp_encoder import BoardTokenMLPEncoder
from nn.dataset import ValueDataset
from nn.inference import load_value_artifacts
from nn.two_stage_trainer import build_teacher_targets, train_fast_variance_model


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FastValueNet variance-head model.")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--teacher-model", required=True, help="Slow model path for distillation")
    parser.add_argument("--teacher-encoder", required=True, help="Slow encoder.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lambda-target", type=float, default=0.5)
    parser.add_argument("--lambda-distill", type=float, default=0.5)
    parser.add_argument("--lambda-uncertainty", type=float, default=1.0)
    parser.add_argument("--position-mode", choices=["index", "xy"], default="index")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    _set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = ValueDataset.load_jsonl(args.dataset)
    encoder = BoardTokenMLPEncoder.fit(samples=samples, position_mode=args.position_mode)

    teacher_model, teacher_encoder, _ = load_value_artifacts(
        model_path=args.teacher_model,
        vocab_path=None,
        encoder_config_path=args.teacher_encoder,
        device=args.device,
    )
    distill_targets = build_teacher_targets(samples, teacher_model, teacher_encoder, device=args.device)

    model, metrics = train_fast_variance_model(
        samples=samples,
        encoder=encoder,
        output_dir=output_dir,
        distill_targets=distill_targets,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        patience=args.patience,
        lambda_target=args.lambda_target,
        lambda_distill=args.lambda_distill,
        lambda_uncertainty=args.lambda_uncertainty,
        device=args.device,
    )
    del model

    encoder.save(output_dir / "encoder.json")
    config = {
        "dataset": str(Path(args.dataset).resolve()),
        "teacher_model": str(Path(args.teacher_model).resolve()),
        "teacher_encoder": str(Path(args.teacher_encoder).resolve()),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "patience": args.patience,
        "lambda_target": args.lambda_target,
        "lambda_distill": args.lambda_distill,
        "lambda_uncertainty": args.lambda_uncertainty,
        "device": args.device,
        "num_samples": len(samples),
        "best_val_nll": metrics.get("best_val_nll"),
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train_fast_variance_net] saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
