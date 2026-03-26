from __future__ import annotations

"""Train value models from JSONL self-play datasets."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from encoding.board_token_encoder import BoardTokenEncoder
from encoding.board_token_mlp_encoder import BoardTokenMLPEncoder
from encoding.fact_vector_encoder import FactVectorEncoder
from encoding.vocab import FactVocabulary
from nn.dataset import ValueDataset
from nn.trainer import train_value_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train value model from JSONL dataset.")
    parser.add_argument("--dataset", required=True, help="Input JSONL dataset path")
    parser.add_argument("--encoder", choices=["fact_vector", "board_token"], default="fact_vector")
    parser.add_argument("--model", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--position-mode", choices=["index", "xy"], default="index")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--position-encoding", choices=["sinusoidal", "learned"], default="sinusoidal")
    parser.add_argument("--max-positions", type=int, default=4096)
    parser.add_argument("--disable-global-features", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = ValueDataset.load_jsonl(args.dataset)
    if not samples:
        raise ValueError("Dataset is empty.")

    roles = sorted({str(s.get("acting_role", "")) for s in samples if s.get("acting_role")})
    vocab = None

    if args.encoder == "fact_vector":
        if args.model != "mlp":
            raise ValueError("encoder=fact_vector only supports model=mlp")
        vocab = FactVocabulary.fit((s["state_facts"] for s in samples))
        encoder = FactVectorEncoder(
            vocab=vocab,
            roles=roles,
            include_role=True,
            include_turn_features=True,
        )
    else:
        if args.model == "transformer":
            encoder = BoardTokenEncoder(
                position_mode=args.position_mode,
                include_player_feature=not args.disable_global_features,
                include_turn_features=not args.disable_global_features,
            ).fit(samples)
        elif args.model == "mlp":
            encoder = BoardTokenMLPEncoder.fit(
                samples=samples,
                position_mode=args.position_mode,
                include_player_feature=not args.disable_global_features,
                include_turn_features=not args.disable_global_features,
                normalize_counts=True,
            )
        else:
            raise ValueError(f"Unsupported model for board_token encoder: {args.model}")

    model, metrics = train_value_model(
        samples=samples,
        encoder=encoder,
        output_dir=output_dir,
        model_name=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        transformer_kwargs={
            "num_tokens": int(getattr(encoder, "num_tokens", 0)),
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "position_encoding": args.position_encoding,
            "max_positions": args.max_positions,
            "use_global_features": not args.disable_global_features,
        },
        loss_name=args.loss,
        patience=args.patience,
        device=args.device,
    )

    encoder.save(output_dir / "encoder.json")
    if vocab is not None:
        vocab.save(output_dir / "vocab.json")

    config = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "loss": args.loss,
        "patience": args.patience,
        "device": args.device,
        "roles": roles,
        "num_samples": len(samples),
    }

    if args.model == "mlp":
        config["input_dim"] = encoder.input_dim
        if args.encoder == "board_token":
            config.update(
                {
                    "position_mode": args.position_mode,
                    "num_tokens": encoder.board_encoder.num_tokens,
                    "use_global_features": (not args.disable_global_features),
                    "global_feature_dim": encoder.board_encoder.global_feature_dim,
                    "token_pooling": "normalized_histogram",
                }
            )
    else:
        config.update(
            {
                "position_mode": args.position_mode,
                "num_tokens": encoder.num_tokens,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dim_feedforward": args.dim_feedforward,
                "position_encoding": args.position_encoding,
                "max_positions": args.max_positions,
                "use_global_features": (not args.disable_global_features),
                "global_feature_dim": encoder.global_feature_dim,
            }
        )

    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[train_value_model] saved artifacts to {output_dir}")
    print(
        "[train_value_model] "
        f"test_loss={metrics['test_loss']:.6f}, test_sign_acc={metrics['test_sign_acc']:.4f}"
    )


if __name__ == "__main__":
    main()
