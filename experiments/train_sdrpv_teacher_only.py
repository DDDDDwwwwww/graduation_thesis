from __future__ import annotations

"""Teacher-only training entry for SDRPV datasets.

This script converts SDRPV rows to the existing training sample schema and
trains a value model supervised by teacher targets (q_t by default).
"""

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
from encoding.vocab import FactVocabulary
from nn.trainer import train_value_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clip_unit(x: float) -> float:
    return max(-1.0, min(1.0, float(x)))


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def convert_sdrpv_to_training_samples(
    rows: list[dict],
    target_field: str = "q_t",
    game_filter: str | None = None,
) -> tuple[list[dict], dict]:
    samples: list[dict] = []
    stats = {
        "total_input": len(rows),
        "kept": 0,
        "skipped": 0,
        "missing_s": 0,
        "missing_target": 0,
        "game_filtered": 0,
    }
    for row in rows:
        if game_filter and str(row.get("game_name", "")) != str(game_filter):
            stats["game_filtered"] += 1
            stats["skipped"] += 1
            continue

        s = row.get("s")
        if not isinstance(s, dict):
            stats["missing_s"] += 1
            stats["skipped"] += 1
            continue

        if target_field not in row:
            stats["missing_target"] += 1
            stats["skipped"] += 1
            continue

        facts = s.get("state_facts")
        role = s.get("acting_role")
        if not isinstance(facts, list) or role is None:
            stats["missing_s"] += 1
            stats["skipped"] += 1
            continue

        y = clip_unit(float(row[target_field]))
        sample = {
            "state_facts": [str(x) for x in facts],
            "acting_role": str(role),
            "ply_index": int(s.get("ply_index", row.get("ply_index", 0)) or 0),
            "terminal": bool(s.get("terminal", False)),
            "value_target": y,
            "game_name": row.get("game_name"),
            "match_id": row.get("match_id"),
            "source_agent": row.get("source_agent"),
            "teacher_sims": row.get("teacher_sims"),
            "baseline_mode": row.get("baseline_mode"),
        }
        samples.append(sample)
        stats["kept"] += 1
    return samples, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train teacher-only value model from SDRPV JSONL.")
    parser.add_argument("--dataset", required=True, help="Input SDRPV JSONL path.")
    parser.add_argument("--target-field", choices=["q_t", "z"], default="q_t")
    parser.add_argument("--game", default=None, help="Optional game_name filter")
    parser.add_argument("--encoder", choices=["fact_vector", "board_token"], default="board_token")
    parser.add_argument("--model", choices=["mlp", "transformer"], default="transformer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.1)
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.dataset)
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]
    if not rows:
        raise ValueError("Dataset is empty.")

    samples, convert_stats = convert_sdrpv_to_training_samples(
        rows=rows,
        target_field=args.target_field,
        game_filter=args.game,
    )
    if not samples:
        raise ValueError(f"No valid samples after conversion. stats={convert_stats}")

    roles = sorted({str(s.get("acting_role", "")) for s in samples if s.get("acting_role")})
    vocab = None
    if args.encoder == "fact_vector":
        if args.model != "mlp":
            raise ValueError("encoder=fact_vector only supports model=mlp")
        vocab = FactVocabulary.fit((s["state_facts"] for s in samples))
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

    _, metrics = train_value_model(
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
        "mode": "teacher_only",
        "source_dataset": args.dataset,
        "target_field": args.target_field,
        "game_filter": args.game,
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
        "num_input_rows": len(rows),
        "num_samples_used": len(samples),
        "conversion_stats": convert_stats,
        "test_loss": metrics["test_loss"],
        "test_sign_acc": metrics["test_sign_acc"],
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[train_sdrpv_teacher_only] saved artifacts to {output_dir}")
    print(
        "[train_sdrpv_teacher_only] "
        f"samples={len(samples)}, test_loss={metrics['test_loss']:.6f}, "
        f"test_sign_acc={metrics['test_sign_acc']:.4f}"
    )


if __name__ == "__main__":
    main()

