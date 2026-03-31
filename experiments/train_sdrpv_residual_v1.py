from __future__ import annotations

"""Residual-v1 training entry for SDRPV datasets.

This script trains a value model to predict clipped residual targets:
  delta = clip(target - b, -1, 1)
where ``target`` is ``q_t`` by default and can optionally be ``z`` for
ablation runs.

Offline validation reports compare:
  - corr(v_hat, target) / rank correlation
  - MAE(v_hat, target) vs MAE(b, target)
where v_hat = clip(b + delta_hat, -1, 1).
"""

import argparse
import json
import math
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
from nn.inference import predict_value
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


def _split_dataset(samples: list[dict], train_ratio=0.8, val_ratio=0.1, seed=42):
    idx = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = idx[:n_train]
    val_ids = idx[n_train:n_train + n_val]
    test_ids = idx[n_train + n_val:]
    return (
        [samples[i] for i in train_ids],
        [samples[i] for i in val_ids],
        [samples[i] for i in test_ids],
    )


def convert_sdrpv_to_residual_samples(
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
        "missing_b": 0,
        "missing_target": 0,
        "game_filtered": 0,
        "residual_clipped_count": 0,
        "target_field": target_field,
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

        if "b" not in row:
            stats["missing_b"] += 1
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

        b = clip_unit(float(row["b"]))
        target_value = clip_unit(float(row[target_field]))
        residual_raw = target_value - b
        residual_target = clip_unit(residual_raw)
        if abs(residual_raw - residual_target) > 1e-12:
            stats["residual_clipped_count"] += 1

        sample = {
            "state_facts": [str(x) for x in facts],
            "acting_role": str(role),
            "ply_index": int(s.get("ply_index", row.get("ply_index", 0)) or 0),
            "terminal": bool(s.get("terminal", False)),
            "value_target": float(residual_target),
            "baseline_b": float(b),
            "supervision_target": float(target_value),
            "target_field": str(target_field),
            "game_name": row.get("game_name"),
            "match_id": row.get("match_id"),
            "source_agent": row.get("source_agent"),
            "teacher_sims": row.get("teacher_sims"),
            "baseline_mode": row.get("baseline_mode"),
        }
        samples.append(sample)
        stats["kept"] += 1
    return samples, stats


def _rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    n = int(values.size)
    if n == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = ((i + 1) + j) / 2.0
        ranks_sorted[i:j] = avg_rank
        i = j
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    am = a.mean()
    bm = b.mean()
    da = a - am
    db = b - bm
    num = float(np.sum(da * db))
    den = math.sqrt(float(np.sum(da * da)) * float(np.sum(db * db)))
    if den <= 1e-12:
        return 0.0
    return num / den


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata_average_ties(a)
    rb = _rankdata_average_ties(b)
    return _pearson_corr(ra, rb)


def evaluate_residual_offline(
    samples: list[dict],
    model,
    encoder,
    device: str,
    target_field: str,
) -> dict:
    if not samples:
        out = {
            "count": 0,
            "target_field": target_field,
            "corr_vhat_target": 0.0,
            "rank_corr_vhat_target": 0.0,
            "mae_vhat_target": 0.0,
            "mae_b_target": 0.0,
            "mae_gain_over_b": 0.0,
            "mae_gain_ratio_over_b": 0.0,
            "corr_b_target": 0.0,
            "rank_corr_b_target": 0.0,
        }
        if target_field == "q_t":
            out.update(
                {
                    "corr_vhat_qt": 0.0,
                    "rank_corr_vhat_qt": 0.0,
                    "mae_vhat_qt": 0.0,
                    "mae_b_qt": 0.0,
                    "corr_b_qt": 0.0,
                    "rank_corr_b_qt": 0.0,
                }
            )
        return out

    targets: list[float] = []
    bs: list[float] = []
    v_hats: list[float] = []
    with torch.no_grad():
        for s in samples:
            facts = s["state_facts"]
            role = s.get("acting_role")
            ply_index = int(s.get("ply_index", 0) or 0)
            terminal = bool(s.get("terminal", False))
            if hasattr(encoder, "encode_facts"):
                x = encoder.encode_facts(
                    facts,
                    role=role,
                    ply_index=ply_index,
                    terminal=terminal,
                )
            else:
                x = encoder.encode(
                    facts,
                    game=None,
                    role=role,
                    ply_index=ply_index,
                    terminal=terminal,
                )
            delta_hat = float(predict_value(model, x, device=device))
            b = float(s["baseline_b"])
            target_value = float(s["supervision_target"])
            v_hat = clip_unit(b + delta_hat)
            bs.append(b)
            targets.append(target_value)
            v_hats.append(v_hat)

    arr_target = np.asarray(targets, dtype=np.float64)
    arr_b = np.asarray(bs, dtype=np.float64)
    arr_v = np.asarray(v_hats, dtype=np.float64)
    mae_vhat_target = float(np.mean(np.abs(arr_v - arr_target)))
    mae_b_target = float(np.mean(np.abs(arr_b - arr_target)))
    mae_gain = mae_b_target - mae_vhat_target
    mae_gain_ratio = 0.0 if mae_b_target <= 1e-12 else mae_gain / mae_b_target
    out = {
        "count": int(arr_target.size),
        "target_field": target_field,
        "corr_vhat_target": _pearson_corr(arr_v, arr_target),
        "rank_corr_vhat_target": _spearman_corr(arr_v, arr_target),
        "mae_vhat_target": mae_vhat_target,
        "mae_b_target": mae_b_target,
        "mae_gain_over_b": mae_gain,
        "mae_gain_ratio_over_b": float(mae_gain_ratio),
        "corr_b_target": _pearson_corr(arr_b, arr_target),
        "rank_corr_b_target": _spearman_corr(arr_b, arr_target),
    }
    if target_field == "q_t":
        out.update(
            {
                "corr_vhat_qt": out["corr_vhat_target"],
                "rank_corr_vhat_qt": out["rank_corr_vhat_target"],
                "mae_vhat_qt": out["mae_vhat_target"],
                "mae_b_qt": out["mae_b_target"],
                "corr_b_qt": out["corr_b_target"],
                "rank_corr_b_qt": out["rank_corr_b_target"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual_v1 model from SDRPV JSONL.")
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
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
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

    samples, convert_stats = convert_sdrpv_to_residual_samples(
        rows=rows,
        target_field=args.target_field,
        game_filter=args.game,
    )
    if not samples:
        raise ValueError(f"No valid samples after conversion. stats={convert_stats}")

    train_samples, val_samples, test_samples = _split_dataset(samples=samples, seed=args.seed)

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
        num_workers=args.num_workers,
    )

    encoder.save(output_dir / "encoder.json")
    if vocab is not None:
        vocab.save(output_dir / "vocab.json")

    offline_metrics = {
        "all": evaluate_residual_offline(
            samples=samples,
            model=model,
            encoder=encoder,
            device=args.device,
            target_field=args.target_field,
        ),
        "train": evaluate_residual_offline(
            samples=train_samples,
            model=model,
            encoder=encoder,
            device=args.device,
            target_field=args.target_field,
        ),
        "val": evaluate_residual_offline(
            samples=val_samples,
            model=model,
            encoder=encoder,
            device=args.device,
            target_field=args.target_field,
        ),
        "test": evaluate_residual_offline(
            samples=test_samples,
            model=model,
            encoder=encoder,
            device=args.device,
            target_field=args.target_field,
        ),
    }
    (output_dir / "offline_metrics.json").write_text(
        json.dumps(offline_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config = {
        "mode": "residual_v1",
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
        "num_workers": args.num_workers,
        "roles": roles,
        "num_input_rows": len(rows),
        "num_samples_used": len(samples),
        "conversion_stats": convert_stats,
        "split_sizes": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "test_loss_delta": metrics["test_loss"],
        "test_sign_acc_delta": metrics["test_sign_acc"],
        "offline_test_metrics": offline_metrics["test"],
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[train_sdrpv_residual_v1] saved artifacts to {output_dir}")
    print(
        "[train_sdrpv_residual_v1] "
        f"samples={len(samples)}, test_loss_delta={metrics['test_loss']:.6f}, "
        f"target_field={args.target_field}, "
        f"test_corr_vhat_target={offline_metrics['test']['corr_vhat_target']:.4f}, "
        f"test_rank_corr_vhat_target={offline_metrics['test']['rank_corr_vhat_target']:.4f}, "
        f"test_mae_vhat_target={offline_metrics['test']['mae_vhat_target']:.6f}, "
        f"test_mae_b_target={offline_metrics['test']['mae_b_target']:.6f}"
    )


if __name__ == "__main__":
    main()
