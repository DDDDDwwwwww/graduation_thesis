from __future__ import annotations

"""价值模型训练脚本。

流程：
1. 读取 JSONL 样本；
2. 构建词汇表与事实向量编码器；
3. 训练 MLP 价值网络并保存产物。
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
# 允许脚本在项目根目录直接执行。
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from encoding.fact_vector_encoder import FactVectorEncoder
from encoding.vocab import FactVocabulary
from nn.dataset import ValueDataset
from nn.trainer import train_value_model


def set_seed(seed: int):
    """统一设置随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="Train value model from JSONL dataset.")
    parser.add_argument("--dataset", required=True, help="Input JSONL dataset path")
    parser.add_argument("--encoder", choices=["fact_vector"], default="fact_vector")
    parser.add_argument("--model", choices=["mlp"], default="mlp")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = ValueDataset.load_jsonl(args.dataset)
    if not samples:
        raise ValueError("Dataset is empty.")

    # 基于数据集自动提取角色集合与事实词汇。
    roles = sorted({str(s.get("acting_role", "")) for s in samples if s.get("acting_role")})
    vocab = FactVocabulary.fit((s["state_facts"] for s in samples))
    encoder = FactVectorEncoder(
        vocab=vocab,
        roles=roles,
        include_role=True,
        include_turn_features=True,
    )

    model, metrics = train_value_model(
        samples=samples,
        encoder=encoder,
        output_dir=output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        loss_name=args.loss,
        patience=args.patience,
        device=args.device,
    )

    vocab.save(output_dir / "vocab.json")
    encoder.save(output_dir / "encoder.json")
    # 保存配置，便于后续复现实验。
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
        "input_dim": encoder.input_dim,
        "roles": roles,
        "num_samples": len(samples),
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[train_value_model] saved artifacts to {output_dir}")
    print(
        "[train_value_model] "
        f"test_loss={metrics['test_loss']:.6f}, test_sign_acc={metrics['test_sign_acc']:.4f}"
    )


if __name__ == "__main__":
    main()
