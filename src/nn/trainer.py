from __future__ import annotations

"""价值模型训练器。

提供数据切分、训练循环、验证早停、测试评估与模型落盘。
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .dataset import ValueDataset
from .value_net import MLPValueNet, TransformerValueNet


def _set_seed(seed: int) -> None:
    """统一设置 Python / NumPy / Torch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_dataset(samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    """随机切分训练/验证/测试集。"""
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


def _eval_model(model, dataloader, criterion, device):
    """在验证或测试集上评估损失与符号准确率。"""
    model.eval()
    loss_sum = 0.0
    n = 0
    sign_correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = _move_to_device(x, device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            b = y.size(0)
            loss_sum += float(loss.item()) * b
            n += b
            sign_correct += int((torch.sign(pred) == torch.sign(y)).sum().item())
    if n == 0:
        return {"loss": 0.0, "sign_acc": 0.0}
    return {"loss": loss_sum / n, "sign_acc": sign_correct / n}


def _move_to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


def train_value_model(
    samples,
    encoder,
    output_dir,
    model_name="mlp",
    seed=42,
    epochs=20,
    batch_size=128,
    learning_rate=1e-3,
    weight_decay=1e-4,
    hidden_dims=(256, 128),
    dropout=0.1,
    transformer_kwargs=None,
    loss_name="mse",
    patience=5,
    device="cpu",
    num_workers=0,
):
    """训练价值网络并保存 checkpoint 与指标。"""
    _set_seed(int(seed))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(model_name).lower()
    transformer_kwargs = dict(transformer_kwargs or {})

    train_samples, val_samples, test_samples = _split_dataset(samples, seed=seed)
    train_ds = ValueDataset(train_samples, encoder)
    val_ds = ValueDataset(val_samples, encoder)
    test_ds = ValueDataset(test_samples, encoder)

    collate_fn = None
    if model_name == "transformer":
        collate_fn = ValueDataset.collate_board_tokens

    loader_workers = max(0, int(num_workers))
    loader_kwargs = {
        "num_workers": loader_workers,
        "pin_memory": str(device).lower().startswith("cuda"),
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    if model_name == "mlp":
        model = MLPValueNet(
            input_dim=encoder.input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)
    elif model_name == "transformer":
        model = TransformerValueNet(
            num_tokens=int(transformer_kwargs.get("num_tokens", getattr(encoder, "num_tokens"))),
            d_model=int(transformer_kwargs.get("d_model", 128)),
            n_heads=int(transformer_kwargs.get("n_heads", 4)),
            n_layers=int(transformer_kwargs.get("n_layers", 3)),
            dim_feedforward=int(transformer_kwargs.get("dim_feedforward", 256)),
            dropout=float(transformer_kwargs.get("dropout", dropout)),
            position_encoding=str(transformer_kwargs.get("position_encoding", "sinusoidal")),
            use_global_features=bool(transformer_kwargs.get("use_global_features", True)),
            max_positions=int(transformer_kwargs.get("max_positions", 4096)),
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss() if loss_name.lower() == "mse" else torch.nn.HuberLoss()

    # 训练轨迹：每个 epoch 一条记录，便于后续画图和分析。
    history = []
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for x, y in train_loader:
            x = _move_to_device(x, device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            # 防止梯度爆炸，稳定训练过程。
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            b = y.size(0)
            train_loss_sum += float(loss.item()) * b
            train_n += b

        train_loss = 0.0 if train_n == 0 else train_loss_sum / train_n
        val_metrics = _eval_model(model, val_loader, criterion, device=device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_metrics["loss"], "val_sign_acc": val_metrics["sign_acc"]}
        history.append(row)
        print(
            "[train_value_model] "
            f"epoch={epoch}/{int(epochs)} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={row['val_loss']:.6f} "
            f"val_sign_acc={row['val_sign_acc']:.4f}",
            flush=True,
        )

        # 早停：验证集损失不再提升时计数并提前停止。
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
            print(
                "[train_value_model] new_best "
                f"epoch={epoch} val_loss={best_val:.6f}",
                flush=True,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                print(
                    "[train_value_model] early_stop "
                    f"epoch={epoch} bad_epochs={bad_epochs} patience={int(patience)}",
                    flush=True,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _eval_model(model, test_loader, criterion, device=device)
    metrics = {
        "history": history,
        "best_val_loss": best_val if best_val != float("inf") else None,
        "test_loss": test_metrics["loss"],
        "test_sign_acc": test_metrics["sign_acc"],
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }
    print(
        "[train_value_model] "
        f"test_loss={metrics['test_loss']:.6f} "
        f"test_sign_acc={metrics['test_sign_acc']:.4f} "
        f"n_train={metrics['n_train']} n_val={metrics['n_val']} n_test={metrics['n_test']}",
        flush=True,
    )

    checkpoint = {"model_type": model_name, "state_dict": model.state_dict()}
    if model_name == "mlp":
        checkpoint.update(
            {
                "input_dim": encoder.input_dim,
                "hidden_dims": list(hidden_dims),
                "dropout": float(dropout),
            }
        )
    else:
        checkpoint.update(
            {
                "num_tokens": int(getattr(encoder, "num_tokens")),
                "d_model": int(getattr(model, "d_model")),
                "n_heads": int(transformer_kwargs.get("n_heads", 4)),
                "n_layers": int(transformer_kwargs.get("n_layers", 3)),
                "dim_feedforward": int(transformer_kwargs.get("dim_feedforward", 256)),
                "dropout": float(transformer_kwargs.get("dropout", dropout)),
                "position_encoding": str(transformer_kwargs.get("position_encoding", "sinusoidal")),
                "use_global_features": bool(transformer_kwargs.get("use_global_features", True)),
                "max_positions": int(transformer_kwargs.get("max_positions", 4096)),
            }
        )
    torch.save(checkpoint, output_dir / "model.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return model, metrics
