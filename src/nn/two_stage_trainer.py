from __future__ import annotations

"""Training helpers for fast variance value net with optional distillation."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from .value_net import MLPVarianceValueNet


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_indices(n: int, train_ratio=0.8, val_ratio=0.1, seed=42):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


class _FastTrainDataset(Dataset):
    def __init__(self, xs, ys, ts):
        self.xs = xs
        self.ys = ys
        self.ts = ts

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], self.ts[idx]


def build_teacher_targets(samples, teacher_model, teacher_encoder, device="cpu") -> list[float]:
    teacher_model.eval()
    out = []
    with torch.no_grad():
        for sample in samples:
            x = teacher_encoder.encode_facts(
                sample["state_facts"],
                role=sample.get("acting_role"),
                ply_index=sample.get("ply_index", 0),
                terminal=sample.get("terminal", False),
            )
            if isinstance(x, dict):
                batch = {}
                for k, v in x.items():
                    t = torch.as_tensor(v, device=device)
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    elif t.dim() == 2 and k == "tile_positions" and t.size(-1) == 2:
                        t = t.unsqueeze(0)
                    batch[k] = t
                pred = teacher_model(batch)
            else:
                pred = teacher_model(torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0))
            if isinstance(pred, dict):
                out.append(float(pred["value"].item()))
            else:
                out.append(float(pred.item()))
    return out


def train_fast_variance_model(
    samples,
    encoder,
    output_dir,
    *,
    distill_targets: list[float] | None = None,
    seed=42,
    epochs=20,
    batch_size=128,
    learning_rate=1e-3,
    weight_decay=1e-4,
    hidden_dims=(256, 128),
    dropout=0.1,
    patience=5,
    lambda_target=0.5,
    lambda_distill=0.5,
    lambda_uncertainty=1.0,
    min_variance=1e-4,
    device="cpu",
):
    _set_seed(int(seed))
    samples = list(samples)
    if distill_targets is not None and len(distill_targets) != len(samples):
        raise ValueError("distill_targets size mismatch.")

    xs = []
    ys = []
    ts = []
    for i, sample in enumerate(samples):
        x = encoder.encode_facts(
            sample["state_facts"],
            role=sample.get("acting_role"),
            ply_index=sample.get("ply_index", 0),
            terminal=sample.get("terminal", False),
        )
        xs.append(torch.tensor(x, dtype=torch.float32))
        ys.append(torch.tensor([float(sample["value_target"])], dtype=torch.float32))
        t = float(sample["value_target"]) if distill_targets is None else float(distill_targets[i])
        ts.append(torch.tensor([t], dtype=torch.float32))

    train_idx, val_idx, test_idx = _split_indices(len(samples), seed=seed)

    def _subset(ids):
        return _FastTrainDataset([xs[i] for i in ids], [ys[i] for i in ids], [ts[i] for i in ids])

    train_loader = DataLoader(_subset(train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_subset(val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(_subset(test_idx), batch_size=batch_size, shuffle=False)

    model = MLPVarianceValueNet(
        input_dim=int(encoder.input_dim),
        hidden_dims=hidden_dims,
        dropout=dropout,
        min_variance=min_variance,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _eval(loader):
        model.eval()
        n = 0
        nll_sum = 0.0
        mse_sum = 0.0
        with torch.no_grad():
            for x, y, t in loader:
                x = x.to(device)
                y = y.to(device)
                t = t.to(device)
                out = model(x)
                mu = out["value"]
                var = out["variance"]
                nll = 0.5 * (((t - mu) ** 2) / var + torch.log(var))
                mse = torch.mean((y - mu) ** 2)
                b = y.size(0)
                n += b
                nll_sum += float(torch.mean(nll).item()) * b
                mse_sum += float(mse.item()) * b
        if n == 0:
            return {"nll": 0.0, "mse": 0.0}
        return {"nll": nll_sum / n, "mse": mse_sum / n}

    history = []
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        n = 0
        train_loss_sum = 0.0
        for x, y, t in train_loader:
            x = x.to(device)
            y = y.to(device)
            t = t.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            mu = out["value"]
            var = out["variance"]

            nll = torch.mean(0.5 * (((t - mu) ** 2) / var + torch.log(var)))
            mse_target = torch.mean((y - mu) ** 2)
            mse_distill = torch.mean((t - mu) ** 2)
            loss = (
                float(lambda_target) * mse_target
                + float(lambda_distill) * mse_distill
                + float(lambda_uncertainty) * nll
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            b = y.size(0)
            n += b
            train_loss_sum += float(loss.item()) * b

        train_loss = 0.0 if n == 0 else train_loss_sum / n
        val_metrics = _eval(val_loader)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_nll": val_metrics["nll"],
            "val_mse": val_metrics["mse"],
        }
        history.append(row)
        if row["val_nll"] < best_val:
            best_val = row["val_nll"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = _eval(test_loader)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": "mlp_variance",
        "state_dict": model.state_dict(),
        "input_dim": int(encoder.input_dim),
        "hidden_dims": list(hidden_dims),
        "dropout": float(dropout),
        "min_variance": float(min_variance),
    }
    torch.save(checkpoint, output_dir / "model.pt")
    metrics = {
        "history": history,
        "best_val_nll": best_val if best_val != float("inf") else None,
        "test_nll": test_metrics["nll"],
        "test_mse": test_metrics["mse"],
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "lambda_target": float(lambda_target),
        "lambda_distill": float(lambda_distill),
        "lambda_uncertainty": float(lambda_uncertainty),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return model, metrics
