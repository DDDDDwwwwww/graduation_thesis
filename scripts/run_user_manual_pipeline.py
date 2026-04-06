from __future__ import annotations

"""Unified pipeline for thesis User Manual reproduction.

This script avoids legacy shell wrappers and calls experiment entries directly.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"


def _run(cmd: list[str], cwd: Path) -> None:
    print("[user_manual_pipeline] run:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SDRPV user-manual pipeline.")
    parser.add_argument(
        "--dataset",
        default="outputs/datasets/sdrpv_dataset_v3_parallel.jsonl",
        help="SDRPV dataset JSONL path.",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed-train", type=int, default=42)
    parser.add_argument("--seed-eval", default="242")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small quick-check setup (fast smoke run).",
    )
    args = parser.parse_args()

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else Path("outputs/experiments") / f"UM_pipeline_{run_tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    epochs = 2 if args.quick else int(args.epochs)
    rounds = 4 if args.quick else int(args.rounds)
    max_samples_args = ["--max-samples", "400"] if args.quick else []

    residual_dir = out_root / "residual_v1_model"
    main_benchmark_dir = out_root / "main_benchmark"
    manifest_path = out_root / "meta" / "run_manifest.json"

    cmd_train = [
        args.python_bin,
        str(EXP_DIR / "train_sdrpv_residual_v1.py"),
        "--dataset",
        str(dataset_path),
        "--target-field",
        "q_t",
        "--encoder",
        "board_token",
        "--model",
        "transformer",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(int(args.batch_size)),
        "--loss",
        "huber",
        "--seed",
        str(int(args.seed_train)),
        "--num-workers",
        "0",
        "--output-dir",
        str(residual_dir),
        "--device",
        str(args.device),
        *max_samples_args,
    ]
    _run(cmd_train, cwd=ROOT)

    cmd_main = [
        args.python_bin,
        str(EXP_DIR / "run_sdrpv_residual_v1_baseline_benchmark.py"),
        "--model-dir",
        str(residual_dir),
        "--games",
        "games/breakthrough.kif",
        "games/connectFour.kif",
        "games/hex.kif",
        "--rounds",
        str(rounds),
        "--seeds",
        str(args.seed_eval),
        "--fixed-sims",
        "120",
        "--fixed-sims-playclock",
        "0.5",
        "--fixed-time",
        "0.5",
        "--fixed-time-iters",
        "120",
        "--device",
        str(args.device),
        "--out-dir",
        str(main_benchmark_dir),
    ]
    _run(cmd_main, cwd=ROOT)

    ablation_train_dir = None
    ablation_benchmark_dir = None
    if args.run_ablation:
        ablation_train_dir = out_root / "ablation_train"
        ablation_benchmark_dir = out_root / "ablation_benchmark"

        cmd_ablation_train = [
            args.python_bin,
            str(EXP_DIR / "run_sdrpv_ablation_train.py"),
            "--dataset",
            str(dataset_path),
            "--encoder",
            "board_token",
            "--model",
            "transformer",
            "--epochs",
            str(epochs),
            "--batch-size",
            str(int(args.batch_size)),
            "--loss",
            "huber",
            "--seed",
            str(int(args.seed_train)),
            "--device",
            str(args.device),
            "--out-dir",
            str(ablation_train_dir),
            *max_samples_args,
        ]
        _run(cmd_ablation_train, cwd=ROOT)

        models_root = ablation_train_dir / "models"
        cmd_ablation_benchmark = [
            args.python_bin,
            str(EXP_DIR / "run_sdrpv_ablation_benchmark.py"),
            "--models-root",
            str(models_root),
            "--games",
            "games/breakthrough.kif",
            "games/connectFour.kif",
            "games/hex.kif",
            "--rounds",
            str(rounds),
            "--seeds",
            str(args.seed_eval),
            "--fixed-sims",
            "120",
            "--fixed-sims-playclock",
            "0.5",
            "--fixed-time",
            "0.5",
            "--fixed-time-iters",
            "120",
            "--device",
            str(args.device),
            "--out-dir",
            str(ablation_benchmark_dir),
        ]
        _run(cmd_ablation_benchmark, cwd=ROOT)

    _write_json(
        manifest_path,
        {
            "pipeline": "user_manual",
            "run_tag": run_tag,
            "quick": bool(args.quick),
            "run_ablation": bool(args.run_ablation),
            "dataset": str(dataset_path),
            "device": str(args.device),
            "seed_train": int(args.seed_train),
            "seed_eval": str(args.seed_eval),
            "epochs": epochs,
            "rounds": rounds,
            "residual_model_dir": str(residual_dir),
            "main_benchmark_dir": str(main_benchmark_dir),
            "ablation_train_dir": (str(ablation_train_dir) if ablation_train_dir else None),
            "ablation_benchmark_dir": (str(ablation_benchmark_dir) if ablation_benchmark_dir else None),
        },
    )
    print(f"[user_manual_pipeline] done. out_root={out_root}", flush=True)


if __name__ == "__main__":
    main()

