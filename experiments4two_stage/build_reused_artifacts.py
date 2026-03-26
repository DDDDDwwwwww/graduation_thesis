from __future__ import annotations

"""Build two-stage artifact map by reusing previous experiment outputs."""

import argparse
import json
from pathlib import Path

from experiment_utils import ROOT, ensure_dir, write_json


def _find_existing_artifact_manifest() -> Path:
    candidates = [
        ROOT / "outputs" / "experiments" / "D_dataset_size_sensitivity" / "artifacts" / "artifact_manifest.json",
        ROOT / "Preliminary_experimental" / "outputs" / "experiments" / "D_dataset_size_sensitivity" / "artifacts" / "artifact_manifest.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find previous D_dataset_size_sensitivity artifact_manifest.json in outputs/ or Preliminary_experimental/outputs/.")


def _resolve_from_manifest(manifest_path: Path, size_key: str, model_key: str) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = payload[str(size_key)][model_key]

    def _to_abs(path_like):
        p = Path(path_like)
        if p.is_absolute():
            return str(p)
        base1 = ROOT / path_like
        if base1.exists():
            return str(base1.resolve())
        root_from_manifest = manifest_path.parent.parent.parent.parent
        parts = list(p.parts)
        if parts and parts[0].lower() == "outputs":
            parts = parts[1:]
        base2 = root_from_manifest.joinpath(*parts)
        return str(base2.resolve())

    return {
        "model_path": _to_abs(row["model_path"]),
        "encoder_config_path": _to_abs(row["encoder_config_path"]),
        "vocab_path": None if row.get("vocab_path") in (None, "") else _to_abs(row["vocab_path"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reusable two-stage artifacts map.")
    parser.add_argument("--manifest", default=None, help="Optional path to previous D artifact_manifest.json")
    parser.add_argument("--fast-size", type=int, default=1000)
    parser.add_argument("--slow-size", type=int, default=2000)
    parser.add_argument("--out-dir", default="outputs4two_stage/artifacts")
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else _find_existing_artifact_manifest()
    fast = _resolve_from_manifest(manifest_path, str(args.fast_size), "token_mlp")
    slow = _resolve_from_manifest(manifest_path, str(args.slow_size), "token_transformer")

    payload = {
        "token_mlp": fast,
        "token_transformer": slow,
        "two_stage_neural_mcts": {
            "fast_model_path": fast["model_path"],
            "fast_encoder_config_path": fast["encoder_config_path"],
            "fast_vocab_path": fast["vocab_path"],
            "slow_model_path": slow["model_path"],
            "slow_encoder_config_path": slow["encoder_config_path"],
            "slow_vocab_path": slow["vocab_path"],
        },
        "reuse_meta": {
            "source_manifest": str(manifest_path.resolve()),
            "fast_source": f"token_mlp@size_{args.fast_size}",
            "slow_source": f"token_transformer@size_{args.slow_size}",
        },
    }

    out_dir = ensure_dir(args.out_dir)
    out_path = out_dir / "two_stage_artifacts.json"
    write_json(out_path, payload)
    print(f"[build_reused_artifacts] saved: {out_path}")


if __name__ == "__main__":
    main()
