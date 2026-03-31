from __future__ import annotations

import argparse
from pathlib import Path

from experiment_utils import read_json, summarize_series, write_csv, write_json


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n_rows": 0,
            "n_matches": 0,
            "win_rate_mean": 0.0,
            "draw_rate_mean": 0.0,
            "avg_decision_sec_a_mean": 0.0,
            "avg_eval_calls_neural_a_mean": 0.0,
            "avg_eval_calls_fallback_a_mean": 0.0,
        }
    n_rows = len(rows)
    n_matches = sum(int(r.get("n_matches", 0)) for r in rows)
    return {
        "n_rows": n_rows,
        "n_matches": n_matches,
        "win_rate_mean": sum(float(r.get("win_rate_a", 0.0)) for r in rows) / n_rows,
        "draw_rate_mean": sum(float(r.get("draw_rate", 0.0)) for r in rows) / n_rows,
        "avg_decision_sec_a_mean": sum(float(r.get("avg_decision_sec_a", 0.0)) for r in rows) / n_rows,
        "avg_eval_calls_neural_a_mean": sum(float(r.get("avg_eval_calls_neural_a", 0.0)) for r in rows) / n_rows,
        "avg_eval_calls_fallback_a_mean": sum(float(r.get("avg_eval_calls_fallback_a", 0.0)) for r in rows) / n_rows,
    }


def _opponents(base_dir: Path) -> list[str]:
    out = []
    for p in sorted(base_dir.iterdir()):
        if p.is_dir() and p.name not in {"summary", "meta", "raw", "metrics", "artifacts", "datasets", "models"}:
            out.append(p.name)
    return out


def _seed_dirs(opponent_dir: Path) -> list[Path]:
    return sorted([p for p in opponent_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])


def _mode_paths(seed_dir: Path, mode: str) -> tuple[Path, Path]:
    mode_dir = seed_dir / mode
    return mode_dir / "raw" / "matches.json", mode_dir / "summary" / "by_game.json"


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected list JSON: {path}")


def _write_mode(seed_dir: Path, mode: str, raw_rows: list[dict]) -> list[dict]:
    raw_path, summary_json_path = _mode_paths(seed_dir, mode)
    summary_csv_path = summary_json_path.with_suffix(".csv")
    by_game: dict[str, list[dict]] = {}
    for row in raw_rows:
        by_game.setdefault(str(row["game"]), []).append(row)
    summary_rows = [summarize_series(by_game[g]) for g in sorted(by_game.keys())]
    write_json(raw_path, raw_rows)
    write_json(summary_json_path, summary_rows)
    write_csv(summary_csv_path, summary_rows)
    return summary_rows


def _parse_seed(seed_dir_name: str) -> int:
    return int(seed_dir_name.split("_", 1)[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge hex-only baseline benchmark outputs into an existing baseline benchmark directory."
    )
    parser.add_argument(
        "--base-dir",
        default="outputs/experiments/SDRPV_residual_v1_baseline_benchmark",
        help="Existing 3-game baseline benchmark directory.",
    )
    parser.add_argument(
        "--hex-dir",
        required=True,
        help="Hex-only run directory produced by run_sdrpv_residual_v1_baseline_benchmark.py --games games/hex.kif",
    )
    parser.add_argument("--hex-game-name", default="hex", help="Game name to merge (default: hex).")
    parser.add_argument("--hex-game-file", default="games/hex.kif", help="Game file path used in manifest update.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    hex_dir = Path(args.hex_dir)
    hex_game_name = str(args.hex_game_name)
    hex_game_file = str(args.hex_game_file)

    if not base_dir.exists():
        raise FileNotFoundError(f"base dir not found: {base_dir}")
    if not hex_dir.exists():
        raise FileNotFoundError(f"hex dir not found: {hex_dir}")

    per_seed_rows: list[dict] = []
    by_game_average_rows: list[dict] = []
    aggregate_rows: list[dict] = []

    opponents = _opponents(base_dir)
    for opponent_tag in opponents:
        base_opponent_dir = base_dir / opponent_tag
        hex_opponent_dir = hex_dir / opponent_tag
        if not hex_opponent_dir.exists():
            raise FileNotFoundError(f"hex opponent dir missing: {hex_opponent_dir}")

        all_fixed_sims_rows: list[dict] = []
        all_fixed_time_rows: list[dict] = []
        agent_a_key = None
        agent_b_key = None

        for seed_dir in _seed_dirs(base_opponent_dir):
            seed = _parse_seed(seed_dir.name)
            hex_seed_dir = hex_opponent_dir / seed_dir.name
            if not hex_seed_dir.exists():
                raise FileNotFoundError(f"hex seed dir missing: {hex_seed_dir}")

            seed_summaries: dict[str, list[dict]] = {}
            for mode in ("fixed_sims", "fixed_time"):
                base_raw_path, _ = _mode_paths(seed_dir, mode)
                hex_raw_path, _ = _mode_paths(hex_seed_dir, mode)
                base_raw = _load_rows(base_raw_path)
                hex_raw = _load_rows(hex_raw_path)

                if not hex_raw:
                    raise ValueError(f"hex raw rows empty: {hex_raw_path}")

                filtered_base = [row for row in base_raw if str(row.get("game")) != hex_game_name]
                filtered_hex = [row for row in hex_raw if str(row.get("game")) == hex_game_name]
                merged_raw = filtered_base + filtered_hex
                summary_rows = _write_mode(seed_dir, mode, merged_raw)
                seed_summaries[mode] = summary_rows

                if summary_rows:
                    agent_a_key = str(summary_rows[0].get("agent_a_key", agent_a_key or ""))
                    agent_b_key = str(summary_rows[0].get("agent_b_key", agent_b_key or ""))

                all_fixed_sims_rows.extend(summary_rows if mode == "fixed_sims" else [])
                all_fixed_time_rows.extend(summary_rows if mode == "fixed_time" else [])

                for row in summary_rows:
                    by_game_average_rows.append(
                        {
                            "opponent_tag": opponent_tag,
                            "mode": mode,
                            "seed": seed,
                            "game": str(row["game"]),
                            "n_rows": 1,
                            "n_matches": int(row.get("n_matches", 0)),
                            "win_rate_mean": float(row.get("win_rate_a", 0.0)),
                            "draw_rate_mean": float(row.get("draw_rate", 0.0)),
                            "avg_decision_sec_a_mean": float(row.get("avg_decision_sec_a", 0.0)),
                            "avg_eval_calls_neural_a_mean": float(row.get("avg_eval_calls_neural_a", 0.0)),
                            "avg_eval_calls_fallback_a_mean": float(row.get("avg_eval_calls_fallback_a", 0.0)),
                        }
                    )

            sims_agg = _aggregate(seed_summaries["fixed_sims"])
            time_agg = _aggregate(seed_summaries["fixed_time"])
            per_seed_rows.append(
                {
                    "opponent_tag": opponent_tag,
                    "seed": seed,
                    "games_count": len(seed_summaries["fixed_sims"]),
                    "rounds": int(seed_summaries["fixed_sims"][0]["n_matches"]) if seed_summaries["fixed_sims"] else 0,
                    "fixed_sims_win_rate_mean": sims_agg["win_rate_mean"],
                    "fixed_sims_n_matches": sims_agg["n_matches"],
                    "fixed_time_win_rate_mean": time_agg["win_rate_mean"],
                    "fixed_time_n_matches": time_agg["n_matches"],
                    "fixed_sims_avg_decision_sec_a_mean": sims_agg["avg_decision_sec_a_mean"],
                    "fixed_time_avg_decision_sec_a_mean": time_agg["avg_decision_sec_a_mean"],
                    "fixed_sims_avg_eval_calls_neural_a_mean": sims_agg["avg_eval_calls_neural_a_mean"],
                    "fixed_time_avg_eval_calls_neural_a_mean": time_agg["avg_eval_calls_neural_a_mean"],
                    "fixed_sims_avg_eval_calls_fallback_a_mean": sims_agg["avg_eval_calls_fallback_a_mean"],
                    "fixed_time_avg_eval_calls_fallback_a_mean": time_agg["avg_eval_calls_fallback_a_mean"],
                }
            )

        sims_global = _aggregate(all_fixed_sims_rows)
        time_global = _aggregate(all_fixed_time_rows)
        seeds_count = len({int(r["seed"]) for r in per_seed_rows if r["opponent_tag"] == opponent_tag})
        games_count = len({str(r["game"]) for r in by_game_average_rows if r["opponent_tag"] == opponent_tag and r["mode"] == "fixed_sims"})
        aggregate_rows.append(
            {
                "opponent_tag": opponent_tag,
                "games_count": games_count,
                "seeds_count": seeds_count,
                "rounds": int(all_fixed_sims_rows[0]["n_matches"]) if all_fixed_sims_rows else 0,
                "fixed_sims_win_rate_mean": sims_global["win_rate_mean"],
                "fixed_sims_n_matches": sims_global["n_matches"],
                "fixed_time_win_rate_mean": time_global["win_rate_mean"],
                "fixed_time_n_matches": time_global["n_matches"],
                "fixed_sims_avg_decision_sec_a_mean": sims_global["avg_decision_sec_a_mean"],
                "fixed_time_avg_decision_sec_a_mean": time_global["avg_decision_sec_a_mean"],
                "fixed_sims_avg_eval_calls_neural_a_mean": sims_global["avg_eval_calls_neural_a_mean"],
                "fixed_time_avg_eval_calls_neural_a_mean": time_global["avg_eval_calls_neural_a_mean"],
                "fixed_sims_avg_eval_calls_fallback_a_mean": sims_global["avg_eval_calls_fallback_a_mean"],
                "fixed_time_avg_eval_calls_fallback_a_mean": time_global["avg_eval_calls_fallback_a_mean"],
                "agent_a_key": agent_a_key,
                "agent_b_key": agent_b_key,
            }
        )

    summary_dir = base_dir / "summary"
    per_seed_rows.sort(key=lambda r: (str(r["opponent_tag"]), int(r["seed"])))
    by_game_average_rows.sort(key=lambda r: (str(r["opponent_tag"]), int(r["seed"]), str(r["mode"]), str(r["game"])))
    aggregate_rows.sort(key=lambda r: float(r["fixed_time_win_rate_mean"]), reverse=True)

    write_json(summary_dir / "per_seed_summary.json", per_seed_rows)
    write_csv(summary_dir / "per_seed_summary.csv", per_seed_rows)
    write_json(summary_dir / "by_game_average.json", by_game_average_rows)
    write_csv(summary_dir / "by_game_average.csv", by_game_average_rows)
    write_json(summary_dir / "aggregate_summary.json", aggregate_rows)
    write_csv(summary_dir / "aggregate_summary.csv", aggregate_rows)

    run_manifest_path = base_dir / "meta" / "run_manifest.json"
    if run_manifest_path.exists():
        manifest = read_json(run_manifest_path)
        args_payload = dict(manifest.get("args") or {})
        games = list(args_payload.get("games") or [])
        if hex_game_file not in games:
            games.append(hex_game_file)
        args_payload["games"] = games
        manifest["args"] = args_payload
        write_json(run_manifest_path, manifest)

    print(f"[merge_baseline_benchmark_hex] done. base_dir={base_dir} hex_dir={hex_dir}")


if __name__ == "__main__":
    main()
