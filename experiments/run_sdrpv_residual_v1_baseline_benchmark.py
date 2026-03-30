from __future__ import annotations

"""Baseline benchmark for SDRPV residual_v1 value_full.

This script aligns with the short-cycle plan in sdrpv_main_benchmark_report.md:
1) Freeze the current SDRPV residual_v1 model and use value_full as the main method.
2) Compare it directly against four baselines:
   - pure_mct
   - token_transformer size_2000
   - fact_mlp size_2000
   - heuristic_mcts
3) Evaluate with both fixed-sims and fixed-time over multiple games and seeds.

The script is intentionally verbose and writes intermediate artifacts after each
seed/mode so long runs can be monitored safely.
"""

import argparse
import time
from pathlib import Path

from experiment_utils import (
    ValueArtifact,
    default_games,
    init_output_layout,
    run_match_grid,
    write_csv,
    write_json,
)


MAIN_AGENT_KEY = "neural_mcts:sdrpv_value_full"


def _parse_seeds(raw: str) -> list[int]:
    out = []
    for token in raw.replace(",", " ").split():
        if token.strip():
            out.append(int(token.strip()))
    if not out:
        raise ValueError("At least one seed is required.")
    return out


def _artifact_from_dir(model_dir: Path, require_vocab: bool = False) -> ValueArtifact:
    model_path = model_dir / "model.pt"
    encoder_path = model_dir / "encoder.json"
    vocab_path = model_dir / "vocab.json"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"encoder not found: {encoder_path}")
    if require_vocab and not vocab_path.exists():
        raise FileNotFoundError(f"vocab not found: {vocab_path}")
    return ValueArtifact(
        model_path=str(model_path),
        encoder_config_path=str(encoder_path),
        vocab_path=str(vocab_path) if vocab_path.exists() else None,
    )


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


def _by_game_average(rows: list[dict], opponent_tag: str, mode: str, seed: int) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row["game"]), []).append(row)

    out = []
    for game, game_rows in sorted(buckets.items()):
        agg = _aggregate(game_rows)
        out.append(
            {
                "opponent_tag": opponent_tag,
                "mode": mode,
                "seed": int(seed),
                "game": game,
                "n_rows": agg["n_rows"],
                "n_matches": agg["n_matches"],
                "win_rate_mean": agg["win_rate_mean"],
                "draw_rate_mean": agg["draw_rate_mean"],
                "avg_decision_sec_a_mean": agg["avg_decision_sec_a_mean"],
                "avg_eval_calls_neural_a_mean": agg["avg_eval_calls_neural_a_mean"],
                "avg_eval_calls_fallback_a_mean": agg["avg_eval_calls_fallback_a_mean"],
            }
        )
    return out


def _opponents() -> list[dict]:
    return [
        {"tag": "pure_mct", "agent_b_key": "pure_mct"},
        {"tag": "baseline_token_transformer_2000", "agent_b_key": "neural_mcts:baseline_token_transformer_2000"},
        {"tag": "baseline_fact_mlp_2000", "agent_b_key": "neural_mcts:baseline_fact_mlp_2000"},
        {"tag": "heuristic_mcts", "agent_b_key": "heuristic_mcts"},
    ]


def _log(message: str) -> None:
    print(f"[baseline_benchmark] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SDRPV residual_v1 value_full baseline benchmark.")
    parser.add_argument("--model-dir", required=True, help="Residual SDRPV model dir with model.pt + encoder.json")
    parser.add_argument(
        "--baseline-token-dir",
        default="outputs/experiments/D_dataset_size_sensitivity/models/size_2000/token_transformer",
        help="Baseline token_transformer model dir",
    )
    parser.add_argument(
        "--baseline-fact-dir",
        default="outputs/experiments/D_dataset_size_sensitivity/models/size_2000/fact_mlp",
        help="Baseline fact_mlp model dir",
    )
    parser.add_argument("--games", nargs="+", default=default_games(multi=True))
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seeds", default="42,142,242", help="Comma/space separated seeds.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fixed-sims", type=int, default=120)
    parser.add_argument("--fixed-sims-playclock", type=float, default=0.5)
    parser.add_argument("--fixed-time", type=float, default=0.5)
    parser.add_argument("--fixed-time-iters", type=int, default=120)
    parser.add_argument("--out-dir", default="outputs/experiments/SDRPV_residual_v1_baseline_benchmark")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    model_dir = Path(args.model_dir)
    baseline_token_dir = Path(args.baseline_token_dir)
    baseline_fact_dir = Path(args.baseline_fact_dir)

    artifacts = {
        "sdrpv_value_full": _artifact_from_dir(model_dir),
        "baseline_token_transformer_2000": _artifact_from_dir(baseline_token_dir),
        "baseline_fact_mlp_2000": _artifact_from_dir(baseline_fact_dir, require_vocab=True),
    }
    agent_overrides = {
        MAIN_AGENT_KEY: {
            "evaluator_mode": "value",
        }
    }

    root_layout = init_output_layout("SDRPV_residual_v1_baseline_benchmark", args.out_dir, args=args)
    opponents = _opponents()
    total_runs = len(opponents) * len(seeds) * 2
    completed_runs = 0
    per_seed_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    by_game_rows: list[dict] = []

    _log(
        "start "
        f"games={len(args.games)} rounds={args.rounds} seeds={seeds} "
        f"opponents={[item['tag'] for item in opponents]} device={args.device}"
    )

    for opponent_index, opponent in enumerate(opponents, start=1):
        opponent_tag = str(opponent["tag"])
        pair = (MAIN_AGENT_KEY, str(opponent["agent_b_key"]))
        opponent_dir = root_layout["root"] / opponent_tag
        init_output_layout(
            "SDRPV_residual_v1_baseline_benchmark_opponent",
            opponent_dir,
            args={"opponent_tag": opponent_tag, "pair": pair},
        )
        _log(
            f"opponent_start {opponent_index}/{len(opponents)} "
            f"tag={opponent_tag} pair={pair[0]}__vs__{pair[1]}"
        )

        all_fixed_sims_rows: list[dict] = []
        all_fixed_time_rows: list[dict] = []
        for seed_index, seed in enumerate(seeds, start=1):
            seed_dir = opponent_dir / f"seed_{seed}"
            seed_layout = init_output_layout(
                "SDRPV_residual_v1_baseline_benchmark_seed",
                seed_dir,
                args={"seed": seed, "opponent_tag": opponent_tag},
            )
            _log(
                f"seed_start opponent={opponent_tag} "
                f"seed={seed} ({seed_index}/{len(seeds)})"
            )

            fixed_sims_dir = seed_layout["root"] / "fixed_sims"
            fixed_sims_layout = init_output_layout(
                "SDRPV_residual_v1_baseline_benchmark_fixed_sims",
                fixed_sims_dir,
                args={"fixed_sims": args.fixed_sims, "playclock": args.fixed_sims_playclock},
            )
            _log(
                f"mode_start run={completed_runs + 1}/{total_runs} "
                f"opponent={opponent_tag} seed={seed} mode=fixed_sims "
                f"iterations={args.fixed_sims} playclock={args.fixed_sims_playclock}"
            )
            t0 = time.perf_counter()
            raw_sims, summary_sims = run_match_grid(
                games=list(args.games),
                pairs=[pair],
                rounds=int(args.rounds),
                playclock=float(args.fixed_sims_playclock),
                iterations=int(args.fixed_sims),
                seed=int(seed),
                cache_enabled=True,
                artifacts=artifacts,
                device=args.device,
                agent_overrides=agent_overrides,
            )
            write_json(fixed_sims_layout["raw"] / "matches.json", raw_sims)
            write_json(fixed_sims_layout["summary"] / "by_game.json", summary_sims)
            write_csv(fixed_sims_layout["summary"] / "by_game.csv", summary_sims)
            sims_elapsed = time.perf_counter() - t0
            sims_agg = _aggregate(summary_sims)
            _log(
                f"mode_done run={completed_runs + 1}/{total_runs} "
                f"opponent={opponent_tag} seed={seed} mode=fixed_sims "
                f"win_rate_mean={sims_agg['win_rate_mean']:.4f} "
                f"n_matches={sims_agg['n_matches']} elapsed_sec={sims_elapsed:.2f}"
            )
            completed_runs += 1

            fixed_time_dir = seed_layout["root"] / "fixed_time"
            fixed_time_layout = init_output_layout(
                "SDRPV_residual_v1_baseline_benchmark_fixed_time",
                fixed_time_dir,
                args={"time_budget": args.fixed_time, "iterations_cap": args.fixed_time_iters},
            )
            _log(
                f"mode_start run={completed_runs + 1}/{total_runs} "
                f"opponent={opponent_tag} seed={seed} mode=fixed_time "
                f"time_budget={args.fixed_time} iterations_cap={args.fixed_time_iters}"
            )
            t1 = time.perf_counter()
            raw_time, summary_time = run_match_grid(
                games=list(args.games),
                pairs=[pair],
                rounds=int(args.rounds),
                playclock=float(args.fixed_time),
                iterations=int(args.fixed_time_iters),
                seed=int(seed + 5000),
                cache_enabled=True,
                artifacts=artifacts,
                device=args.device,
                agent_overrides=agent_overrides,
            )
            write_json(fixed_time_layout["raw"] / "matches.json", raw_time)
            write_json(fixed_time_layout["summary"] / "by_game.json", summary_time)
            write_csv(fixed_time_layout["summary"] / "by_game.csv", summary_time)
            time_elapsed = time.perf_counter() - t1
            time_agg = _aggregate(summary_time)
            _log(
                f"mode_done run={completed_runs + 1}/{total_runs} "
                f"opponent={opponent_tag} seed={seed} mode=fixed_time "
                f"win_rate_mean={time_agg['win_rate_mean']:.4f} "
                f"n_matches={time_agg['n_matches']} elapsed_sec={time_elapsed:.2f}"
            )
            completed_runs += 1

            all_fixed_sims_rows.extend(summary_sims)
            all_fixed_time_rows.extend(summary_time)
            by_game_rows.extend(_by_game_average(summary_sims, opponent_tag, "fixed_sims", seed))
            by_game_rows.extend(_by_game_average(summary_time, opponent_tag, "fixed_time", seed))

            per_seed_rows.append(
                {
                    "opponent_tag": opponent_tag,
                    "seed": int(seed),
                    "games_count": len(args.games),
                    "rounds": int(args.rounds),
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

            write_json(root_layout["summary"] / "per_seed_summary.json", per_seed_rows)
            write_csv(root_layout["summary"] / "per_seed_summary.csv", per_seed_rows)
            write_json(root_layout["summary"] / "by_game_average.json", by_game_rows)
            write_csv(root_layout["summary"] / "by_game_average.csv", by_game_rows)
            _log(
                f"seed_done opponent={opponent_tag} seed={seed} "
                f"fixed_sims_win_rate_mean={sims_agg['win_rate_mean']:.4f} "
                f"fixed_time_win_rate_mean={time_agg['win_rate_mean']:.4f}"
            )

        sims_global = _aggregate(all_fixed_sims_rows)
        time_global = _aggregate(all_fixed_time_rows)
        aggregate_rows.append(
            {
                "opponent_tag": opponent_tag,
                "games_count": len(args.games),
                "seeds_count": len(seeds),
                "rounds": int(args.rounds),
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
                "agent_a_key": pair[0],
                "agent_b_key": pair[1],
            }
        )
        write_json(root_layout["summary"] / "aggregate_summary.json", aggregate_rows)
        write_csv(root_layout["summary"] / "aggregate_summary.csv", aggregate_rows)
        _log(
            f"opponent_done {opponent_index}/{len(opponents)} tag={opponent_tag} "
            f"fixed_sims_win_rate_mean={sims_global['win_rate_mean']:.4f} "
            f"fixed_time_win_rate_mean={time_global['win_rate_mean']:.4f}"
        )

    aggregate_rows.sort(key=lambda row: row["fixed_time_win_rate_mean"], reverse=True)
    write_json(root_layout["summary"] / "aggregate_summary.json", aggregate_rows)
    write_csv(root_layout["summary"] / "aggregate_summary.csv", aggregate_rows)
    _log(f"done output={root_layout['root']}")


if __name__ == "__main__":
    main()
