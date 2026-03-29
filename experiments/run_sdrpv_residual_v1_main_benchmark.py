from __future__ import annotations

"""Main-paper benchmark for SDRPV residual_v1 (without phase-aware).

Design (aligned with current project status):
1) Keep model/backbone fixed.
2) Compare integration strategies on the same model:
   - selective (candidate main method)
   - value_full (ablation: no selective)
3) Evaluate both fixed-sims and fixed-time over multiple games and seeds.
"""

import argparse
from pathlib import Path

from experiment_utils import (
    ValueArtifact,
    default_games,
    init_output_layout,
    run_match_grid,
    write_csv,
    write_json,
)


def _parse_seeds(raw: str) -> list[int]:
    out = []
    for token in raw.replace(",", " ").split():
        if token.strip():
            out.append(int(token.strip()))
    if not out:
        raise ValueError("At least one seed is required.")
    return out


def _integration_configs(include_value_full: bool) -> list[dict]:
    cfgs = [
        {
            "tag": "sel_cap20",
            "override": {
                "evaluator_mode": "selective",
                "selective_max_neural_evals_per_move": 20,
                "selective_alpha": 1.0,
            },
        }
    ]
    if include_value_full:
        cfgs.append(
            {
                "tag": "value_full",
                "override": {
                    "evaluator_mode": "value",
                },
            }
        )
    return cfgs


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SDRPV residual_v1 main benchmark (no phase-aware).")
    parser.add_argument("--model-dir", required=True, help="Residual model dir with model.pt + encoder.json")
    parser.add_argument("--games", nargs="+", default=default_games(multi=True))
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seeds", default="42,142,242", help="Comma/space separated seeds.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fixed-sims", type=int, default=120)
    parser.add_argument("--fixed-sims-playclock", type=float, default=0.5)
    parser.add_argument("--fixed-time", type=float, default=0.5)
    parser.add_argument("--fixed-time-iters", type=int, default=120)
    parser.add_argument("--include-value-full", action="store_true", help="Include no-selective ablation.")
    parser.add_argument("--out-dir", default="outputs/experiments/SDRPV_residual_v1_main_benchmark")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    model_dir = Path(args.model_dir)
    model_path = model_dir / "model.pt"
    encoder_path = model_dir / "encoder.json"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"encoder not found: {encoder_path}")

    artifacts = {
        "token_transformer": ValueArtifact(
            model_path=str(model_path),
            encoder_config_path=str(encoder_path),
            vocab_path=None,
        )
    }
    pairs = [("neural_mcts:token_transformer", "pure_mct")]
    configs = _integration_configs(include_value_full=bool(args.include_value_full))

    root_layout = init_output_layout("SDRPV_residual_v1_main_benchmark", args.out_dir, args=args)
    per_seed_rows: list[dict] = []
    aggregate_rows: list[dict] = []

    for cfg in configs:
        tag = str(cfg["tag"])
        override = dict(cfg["override"])
        config_dir = root_layout["root"] / tag
        init_output_layout(
            "SDRPV_residual_v1_main_benchmark_cfg",
            config_dir,
            args={"tag": tag, "override": override},
        )
        agent_overrides = {"neural_mcts:token_transformer": dict(override)}

        all_fixed_sims_rows = []
        all_fixed_time_rows = []
        for seed in seeds:
            seed_dir = config_dir / f"seed_{seed}"
            seed_layout = init_output_layout(
                "SDRPV_residual_v1_main_benchmark_seed",
                seed_dir,
                args={"seed": seed, "tag": tag},
            )

            fixed_sims_dir = seed_layout["root"] / "fixed_sims"
            fixed_sims_layout = init_output_layout(
                "SDRPV_residual_v1_main_benchmark_fixed_sims",
                fixed_sims_dir,
                args={"fixed_sims": args.fixed_sims, "playclock": args.fixed_sims_playclock},
            )
            raw_sims, summary_sims = run_match_grid(
                games=list(args.games),
                pairs=pairs,
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

            fixed_time_dir = seed_layout["root"] / "fixed_time"
            fixed_time_layout = init_output_layout(
                "SDRPV_residual_v1_main_benchmark_fixed_time",
                fixed_time_dir,
                args={"time_budget": args.fixed_time, "iterations_cap": args.fixed_time_iters},
            )
            raw_time, summary_time = run_match_grid(
                games=list(args.games),
                pairs=pairs,
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

            all_fixed_sims_rows.extend(summary_sims)
            all_fixed_time_rows.extend(summary_time)

            sims_agg = _aggregate(summary_sims)
            time_agg = _aggregate(summary_time)
            seed_row = {
                "config_tag": tag,
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
            per_seed_rows.append(seed_row)
            print(
                f"[main_benchmark][seed_done] cfg={tag} seed={seed} "
                f"fixed_sims_win_rate_mean={seed_row['fixed_sims_win_rate_mean']:.4f} "
                f"fixed_time_win_rate_mean={seed_row['fixed_time_win_rate_mean']:.4f}",
                flush=True,
            )

        sims_global = _aggregate(all_fixed_sims_rows)
        time_global = _aggregate(all_fixed_time_rows)
        aggregate_rows.append(
            {
                "config_tag": tag,
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
                "override": override,
            }
        )

    aggregate_rows.sort(key=lambda r: r["fixed_time_win_rate_mean"], reverse=True)
    write_json(root_layout["summary"] / "per_seed_summary.json", per_seed_rows)
    write_csv(root_layout["summary"] / "per_seed_summary.csv", per_seed_rows)
    write_json(root_layout["summary"] / "aggregate_summary.json", aggregate_rows)
    write_csv(root_layout["summary"] / "aggregate_summary.csv", aggregate_rows)

    gate = {"has_selective_vs_value": False, "delta_fixed_time": None, "delta_fixed_sims": None}
    by_tag = {row["config_tag"]: row for row in aggregate_rows}
    if "sel_cap20" in by_tag and "value_full" in by_tag:
        gate["has_selective_vs_value"] = True
        gate["delta_fixed_time"] = (
            float(by_tag["sel_cap20"]["fixed_time_win_rate_mean"])
            - float(by_tag["value_full"]["fixed_time_win_rate_mean"])
        )
        gate["delta_fixed_sims"] = (
            float(by_tag["sel_cap20"]["fixed_sims_win_rate_mean"])
            - float(by_tag["value_full"]["fixed_sims_win_rate_mean"])
        )
    write_json(root_layout["summary"] / "gate_check.json", gate)
    print(f"[main_benchmark] done. output={root_layout['root']}")


if __name__ == "__main__":
    main()
