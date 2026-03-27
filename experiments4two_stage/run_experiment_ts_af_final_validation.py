from __future__ import annotations

import argparse
import atexit
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments4two_stage"
OUTPUT_ROOT = ROOT / "outputs4two_stage" / "experiments"
_LOG_FH: TextIO | None = None
_ORIG_STDOUT = sys.stdout
_ORIG_STDERR = sys.stderr

EXP_NAME = "TS_AF_final_validation"


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _set_log_file(path: str | Path) -> Path:
    global _LOG_FH
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if _LOG_FH is not None:
        _LOG_FH.close()
    _LOG_FH = p.open("w", encoding="utf-8")
    sys.stdout = _LOG_FH
    sys.stderr = _LOG_FH
    return p


def _close_log_file() -> None:
    global _LOG_FH
    if _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None
    sys.stdout = _ORIG_STDOUT
    sys.stderr = _ORIG_STDERR


atexit.register(_close_log_file)


def _log(msg: str) -> None:
    line = f"[{_now_ts()}][{EXP_NAME}] {msg}\n"
    if _LOG_FH is not None:
        _LOG_FH.write(line)
        _LOG_FH.flush()
        return
    print(line, end="", flush=True)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(EXP_DIR / script), *args]
    t0 = time.perf_counter()
    _log(f"start script={script} cmd={' '.join(cmd)}")
    subprocess.run(
        cmd,
        check=True,
        stdout=_LOG_FH,
        stderr=subprocess.STDOUT,
    )
    _log(f"done script={script} elapsed={_fmt_seconds(time.perf_counter() - t0)}")


def _load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(row: dict, key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def _collect_candidate_metrics(config: dict, cross_rows: list[dict], target_slow_ratio: float) -> dict:
    selected = [
        row for row in cross_rows
        if row.get("agent_a_key") == "two_stage_neural_mcts"
        and row.get("agent_b_key") in {"neural_mcts:token_transformer", "pure_mct"}
    ]
    by_opponent = {row["agent_b_key"]: row for row in selected}
    vs_transformer = by_opponent.get("neural_mcts:token_transformer", {})
    vs_pure = by_opponent.get("pure_mct", {})

    slow_ratio = _to_float(vs_transformer, "mean_mean_slow_call_ratio_a")
    return {
        "candidate_name": config["name"],
        "tau": config["tau"],
        "visit_threshold": config["visit_threshold"],
        "slow_budget_per_move": config["slow_budget_per_move"],
        "a_out_dir": config["a_out_dir"],
        "vs_transformer_win_rate": _to_float(vs_transformer, "mean_win_rate_a"),
        "vs_transformer_decision_sec": _to_float(vs_transformer, "mean_avg_decision_sec_a"),
        "vs_transformer_slow_call_ratio": slow_ratio,
        "vs_transformer_slow_trigger_rate": _to_float(vs_transformer, "mean_mean_slow_trigger_rate_a"),
        "vs_transformer_uncertainty_ok_rate": _to_float(vs_transformer, "mean_mean_uncertainty_ok_rate_a"),
        "vs_transformer_visit_ok_rate": _to_float(vs_transformer, "mean_mean_visit_ok_rate_a"),
        "vs_pure_mct_win_rate": _to_float(vs_pure, "mean_win_rate_a"),
        "vs_pure_mct_decision_sec": _to_float(vs_pure, "mean_avg_decision_sec_a"),
        "vs_pure_mct_slow_call_ratio": _to_float(vs_pure, "mean_mean_slow_call_ratio_a"),
        "slow_ratio_distance_to_target": abs(slow_ratio - float(target_slow_ratio)),
    }


def _pick_best_candidate(rows: list[dict]) -> dict:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["vs_transformer_win_rate"]),
            -float(row["vs_pure_mct_win_rate"]),
            float(row["vs_transformer_decision_sec"]),
            float(row["slow_ratio_distance_to_target"]),
        ),
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run three final A candidates, pick the best, then rerun F with the winner."
    )
    parser.add_argument("--artifacts", default="outputs4two_stage/artifacts/two_stage_artifacts.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-slow-ratio", type=float, default=0.08)
    parser.add_argument("--rounds-a", type=int, default=20)
    parser.add_argument("--rounds-f", type=int, default=20)
    parser.add_argument("--playclock", type=float, default=0.7)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uncertainty-type", choices=["margin", "variance_head"], default="variance_head")
    parser.add_argument("--gate-type", default="combined")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--a-prefix", default="TS_A_main_benchmark_final")
    parser.add_argument("--f-prefix", default="TS_F_overhead_analysis_final")
    args = parser.parse_args()

    meta_root = OUTPUT_ROOT / EXP_NAME / "meta"
    summary_root = OUTPUT_ROOT / EXP_NAME / "summary"
    log_path = Path(args.log_file) if args.log_file else (meta_root / "run.log")
    _set_log_file(log_path)

    candidates = [
        {"name": "tau1e4_v0_b8", "tau": 0.0001, "visit_threshold": 0, "slow_budget_per_move": 8},
        {"name": "tau3e4_v0_b8", "tau": 0.0003, "visit_threshold": 0, "slow_budget_per_move": 8},
        {"name": "tau1e4_v0_b4", "tau": 0.0001, "visit_threshold": 0, "slow_budget_per_move": 4},
    ]

    _log(
        "suite-start "
        f"device={args.device} artifacts={args.artifacts} candidates={len(candidates)} "
        f"rounds_a={args.rounds_a} rounds_f={args.rounds_f}"
    )

    rankings = []
    for idx, config in enumerate(candidates, start=1):
        a_out_dir = f"outputs4two_stage/experiments/{args.a_prefix}_{config['name']}"
        config["a_out_dir"] = a_out_dir
        _log(
            f"A-candidate-start {idx}/{len(candidates)} "
            f"name={config['name']} tau={config['tau']} "
            f"visit_threshold={config['visit_threshold']} slow_budget={config['slow_budget_per_move']}"
        )
        _run(
            "run_experiment_ts_a_main_benchmark.py",
            [
                "--artifacts", args.artifacts,
                "--device", args.device,
                "--rounds", str(args.rounds_a),
                "--playclock", str(args.playclock),
                "--iterations", str(args.iterations),
                "--seed", str(args.seed),
                "--tau", str(config["tau"]),
                "--visit-threshold", str(config["visit_threshold"]),
                "--slow-budget-per-move", str(config["slow_budget_per_move"]),
                "--uncertainty-type", args.uncertainty_type,
                "--gate-type", args.gate_type,
                "--out-dir", a_out_dir,
            ],
        )
        cross_rows = _load_csv_rows(ROOT / a_out_dir / "summary" / "cross_game.csv")
        metrics = _collect_candidate_metrics(config, cross_rows, target_slow_ratio=args.target_slow_ratio)
        rankings.append(metrics)
        _log(
            "A-candidate-end "
            f"name={config['name']} "
            f"vs_transformer_win={metrics['vs_transformer_win_rate']:.3f} "
            f"vs_pure_win={metrics['vs_pure_mct_win_rate']:.3f} "
            f"slow_ratio={metrics['vs_transformer_slow_call_ratio']:.4f}"
        )

    best = _pick_best_candidate(rankings)
    _write_csv(summary_root / "a_candidate_ranking.csv", rankings)
    _write_json(summary_root / "a_candidate_ranking.json", rankings)
    _write_json(summary_root / "best_candidate.json", best)
    _log(
        "best-candidate "
        f"name={best['candidate_name']} "
        f"tau={best['tau']} visit_threshold={best['visit_threshold']} "
        f"slow_budget={best['slow_budget_per_move']}"
    )

    f_out_dir = f"outputs4two_stage/experiments/{args.f_prefix}_{best['candidate_name']}"
    _run(
        "run_experiment_ts_f_overhead_analysis.py",
        [
            "--artifacts", args.artifacts,
            "--device", args.device,
            "--rounds", str(args.rounds_f),
            "--playclock", str(args.playclock),
            "--iterations", str(args.iterations),
            "--seed", str(args.seed),
            "--tau", str(best["tau"]),
            "--visit-threshold", str(best["visit_threshold"]),
            "--slow-budget-per-move", str(best["slow_budget_per_move"]),
            "--uncertainty-type", args.uncertainty_type,
            "--gate-type", args.gate_type,
            "--out-dir", f_out_dir,
        ],
    )
    _write_json(
        summary_root / "final_selection.json",
        {
            "best_candidate": best,
            "f_out_dir": f_out_dir,
        },
    )
    _log(f"suite-done f_out_dir={f_out_dir}")


if __name__ == "__main__":
    main()
