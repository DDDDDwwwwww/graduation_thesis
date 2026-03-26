from __future__ import annotations

"""Run required experiment blocks from section 15."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EXPERIMENTS = ["A", "B", "C", "D", "E", "H"]
OPTIONAL_EXPERIMENTS = ["F", "G", "I"]


def _run(script_name: str, args: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "experiments" / script_name), *args]
    print("[run_all_required_experiments]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run section-15 required experiments in batch.")
    parser.add_argument("--artifacts", help="Artifact map JSON required by A/B/C/E/F/H")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also run optional heavy experiments: F/G/I.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Subset to run. Choices in {A,B,C,D,E,F,G,H,I}",
    )
    args = parser.parse_args()

    if args.experiments:
        selected = [e.upper() for e in args.experiments]
    else:
        selected = list(REQUIRED_EXPERIMENTS)
        if args.include_optional:
            selected.extend(OPTIONAL_EXPERIMENTS)

    exp_set = set(selected)
    print("[run_all_required_experiments] selected:", " ".join(selected))
    need_artifacts = {"A", "B", "C", "E", "F", "H"}
    if exp_set.intersection(need_artifacts) and not args.artifacts:
        raise ValueError("--artifacts is required when running any of A/B/C/E/F/H.")

    if "A" in exp_set:
        _run("run_experiment_a_baseline_strength.py", ["--artifacts", args.artifacts, "--device", args.device])
    if "B" in exp_set:
        _run("run_experiment_b_time_budget.py", ["--artifacts", args.artifacts, "--device", args.device])
    if "C" in exp_set:
        _run("run_experiment_c_search_budget.py", ["--artifacts", args.artifacts, "--device", args.device])
    if "D" in exp_set:
        _run("run_experiment_d_dataset_size.py", ["--device", args.device])
    if "E" in exp_set:
        _run("run_experiment_e_encoder_model_ablation.py", ["--artifacts", args.artifacts, "--device", args.device])
    if "F" in exp_set:
        _run("run_experiment_f_cache_performance.py", ["--artifacts", args.artifacts, "--device", args.device])
    if "G" in exp_set:
        _run("run_experiment_g_single_vs_multi.py", ["--device", args.device])
    if "H" in exp_set:
        _run("run_experiment_h_multi_game_benchmark.py", ["--artifacts", args.artifacts, "--device", args.device])
    if "I" in exp_set:
        _run("run_experiment_i_cross_game_generalization.py", ["--device", args.device])

    print("[run_all_required_experiments] done")


if __name__ == "__main__":
    main()
