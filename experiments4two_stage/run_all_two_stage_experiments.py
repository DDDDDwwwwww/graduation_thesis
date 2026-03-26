from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments4two_stage"


def _run(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(EXP_DIR / script), *args]
    print("[run_all_two_stage_experiments]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-stage experiment suite.")
    parser.add_argument("--artifacts", default="outputs4two_stage/artifacts/two_stage_artifacts.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiments", nargs="+", default=["A", "B", "C", "D", "E", "F"])
    args = parser.parse_args()

    selected = [s.upper() for s in args.experiments]
    _run("build_reused_artifacts.py", ["--out-dir", "outputs4two_stage/artifacts"])
    if "A" in selected:
        _run("run_experiment_ts_a_main_benchmark.py", ["--artifacts", args.artifacts, "--device", args.device])
    if "B" in selected:
        _run("run_experiment_ts_b_time_budget.py", ["--artifacts", args.artifacts, "--device", args.device])
    if "C" in selected:
        _run("run_experiment_ts_c_search_budget.py", ["--artifacts", args.artifacts, "--device", args.device])
    if "D" in selected:
        _run("run_experiment_ts_d_gate_ablation.py", ["--artifacts", args.artifacts, "--device", args.device])
    if "E" in selected:
        _run("run_experiment_ts_e_uncertainty_ablation.py", ["--artifacts", args.artifacts, "--device", args.device])
    if "F" in selected:
        _run("run_experiment_ts_f_overhead_analysis.py", ["--artifacts", args.artifacts, "--device", args.device])
    print("[run_all_two_stage_experiments] done")


if __name__ == "__main__":
    main()
