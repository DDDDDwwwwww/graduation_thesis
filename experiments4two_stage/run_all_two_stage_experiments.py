from __future__ import annotations

import argparse
import atexit
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments4two_stage"
_LOG_FH: TextIO | None = None
_ORIG_STDOUT = sys.stdout
_ORIG_STDERR = sys.stderr


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _default_log_path() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return ROOT / "outputs4two_stage" / "logs" / f"run_all_{ts}.log"


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
    line = f"[{_now_ts()}][run_all_two_stage_experiments] {msg}\n"
    if _LOG_FH is not None:
        _LOG_FH.write(line)
        _LOG_FH.flush()
        return


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-stage experiment suite.")
    parser.add_argument("--artifacts", default="outputs4two_stage/artifacts/two_stage_artifacts.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiments", nargs="+", default=["A", "B", "C", "D", "E", "F", "G"])
    parser.add_argument("--log-file", default=None)
    parser.add_argument(
        "--rebuild-artifacts",
        action="store_true",
        help="Regenerate outputs4two_stage/artifacts/two_stage_artifacts.json before running experiments.",
    )
    args = parser.parse_args()
    log_path = _set_log_file(Path(args.log_file) if args.log_file else _default_log_path())

    all_start = time.perf_counter()
    _log(
        "suite-start "
        f"log_file={log_path} "
        f"device={args.device} artifacts={args.artifacts} "
        f"experiments={','.join([s.upper() for s in args.experiments])}"
    )
    selected = [s.upper() for s in args.experiments]
    if args.rebuild_artifacts:
        _run("build_reused_artifacts.py", ["--out-dir", "outputs4two_stage/artifacts"])
    else:
        _log("skip rebuilding artifacts (using existing file)")
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
    if "G" in selected:
        _run("run_experiment_ts_g_gate_sweep.py", ["--artifacts", args.artifacts, "--device", args.device])
    _log(f"suite-done elapsed={_fmt_seconds(time.perf_counter() - all_start)}")


if __name__ == "__main__":
    main()
