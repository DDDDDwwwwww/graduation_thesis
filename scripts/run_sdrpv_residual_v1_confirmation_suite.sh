#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/logs outputs/experiments

PYTHON_BIN="${PYTHON_BIN:-python}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-confirm_suite_${TS}}"
MODEL_DIR="${MODEL_DIR:-}"
GAME="${GAME:-games/connectFour.kif}"
ROUNDS="${ROUNDS:-50}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42 142 242}"

if [[ -z "${MODEL_DIR}" ]]; then
  echo "[CONFIRM_SUITE][ERROR] MODEL_DIR is required."
  echo "Example:"
  echo "  MODEL_DIR=outputs/experiments/SDRPV_residual_v1_residual_v1_gpu0_20260329_114156 \\"
  echo "  bash scripts/run_sdrpv_residual_v1_confirmation_suite.sh"
  exit 1
fi

OUT_ROOT="${OUT_ROOT:-outputs/experiments/SDRPV_residual_v1_confirmation_suite_${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-outputs/logs/SDRPV_residual_v1_confirmation_suite_${RUN_TAG}.log}"
mkdir -p "${OUT_ROOT}"

# 全流程统一写入同一个日志文件，并同步输出到终端。
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[CONFIRM_SUITE] start $(date)"
echo "[CONFIRM_SUITE] run_tag=${RUN_TAG}"
echo "[CONFIRM_SUITE] model_dir=${MODEL_DIR}"
echo "[CONFIRM_SUITE] game=${GAME}"
echo "[CONFIRM_SUITE] rounds=${ROUNDS}"
echo "[CONFIRM_SUITE] device=${DEVICE}"
echo "[CONFIRM_SUITE] seeds=${SEEDS}"
echo "[CONFIRM_SUITE] out_root=${OUT_ROOT}"
echo "[CONFIRM_SUITE] log_file=${LOG_FILE}"

run_case() {
  local cfg_tag="$1"
  local seed="$2"
  shift 2
  local out_dir="${OUT_ROOT}/${cfg_tag}_seed${seed}"

  echo ""
  echo "============================================================"
  echo "[CONFIRM_SUITE] case_start $(date)"
  echo "[CONFIRM_SUITE] cfg=${cfg_tag} seed=${seed}"
  echo "[CONFIRM_SUITE] out_dir=${out_dir}"
  echo "============================================================"

  "${PYTHON_BIN}" -u experiments/run_sdrpv_residual_v1_mcts_smoke.py \
    --model-dir "${MODEL_DIR}" \
    --game "${GAME}" \
    --rounds "${ROUNDS}" \
    --seed "${seed}" \
    --fixed-sims 120 \
    --fixed-sims-playclock 0.5 \
    --fixed-time 0.5 \
    --fixed-time-iters 120 \
    --device "${DEVICE}" \
    --out-dir "${out_dir}" \
    "$@"

  echo "[CONFIRM_SUITE] case_done $(date) cfg=${cfg_tag} seed=${seed}"
}

for seed in ${SEEDS}; do
  run_case "value_full" "${seed}" \
    --integration-mode value
done

for seed in ${SEEDS}; do
  run_case "sel_cap20" "${seed}" \
    --integration-mode selective \
    --selective-max-neural-evals-per-move 20 \
    --selective-alpha 1.0
done

SUMMARY_DIR="${OUT_ROOT}/summary"
mkdir -p "${SUMMARY_DIR}"

echo ""
echo "[CONFIRM_SUITE] collecting summary..."
export CONFIRM_SUITE_OUT_ROOT="${OUT_ROOT}"
"${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

out_root = Path(os.environ["CONFIRM_SUITE_OUT_ROOT"])
summary_dir = out_root / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)

rows = []
for case_dir in sorted([p for p in out_root.iterdir() if p.is_dir() and p.name != "summary"]):
    stage_path = case_dir / "summary" / "stage_summary.json"
    if not stage_path.exists():
        continue
    stage = json.loads(stage_path.read_text(encoding="utf-8"))

    name = case_dir.name  # e.g. value_full_seed42
    if "_seed" not in name:
        continue
    cfg_tag, seed_text = name.rsplit("_seed", 1)
    seed = int(seed_text)

    fs = stage["fixed_sims"]["summary_rows"][0]
    ft = stage["fixed_time"]["summary_rows"][0]
    rows.append(
        {
            "cfg_tag": cfg_tag,
            "seed": seed,
            "n_matches": int(fs["n_matches"]),
            "fixed_sims_win_rate": float(fs["win_rate_a"]),
            "fixed_time_win_rate": float(ft["win_rate_a"]),
            "fixed_sims_wins": int(fs["wins_a"]),
            "fixed_time_wins": int(ft["wins_a"]),
        }
    )

rows.sort(key=lambda r: (r["cfg_tag"], r["seed"]))
(summary_dir / "per_run.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

with (summary_dir / "per_run.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "cfg_tag",
            "seed",
            "n_matches",
            "fixed_sims_wins",
            "fixed_sims_win_rate",
            "fixed_time_wins",
            "fixed_time_win_rate",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

agg = {}
for r in rows:
    g = agg.setdefault(r["cfg_tag"], {"n": 0, "ws": 0, "wt": 0, "seeds": []})
    g["n"] += int(r["n_matches"])
    g["ws"] += int(r["fixed_sims_wins"])
    g["wt"] += int(r["fixed_time_wins"])
    g["seeds"].append(r)

agg_rows = []
for cfg_tag, g in sorted(agg.items()):
    n = g["n"]
    sims_rate = (g["ws"] / n) if n else 0.0
    time_rate = (g["wt"] / n) if n else 0.0
    agg_rows.append(
        {
            "cfg_tag": cfg_tag,
            "n_total_matches": n,
            "fixed_sims_win_rate": sims_rate,
            "fixed_time_win_rate": time_rate,
        }
    )

(summary_dir / "aggregate.json").write_text(json.dumps(agg_rows, ensure_ascii=False, indent=2), encoding="utf-8")
with (summary_dir / "aggregate.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["cfg_tag", "n_total_matches", "fixed_sims_win_rate", "fixed_time_win_rate"],
    )
    writer.writeheader()
    writer.writerows(agg_rows)

def get_cfg(name):
    for r in agg_rows:
        if r["cfg_tag"] == name:
            return r
    return None

value_full = get_cfg("value_full")
sel_cap20 = get_cfg("sel_cap20")

gate = {
    "gate_1_fixed_time_delta_ge_0.05": False,
    "gate_2_seed_majority_fixed_time": False,
    "gate_3_no_regression_fixed_sims": False,
    "passed_all": False,
    "details": {},
}

if value_full and sel_cap20:
    delta_time = float(sel_cap20["fixed_time_win_rate"]) - float(value_full["fixed_time_win_rate"])
    delta_sims = float(sel_cap20["fixed_sims_win_rate"]) - float(value_full["fixed_sims_win_rate"])
    gate["details"]["delta_fixed_time"] = delta_time
    gate["details"]["delta_fixed_sims"] = delta_sims

    gate["gate_1_fixed_time_delta_ge_0.05"] = delta_time >= 0.05
    gate["gate_3_no_regression_fixed_sims"] = delta_sims >= 0.0

    per_seed = {}
    for r in rows:
        per_seed.setdefault(int(r["seed"]), {})[r["cfg_tag"]] = r
    better = 0
    total = 0
    for seed, d in sorted(per_seed.items()):
        if "value_full" in d and "sel_cap20" in d:
            total += 1
            ok = float(d["sel_cap20"]["fixed_time_win_rate"]) > float(d["value_full"]["fixed_time_win_rate"])
            better += int(ok)
    gate["details"]["seed_fixed_time_better_count"] = better
    gate["details"]["seed_pair_count"] = total
    gate["gate_2_seed_majority_fixed_time"] = better >= 2

gate["passed_all"] = (
    gate["gate_1_fixed_time_delta_ge_0.05"]
    and gate["gate_2_seed_majority_fixed_time"]
    and gate["gate_3_no_regression_fixed_sims"]
)

(summary_dir / "gates.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
print("[CONFIRM_SUITE][SUMMARY] aggregate:", json.dumps(agg_rows, ensure_ascii=False))
print("[CONFIRM_SUITE][SUMMARY] gates:", json.dumps(gate, ensure_ascii=False))
PY

echo "[CONFIRM_SUITE] done $(date)"
echo "[CONFIRM_SUITE] summary_dir=${SUMMARY_DIR}"
echo "[CONFIRM_SUITE] per_run_json=${SUMMARY_DIR}/per_run.json"
echo "[CONFIRM_SUITE] aggregate_json=${SUMMARY_DIR}/aggregate.json"
echo "[CONFIRM_SUITE] gates_json=${SUMMARY_DIR}/gates.json"
