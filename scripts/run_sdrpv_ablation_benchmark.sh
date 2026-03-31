#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/logs outputs/experiments

PYTHON_BIN="${PYTHON_BIN:-python}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-ablation_benchmark_${TS}}"
MODELS_ROOT="${MODELS_ROOT:-}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42,142,242}"
ROUNDS="${ROUNDS:-50}"
OUT_DIR="${OUT_DIR:-outputs/experiments/SDRPV_ablation_benchmark_${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-outputs/logs/SDRPV_ablation_benchmark_${RUN_TAG}.log}"

if [[ -z "${MODELS_ROOT}" && -z "${FULL_MODEL_DIR:-}" ]]; then
  echo "[ABLATION_BENCHMARK][ERROR] MODELS_ROOT or explicit model dirs are required."
  echo "Example:"
  echo "  MODELS_ROOT=outputs/experiments/SDRPV_ablation_train_xxx/models bash scripts/run_sdrpv_ablation_benchmark.sh"
  exit 1
fi

if [[ -n "${GAMES:-}" ]]; then
  # shellcheck disable=SC2206
  GAME_ARR=(${GAMES})
else
  GAME_ARR=(games/hex.kif games/connectFour.kif games/breakthrough.kif)
fi

exec >>"${LOG_FILE}" 2>&1

echo "[ABLATION_BENCHMARK] start $(date)"
echo "[ABLATION_BENCHMARK] models_root=${MODELS_ROOT}"
echo "[ABLATION_BENCHMARK] device=${DEVICE}"
echo "[ABLATION_BENCHMARK] seeds=${SEEDS}"
echo "[ABLATION_BENCHMARK] rounds=${ROUNDS}"
echo "[ABLATION_BENCHMARK] games=${GAME_ARR[*]}"
echo "[ABLATION_BENCHMARK] out_dir=${OUT_DIR}"
echo "[ABLATION_BENCHMARK] log_file=${LOG_FILE}"

ARGS=(
  -u experiments/run_sdrpv_ablation_benchmark.py
  --games "${GAME_ARR[@]}"
  --rounds "${ROUNDS}"
  --seeds "${SEEDS}"
  --fixed-sims 120
  --fixed-sims-playclock 0.5
  --fixed-time 0.5
  --fixed-time-iters 120
  --device "${DEVICE}"
  --out-dir "${OUT_DIR}"
)

if [[ -n "${MODELS_ROOT}" ]]; then
  ARGS+=(--models-root "${MODELS_ROOT}")
fi
if [[ -n "${FULL_MODEL_DIR:-}" ]]; then
  ARGS+=(--full-model-dir "${FULL_MODEL_DIR}")
fi
if [[ -n "${NO_RESIDUAL_MODEL_DIR:-}" ]]; then
  ARGS+=(--no-residual-model-dir "${NO_RESIDUAL_MODEL_DIR}")
fi
if [[ -n "${NO_TEACHER_MODEL_DIR:-}" ]]; then
  ARGS+=(--no-teacher-model-dir "${NO_TEACHER_MODEL_DIR}")
fi

"${PYTHON_BIN}" "${ARGS[@]}"

echo "[ABLATION_BENCHMARK] done $(date)"
echo "[ABLATION_BENCHMARK] aggregate_csv=${OUT_DIR}/summary/aggregate_summary.csv"
echo "[ABLATION_BENCHMARK] delta_csv=${OUT_DIR}/summary/delta_vs_full.csv"
