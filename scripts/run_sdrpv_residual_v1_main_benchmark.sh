#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/logs outputs/experiments

PYTHON_BIN="${PYTHON_BIN:-python}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-main_benchmark_${TS}}"
MODEL_DIR="${MODEL_DIR:-}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42,142,242}"
ROUNDS="${ROUNDS:-50}"
INCLUDE_VALUE_FULL="${INCLUDE_VALUE_FULL:-1}"
OUT_DIR="${OUT_DIR:-outputs/experiments/SDRPV_residual_v1_main_benchmark_${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-outputs/logs/SDRPV_residual_v1_main_benchmark_${RUN_TAG}.log}"

if [[ -z "${MODEL_DIR}" ]]; then
  echo "[MAIN_BENCHMARK][ERROR] MODEL_DIR is required."
  echo "Example:"
  echo "  MODEL_DIR=outputs/experiments/SDRPV_residual_v1_residual_v1_gpu0_20260329_114156 \\"
  echo "  bash scripts/run_sdrpv_residual_v1_main_benchmark.sh"
  exit 1
fi

# 默认多游戏。若传入 GAMES，按空格分隔覆盖。
if [[ -n "${GAMES:-}" ]]; then
  # shellcheck disable=SC2206
  GAME_ARR=(${GAMES})
else
  GAME_ARR=(games/ticTacToe.kif games/connectFour.kif games/breakthrough.kif)
fi

exec >>"${LOG_FILE}" 2>&1

echo "[MAIN_BENCHMARK] start $(date)"
echo "[MAIN_BENCHMARK] run_tag=${RUN_TAG}"
echo "[MAIN_BENCHMARK] model_dir=${MODEL_DIR}"
echo "[MAIN_BENCHMARK] device=${DEVICE}"
echo "[MAIN_BENCHMARK] seeds=${SEEDS}"
echo "[MAIN_BENCHMARK] rounds=${ROUNDS}"
echo "[MAIN_BENCHMARK] games=${GAME_ARR[*]}"
echo "[MAIN_BENCHMARK] out_dir=${OUT_DIR}"
echo "[MAIN_BENCHMARK] log_file=${LOG_FILE}"

ARGS=(
  -u experiments/run_sdrpv_residual_v1_main_benchmark.py
  --model-dir "${MODEL_DIR}"
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

if [[ "${INCLUDE_VALUE_FULL}" == "1" ]]; then
  ARGS+=(--include-value-full)
fi

"${PYTHON_BIN}" "${ARGS[@]}"

echo "[MAIN_BENCHMARK] done $(date)"
echo "[MAIN_BENCHMARK] aggregate_json=${OUT_DIR}/summary/aggregate_summary.json"
echo "[MAIN_BENCHMARK] gate_json=${OUT_DIR}/summary/gate_check.json"
