#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/logs outputs/experiments

PYTHON_BIN="${PYTHON_BIN:-python}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-ablation_all_${TS}}"
DATASET="${DATASET:-}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
SEEDS="${SEEDS:-42,142,242}"
ROUNDS="${ROUNDS:-50}"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-outputs/experiments/SDRPV_ablation_train_${RUN_TAG}}"
BENCH_OUT_DIR="${BENCH_OUT_DIR:-outputs/experiments/SDRPV_ablation_benchmark_${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-outputs/logs/SDRPV_ablation_all_${RUN_TAG}.log}"

if [[ -z "${DATASET}" ]]; then
  echo "[ABLATION_ALL][ERROR] DATASET is required."
  echo "Example:"
  echo "  DATASET=outputs/datasets/sdrpv_dataset_v3_parallel.jsonl DEVICE=cuda bash scripts/run_sdrpv_ablation_all.sh"
  exit 1
fi

if [[ -n "${GAMES:-}" ]]; then
  # shellcheck disable=SC2206
  GAME_ARR=(${GAMES})
else
  GAME_ARR=(games/hex.kif games/connectFour.kif games/breakthrough.kif)
fi

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[ABLATION_ALL] start $(date)"
echo "[ABLATION_ALL] run_tag=${RUN_TAG}"
echo "[ABLATION_ALL] dataset=${DATASET}"
echo "[ABLATION_ALL] device=${DEVICE}"
echo "[ABLATION_ALL] train_seed=${SEED}"
echo "[ABLATION_ALL] benchmark_seeds=${SEEDS}"
echo "[ABLATION_ALL] rounds=${ROUNDS}"
echo "[ABLATION_ALL] games=${GAME_ARR[*]}"
echo "[ABLATION_ALL] train_out_dir=${TRAIN_OUT_DIR}"
echo "[ABLATION_ALL] bench_out_dir=${BENCH_OUT_DIR}"
echo "[ABLATION_ALL] log_file=${LOG_FILE}"

TRAIN_ARGS=(
  -u experiments/run_sdrpv_ablation_train.py
  --dataset "${DATASET}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --out-dir "${TRAIN_OUT_DIR}"
)
if [[ -n "${GAME:-}" ]]; then
  TRAIN_ARGS+=(--game "${GAME}")
fi

BENCH_ARGS=(
  -u experiments/run_sdrpv_ablation_benchmark.py
  --models-root "${TRAIN_OUT_DIR}/models"
  --games "${GAME_ARR[@]}"
  --rounds "${ROUNDS}"
  --seeds "${SEEDS}"
  --fixed-sims 120
  --fixed-sims-playclock 0.5
  --fixed-time 0.5
  --fixed-time-iters 120
  --device "${DEVICE}"
  --out-dir "${BENCH_OUT_DIR}"
)

echo "[ABLATION_ALL] stage=train begin $(date)"
"${PYTHON_BIN}" "${TRAIN_ARGS[@]}"
echo "[ABLATION_ALL] stage=train done $(date)"

echo "[ABLATION_ALL] stage=benchmark begin $(date)"
"${PYTHON_BIN}" "${BENCH_ARGS[@]}"
echo "[ABLATION_ALL] stage=benchmark done $(date)"

echo "[ABLATION_ALL] done $(date)"
echo "[ABLATION_ALL] model_manifest=${TRAIN_OUT_DIR}/summary/model_manifest.csv"
echo "[ABLATION_ALL] aggregate_csv=${BENCH_OUT_DIR}/summary/aggregate_summary.csv"
echo "[ABLATION_ALL] by_game_csv=${BENCH_OUT_DIR}/summary/by_game_average.csv"
echo "[ABLATION_ALL] delta_csv=${BENCH_OUT_DIR}/summary/delta_vs_full.csv"
