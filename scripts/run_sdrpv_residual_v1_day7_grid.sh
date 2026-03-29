#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/logs outputs/experiments

PYTHON_BIN="${PYTHON_BIN:-python}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-day7_grid_${TS}}"
DEVICE="${DEVICE:-cpu}"
MODEL_DIR="${MODEL_DIR:-}"
LOG_FILE="${LOG_FILE:-outputs/logs/sdrpv_residual_v1_day7_grid_${RUN_TAG}.log}"
OUT_DIR="${OUT_DIR:-outputs/experiments/SDRPV_residual_v1_mcts_grid_${RUN_TAG}}"
ROUNDS="${ROUNDS:-20}"

if [[ -z "${MODEL_DIR}" ]]; then
  echo "[DAY7_GRID][ERROR] MODEL_DIR is required."
  echo "Example: MODEL_DIR=outputs/experiments/SDRPV_residual_v1_xxx bash scripts/run_sdrpv_residual_v1_day7_grid.sh"
  exit 1
fi

exec >>"${LOG_FILE}" 2>&1

echo "[DAY7_GRID] start $(date)"
"${PYTHON_BIN}" -u experiments/run_sdrpv_residual_v1_mcts_grid.py \
  --model-dir "${MODEL_DIR}" \
  --game games/connectFour.kif \
  --rounds "${ROUNDS}" \
  --fixed-sims 120 \
  --fixed-sims-playclock 0.5 \
  --fixed-time 0.5 \
  --fixed-time-iters 120 \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}"
echo "[DAY7_GRID] done $(date)"
echo "[DAY7_GRID] log_file=${LOG_FILE}"
echo "[DAY7_GRID] out_dir=${OUT_DIR}"
