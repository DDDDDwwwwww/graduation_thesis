#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/logs outputs/experiments

PYTHON_BIN="${PYTHON_BIN:-python}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-ablation_train_${TS}}"
DATASET="${DATASET:-}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-outputs/experiments/SDRPV_ablation_train_${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-outputs/logs/SDRPV_ablation_train_${RUN_TAG}.log}"

if [[ -z "${DATASET}" ]]; then
  echo "[ABLATION_TRAIN][ERROR] DATASET is required."
  echo "Example:"
  echo "  DATASET=outputs/datasets/your_sdrpv.jsonl bash scripts/run_sdrpv_ablation_train.sh"
  exit 1
fi

exec >>"${LOG_FILE}" 2>&1

echo "[ABLATION_TRAIN] start $(date)"
echo "[ABLATION_TRAIN] dataset=${DATASET}"
echo "[ABLATION_TRAIN] device=${DEVICE}"
echo "[ABLATION_TRAIN] seed=${SEED}"
echo "[ABLATION_TRAIN] out_dir=${OUT_DIR}"
echo "[ABLATION_TRAIN] log_file=${LOG_FILE}"

ARGS=(
  -u experiments/run_sdrpv_ablation_train.py
  --dataset "${DATASET}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --out-dir "${OUT_DIR}"
)

if [[ -n "${GAME:-}" ]]; then
  ARGS+=(--game "${GAME}")
fi

"${PYTHON_BIN}" "${ARGS[@]}"

echo "[ABLATION_TRAIN] done $(date)"
echo "[ABLATION_TRAIN] model_manifest=${OUT_DIR}/summary/model_manifest.csv"
