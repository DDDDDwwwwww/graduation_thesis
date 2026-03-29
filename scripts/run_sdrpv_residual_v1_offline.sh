#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/logs outputs/experiments

PYTHON_BIN="${PYTHON_BIN:-python}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-${TS}}"
DEVICE="${DEVICE:-cpu}"
DATASET_PATH="${DATASET_PATH:-outputs/datasets/sdrpv_dataset_v3_parallel.jsonl}"
TRAIN_LOG="outputs/logs/sdrpv_residual_v1_${RUN_TAG}.log"
MCTS_LOG="outputs/logs/sdrpv_residual_v1_mcts_smoke_${RUN_TAG}.log"
MODEL_DIR="${MODEL_DIR:-outputs/experiments/SDRPV_residual_v1_${RUN_TAG}}"
OFFLINE_METRICS="${MODEL_DIR}/offline_metrics.json"
MCTS_OUT_DIR="${MCTS_OUT_DIR:-outputs/experiments/SDRPV_residual_v1_mcts_smoke_${RUN_TAG}}"

echo "[RESIDUAL_V1] start $(date)"

"${PYTHON_BIN}" -u experiments/train_sdrpv_residual_v1.py \
  --dataset "${DATASET_PATH}" \
  --encoder board_token \
  --model transformer \
  --epochs 20 \
  --batch-size 128 \
  --loss huber \
  --output-dir "${MODEL_DIR}" \
  --device "${DEVICE}" \
  >"${TRAIN_LOG}" 2>&1

echo "[RESIDUAL_V1] train done $(date)"
echo "[RESIDUAL_V1] train_log=${TRAIN_LOG}"

if [[ ! -f "${OFFLINE_METRICS}" ]]; then
  echo "[RESIDUAL_V1][ERROR] offline metrics not found: ${OFFLINE_METRICS}"
  exit 1
fi

RES="$("${PYTHON_BIN}" -c "import json; d=json.load(open('${OFFLINE_METRICS}','r',encoding='utf-8')); t=d['test']; print(t['mae_vhat_qt'], t['mae_b_qt'], t['corr_vhat_qt'], t['rank_corr_vhat_qt'])")"
MAE_V="$(echo "${RES}" | awk '{print $1}')"
MAE_B="$(echo "${RES}" | awk '{print $2}')"
CORR="$(echo "${RES}" | awk '{print $3}')"
RANK_CORR="$(echo "${RES}" | awk '{print $4}')"

echo "[RESIDUAL_V1] test_mae(v_hat,q_t)=${MAE_V}"
echo "[RESIDUAL_V1] test_mae(b,q_t)=${MAE_B}"
echo "[RESIDUAL_V1] test_corr(v_hat,q_t)=${CORR}"
echo "[RESIDUAL_V1] test_rank_corr(v_hat,q_t)=${RANK_CORR}"

PASS="$("${PYTHON_BIN}" -c "import json; d=json.load(open('${OFFLINE_METRICS}','r',encoding='utf-8')); t=d['test']; ok=(t['mae_vhat_qt'] < t['mae_b_qt']); print(1 if ok else 0)")"
if [[ "${PASS}" != "1" ]]; then
  echo "[RESIDUAL_V1][STOP] offline residual not better than baseline MAE; do not enable MCTS yet."
  exit 2
fi

echo "[RESIDUAL_V1] offline gate passed; ready for small-scale MCTS matches."

echo "[RESIDUAL_V1] step2 small-scale MCTS smoke start $(date)"
"${PYTHON_BIN}" -u experiments/run_sdrpv_residual_v1_mcts_smoke.py \
  --model-dir "${MODEL_DIR}" \
  --game games/connectFour.kif \
  --rounds 4 \
  --fixed-sims 120 \
  --fixed-sims-playclock 0.5 \
  --fixed-time 0.5 \
  --fixed-time-iters 120 \
  --out-dir "${MCTS_OUT_DIR}" \
  --device "${DEVICE}" \
  >"${MCTS_LOG}" 2>&1

echo "[RESIDUAL_V1] step2 small-scale MCTS smoke done $(date)"
echo "[RESIDUAL_V1] mcts_log=${MCTS_LOG}"
echo "[RESIDUAL_V1] done $(date)"
