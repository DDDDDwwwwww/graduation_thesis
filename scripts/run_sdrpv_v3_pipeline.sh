#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/logs outputs/datasets outputs/cache outputs/experiments

TS="$(date +%Y%m%d_%H%M%S)"
CONVERT_LOG="outputs/logs/sdrpv_convert_v3_${TS}.log"
TRAIN_LOG="outputs/logs/sdrpv_teacher_only_v3_${TS}.log"
STATS_FILE="outputs/datasets/sdrpv_dataset_v3_parallel.jsonl.stats.json"

echo "[PIPELINE] start $(date)"

echo "[PIPELINE] step1 convert start $(date)"
python -u experiments/convert_dataset_to_sdrpv.py \
  --input outputs/experiments/D_dataset_size_sensitivity/datasets/dataset_200.jsonl \
  --output outputs/datasets/sdrpv_dataset_v3_parallel.jsonl \
  --baseline-cache outputs/cache/sdrpv_baseline_cache_v3.jsonl \
  --teacher-cache outputs/cache/sdrpv_teacher_cache_v3.jsonl \
  --baseline-mode shallow_mcts \
  --baseline-sims 32 \
  --student-sims 120 \
  --teacher-sims 600 \
  --num-workers 8 \
  --log-every 20 \
  --heartbeat-sec 30 \
  --resume \
  >"${CONVERT_LOG}" 2>&1

echo "[PIPELINE] step1 convert done $(date)"
echo "[PIPELINE] convert_log=${CONVERT_LOG}"

if [[ ! -f "${STATS_FILE}" ]]; then
  echo "[PIPELINE][ERROR] stats file not found: ${STATS_FILE}"
  exit 1
fi

METRICS="$(python -c "import json; d=json.load(open('${STATS_FILE}','r',encoding='utf-8')); print(d['q_t']['std'], d['b']['std'], d['|q_t-b|']['mean'])")"
QT_STD="$(echo "${METRICS}" | awk '{print $1}')"
B_STD="$(echo "${METRICS}" | awk '{print $2}')"
DIFF_MEAN="$(echo "${METRICS}" | awk '{print $3}')"

echo "[PIPELINE] q_t.std=${QT_STD} b.std=${B_STD} |q_t-b|.mean=${DIFF_MEAN}"

PASS="$(python -c "import json; d=json.load(open('${STATS_FILE}','r',encoding='utf-8')); ok=(d['q_t']['std']>0 and d['b']['std']>0 and d['|q_t-b|']['mean']>0); print(1 if ok else 0)")"
if [[ "${PASS}" != "1" ]]; then
  echo "[PIPELINE][STOP] distribution check failed, skip teacher-only training."
  exit 2
fi

echo "[PIPELINE] step2 teacher-only training start $(date)"
python -u experiments/train_sdrpv_teacher_only.py \
  --dataset outputs/datasets/sdrpv_dataset_v3_parallel.jsonl \
  --target-field q_t \
  --encoder board_token \
  --model transformer \
  --epochs 20 \
  --batch-size 128 \
  --loss huber \
  --output-dir outputs/experiments/SDRPV_teacher_only_v3 \
  --device cpu \
  >"${TRAIN_LOG}" 2>&1

echo "[PIPELINE] step2 teacher-only training done $(date)"
echo "[PIPELINE] train_log=${TRAIN_LOG}"
echo "[PIPELINE] done $(date)"

