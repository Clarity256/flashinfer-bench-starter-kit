#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-mlsys26-contest}"
OUT_DIR="${2:-profile_out}"

mkdir -p "${OUT_DIR}"

echo "[1/3] Triton vs Torch latency benchmark..."
python scripts/benchmark_dsa_indexer.py \
  --dataset-root "${DATASET_ROOT}" \
  --num-workloads 10 \
  --warmup 20 \
  --iters 100 \
  --torch-baseline | tee "${OUT_DIR}/latency.txt"

echo "[2/3] NSYS timeline capture..."
nsys profile \
  --force-overwrite true \
  --trace cuda,nvtx,osrt \
  --output "${OUT_DIR}/nsys_dsa_indexer" \
  python scripts/benchmark_dsa_indexer.py \
    --dataset-root "${DATASET_ROOT}" \
    --num-workloads 1 \
    --warmup 10 \
    --iters 30

echo "[3/3] NCU kernel metrics..."
ncu \
  --set full \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name "regex:.*weighted_relu_score_kernel.*" \
  --launch-skip 10 \
  --launch-count 20 \
  --csv \
  --log-file "${OUT_DIR}/ncu_dsa_indexer.csv" \
  python scripts/benchmark_dsa_indexer.py \
    --dataset-root "${DATASET_ROOT}" \
    --num-workloads 1 \
    --warmup 20 \
    --iters 80

echo "Profiling artifacts are saved under: ${OUT_DIR}"
