#!/usr/bin/env bash
#
# End-to-end reproducibility script.
#
# Runs the ten-seed SDViT training, the ten-seed UDAM adaptation, the
# six ablation groups, and the deployment pipeline, following Sections
# 4.1 to 4.5 of the thesis.
#
# Usage:
#   bash run_all.sh
#
# Environment:
#   CUDA_VISIBLE_DEVICES controls which GPUs are used.
#   The data roots and SNPE / TensorRT paths are read from
#   configs/default.yaml.
#
set -euo pipefail

CFG="configs/default.yaml"
SEEDS=(0 1 2 3 4 5 6 7 8 9)
RUNS_DIR="runs"

mkdir -p "${RUNS_DIR}"

echo "================================================="
echo "  SDViT + UDAM reproducibility pipeline"
echo "  config: ${CFG}"
echo "  runs:   ${RUNS_DIR}"
echo "================================================="

# ---------------------------------------------------------------------
# 1. Supervised SDViT training (ten seeds)
# ---------------------------------------------------------------------
for seed in "${SEEDS[@]}"; do
    echo "-------------------------------------------------"
    echo "  SDViT training | seed=${seed}"
    echo "-------------------------------------------------"
    python train_sdvit.py --cfg "${CFG}" --seed "${seed}"
done

# ---------------------------------------------------------------------
# 2. UDAM source-free adaptation (ten seeds)
# ---------------------------------------------------------------------
for seed in "${SEEDS[@]}"; do
    echo "-------------------------------------------------"
    echo "  UDAM adaptation | seed=${seed}"
    echo "-------------------------------------------------"
    python train_udam.py --cfg "${CFG}" --seed "${seed}"
done

# ---------------------------------------------------------------------
# 3. Ablation study (six groups, seed 0 as reference)
# ---------------------------------------------------------------------
for variant in full no_prune prune_only sd_only lambda_zero fixed_thr; do
    echo "-------------------------------------------------"
    echo "  Ablation | variant=${variant}"
    echo "-------------------------------------------------"
    python eval_ablation.py --cfg "${CFG}" --variant "${variant}" --seed 0
done

# ---------------------------------------------------------------------
# 4. Deployment pipeline (ONNX -> SNPE / TensorRT -> latency / power)
# ---------------------------------------------------------------------
echo "-------------------------------------------------"
echo "  Deployment | ONNX export"
echo "-------------------------------------------------"
python -m deploy.export_onnx

echo "-------------------------------------------------"
echo "  Deployment | operator compatibility check"
echo "-------------------------------------------------"
python -m deploy.check_ops

echo "-------------------------------------------------"
echo "  Deployment | SNPE engine"
echo "-------------------------------------------------"
python -m deploy.build_snpe

echo "-------------------------------------------------"
echo "  Deployment | TensorRT engine"
echo "-------------------------------------------------"
python -m deploy.build_tensorrt

echo "-------------------------------------------------"
echo "  Deployment | latency measurement"
echo "-------------------------------------------------"
python -m deploy.measure_latency

echo "-------------------------------------------------"
echo "  Deployment | power measurement"
echo "-------------------------------------------------"
python -m deploy.measure_power

echo "================================================="
echo "  Pipeline finished."
echo "================================================="