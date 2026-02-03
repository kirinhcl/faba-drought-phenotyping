#!/bin/bash
#SBATCH --job-name=faba-analyze
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/analyze_%j.out
#SBATCH --error=logs/analyze_%j.err

# Usage:
#   sbatch scripts/slurm/analyze.sh results/full_model analysis/full_model configs/default.yaml
#   sbatch scripts/slurm/analyze.sh results/image_only analysis/image_only configs/ablation/variant_2_image_only.yaml

set -euo pipefail

REPO_DIR="/scratch/project_2013932/chenghao/faba-drought-phenotyping"
cd "${REPO_DIR}"

# Load modules
module purge
module load pytorch/2.4

# Activate project virtualenv
source .venv/bin/activate

# Create logs directory
mkdir -p logs

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export WANDB_MODE=disabled

# Parse arguments
RESULTS_DIR=${1:-results/full_model}
OUTPUT_DIR=${2:-analysis/full_model}
CONFIG=${3:-configs/default.yaml}
FOLD=${4:-0}

echo "=== XAI Analysis ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Results dir: ${RESULTS_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Config: ${CONFIG}"
echo "Fold: ${FOLD}"
echo "Start: $(date)"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# 1. Attention analysis (requires model inference)
echo ""
echo ">>> Running attention analysis..."
python scripts/analyze_attention.py \
    --model_dir "${RESULTS_DIR}" \
    --output_dir "${OUTPUT_DIR}/attention" \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    --device cuda

# 2. Ranking analysis (requires embedding extraction)
echo ""
echo ">>> Running ranking analysis..."
python scripts/analyze_ranking.py \
    --results_dir "${RESULTS_DIR}" \
    --output_dir "${OUTPUT_DIR}/ranking" \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    --device cuda

echo ""
echo "End: $(date)"
echo "=== Analysis completed ==="
