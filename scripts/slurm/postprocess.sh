#!/bin/bash
#SBATCH --job-name=faba-postprocess
#SBATCH --account=project_2013932
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/postprocess_%j.out
#SBATCH --error=logs/postprocess_%j.err

# Usage:
#   sbatch scripts/slurm/postprocess.sh
#
# This script runs all CPU-based post-processing:
#   1. Evaluate all 8 variants (aggregate 44-fold metrics)
#   2. Run classical ML baselines
#   3. Generate paper figures

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

echo "=== Post-processing Pipeline ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"

# =============================================================================
# 1. Evaluate all variants
# =============================================================================
echo ""
echo "=========================================="
echo "[1/3] Evaluating all variants..."
echo "=========================================="

VARIANTS=(
    "results/full_model"
    "results/image_only"
    "results/image_fluor"
    "results/no_temporal"
    "results/single_task"
    "results/lora"
    "results/clip_full"
    "results/bioclip_full"
)

for RESULTS_DIR in "${VARIANTS[@]}"; do
    if [ -d "${RESULTS_DIR}" ]; then
        VARIANT_NAME=$(basename "${RESULTS_DIR}")
        echo ""
        echo ">>> Evaluating ${VARIANT_NAME}..."
        python scripts/evaluate.py \
            --results_dir "${RESULTS_DIR}" \
            --output "${RESULTS_DIR}/main_results.json"
    else
        echo ">>> Skipping ${RESULTS_DIR} (not found)"
    fi
done

# =============================================================================
# 2. Run classical ML baselines
# =============================================================================
echo ""
echo "=========================================="
echo "[2/3] Running classical ML baselines..."
echo "=========================================="

python scripts/run_baselines.py \
    --config configs/default.yaml \
    --output_dir results/baselines/

# =============================================================================
# 3. Generate paper figures
# =============================================================================
echo ""
echo "=========================================="
echo "[3/3] Generating paper figures..."
echo "=========================================="

python scripts/generate_figures.py \
    --results_dir results/ \
    --output_dir paper/figures/

echo ""
echo "End: $(date)"
echo "=== Post-processing completed ==="
echo ""
echo "Output locations:"
echo "  - Variant metrics: results/*/main_results.json"
echo "  - Baseline results: results/baselines/"
echo "  - Paper figures: paper/figures/"
