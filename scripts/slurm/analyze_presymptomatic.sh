#!/bin/bash
#SBATCH --job-name=faba-presym
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/presymptomatic_%j.out
#SBATCH --error=logs/presymptomatic_%j.err

set -euo pipefail

REPO_DIR="/scratch/project_2013932/chenghao/faba-drought-phenotyping"
cd "${REPO_DIR}"

# Load modules
module purge
module load pytorch/2.4

# Activate project virtualenv
source .venv/bin/activate

# Create directories
mkdir -p logs
mkdir -p results/presymptomatic_analysis

# Set environment
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "=== Pre-Symptomatic Detection Validation ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

# Run analysis (inference on all 44 folds + gates + triangulation)
python scripts/analyze_presymptomatic.py \
    --results_dir results/stress/checkpoints \
    --config configs/stress.yaml \
    --data_dir data \
    --output_dir results/presymptomatic_analysis

echo ""
echo "=== Analysis Complete ==="
echo "End: $(date)"

# Also run fluorescence divergence analysis (no GPU needed, but convenient)
echo ""
echo "=== Running Fluorescence Divergence Analysis ==="
python scripts/analyze_fluorescence.py \
    --data_dir data \
    --output_dir results/fluorescence_analysis

echo ""
echo "=== All Done ==="
echo "Output files:"
ls -la results/presymptomatic_analysis/
ls -la results/fluorescence_analysis/
