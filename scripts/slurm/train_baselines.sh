#!/bin/bash
#SBATCH --job-name=faba-baselines
#SBATCH --account=project_2013932
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err

# =============================================================================
# Train classical ML baselines (RF, LR, XGBoost) — CPU only, no GPU needed.
# Runs all 3 seeds sequentially in one job (~30-60 min total).
#
# Usage:
#   sbatch scripts/slurm/train_baselines.sh
# =============================================================================

set -euo pipefail

REPO_DIR="/scratch/project_2013932/chenghao/faba-drought-phenotyping"
cd "${REPO_DIR}"

module purge
module load pytorch/2.4
source .venv/bin/activate

mkdir -p logs
mkdir -p results/baselines

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

SEEDS=(42 123 456)

echo "=== Training classical ML baselines ==="
echo "Job: ${SLURM_JOB_ID}, Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "=== Seed: ${SEED} ==="
    python scripts/train_baselines.py \
        --config configs/stress_v3.yaml \
        --seed "${SEED}" \
        --output_dir results/baselines/
    echo ""
done

echo "=== All baselines complete ==="
echo "End: $(date)"
