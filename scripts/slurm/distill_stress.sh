#!/bin/bash
#SBATCH --job-name=faba-distill
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-43%10
#SBATCH --output=logs/distill_stress_%A_%a.out
#SBATCH --error=logs/distill_stress_%A_%a.err

# =============================================================================
# Train distillation for a single seed (all 44 folds via array)
#
# Usage:
#   sbatch scripts/slurm/distill_stress.sh 42
#   sbatch scripts/slurm/distill_stress.sh 123
#   sbatch scripts/slurm/distill_stress.sh 456
#
# Or all three seeds:
#   for s in 42 123 456; do sbatch scripts/slurm/distill_stress.sh $s; done
#
# Arguments:
#   $1 = random seed (e.g., 42, 123, 456)
# =============================================================================

set -euo pipefail

SEED="${1:?Usage: sbatch distill_stress.sh <seed>}"

REPO_DIR="/scratch/project_2013932/chenghao/faba-drought-phenotyping"
cd "${REPO_DIR}"

module purge
module load pytorch/2.4
source .venv/bin/activate

mkdir -p logs
mkdir -p results/distillation

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export WANDB_MODE=disabled

FOLD_ID=$SLURM_ARRAY_TASK_ID

echo "=== Distillation | Seed: ${SEED} | Fold: ${FOLD_ID} ==="
echo "Job: ${SLURM_JOB_ID}, Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/train_distill_stress.py \
    --config configs/distillation_stress.yaml \
    --fold "${FOLD_ID}" \
    --seed "${SEED}"

echo "End: $(date)"
