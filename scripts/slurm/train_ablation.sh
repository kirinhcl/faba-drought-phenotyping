#!/bin/bash
#SBATCH --job-name=faba-ablation
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-43%10
#SBATCH --output=logs/ablation_%x_%A_%a.out
#SBATCH --error=logs/ablation_%x_%A_%a.err

# =============================================================================
# Train a single ablation experiment (all 44 folds, one seed)
#
# Usage:
#   sbatch --job-name=image_only scripts/slurm/train_ablation.sh \
#       configs/ablation/stress/image_only.yaml 42
#
# Arguments:
#   $1 = config file path (e.g., configs/ablation/stress/image_only.yaml)
#   $2 = random seed (e.g., 42, 123, 456)
# =============================================================================

set -euo pipefail

CONFIG="${1:?Usage: sbatch train_ablation.sh <config.yaml> <seed>}"
SEED="${2:?Usage: sbatch train_ablation.sh <config.yaml> <seed>}"

REPO_DIR="/scratch/project_2013932/chenghao/faba-drought-phenotyping"
cd "${REPO_DIR}"

module purge
module load pytorch/2.4
source .venv/bin/activate

mkdir -p logs

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export WANDB_MODE=disabled

FOLD_ID=$SLURM_ARRAY_TASK_ID

echo "=== Ablation: ${CONFIG} | Seed: ${SEED} | Fold: ${FOLD_ID} ==="
echo "Job: ${SLURM_JOB_ID}, Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/train_stress.py \
    --config "${CONFIG}" \
    --fold "${FOLD_ID}" \
    --seed "${SEED}"

echo "End: $(date)"
