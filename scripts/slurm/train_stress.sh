#!/bin/bash
#SBATCH --job-name=faba-stress
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-43%10
#SBATCH --output=logs/stress_%A_%a.out
#SBATCH --error=logs/stress_%A_%a.err

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

# Feature directory: read directly from scratch (no NVMe needed)
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export WANDB_MODE=disabled
FEATURE_DIR="${REPO_DIR}/features"

# Get fold ID from array task ID
FOLD_ID=$SLURM_ARRAY_TASK_ID

# Parse arguments (with defaults)
CONFIG=${1:-configs/stress.yaml}
CHECKPOINT_DIR=${2:-results/stress/checkpoints}

echo "=== Training Stress Detection Fold ${FOLD_ID} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Config: ${CONFIG}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Feature dir: ${FEATURE_DIR}"
echo "Start: $(date)"

# Run training for this fold
python scripts/train_stress.py \
    --config "${CONFIG}" \
    --fold "${FOLD_ID}"

echo "End: $(date)"
echo "=== Stress Detection Fold ${FOLD_ID} completed ==="
