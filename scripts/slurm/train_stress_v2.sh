#!/bin/bash
#SBATCH --job-name=faba-stress-v2
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-43%10
#SBATCH --output=logs/stress_v2_%A_%a.out
#SBATCH --error=logs/stress_v2_%A_%a.err

set -euo pipefail

REPO_DIR="/scratch/project_2013932/chenghao/faba-drought-phenotyping"
cd "${REPO_DIR}"

module purge
module load pytorch/2.4
source .venv/bin/activate

mkdir -p logs
mkdir -p results/stress_v2/checkpoints

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export WANDB_MODE=disabled

FOLD_ID=$SLURM_ARRAY_TASK_ID

echo "=== Stress v2 Fold ${FOLD_ID} ==="
echo "Changes: pos_weight=1.0, fluor_normalize=true"
echo "Job: ${SLURM_JOB_ID}, Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/train_stress.py \
    --config configs/stress_v2.yaml \
    --fold "${FOLD_ID}"

echo "End: $(date)"
