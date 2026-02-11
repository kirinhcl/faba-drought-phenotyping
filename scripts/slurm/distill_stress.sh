#!/bin/bash
#SBATCH --job-name=faba-distill-stress
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2%3
#SBATCH --output=logs/distill_stress_%A_%a.out
#SBATCH --error=logs/distill_stress_%A_%a.err

set -euo pipefail

REPO_DIR="/scratch/project_2013932/chenghao/faba-drought-phenotyping"
cd "${REPO_DIR}"

module purge
module load pytorch/2.4
source .venv/bin/activate

mkdir -p logs
mkdir -p results/distillation

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export WANDB_MODE=disabled

# Map array task ID to seed
SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

mkdir -p results/distillation/seed_${SEED}

echo "=== Distillation Training (Seed ${SEED}) ==="
echo "Job: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

# Train all 44 folds sequentially
for FOLD_ID in {0..43}; do
    echo ""
    echo "=== Fold ${FOLD_ID} / 43 ==="
    python scripts/train_distill_stress.py \
        --config configs/distillation_stress.yaml \
        --fold "${FOLD_ID}" \
        --seed "${SEED}"
done

echo ""
echo "=== All folds complete for seed ${SEED} ==="
echo "End: $(date)"
