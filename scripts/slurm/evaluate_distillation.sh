#!/bin/bash
#SBATCH --job-name=eval-distill
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/eval_distill_%j.out
#SBATCH --error=logs/eval_distill_%j.err

set -euo pipefail

REPO_DIR="/scratch/project_2013932/chenghao/faba-drought-phenotyping"
cd "${REPO_DIR}"

module purge
module load pytorch/2.4
source .venv/bin/activate

mkdir -p logs

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export WANDB_MODE=disabled

echo "=== Evaluate Distillation (3 seeds × 44 folds) ==="
echo "Start: $(date)"

python scripts/evaluate_ablation.py \
    --results_dir results/distillation/checkpoints \
    --config configs/distillation_stress.yaml \
    --seeds 42,123,456

echo "End: $(date)"
