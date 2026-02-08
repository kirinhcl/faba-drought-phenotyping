#!/bin/bash
#SBATCH --job-name=faba-v3-eval
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/v3_multiseed_eval_%j.out
#SBATCH --error=logs/v3_multiseed_eval_%j.err

set -euo pipefail

REPO_DIR="/scratch/project_2013932/chenghao/faba-drought-phenotyping"
cd "${REPO_DIR}"

module purge
module load pytorch/2.4
source .venv/bin/activate

mkdir -p logs
mkdir -p results/ablation/summary

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "=== Evaluating v3 full model (3 seeds) ==="
echo "Start: $(date)"

python scripts/evaluate_ablation.py \
    --results_dir results/stress_v3/checkpoints \
    --config configs/stress_v3.yaml \
    --seeds 42,123,456 \
    --output_dir results/ablation/summary

echo "=== Done ==="
echo "End: $(date)"
