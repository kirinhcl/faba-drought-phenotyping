#!/bin/bash
#SBATCH --job-name=faba-stress-eval
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stress_eval_%j.out
#SBATCH --error=logs/stress_eval_%j.err

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

# Set environment
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "=== Stress Detection Evaluation ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

# Run evaluation
python scripts/evaluate_stress.py \
    --results_dir results/stress/checkpoints/ \
    --config configs/stress.yaml

echo ""
echo "=== Evaluation Complete ==="
echo "End: $(date)"

# Show results summary
echo ""
echo "=== Results Summary ==="
if [ -f "results/stress/checkpoints/evaluation_results.json" ]; then
    cat results/stress/checkpoints/evaluation_results.json
fi
