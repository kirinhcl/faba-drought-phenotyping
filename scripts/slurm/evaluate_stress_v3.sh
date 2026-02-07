#!/bin/bash
#SBATCH --job-name=faba-stress-v3-eval
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stress_v3_eval_%j.out
#SBATCH --error=logs/stress_v3_eval_%j.err

set -euo pipefail

REPO_DIR="/scratch/project_2013932/chenghao/faba-drought-phenotyping"
cd "${REPO_DIR}"

module purge
module load pytorch/2.4
source .venv/bin/activate

mkdir -p logs
mkdir -p results/presymptomatic_v3
mkdir -p results/threshold_v3

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "=== Stress v3 Full Evaluation Pipeline ==="
echo "Job: ${SLURM_JOB_ID}, Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

echo ""
echo "=== Step 1: Evaluate stress v3 model ==="
python scripts/evaluate_stress.py \
    --results_dir results/stress_v3/checkpoints \
    --config configs/stress_v3.yaml

echo ""
echo "=== Step 2: Pre-symptomatic analysis ==="
python scripts/analyze_presymptomatic.py \
    --results_dir results/stress_v3/checkpoints \
    --config configs/stress_v3.yaml \
    --output_dir results/presymptomatic_v3

echo ""
echo "=== Step 3: Threshold analysis ==="
python scripts/analyze_threshold.py \
    --predictions results/presymptomatic_v3/plant_predictions.csv \
    --output_dir results/threshold_v3

echo ""
echo "=== All Done ==="
echo "End: $(date)"
