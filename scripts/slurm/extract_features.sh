#!/bin/bash
#SBATCH --job-name=faba-features
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=results/logs/features_%j.out
#SBATCH --error=results/logs/features_%j.err

set -euo pipefail

REPO_DIR="/scratch/project_2013932/chenghao/faba-drought-phenotyping"
cd "${REPO_DIR}"

module purge
module load pytorch/2.4
source .venv/bin/activate

mkdir -p results/logs features

BACKBONE=${1:-dinov2}
echo "=== Extracting features: ${BACKBONE} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python3 scripts/extract_features.py \
    --backbone "${BACKBONE}" \
    --output "features/${BACKBONE}_features.h5" \
    --batch_size 64 \
    --num_workers 8

echo "End: $(date)"
echo "=== Done: ${BACKBONE} ==="
