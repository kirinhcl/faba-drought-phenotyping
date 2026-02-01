#!/bin/bash
# Submit feature extraction jobs for all 3 backbones
# Usage: bash scripts/slurm/extract_all_features.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p results/logs

echo "Submitting feature extraction jobs..."

JOB1=$(sbatch --parsable "${SCRIPT_DIR}/extract_features.sh" dinov2)
echo "DINOv2:  job ${JOB1}"

JOB2=$(sbatch --parsable "${SCRIPT_DIR}/extract_features.sh" clip)
echo "CLIP:    job ${JOB2}"

JOB3=$(sbatch --parsable "${SCRIPT_DIR}/extract_features.sh" bioclip)
echo "BioCLIP: job ${JOB3}"

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f results/logs/features_*.out"
