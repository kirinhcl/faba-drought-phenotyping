#!/bin/bash
# =============================================================================
# Ablation Study SLURM Sweep Script
# =============================================================================
# Submits all 8 ablation variant training jobs to SLURM.
# Each variant runs 44 folds as an array job (0-43).
#
# Usage:
#   bash scripts/slurm/ablation_sweep.sh
#
# This script will submit 8 separate SLURM jobs, one for each variant.
# Monitor progress with: squeue -u $USER
# =============================================================================

set -euo pipefail

REPO_DIR="/scratch/project_2013932/chenghao/faba-drought-phenotyping"

# Array of (config_path:output_checkpoint_dir) pairs
VARIANTS=(
    "configs/ablation/variant_1_full_model.yaml:results/full_model"
    "configs/ablation/variant_2_image_only.yaml:results/image_only"
    "configs/ablation/variant_3_image_fluor.yaml:results/image_fluor"
    "configs/ablation/variant_4_no_temporal.yaml:results/no_temporal"
    "configs/ablation/variant_5_single_task.yaml:results/single_task"
    "configs/ablation/variant_6_lora.yaml:results/lora"
    "configs/ablation/variant_7_clip.yaml:results/clip_full"
    "configs/ablation/variant_8_bioclip.yaml:results/bioclip_full"
)

echo "=== Faba Bean Ablation Study SLURM Sweep ==="
echo "Repository: ${REPO_DIR}"
echo "Submitting ${#VARIANTS[@]} variants..."
echo ""

for entry in "${VARIANTS[@]}"; do
    IFS=':' read -r config checkpoint_dir <<< "$entry"
    echo "Submitting: $config -> $checkpoint_dir"
    sbatch scripts/slurm/train.sh "$config" "$checkpoint_dir"
done

echo ""
echo "All ${#VARIANTS[@]} variants submitted."
echo "Monitor with: squeue -u $USER"
echo "View logs with: tail -f logs/train_*.out"
