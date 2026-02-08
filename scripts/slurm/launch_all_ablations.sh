#!/bin/bash
# =============================================================================
# Launch ALL ablation experiments: 11 ablations × 3 seeds = 33 SLURM array jobs
# Each array job has 44 tasks (one per fold), with max 10 concurrent per job.
#
# Total: 33 jobs × 44 folds = 1,452 training runs
# Estimated wall time: ~1h per job (early stopping at ~30-50 epochs)
# Estimated GPU-hours: ~1,452 × 0.5h = ~726 GPU-hours
#
# Usage:
#   bash scripts/slurm/launch_all_ablations.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_ablation.sh"

SEEDS=(42 123 456)

# All ablation configs
ABLATIONS=(
    # A1: Single modality
    "image_only"
    "fluor_only"
    "env_only"
    "vi_only"
    # A2: Leave-one-out
    "drop_image"
    "drop_fluor"
    "drop_env"
    "drop_vi"
    # A3: Causal masking
    "causal"
    # A4: No temporal
    "no_temporal"
    # A5: Concat fusion
    "concat_fusion"
)

echo "=== Launching ablation experiments ==="
echo "Ablations: ${#ABLATIONS[@]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total jobs: $(( ${#ABLATIONS[@]} * ${#SEEDS[@]} ))"
echo ""

JOB_COUNT=0

for ABLATION in "${ABLATIONS[@]}"; do
    CONFIG="configs/ablation/stress/${ABLATION}.yaml"

    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="abl-${ABLATION}-s${SEED}"

        echo "Submitting: ${JOB_NAME} (config=${CONFIG}, seed=${SEED})"

        sbatch \
            --job-name="${JOB_NAME}" \
            "${TRAIN_SCRIPT}" \
            "${CONFIG}" \
            "${SEED}"

        JOB_COUNT=$((JOB_COUNT + 1))
    done
done

echo ""
echo "=== Submitted ${JOB_COUNT} array jobs ==="
echo "Monitor with: squeue -u \$USER"
echo "After completion, run: bash scripts/slurm/evaluate_all_ablations.sh"
