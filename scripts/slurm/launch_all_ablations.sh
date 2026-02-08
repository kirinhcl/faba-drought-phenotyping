#!/bin/bash
# =============================================================================
# Launch ALL ablation experiments sequentially
# Respects MaxSubmitJobs=4 by waiting for each ablation to complete.
#
# Strategy: Submit 3 seeds for one ablation (3 array jobs), wait for all to
# finish, then submit the next ablation's 3 seeds. Each ablation takes ~1h.
#
# Total: 11 ablations × 3 seeds × 44 folds = 1,452 training runs
# Total sequential time: ~11 hours (run in tmux/screen)
#
# Usage:
#   # Run in tmux/screen so it persists after logout:
#   tmux new -s ablation
#   bash scripts/slurm/launch_all_ablations.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_ablation.sh"

SEEDS=(42 123 456)

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

wait_for_jobs() {
    # Wait until all jobs with given IDs are finished
    local job_ids=("$@")
    echo "  Waiting for ${#job_ids[@]} jobs to complete..."
    while true; do
        local still_running=0
        for jid in "${job_ids[@]}"; do
            # Check if job is still in queue (PENDING, RUNNING, etc.)
            if squeue -j "${jid}" -h 2>/dev/null | grep -q "${jid}"; then
                still_running=$((still_running + 1))
            fi
        done
        if [ "${still_running}" -eq 0 ]; then
            break
        fi
        echo "  ... ${still_running} jobs still active ($(date +%H:%M:%S))"
        sleep 60
    done
    echo "  All jobs complete."
}

echo "=== Launching ablation experiments (sequential per ablation) ==="
echo "Ablations: ${#ABLATIONS[@]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total jobs: $(( ${#ABLATIONS[@]} * ${#SEEDS[@]} ))"
echo "Start: $(date)"
echo ""

TOTAL_DONE=0

for ABLATION in "${ABLATIONS[@]}"; do
    CONFIG="configs/ablation/stress/${ABLATION}.yaml"
    echo "=== [$(( TOTAL_DONE + 1 ))/${#ABLATIONS[@]}] ${ABLATION} ==="

    JOB_IDS=()
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="abl-${ABLATION}-s${SEED}"
        echo "  Submitting: ${JOB_NAME}"

        JOB_ID=$(sbatch --parsable \
            --job-name="${JOB_NAME}" \
            "${TRAIN_SCRIPT}" \
            "${CONFIG}" \
            "${SEED}")

        JOB_IDS+=("${JOB_ID}")
        echo "  → Job ${JOB_ID}"
    done

    # Wait for all 3 seeds to finish before submitting next ablation
    wait_for_jobs "${JOB_IDS[@]}"

    TOTAL_DONE=$((TOTAL_DONE + 1))
    echo ""
done

echo "=== All ${TOTAL_DONE} ablations complete ==="
echo "End: $(date)"
echo "Run evaluation: sbatch scripts/slurm/evaluate_all_ablations.sh"
