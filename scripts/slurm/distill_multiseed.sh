#!/bin/bash
# =============================================================================
# Run distillation training for all 3 seeds (42, 123, 456)
# Submits each seed as an array job (44 folds), waits for completion
# before submitting the next seed (respects MaxSubmitJobs=4).
#
# Usage:
#   tmux new -s distill
#   bash scripts/slurm/distill_multiseed.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTILL_SCRIPT="${SCRIPT_DIR}/distill_stress.sh"

SEEDS=(42 123 456)

wait_for_jobs() {
    local job_ids=("$@")
    echo "  Waiting for ${#job_ids[@]} jobs to complete..."
    while true; do
        local still_running=0
        for jid in "${job_ids[@]}"; do
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

echo "=== Distillation training: 3 seeds × 44 folds ==="
echo "Seeds: ${SEEDS[*]}"
echo "Start: $(date)"
echo ""

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="distill-s${SEED}"
    echo "=== Submitting seed ${SEED} ==="

    JOB_ID=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        "${DISTILL_SCRIPT}" \
        "${SEED}")

    echo "  → Job ${JOB_ID}"
    JOB_IDS=("${JOB_ID}")

    wait_for_jobs "${JOB_IDS[@]}"
    echo ""
done

echo "=== All distillation seeds complete ==="
echo "End: $(date)"
echo ""
echo "Results: results/distillation/seed_{42,123,456}/"
echo ""
echo "Next: sync results to local"
echo "  rsync -avz mahti:/scratch/.../results/distillation/ ./results/distillation/"
