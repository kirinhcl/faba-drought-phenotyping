#!/bin/bash
# =============================================================================
# Train v3 full model with additional seeds (123, 456)
# Seed 42 already exists in results/stress_v3/checkpoints/
# New seeds go to results/stress_v3/checkpoints/seed_123/ and seed_456/
#
# Usage:
#   tmux new -s v3seed
#   bash scripts/slurm/train_v3_multiseed.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_ablation.sh"
CONFIG="configs/stress_v3.yaml"

# Only seeds 123 and 456 (seed 42 already done)
SEEDS=(123 456)

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

echo "=== Training v3 full model with additional seeds ==="
echo "Seeds to run: ${SEEDS[*]}"
echo "Start: $(date)"
echo ""

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="v3-seed-${SEED}"
    echo "=== Submitting seed ${SEED} ==="

    JOB_ID=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        "${TRAIN_SCRIPT}" \
        "${CONFIG}" \
        "${SEED}")

    echo "  → Job ${JOB_ID}"
    JOB_IDS=("${JOB_ID}")

    # Wait before submitting next seed (MaxSubmitJobs=4, stay safe)
    wait_for_jobs "${JOB_IDS[@]}"
    echo ""
done

echo "=== All v3 seeds complete ==="
echo "End: $(date)"
echo ""
echo "Next: evaluate with"
echo "  python scripts/evaluate_ablation.py \\"
echo "    --results_dir results/stress_v3/checkpoints \\"
echo "    --config configs/stress_v3.yaml \\"
echo "    --seeds 42,123,456"
