#!/bin/bash
#SBATCH --job-name=faba-ablation-eval
#SBATCH --account=project_2013932
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ablation_eval_%j.out
#SBATCH --error=logs/ablation_eval_%j.err

# =============================================================================
# Evaluate ALL ablation experiments after training completes
#
# Usage:
#   sbatch scripts/slurm/evaluate_all_ablations.sh
# =============================================================================

set -euo pipefail

REPO_DIR="/scratch/project_2013932/chenghao/faba-drought-phenotyping"
cd "${REPO_DIR}"

module purge
module load pytorch/2.4
source .venv/bin/activate

mkdir -p logs
mkdir -p results/ablation/summary

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

SEEDS="42,123,456"

ABLATIONS=(
    "image_only"
    "fluor_only"
    "env_only"
    "vi_only"
    "drop_image"
    "drop_fluor"
    "drop_env"
    "drop_vi"
    "causal"
    "no_temporal"
    "concat_fusion"
)

echo "=== Evaluating all ablation experiments ==="
echo "Start: $(date)"
echo ""

for ABLATION in "${ABLATIONS[@]}"; do
    RESULTS_DIR="results/ablation/${ABLATION}/checkpoints"
    CONFIG="configs/ablation/stress/${ABLATION}.yaml"
    OUTPUT_DIR="results/ablation/summary"

    if [ ! -d "${RESULTS_DIR}" ]; then
        echo "SKIP: ${ABLATION} (no results directory)"
        continue
    fi

    echo "--- Evaluating: ${ABLATION} ---"

    python scripts/evaluate_ablation.py \
        --results_dir "${RESULTS_DIR}" \
        --config "${CONFIG}" \
        --output_dir "${OUTPUT_DIR}" \
        --seeds "${SEEDS}"

    echo ""
done

echo "=== All evaluations complete ==="
echo "Results in: results/ablation/summary/"
echo "End: $(date)"
