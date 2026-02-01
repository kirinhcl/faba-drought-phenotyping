#!/bin/bash
# Setup script for CSC Mahti supercomputer
# Usage: bash scripts/setup_mahti.sh
set -euo pipefail

PROJECT_DIR="/scratch/project_2013932/chenghao"
REPO_DIR="${PROJECT_DIR}/faba-drought-phenotyping"
DATA_DIR="${PROJECT_DIR}/faba-data"
VENV_DIR="${REPO_DIR}/.venv"

echo "=== Faba Drought Phenotyping — Mahti Setup ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Repo dir:    ${REPO_DIR}"
echo "Data dir:    ${DATA_DIR}"

# Clone repo if not present
if [ ! -d "${REPO_DIR}" ]; then
    echo "Cloning repository..."
    cd "${PROJECT_DIR}"
    git clone git@github.com:kirinhcl/faba-drought-phenotyping.git
fi

cd "${REPO_DIR}"

# Load CSC modules
module purge
module load pytorch/2.4

# Create venv with system site-packages (inherits CSC's PyTorch + CUDA)
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv --system-site-packages "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# Install additional deps not in CSC's pytorch module
pip install --quiet --upgrade pip
pip install --quiet omegaconf open_clip_torch wandb xgboost openpyxl seaborn

# Verify installation
python3 -c "
import torch
import torchvision
import transformers
import open_clip
import omegaconf
import h5py
import wandb
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'transformers {transformers.__version__}')
print(f'open_clip {open_clip.__version__}')
print('All dependencies OK')
"

# Create data directory structure
mkdir -p "${DATA_DIR}"
mkdir -p "${REPO_DIR}/features"
mkdir -p "${REPO_DIR}/results"

# Symlink data if it exists and symlinks don't yet
if [ -d "${DATA_DIR}/img" ] && [ ! -L "${REPO_DIR}/data/img" ]; then
    echo "Creating data symlinks..."
    ln -sf "${DATA_DIR}/img" "${REPO_DIR}/data/img"
fi

for subdir in "00-Misc" "TimeCourse Datasets" "SinglePoint Datasets" "EndPoint Datasets" "Avg_and_Ranks"; do
    if [ -d "${DATA_DIR}/${subdir}" ] && [ ! -L "${REPO_DIR}/data/${subdir}" ]; then
        ln -sf "${DATA_DIR}/${subdir}" "${REPO_DIR}/data/${subdir}"
    fi
done

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Transfer data from local machine (see below)"
echo "  2. Run: source ${VENV_DIR}/bin/activate"
echo "  3. Verify: python3 -m src.data.metadata"
echo ""
echo "Data transfer command (run on your Mac):"
echo "  rsync -avzP ~/Downloads/Faba/data/ chenghao@mahti.csc.fi:${DATA_DIR}/"
