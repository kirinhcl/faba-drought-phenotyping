# Stress Detection Model - Implementation Guide

## Overview

This document describes the stress detection model implementation - a binary per-timestep classification task for early drought stress detection in faba bean plants.

## Quick Start

### Training on CSC Mahti

```bash
# 1. SSH to Mahti
ssh username@mahti.csc.fi
cd /scratch/project_2013932/chenghao/faba-drought-phenotyping

# 2. Activate environment
source .venv/bin/activate
module load pytorch/2.4

# 3. Submit training job (all 44 folds)
sbatch scripts/slurm/train_stress.sh

# 4. Monitor progress
squeue -u $USER
watch -n 60 'ls results/stress/checkpoints/fold_*/best_model.pt | wc -l'

# 5. After training completes, evaluate
python scripts/evaluate_stress.py --results_dir results/stress/checkpoints/
```

### Training Single Fold (Testing)

```bash
# Train just fold 0 for validation
python scripts/train_stress.py --config configs/stress.yaml --fold 0

# Check outputs
ls -lh results/stress/checkpoints/fold_0/
# Expected: best_model.pt, modality_gates.pt
```

## Architecture

### Model Pipeline

```
Input: 4 modalities per timestep (T=22)
    ↓
1. View Aggregation: 4 views → 1 per timestep
    ↓
2. Modality Projection: Each modality → 128-dim
   - Image (768-dim) → 128-dim
   - Fluorescence (94-dim) → 128-dim
   - Environment (5-dim) → 128-dim
   - Vegetation Index (11-dim) → 128-dim
    ↓
3. Modality Gating: Learn per-timestep weights
   - Concat all modalities → Gate network → Softmax
   - Output: Weighted sum (B, T, 128) + Gates (B, T, 4)
    ↓
4. Temporal Transformer: Reason across timesteps
   - 2 layers, 4 heads, dim=128
    ↓
5. Stress Head: Binary classification per timestep
   - MLP → (B, T) logits
    ↓
Loss: BCEWithLogitsLoss with auto pos_weight
```

### Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Task** | Binary per-timestep classification | Simpler than DAG regression, clinically relevant |
| **Fusion** | Modality Gating | Learnable weights > fixed concat, interpretable |
| **Temporal dim** | 128 (not 256) | Smaller model, faster training, sufficient capacity |
| **Loss** | BCE with auto pos_weight | Handles class imbalance automatically |
| **Evaluation** | Onset detection | Early detection is the core clinical metric |

## Files Created

### Core Model (3 files)

1. **`src/model/gating.py`** (124 lines)
   - `ModalityProjection`: Projects each modality to 128-dim
   - `ModalityGating`: Learns per-timestep importance weights
   - Handles missing data with learnable mask tokens

2. **`src/model/stress_model.py`** (147 lines)
   - `StressHead`: MLP for binary classification
   - `StressDetectionModel`: Full 5-stage pipeline
   - Returns: `{'stress_logits': (B,T), 'modality_gates': (B,T,4)}`

3. **`src/training/stress_loss.py`** (71 lines)
   - `StressLoss`: BCE loss with masking
   - Auto-computes `pos_weight = num_neg / (num_pos + 1e-6)`
   - Returns `(loss_tensor, loss_dict)` tuple

### Data Pipeline (2 files modified)

4. **`src/data/dataset.py`** (modified)
   - Added `stress_labels` generation (T=22,) binary 0/1
   - Added `stress_mask` (T=22,) bool for valid timesteps
   - Logic: WHC-30 → label=1 when current_DAG >= threshold_DAG
   - WHC-80 → all labels=0 (never stressed)

5. **`src/data/collate.py`** (modified)
   - Batch `stress_labels` (B, T=22) long
   - Batch `stress_mask` (B, T=22) bool

### Configuration & Scripts (4 files)

6. **`configs/stress.yaml`** (89 lines)
   - Model config: temporal.dim=128, modality.hidden_dim=128
   - Training: 100 epochs, patience=20, 44-fold LOGO-CV
   - Logging: results/stress/ directory

7. **`scripts/train_stress.py`** (471 lines)
   - Custom training loop (Trainer incompatible with single-task)
   - Mixed precision (bfloat16), early stopping, gradient clipping
   - Saves modality gates for analysis

8. **`scripts/evaluate_stress.py`** (252 lines)
   - Per-timestep metrics: accuracy, precision, recall, F1, AUC
   - **Onset detection** (core): MAE, early detection rate, mean early days
   - Aggregates across 44 folds

9. **`scripts/slurm/train_stress.sh`** (57 lines)
   - SLURM array job: 0-43 (44 folds), max 10 concurrent
   - Resources: 1 A100 GPU, 32GB RAM, 1 hour per fold

## Stress Label Generation

### Logic

```python
# For each plant at each timestep
if treatment == 'WHC-30':
    if current_DAG >= dag_drought_onset:
        stress_label = 1  # Stressed
    else:
        stress_label = 0  # Not stressed yet
else:  # WHC-80 (control)
    stress_label = 0  # Never stressed
```

### Example

Plant with `dag_drought_onset = 20`:
- Rounds 2-11 (DAG 4-19): label = 0 (not stressed)
- Rounds 12-23 (DAG 20-38): label = 1 (stressed)

WHC-80 plant:
- All rounds: label = 0 (control, never stressed)

## Evaluation Metrics

### Per-Timestep Classification

- **Accuracy**: Overall correctness
- **Precision**: Of predicted stress, how many are true
- **Recall**: Of true stress, how many detected
- **F1**: Harmonic mean of precision/recall
- **AUC**: Area under ROC curve

### Onset Detection (Core Metric)

**Goal**: Detect stress onset as early as possible

**Metrics**:
1. **Onset MAE**: Mean absolute error in DAG prediction
   - Lower is better
   - Measures accuracy of onset timing

2. **Early Detection Rate**: % of WHC-30 plants detected before true onset
   - Higher is better
   - Measures ability to predict stress before symptoms

3. **Mean Early Days**: Average lead time for early detections
   - Higher is better
   - Measures how many days in advance we can detect

**Computation**:
```python
# True onset: first timestep with label=1
true_onset_idx = (labels == 1).nonzero()[0]
true_onset_dag = ROUND_TO_DAG[true_onset_idx + 2]

# Predicted onset: first timestep with prob > 0.5
pred_onset_idx = (probs > 0.5).nonzero()[0]
pred_onset_dag = ROUND_TO_DAG[pred_onset_idx + 2]

# Error (negative = early detection)
error = pred_onset_dag - true_onset_dag
early = (error < 0)
```

## Expected Results

### Training

- **Initial loss**: ~0.5-0.7 (random initialization)
- **Final loss**: <0.3 (after convergence)
- **Training time**: ~30-60 minutes per fold on A100
- **Total time**: ~22-44 hours for all 44 folds

### Evaluation

**Expected Ranges** (based on similar tasks):
- Accuracy: 0.75-0.90
- F1: 0.70-0.85
- AUC: 0.80-0.95
- Onset MAE: 3-7 days
- Early detection rate: 0.40-0.70
- Mean early days: 2-5 days

### Modality Gates

**Expected Patterns**:
- **Image**: Higher weights early (visual symptoms appear first)
- **Fluorescence**: Higher weights mid-late (physiological stress)
- **Environment**: Consistent baseline (always relevant)
- **Vegetation Index**: Correlated with image (derived from RGB)

## Validation Checklist

### Pre-Training Validation

```bash
# 1. Verify imports
python -c "from src.model.stress_model import StressDetectionModel; print('✓')"
python -c "from src.training.stress_loss import StressLoss; print('✓')"
python -c "from src.data.dataset import FabaDroughtDataset; print('✓')"

# 2. Test model instantiation
python -c "
from src.model.stress_model import StressDetectionModel
from src.utils.config import load_config
cfg = load_config('configs/stress.yaml')
model = StressDetectionModel(cfg)
print(f'✓ Model: {sum(p.numel() for p in model.parameters())} params')
"

# 3. Test dataset loading
python -c "
from src.data.dataset import FabaDroughtDataset
from src.utils.config import load_config
cfg = load_config('configs/stress.yaml')
ds = FabaDroughtDataset(cfg)
sample = ds[0]
print(f'✓ Dataset: {len(ds)} samples')
print(f'✓ stress_labels: {sample[\"stress_labels\"].shape}')
print(f'✓ stress_mask: {sample[\"stress_mask\"].shape}')
"
```

### Post-Training Validation

```bash
# 1. Verify checkpoint exists
ls -lh results/stress/checkpoints/fold_0/best_model.pt

# 2. Load checkpoint
python -c "
import torch
ckpt = torch.load('results/stress/checkpoints/fold_0/best_model.pt')
print(f'✓ Checkpoint: {len(ckpt)} keys')
"

# 3. Run evaluation
python scripts/evaluate_stress.py --results_dir results/stress/checkpoints/

# 4. Verify metrics
python -c "
import json
with open('results/stress/checkpoints/evaluation_results.json') as f:
    results = json.load(f)
print('✓ Metrics:')
for key, value in results['aggregated'].items():
    print(f'  {key}: {value:.4f}')
"

# 5. Verify modality gates
python -c "
import numpy as np
gates = np.load('results/stress/checkpoints/fold_0/test_modality_gates.npy')
print(f'✓ Gates shape: {gates.shape}')  # (N, 22, 4)
print(f'✓ Sum to 1: {np.allclose(gates.sum(axis=-1), 1.0)}')
"
```

## Visualization

### Modality Gates Over Time

```python
import numpy as np
import matplotlib.pyplot as plt

# Load gates from a fold
gates = np.load('results/stress/checkpoints/fold_0/test_modality_gates.npy')
mean_gates = gates.mean(axis=0)  # Average across samples (N, 22, 4) → (22, 4)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(mean_gates[:, 0], label='Image', marker='o')
plt.plot(mean_gates[:, 1], label='Fluorescence', marker='s')
plt.plot(mean_gates[:, 2], label='Environment', marker='^')
plt.plot(mean_gates[:, 3], label='Vegetation Index', marker='d')
plt.xlabel('Timestep (Round)')
plt.ylabel('Gate Weight')
plt.title('Mean Modality Gate Weights Over Time')
plt.legend()
plt.grid(True)
plt.savefig('results/stress/modality_gates.png')
```

### Loss Curves

```python
import json
import matplotlib.pyplot as plt

# Load training logs (if saved)
with open('results/stress/checkpoints/fold_0/training_log.json') as f:
    log = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(log['train_loss'], label='Train Loss')
plt.plot(log['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('results/stress/loss_curves.png')
```

## Troubleshooting

### Common Issues

**1. Import errors**
```bash
# Solution: Activate virtual environment
source .venv/bin/activate
module load pytorch/2.4
```

**2. CUDA out of memory**
```bash
# Solution: Reduce batch size in configs/stress.yaml
# Change: batch_size: 16 → batch_size: 8
```

**3. Training too slow**
```bash
# Solution: Use mixed precision (already enabled)
# Or reduce model size: temporal.dim: 128 → 64
```

**4. Poor performance**
```bash
# Check class imbalance
python -c "
from src.data.dataset import FabaDroughtDataset
from src.utils.config import load_config
cfg = load_config('configs/stress.yaml')
ds = FabaDroughtDataset(cfg)
labels = [ds[i]['stress_labels'] for i in range(len(ds))]
import torch
all_labels = torch.stack(labels)
print(f'Positive: {all_labels.sum().item()}')
print(f'Negative: {(all_labels == 0).sum().item()}')
print(f'Ratio: {all_labels.sum() / len(all_labels.flatten()):.3f}')
"
```

## Comparison with Original Model

| Aspect | Original (Multi-task) | Stress Detection |
|--------|----------------------|------------------|
| **Task** | DAG reg/cls + biomass + trajectory | Binary stress per timestep |
| **Fusion dim** | 256 | 128 |
| **Temporal dim** | 256 | 128 |
| **Fusion method** | Fixed concat | Modality Gating (learnable) |
| **Loss** | Multi-task weighted | Single BCE with pos_weight |
| **Evaluation** | DAG MAE, biomass R² | Onset detection (early days) |
| **Training time** | ~2 hours/fold | ~1 hour/fold |
| **Model size** | ~15M params | ~8M params |

## References

- Plan: `.sisyphus/plans/stress-detection-model.md`
- Learnings: `.sisyphus/notepads/faba-stress-refactor/learnings.md`
- Config: `configs/stress.yaml`
- Training: `scripts/train_stress.py`
- Evaluation: `scripts/evaluate_stress.py`

## Support

For issues or questions:
1. Check validation checklist above
2. Review learnings notepad for troubleshooting
3. Verify all files are committed: `git status`
4. Check SLURM logs: `cat logs/stress_*.out`
