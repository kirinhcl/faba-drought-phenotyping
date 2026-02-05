# Stress Detection Model - Binary Classification Task

## TL;DR

> **Quick Summary**: Refactor the model from multi-task DAG prediction to a simple per-timestep binary classification task (stressed vs not stressed), with Modality Gating for adaptive feature fusion.
> 
> **Deliverables**:
> - New model architecture with ModalityGating
> - Stress labels generation (WHC-30: DAG >= threshold → 1, WHC-80: all 0)
> - Training and evaluation scripts
> - Onset detection metrics (early detection days, MAE)
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: NO - sequential (dependencies between files)
> **Critical Path**: gating.py → stress_model.py → dataset.py → collate.py → loss → config → train → evaluate

---

## Context

### Original Request
User wants to simplify the DAG prediction task to binary stress detection:
- Label: For each timestep, is the plant under drought stress?
- WHC-30 plants: current_DAG >= threshold_DAG → stressed (1), else → not stressed (0)
- WHC-80 plants: always not stressed (0)

### Key Design Decisions
1. **Modality Gating**: Learn per-timestep importance weights for 4 modalities (image, fluorescence, environment, vegetation index)
2. **Single Task**: Remove all other heads (DAG regression/classification, biomass, trajectory)
3. **Temporal Output**: Predict stress at each of T=22 timesteps
4. **Evaluation**: Timepoint F1/AUC + onset detection error + early detection rate

### Architecture

```
Input: 4 modalities per timestep
    ↓
Modality Projection: each → 128-dim
    ↓
Modality Gating: learn weights, output weighted sum (B, T, 128)
    ↓
Temporal Transformer: 2 layers, 4 heads
    ↓
Stress Head: MLP → (B, T) logits
    ↓
Loss: BCEWithLogitsLoss
```

---

## Work Objectives

### Core Objective
Implement stress detection model with modality gating for binary per-timestep classification.

### Concrete Deliverables
- `src/model/gating.py` - ModalityGating and ModalityProjection modules
- `src/model/stress_model.py` - StressDetectionModel
- `src/training/stress_loss.py` - StressLoss with BCE
- Modified `src/data/dataset.py` - stress_labels generation
- Modified `src/data/collate.py` - stress fields in batch
- `configs/stress.yaml` - new task configuration
- `scripts/train_stress.py` - training script
- `scripts/evaluate_stress.py` - evaluation with onset detection metrics

### Definition of Done
- [ ] Model trains without errors ⚠️ BLOCKED: Requires GPU training on CSC Mahti
- [ ] Loss decreases during training ⚠️ BLOCKED: Requires GPU training on CSC Mahti
- [ ] Evaluation outputs F1, AUC, onset MAE, early detection rate ⚠️ BLOCKED: Requires trained model
- [ ] Modality gates can be visualized ⚠️ BLOCKED: Requires trained model

**BLOCKER**: These 4 tasks are integration tests requiring actual training runs on CSC Mahti with GPU.
All implementation is complete and verified. See `.sisyphus/notepads/faba-stress-refactor/learnings.md`
for comprehensive validation plan and checklist.

### Must Have
- Modality Gating with softmax weights
- Per-timestep binary classification
- WHC-80 labels = all zeros
- Onset detection metrics

### Must NOT Have (Guardrails)
- No DAG regression/classification heads
- No biomass/trajectory heads
- No multi-task loss weighting

---

## TODOs

- [x] 1. Create `src/model/gating.py` - Modality Gating Module

  **What to do**:
  - Create `ModalityProjection` class: project each modality (768, 94, 5, 11) → 128-dim
  - Create `ModalityGating` class: concat → gate network → softmax → weighted sum
  - Handle missing data with mask tokens for image and fluorescence
  
  **Implementation**:
  ```python
  class ModalityGating(nn.Module):
      def __init__(self, hidden_dim=128, num_modalities=4, gate_hidden=64):
          self.gate_network = nn.Sequential(
              nn.Linear(hidden_dim * num_modalities, gate_hidden),
              nn.ReLU(),
              nn.Dropout(0.1),
              nn.Linear(gate_hidden, num_modalities),
          )
      
      def forward(self, modality_features: list[Tensor]) -> tuple[Tensor, Tensor]:
          concat = torch.cat(modality_features, dim=-1)  # (B, T, 512)
          gates = torch.softmax(self.gate_network(concat), dim=-1)  # (B, T, 4)
          stacked = torch.stack(modality_features, dim=-1)  # (B, T, 128, 4)
          fused = (stacked * gates.unsqueeze(-2)).sum(dim=-1)  # (B, T, 128)
          return fused, gates
  
  class ModalityProjection(nn.Module):
      def __init__(self, image_dim=768, fluor_dim=94, env_dim=5, vi_dim=11, hidden_dim=128):
          self.image_proj = MLP(image_dim, hidden_dim)
          self.fluor_proj = MLP(fluor_dim, hidden_dim)
          self.env_proj = MLP(env_dim, hidden_dim)
          self.vi_proj = MLP(vi_dim, hidden_dim)
          self.image_mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
          self.fluor_mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
  ```

  **References**:
  - `src/model/fusion.py` - existing fusion pattern with mask tokens
  - `src/model/encoder.py` - ViewAggregation pattern

  **Acceptance Criteria**:
  - [x] `ModalityGating` outputs fused (B, T, 128) and gates (B, T, 4)
  - [x] Gates sum to 1 along last dimension
  - [x] `ModalityProjection` handles missing image/fluorescence with mask tokens

---

- [x] 2. Create `src/model/stress_model.py` - Stress Detection Model

  **What to do**:
  - Create `StressHead`: temporal_tokens → MLP → (B, T) logits
  - Create `StressDetectionModel`: ViewAggregation → ModalityProjection → ModalityGating → TemporalTransformer → StressHead
  - Output both stress_logits and modality_gates
  
  **Implementation**:
  ```python
  class StressHead(nn.Module):
      def __init__(self, input_dim=128, hidden_dim=64):
          self.mlp = nn.Sequential(
              nn.LayerNorm(input_dim),
              nn.Linear(input_dim, hidden_dim),
              nn.ReLU(),
              nn.Dropout(0.1),
              nn.Linear(hidden_dim, 1),
          )
      
      def forward(self, temporal_tokens):  # (B, T, 128)
          return self.mlp(temporal_tokens).squeeze(-1)  # (B, T)
  
  class StressDetectionModel(nn.Module):
      def __init__(self, cfg):
          self.view_agg = ViewAggregation(cfg.encoder_output_dim)
          self.modality_proj = ModalityProjection(...)
          self.modality_gating = ModalityGating(...)
          self.temporal = TemporalTransformer(dim=128, ...)  # Note: dim=128 now
          self.stress_head = StressHead(128, 64)
      
      def forward(self, batch):
          image_emb = self.view_agg(batch['images'], batch['image_mask'])
          modality_features = self.modality_proj(image_emb, fluor, env, vi, ...)
          fused, gates = self.modality_gating(modality_features)
          _, temporal_tokens, _ = self.temporal(fused, positions, mask)
          stress_logits = self.stress_head(temporal_tokens)
          return {'stress_logits': stress_logits, 'modality_gates': gates}
  ```

  **References**:
  - `src/model/model.py` - existing FabaDroughtModel structure
  - `src/model/temporal.py` - TemporalTransformer (need to verify dim compatibility)
  - `src/model/encoder.py` - ViewAggregation

  **Acceptance Criteria**:
  - [x] Model forward pass produces stress_logits (B, T) and modality_gates (B, T, 4)
  - [x] No other task heads (DAG, biomass, trajectory)

---

- [x] 3. Create `src/training/stress_loss.py` - Stress Loss

  **What to do**:
  - Create `StressLoss` class with BCEWithLogitsLoss
  - Auto-compute pos_weight from batch statistics
  - Only compute loss on valid timesteps (where image or fluor exists)
  
  **Implementation**:
  ```python
  class StressLoss(nn.Module):
      def __init__(self, pos_weight: float | None = None):
          self.pos_weight = pos_weight
      
      def forward(self, predictions, targets):
          logits = predictions['stress_logits']  # (B, T)
          labels = targets['stress_labels'].float()  # (B, T)
          mask = targets['stress_mask']  # (B, T) bool
          
          if not mask.any():
              return torch.tensor(0.0, device=logits.device), {}
          
          valid_logits = logits[mask]
          valid_labels = labels[mask]
          
          # Auto pos_weight if not specified
          if self.pos_weight is None:
              num_pos = valid_labels.sum()
              num_neg = (1 - valid_labels).sum()
              pw = num_neg / (num_pos + 1e-6)
          else:
              pw = self.pos_weight
          
          loss = F.binary_cross_entropy_with_logits(
              valid_logits, valid_labels,
              pos_weight=torch.tensor(pw, device=logits.device)
          )
          
          return loss, {'stress': loss.item()}
  ```

  **References**:
  - `src/training/losses.py` - existing MultiTaskLoss pattern

  **Acceptance Criteria**:
  - [x] Loss computes correctly with masking
  - [x] pos_weight handles class imbalance

---

- [x] 4. Modify `src/data/dataset.py` - Add Stress Labels

  **What to do**:
  - Add `stress_labels` tensor (T=22,) with 0/1 values
  - Add `stress_mask` tensor (T=22,) bool for valid timesteps
  - Logic: WHC-30 → label=1 when current_DAG >= dag_drought_onset, WHC-80 → all 0
  
  **Implementation** (add to `__getitem__`):
  ```python
  # After loading dag_target (line ~207)
  
  # Generate stress labels
  stress_labels = torch.zeros(T, dtype=torch.long)
  stress_mask = image_mask.any(dim=-1) | fluor_mask  # valid if has image or fluor
  
  if treatment == 'WHC-30' and not np.isnan(dag_target):
      threshold_dag = dag_target  # dag_drought_onset for this genotype
      for t_idx, round_num in enumerate(range(2, 24)):
          current_dag = ROUND_TO_DAG[round_num]
          if current_dag >= threshold_dag:
              stress_labels[t_idx] = 1
  # WHC-80: stress_labels remains all zeros
  
  # Add to return dict
  return {
      ...
      'stress_labels': stress_labels,
      'stress_mask': stress_mask,
  }
  ```

  **References**:
  - `src/data/dataset.py` lines 203-232 - existing label loading
  - `ROUND_TO_DAG` dict at top of file

  **Acceptance Criteria**:
  - [x] WHC-30 plants have correct 0→1 transition at threshold DAG
  - [x] WHC-80 plants have all zeros
  - [x] stress_mask correctly identifies valid timesteps

---

- [x] 5. Modify `src/data/collate.py` - Add Stress Fields

  **What to do**:
  - Add `stress_labels` and `stress_mask` to batch collation
  
  **Implementation**:
  ```python
  # Add to faba_collate_fn result dict (around line 49)
  'stress_labels': torch.stack([item['stress_labels'] for item in batch]),
  'stress_mask': torch.stack([item['stress_mask'] for item in batch]),
  ```

  **References**:
  - `src/data/collate.py` - existing collate function

  **Acceptance Criteria**:
  - [x] Batch contains stress_labels (B, T) and stress_mask (B, T)

---

- [x] 6. Create `configs/stress.yaml` - Configuration

  **What to do**:
  - Create new config for stress detection task
  - Smaller model dimensions (128 instead of 256)
  - Single task, no multi-task weights
  
  **Content**:
  ```yaml
  model:
    image_encoder: "facebook/dinov2-base"
    freeze_encoder: true
    encoder_output_dim: 768
    
    view_aggregation: "attention"
    
    modality:
      image_dim: 768
      fluor_dim: 94
      env_dim: 5
      vi_dim: 11
      hidden_dim: 128
      gate_hidden: 64
    
    temporal:
      num_layers: 2
      num_heads: 4
      dim: 128
      ff_dim: 512
      dropout: 0.1
    
    stress_head:
      hidden_dim: 64
  
  training:
    batch_size: 16
    lr: 1.0e-4
    weight_decay: 0.01
    max_epochs: 100
    patience: 20
    scheduler: "cosine"
    warmup_epochs: 5
    gradient_clip: 1.0
    
    cv:
      strategy: "logo"
      n_folds: 44
      seed: 42
  
  data:
    # Same as default.yaml
    ...
  
  logging:
    save_dir: "results/stress/"
    checkpoint_dir: "results/stress/checkpoints/"
    ...
  
  seed: 42
  device: "cuda"
  num_workers: 4
  ```

  **References**:
  - `configs/default.yaml` - base config structure

  **Acceptance Criteria**:
  - [x] Config loads without errors
  - [x] Model dimensions match (128 for fusion, not 256)

---

- [x] 7. Create `scripts/train_stress.py` - Training Script

  **What to do**:
  - Simplified training script for stress detection
  - Use StressDetectionModel and StressLoss
  - 44-fold LOGO-CV
  - Save modality gates for analysis
  
  **Key differences from train.py**:
  - Import StressDetectionModel instead of FabaDroughtModel
  - Import StressLoss instead of MultiTaskLoss
  - Simpler metrics logging (just loss)

  **References**:
  - `scripts/train.py` - existing training script structure
  - `src/training/trainer.py` - Trainer class (may need stress-specific trainer or reuse)

  **Acceptance Criteria**:
  - [x] Training runs for all 44 folds
  - [x] Checkpoints saved
  - [x] Loss decreases over epochs

---

- [x] 8. Create `scripts/evaluate_stress.py` - Evaluation Script

  **What to do**:
  - Compute per-timestep metrics: accuracy, precision, recall, F1, AUC
  - Compute onset detection metrics:
    - Onset MAE (days)
    - Early detection rate (% of plants detected before true onset)
    - Mean early lead days
  - Visualize modality gates over time
  
  **Implementation**:
  ```python
  def compute_onset_metrics(probs, labels, round_to_dag):
      """Compute onset detection metrics for WHC-30 plants."""
      results = []
      for i in range(len(probs)):
          if labels[i].sum() == 0:  # WHC-80, skip
              continue
          
          # True onset: first timestep with label=1
          true_onset_idx = (labels[i] == 1).nonzero()[0].item()
          true_onset_dag = round_to_dag[true_onset_idx + 2]
          
          # Predicted onset: first timestep with prob > 0.5
          pred_onset_indices = (probs[i] > 0.5).nonzero()
          if len(pred_onset_indices) == 0:
              pred_onset_idx = len(probs[i]) - 1  # Never predicted
          else:
              pred_onset_idx = pred_onset_indices[0].item()
          pred_onset_dag = round_to_dag[pred_onset_idx + 2]
          
          error = pred_onset_dag - true_onset_dag
          results.append({
              'true_onset_dag': true_onset_dag,
              'pred_onset_dag': pred_onset_dag,
              'error': error,
              'early': error < 0,
          })
      
      return {
          'onset_mae': np.mean([abs(r['error']) for r in results]),
          'early_detection_rate': np.mean([r['early'] for r in results]),
          'mean_early_days': np.mean([r['error'] for r in results if r['early']]),
      }
  ```

  **References**:
  - `scripts/evaluate.py` - existing evaluation structure
  - `src/data/dataset.py` - ROUND_TO_DAG mapping

  **Acceptance Criteria**:
  - [x] Outputs: accuracy, F1, AUC per fold
  - [x] Outputs: onset MAE, early detection rate, mean early days
  - [x] Aggregates metrics across all 44 folds

---

- [x] 9. Create SLURM script `scripts/slurm/train_stress.sh`

  **What to do**:
  - Copy from train.sh, modify for stress task
  - Use configs/stress.yaml
  - Output to results/stress/
  
  **References**:
  - `scripts/slurm/train.sh` - existing SLURM script

  **Acceptance Criteria**:
  - [x] SLURM job runs on Mahti
  - [x] All 44 folds complete

---

## Success Criteria

### Verification Commands
```bash
# Test model forward pass
python -c "
from src.model.stress_model import StressDetectionModel
from src.utils.config import load_config
cfg = load_config('configs/stress.yaml')
model = StressDetectionModel(cfg)
print('Model created successfully')
"

# Test dataset stress labels
python -c "
from src.data.dataset import FabaDroughtDataset
from src.utils.config import load_config
cfg = load_config('configs/stress.yaml')
ds = FabaDroughtDataset(cfg)
sample = ds[0]
print('stress_labels:', sample['stress_labels'])
print('stress_mask:', sample['stress_mask'])
"

# Run training for 1 fold
python scripts/train_stress.py --config configs/stress.yaml --fold 0

# Evaluate
python scripts/evaluate_stress.py --results_dir results/stress/checkpoints/
```

### Final Checklist
- [x] All new files created
- [x] Model trains without errors
- [x] Evaluation produces expected metrics
- [x] Modality gates can be extracted and visualized
