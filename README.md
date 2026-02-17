# LUPIN: Learning from Physiology, Deploying on Vision

A temporal multimodal framework for pre-symptomatic drought phenotyping in faba bean.

LUPIN (Learning Using Privileged Information Network) bridges the "physiology--visibility gap" by integrating CLIP-based visual representations with chlorophyll fluorescence, environmental metadata, and vegetation indices during training, then distilling this knowledge into an RGB-only student model (LUPIN-D) for field deployment.

## Key Results

| Model | F1 | AUC | MAE (days) |
|-------|-----|------|------------|
| LUPIN (teacher, 4-modality) | 0.660 | 0.947 | 8.3 |
| LUPIN-D (student, RGB-only) | **0.667** | **0.964** | **7.4** |

- 44-fold leave-one-genotype-out cross-validation (LOGO-CV), 3 random seeds
- Pre-symptomatic detection: mean onset error -1.6 days (model detects before visible symptoms)

## Project Structure

```
src/
  data/         # Dataset, collation, metadata
  model/        # Encoder, fusion, temporal transformer, heads, student
  training/     # Trainer, losses, cross-validation
  analysis/     # Attention, fluorescence, ranking, embeddings
  baselines/    # Classical ML baselines (RF, SVM, XGBoost)
  utils/        # Config system
configs/
  default.yaml                # Full model config
  stress.yaml                 # Stress detection config
  distillation_stress.yaml    # LUPIN -> LUPIN-D distillation
  ablation/                   # Ablation variants (single-modality, leave-one-out, backbone, architecture)
scripts/
  train_stress.py             # Train LUPIN (stress detection)
  train_distill_stress.py     # Train LUPIN-D (knowledge distillation)
  evaluate_stress.py          # Evaluation pipeline
  extract_features.py         # CLIP feature extraction
  analyze_*.py                # Analysis (attention, fluorescence, presymptomatic, ranking)
  generate_fig2.py            # Figure 2 generation
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Feature Extraction

```bash
python scripts/extract_features.py --backbone clip --image_dir data/images/ --output_dir features/
```

### Training

```bash
# LUPIN (full multimodal)
python scripts/train_stress.py --config configs/stress.yaml --fold all

# LUPIN-D (distilled RGB-only student)
python scripts/train_distill_stress.py --config configs/distillation_stress.yaml --fold all
```

### Evaluation

```bash
python scripts/evaluate_stress.py --results_dir results/full_model/
```

### Analysis

```bash
python scripts/analyze_attention.py
python scripts/analyze_fluorescence.py
python scripts/analyze_presymptomatic.py
python scripts/analyze_ranking.py
```

## Design

- **Visual encoder**: CLIP-ViT-B/16 (frozen, 768-dim [CLS] token per side-view image)
- **Modality fusion**: Adaptive gating mechanism with learned per-timepoint weights
- **Temporal model**: 2-layer transformer encoder with continuous sinusoidal positional encodings (DAG-based)
- **Validation**: 44-fold LOGO-CV across 264 plants (44 genotypes x 2 treatments x 3 batches)
- **Distillation**: Teacher-student KD with alpha-annealing (0.7->0.3), T=2.0, beta=0.5

## Citation

```bibtex
@article{lu2026lupin,
  title={Learning from Physiology, Deploying on Vision: A Temporal Multimodal Framework for Pre-symptomatic Drought Phenotyping in Faba Bean},
  author={Lu, Chenghao and others},
  journal={Computers and Electronics in Agriculture},
  year={2026},
  note={Under review}
}
```

## License

MIT License
