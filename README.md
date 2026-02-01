# Temporal Multimodal Deep Learning for Faba Bean Drought Phenotyping

Framework leveraging vision foundation model representations and privileged information for pre-symptomatic drought detection.

## Paper
[Temporal Multimodal Deep Learning for Faba Bean Drought Phenotyping] — Nature Machine Intelligence (under review)
DOI: 10.1038/s42256-XXXX-XXXX (placeholder)

## Architecture
See `paper/figures/fig1_architecture.pdf` for the full model architecture diagram, illustrating the DINOv2 backbone, temporal transformer, and privileged information fusion.

## Project Structure
```
src/
  data/       # Dataset, collation, metadata
  model/      # Encoder, fusion, temporal transformer, heads, student
  training/   # Trainer, losses, CV
  analysis/   # Attention, fluorescence, ranking, embeddings
  baselines/  # Classical ML baselines
  utils/      # Config system
configs/      # YAML configs (default, ablation variants, distillation)
scripts/      # Training, evaluation, feature extraction, analysis, figures
paper/        # LaTeX manuscript and figures
```

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
### Feature Extraction
```bash
python scripts/extract_features.py --backbone dinov2 --image_dir data/images/ --output_dir features/
```

### Training
Train a single fold:
```bash
python scripts/train.py --config configs/default.yaml --fold 0
```
Train all folds:
```bash
python scripts/train.py --config configs/default.yaml --fold all
```

### Ablation and Distillation
Run ablation sweep:
```bash
bash scripts/slurm/ablation_sweep.sh
```
Knowledge distillation (RGB student):
```bash
python scripts/train_distill.py --config configs/distillation.yaml --fold 0
```

### Evaluation and Analysis
```bash
python scripts/evaluate.py --results_dir results/full_model/
python scripts/analyze_attention.py
python scripts/analyze_ranking.py
```

### Generate Figures
```bash
python scripts/generate_figures.py --results_dir results/ --output_dir paper/figures/
```

## Key Design Choices
- DINOv2-B/14 primary backbone (frozen, 768-dim representations)
- T=22 canonical timeline with per-plant segmentation masks
- 44-fold leave-one-genotype-out cross-validation
- Chlorophyll fluorescence as privileged information (LUPI paradigm)
- Teacher-student distillation for RGB-only field deployment

## CSC Mahti
SLURM scripts optimized for the CSC Mahti supercomputer are provided in `scripts/slurm/`.

## Citation
```bibtex
@article{faba_drought_2026,
  title={Temporal Multimodal Deep Learning for Faba Bean Drought Phenotyping},
  author={Lastname, Firstname and others},
  journal={Nature Machine Intelligence},
  year={2026},
  note={Under review}
}
```

## License
MIT License
