# Learnings: Feature Analysis Figures

## Conventions
- Color scheme: WHC-80 (control) = #3498DB (blue), WHC-30 (drought) = #E74C3C (red)
- Style: `sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)`
- Save pattern: Both PDF and PNG at 300 DPI
- Figure naming: `paper/figures/fig{N}_{description}.{pdf,png}`
- Script naming: `paper/figures/generate_fig{N}_{description}.py`

## Data Structures
- DINOv2 features: HDF5 keys = `{plant_id}_{round}_{view}`, values = float32[768]
- ROUND_TO_DAG mapping: {2: 4, 3: 5, ..., 23: 38} (from src/data/dataset.py:28-33)
- Plant metadata: `data/plant_metadata.csv` (264 plants, treatment labels)
- Fluorescence: `data/TimeCourse Datasets/FCQ_FabaDr_Auto.xlsx` (94 JIP-test params)

## Known Issues
- Plant ID whitespace: Excel has trailing spaces → use `.str.strip()`
- Missing timepoints: Not all plants have all 22 rounds → handle NaN
- PI_abs column name: May differ in Excel → auto-detect actual column names
- 94 params too many: Filter heatmap to |r| > 0.3 for readability


## [2026-02-09] Task 1: Feature Sync & Dependencies
- **Feature files synced from Mahti:**
  - `dinov2_features.h5`: 8.6 GB (264 plants, 23,250 embeddings)
  - `clip_features.h5`: 40 MB (264 plants)
  - `bioclip_features.h5`: 40 MB (264 plants)
- **HDF5 structure (3-level nested):**
  - Level 1: Plant ID (e.g., `24_FabaDr_001`)
  - Level 2: Round ID (e.g., `10`, `11`, `14`, ..., `23`)
  - Level 3: View ID (e.g., `side_000`, `side_120`, `side_240`, `top`)
  - DINOv2 also includes `*_patches` variants (8 views per round)
  - CLIP/BioCLIP: 4 views per round (no patches)
- **Embedding shape:** float32[768] (consistent across all models)
- **Dependencies installed:** umap-learn, scikit-learn, openpyxl, h5py
- **Verification:** All imports successful, HDF5 files readable

## [2026-02-09 11:55:36] Task 3: Fig 7 Fluorescence Heatmap
- Total fluorescence parameters: 94
- Parameters with |r| > 0.3: 80
- Top 3 correlated params: Fv_Lss (0.85), Fv/Fm_Lss (0.85), Fq_Lss (0.85)
- Correlation metric: Spearman rank

## [2026-02-09 11:56:48.178453] Task 4: Fig 8 Temporal Curves
- Classic params used: ['QY_max', 'Rfd_Lss', 'QY_Lss', 'qP_Lss']
- Data-driven params selected: ['Fv_Lss', 'Fq_Lss', 'Fv/Fm_Lss']
- Normalization: z-score (global)
- Saved: fig8_temporal_curves.pdf

## [2026-02-09 11:57:49] Task 2: Fig 6 Embeddings
- Total embeddings aggregated: 2905
- t-SNE perplexity: 30
- UMAP n_neighbors: 15
- Visual separation: [Pending visual inspection]

## [2026-02-09 11:58:08] Task 3: Fig 7 Fluorescence Heatmap
- Total fluorescence parameters: 94
- Parameters with |r| > 0.3: 80
- Top 3 correlated params: Fv_Lss (0.85), Fv/Fm_Lss (0.85), Fq_Lss (0.85)
- Correlation metric: Spearman rank

## [2026-02-09 11:58:19.732549] Task 4: Fig 8 Temporal Curves
- Classic params used: ['QY_max', 'Rfd_Lss', 'QY_Lss', 'qP_Lss']
- Data-driven params selected: ['Fv_Lss', 'Fq_Lss', 'Fv/Fm_Lss']
- Normalization: z-score (global)
- Saved: fig8_temporal_curves.pdf
