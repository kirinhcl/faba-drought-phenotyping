# Temporal Multimodal Framework for Faba Bean Drought Phenotyping

## TL;DR

> **Quick Summary**: Build a temporal multimodal deep learning framework that leverages DINOv2 vision foundation model representations to predict drought onset, biomass, and stress trajectories in faba bean. The core scientific contribution is **quantifying the physiology→visibility lag**: the framework detects drought stress days before human expert observation, validated by independent chlorophyll fluorescence measurements as physiological ground truth. A teacher-student distillation paradigm demonstrates that fluorescence-guided training produces a deployable RGB-only early warning system. Complete code, experiments, and Nature Machine Intelligence paper draft in 30 days.
>
> **Deliverables**:
> - Complete PyTorch codebase with config-driven experiment system
> - Pre-extracted features from 3 vision foundation models (DINOv2, OpenAI CLIP, BioCLIP) for 11,626 plant images
> - Trained multimodal teacher model + RGB-only student model (distillation)
> - 44-fold leave-genotype-out cross-validation
> - 8 ablation study variants (including backbone comparison: DINOv2 vs CLIP vs BioCLIP) + classical ML baselines + distillation experiment
> - XAI temporal attention analysis with **fluorescence-anchored triangulation**
> - Genotype ranking evaluation against expert ground truth
> - Nature Machine Intelligence paper draft with all figures
>
> **Estimated Effort**: Large (30 days, full-time)
> **Parallel Execution**: YES - 5 waves
> **Critical Path**: TODO 1→2→3→4→5→6→8→8b→9→10→11→12→13

---

## Context

### Original Request
Build a multimodal temporal foundation model framework for faba bean drought phenotyping, fusing Direction A (multimodal temporal FM) with Direction B (pre-symptomatic drought signal discovery). Target Nature Machine Intelligence. Complete code + experiments + paper draft within 1 month.

### Interview Summary
**Key Discussions**:
- Initial CLIP idea assessed as too incremental — enhanced to A+B fusion strategy
- Direction A: DINOv2 backbone → Temporal Transformer → Multimodal Fusion → Multi-task heads
- Direction B: Pre-symptomatic drought signal discovery via temporal attention analysis
- Fusion: Use A's framework with B's "AI detects drought before humans" as core narrative
- Framing: Nature Machine Intelligence (methodology-first, biology as validation)

**Research Findings**:
- ZERO existing work combining foundation models + temporal drought prediction
- ZERO RGB + chlorophyll fluorescence DL fusion models for drought
- ZERO DL-based genotype ranking for drought tolerance
- ZERO temporal XAI for drought biology discovery
- Gold standard: Wu et al. 2021 Genome Biology (classical image analysis, 368 maize genotypes)
- Closest: AgriCLIP (COLING 2025, agriculture CLIP), BioCLIP (CVPR 2024) — neither temporal nor drought
- Classical baseline: Choudhury et al. 2023 (DTW-based drought prediction)

**Team & Resources**:
- Deep Learning expertise (strong)
- Finnish CSC supercomputer (Puhti V100, Mahti A100, LUMI MI250X)
- PyTorch + HuggingFace
- 1-month timeline (code + experiments + paper)

### Metis Review

**Critical Data Corrections**:
- The full experiment spans **22 imaging rounds** (rounds 2–23), but each plant was imaged on ~11 of those rounds. The specific 11-round schedule is **deterministic per replicate**: Rep-1 always gets rounds [2,3,6,7,10,11,14,17,18,19,23]; Rep-3 always gets [2,5,6,9,10,13,14,16,18,21,23]; Rep-2 varies by accession across 4 sub-schedules. Five plants are exceptions (4 with 12 rounds, 1 with 10). Model uses T=22 canonical timeline with per-plant image masks.
- Both side-view AND top-view images contain a **mix of RGB and RGBA** (~60% RGBA, ~40% RGB) — ALL images need consistent RGBA→RGB conversion
- Fluorescence data is sparse: ~5 measurements per plant, with only some rounds overlapping with a given plant's image rounds
- DAG ground truth: 44 values (per-genotype, not per-plant), only 13 unique values
- Temporal spacing is irregular (gaps of 1-3 days between consecutive rounds)

**Identified Gaps (all resolved)**:
- Multi-view fusion strategy: → attention pooling over 4 views per timestep
- DAG task formulation: → both regression + 3-class ordinal classification
- Control plant handling: → biomass task only; censored for drought onset
- DINOv2 variant: → ViT-B/14 (86M params, 768-dim, frozen)
- Baseline models: → XGBoost/RF on pre-computed tabular features + DINOv2+RF (no temporal)
- CV strategy: → 44-fold LOGO stratified by Early/Mid/Late
- Pre-symptomatic protocol: → progressive temporal truncation + attention peak vs DAG comparison
- "Foundation model" naming risk: → frame as "framework leveraging foundation model representations"

**Key Risks Acknowledged**:
- Small dataset (n=264) → freeze DINOv2, small temporal transformer, strong regularization
- 1-month timeline → strict phased schedule with hard cutoffs
- CSC queue times → design resumable jobs, batch experiments
- Statistical power → report 95% CI, frame as proof-of-concept framework

---

## Work Objectives

### Core Objective
Develop and validate a temporal multimodal deep learning framework that integrates vision foundation model representations with phenotypic time-series data for automated crop drought assessment. The framework's central scientific contribution is **exploiting the physiology→visibility lag** in drought stress: using chlorophyll fluorescence as privileged physiological information to train a multimodal teacher model, then distilling into a deployable RGB-only student that achieves early warning without expensive sensors. The three-way validation (model attention ↔ fluorescence change ↔ human annotation) provides uniquely defensible pre-symptomatic detection claims.

### Concrete Deliverables
1. `src/` — Complete PyTorch codebase (data pipeline, model, training, analysis)
2. `configs/` — YAML configuration files for all experiments
3. `scripts/` — Training, evaluation, and CSC SLURM scripts
4. `features/` — Pre-extracted DINOv2 features (HDF5)
5. `results/` — Cross-validation results, ablation results, analysis outputs
6. `paper/figures/` — All paper figures (architecture, results, attention maps, rankings)
7. `paper/main.tex` — Nature Machine Intelligence paper draft

### Definition of Done
- [ ] All 44-fold LOGO-CV completes with results logged
- [ ] 8 ablation variants produce comparable metrics
- [ ] Classical baselines (XGBoost, RF) produce benchmark metrics
- [ ] Teacher-student distillation: RGB-only student trained and evaluated
- [ ] **Triangulation validated**: attention peaks correlate with fluorescence change points AND precede human DAG
- [ ] Control plants show no pre-symptomatic attention peaks (negative control)
- [ ] Genotype ranking correlation (Spearman ρ) computed against expert DAG ranking
- [ ] All figures rendered and saved
- [ ] Paper draft complete in Nature MI format (≤4,500 words main text)

### Must Have
- Frozen vision foundation model as image encoder (DINOv2-B/14 primary; CLIP ViT-B/16 + BioCLIP as ablation backbones; no full fine-tuning)
- Multi-view attention pooling (3 side + 1 top per timestep)
- Temporal transformer with irregular positional encoding
- Multimodal fusion handling sparse fluorescence via mask tokens
- Multi-task learning (drought onset + biomass + stress trajectory)
- **Teacher-student distillation**: multimodal teacher → RGB-only student (fluorescence as privileged information)
- **Fluorescence change point detection**: per-plant fluorescence onset timing for triangulation
- **Three-way triangulation analysis**: model attention peak ↔ fluorescence change point ↔ human DAG
- 44-fold LOGO-CV with stratification
- Progressive temporal truncation for early detection evaluation
- Classical ML baselines on existing tabular features
- Config-driven experiments (no hardcoded hyperparameters)
- CSC SLURM job scripts with checkpoint resumption
- Wandb logging

### Must NOT Have (Guardrails)
- **G1**: MUST NOT fine-tune full DINOv2 backbone (264 plants is insufficient; use frozen or LoRA only)
- **G2**: MUST NOT add custom pre-training, contrastive learning, GAN, graph neural networks, or diffusion components
- **G3**: MUST NOT exceed 8 ablation variants without explicit approval (6 architecture + 2 backbone comparison)
- **G4**: MUST NOT hand-engineer image features (use DINOv2 representations; pre-computed tabular features from Excel are allowed as they come from the phenotyping platform)
- **G5**: MUST NOT implement custom image segmentation or background removal (images are pre-segmented)
- **G6**: MUST NOT call this a "foundation model" in the paper — frame as "framework leveraging vision foundation model representations"
- **G7**: MUST NOT design architecture requiring >40GB GPU RAM per training sample
- **G8**: MUST NOT add GWAS, genomics, field validation, deployment, or web interface
- **G9**: MUST NOT exceed 6 main figures + 4 supplementary in the paper
- **G10**: MUST NOT interpolate missing fluorescence data — use learnable mask tokens
- **G11**: MUST NOT try multiple temporal architectures (LSTM, Mamba, etc.) — Temporal Transformer only
- **G12**: MUST NOT add per-pixel XAI (GradCAM on every image) — temporal attention weights only for v1

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (new project from scratch)
- **User wants tests**: NO (research project, 1-month timeline)
- **Framework**: N/A
- **QA approach**: Automated verification via executable commands

### Verification Approach

All verification is automated and agent-executable. No manual/visual inspection required.

**By Deliverable Type:**

| Type | Verification Tool | Method |
|------|------------------|--------|
| Data pipeline | Bash (Python one-liners) | Assert tensor shapes, dataset sizes, metadata consistency |
| Model architecture | Bash (Python one-liners) | Forward pass sanity check, output shape verification |
| Training | Bash (Python scripts) | Overfit test (10 samples → near-zero loss), convergence check |
| Experiments | Bash (file existence + parsing) | Verify result files exist, parse metrics, check thresholds |
| XAI Analysis | Bash (file existence + parsing) | Attention map files exist, pre-symptomatic counts verified |
| Figures | Bash (file existence + size) | Figure files exist with minimum file size |
| Paper | Bash (word count + structure) | LaTeX compiles, word count within limits |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Days 1-3): Data Foundation
├── TODO 1: Project scaffolding + config system
└── TODO 2: Canonical metadata + data validation

Wave 2 (Days 3-5): Feature Extraction (after Wave 1)
└── TODO 3: DINOv2 feature pre-extraction (CSC GPU job)

Wave 3 (Days 5-10): Model + Training (after Wave 2)
├── TODO 4: PyTorch Dataset class
├── TODO 5: Model architecture (teacher + student)
├── TODO 6: Training pipeline + LOGO-CV
└── TODO 7: Classical ML baselines (parallel with 5,6)

Wave 4 (Days 10-20): Experiments + Distillation + Analysis (after Wave 3)
├── TODO 8: Run all experiments on CSC (ablation + baselines)
├── TODO 8b: Teacher-Student Distillation (after TODO 8)
├── TODO 9: XAI + Fluorescence Triangulation (after TODO 8)
└── TODO 10: Genotype ranking evaluation (parallel with 9, after TODO 8)

Wave 5 (Days 20-30): Figures + Paper (after Wave 4)
├── TODO 11: Generate all figures (after TODO 8b, 9, 10)
├── TODO 12: Paper draft (parallel with 11, start structure early)
└── TODO 13: Supplementary materials + final review (after 11, 12)
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2,3,4,5,6,7 | 2 (partially) |
| 2 | 1 (config) | 3,4 | 1 (partially) |
| 3 | 2 (metadata) | 4,5,6 | None |
| 4 | 2,3 (metadata, features) | 6,8 | 5 |
| 5 | 1 (config) | 6,8,8b | 4,7 |
| 6 | 4,5 | 8 | 7 |
| 7 | 2 (metadata) | 8 | 4,5,6 |
| 8 | 6,7 | 8b,9,10 | None |
| **8b** | **8 (trained teacher)** | **11** | **9, 10** |
| 9 | 8 | 11 | 8b, 10 |
| 10 | 8 | 11 | 8b, 9 |
| 11 | 8b,9,10 | 13 | 12 |
| 12 | 8 (results) | 13 | 11 |
| 13 | 11,12 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Dispatch |
|------|-------|---------------------|
| 1 | 1,2 | `category="unspecified-high"` — scaffolding + data wrangling |
| 2 | 3 | `category="unspecified-high"` — GPU feature extraction pipeline |
| 3 | 4,5,6,7 | `category="ultrabrain"` for model (5), `category="unspecified-high"` for pipeline (4,6,7) |
| 4 | 8,8b,9,10 | `category="unspecified-high"` for experiments (8,8b), analysis (9,10) |
| 5 | 11,12,13 | `category="visual-engineering"` for figures (11), `category="writing"` for paper (12,13) |

---

## TODOs

### TODO 1: Project Scaffolding + Configuration System

- [ ] 1. Project Scaffolding + Configuration System

  **What to do**:
  - Create project directory structure:
    ```
    src/
    ├── data/           # dataset, metadata, transforms
    ├── model/          # architecture components
    ├── training/       # trainer, losses, CV
    ├── analysis/       # XAI, ranking, pre-symptomatic
    ├── baselines/      # classical ML
    └── utils/          # config, logging, constants
    configs/
    ├── default.yaml    # base config
    ├── overfit_test.yaml
    └── ablation/       # per-variant configs
    scripts/
    ├── extract_features.py
    ├── train.py
    ├── evaluate.py
    ├── run_baselines.py
    ├── analyze_attention.py
    ├── analyze_ranking.py
    └── slurm/          # CSC job scripts
    paper/
    ├── figures/
    └── main.tex
    results/
    features/
    ```
  - Implement configuration system using **OmegaConf** (not Hydra — simpler, sufficient for this project). Use `OmegaConf.load()` + `OmegaConf.merge()` for config inheritance.
  - Config must cover: model hyperparams, training hyperparams, data paths, CV settings, logging
  - Create `requirements.txt` or `pyproject.toml` with all dependencies:
    - torch, torchvision, transformers (HuggingFace for DINOv2, CLIP, BioCLIP), open_clip_torch (for OpenAI CLIP)
    - pandas, numpy, openpyxl (Excel reading)
    - scikit-learn (baselines + metrics)
    - xgboost
    - h5py (feature storage)
    - omegaconf or hydra-core (config)
    - wandb (experiment tracking)
    - matplotlib, seaborn (figures)
    - scipy (statistics)

  **Must NOT do**:
  - Do NOT hardcode any paths or hyperparameters in source files
  - Do NOT install unnecessary packages (no tensorflow, no detectron2, no segmentation libraries)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Scaffolding is straightforward directory creation + boilerplate config code
  - **Skills**: [`template-skill`]
    - `template-skill`: Standard project setup pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES (with TODO 2 partially — metadata doesn't need full scaffolding)
  - **Parallel Group**: Wave 1 (with TODO 2)
  - **Blocks**: TODO 3, 4, 5, 6, 7 (all need config system)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Data References** (to configure paths):
  - `data/00-Misc/` — Raw metadata Excel files
  - `data/TimeCourse Datasets/` — Time-series CSV/Excel files
  - `data/EndPoint Datasets/` — Endpoint measurements
  - `data/SinglePoint Datasets/` — Drought impact timing
  - `data/img/side_view/` — 8,718 side-view images (mixed RGB/RGBA)
  - `data/img/top_view/` — 2,908 top-view images (mixed RGB/RGBA)

  **Config Design Reference** (follow this pattern):
  - Default config should mirror the model architecture:
    ```yaml
    model:
      image_encoder: "facebook/dinov2-base"  # HuggingFace model ID. Alternatives: "openai/clip-vit-base-patch16", "imageomics/bioclip"
      freeze_encoder: true
      encoder_output_dim: 768  # All three ViT-B variants output 768-dim; kept as config for safety
      view_aggregation: "attention"  # attention | mean | concat
      temporal:
        num_layers: 2
        num_heads: 4
        dim: 256
        ff_dim: 1024
        dropout: 0.1
      fusion:
        image_dim: 768  # DINOv2-B output dim
        fluor_dim: 93   # ~93-96 fluorescence params (columns 16+ in FCQ_FabaDr_Auto.xlsx; exact count determined at load time after dropping non-numeric metadata)
        env_dim: 5   # li1_Buffer_uE, t1_Buffer_C, rh1_Buffer_%, t2_Tunnel_C, rh2_Tunnel_%
        water_dim: 5  # Water Added, WHC.Bf, WHC.Af, Water Loss, Water Loss per Hours
        hidden_dim: 128
        fused_dim: 256
      heads:
        dag_regression: true
        dag_classification: true  # 3-class: Early/Mid/Late
        biomass_regression: true  # FW + DW
        trajectory: true
    training:
      batch_size: 16
      lr: 1e-4
      weight_decay: 0.01
      max_epochs: 100
      patience: 15
      loss_weights:
        dag_reg: 1.0
        dag_cls: 0.5
        biomass: 1.0
        trajectory: 0.5
      cv:
        strategy: "logo"  # leave-genotype-out
        n_folds: 44
        stratify_by: "drought_category"  # Early/Mid/Late
    data:
      image_size: 224
      num_timepoints: 22  # canonical T=22 (rounds 2-23); per-plant masks indicate which rounds exist
      num_views: 4  # 3 side + 1 top
      feature_dir: "features/"
      normalize: true
    ```

  **Acceptance Criteria**:
  ```bash
  # Verify directory structure exists
  ls src/data src/model src/training src/analysis src/baselines src/utils configs scripts paper/figures results features
  # Assert: all directories exist (exit code 0)

  # Verify config loads
  python3 -c "
  from src.utils.config import load_config
  cfg = load_config('configs/default.yaml')
  assert cfg.model.image_encoder == 'facebook/dinov2-base'
  assert cfg.training.cv.n_folds == 44
  print('Config system PASSED')
  "
  # Assert: output contains "Config system PASSED"

  # Verify dependencies install
  pip install -r requirements.txt --dry-run 2>&1 | tail -1
  # Assert: no errors
  ```

  **Commit**: YES
  - Message: `feat(scaffold): project structure and config system`
  - Files: `src/`, `configs/`, `scripts/`, `paper/`, `requirements.txt`

---

### TODO 2: Canonical Plant Metadata + Data Validation

- [ ] 2. Canonical Plant Metadata + Data Validation

  **What to do**:
  - Create `src/data/metadata.py` that reconciles ALL naming inconsistencies across data files:
    - WHC-70% (FabaDr_Obs.xlsx) ↔ WHC-80% (EndPoint, image dirs) → canonical: `WHC-80`
    - WHC-40% (FabaDr_Obs.xlsx) ↔ WHC-30% (EndPoint, image dirs) → canonical: `WHC-30`
    - Accession name variants: `Mélodie/2` ↔ `Mélodie_2`, spaces, special chars → canonical slug
    - **MUST** apply `unicodedata.normalize('NFC', name)` to ALL string comparisons — macOS HFS+ uses NFD decomposition, so `é` may be stored as `e` + combining accent. Without NFC normalization, string matching between Excel data and filesystem paths will silently fail for `Mélodie` and any other accented names.
  - Generate `data/plant_metadata.csv` with columns:
    - `plant_id`: canonical unique ID (from image directory names)
    - `accession`: genotype name (canonical)
    - `accession_slug`: filesystem-safe version
    - `treatment`: `WHC-80` or `WHC-30`
    - `replicate`: 1, 2, or 3
    - `dag_drought_onset`: integer (WHC-30 only, null for WHC-80)
    - `drought_category`: Early/Mid/Late (WHC-30 only, null for WHC-80)
    - `fw_g`: fresh weight at endpoint (grams)
    - `dw_g`: dry weight at endpoint (grams)
    - `image_side_dir`: path to side-view images
    - `image_top_dir`: path to top-view images
    - `available_timepoints`: JSON string of round numbers with images (e.g., `"[2,3,6,7,10,11,14,17,18,19,23]"`). Parse with `json.loads()` in downstream code.
    - `num_images_side`: count
    - `num_images_top`: count
    - `is_dead`: boolean (from outlier annotations)
  - Create `data/timepoint_metadata.csv` — **canonical T=22 timeline** mapping:
    - `round`: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    - `dag`: days after germination (from `DateIndex.xlsx`)
    - `das`: days after sowing
    - `date`: calendar date
    - `has_images`: boolean (True for all 22 — every round has images for SOME plants)
    - `has_fluorescence`: boolean (True for 15 of 22 rounds: [3,4,5,7,8,9,11,12,13,15,16,17,19,20,21])
    - `rep1_has_images`: boolean (True for rounds [2,3,6,7,10,11,14,17,18,19,23])
    - `rep2_has_images`: boolean (True for Rep-2's primary schedule rounds)
    - `rep3_has_images`: boolean (True for rounds [2,5,6,9,10,13,14,16,18,21,23])
  - Validate data completeness:
    - Flag plants with <8 image timepoints
    - Flag plants marked as "died"
    - Flag genotypes where all 3 drought reps are dead
    - Verify all 264 plants have endpoint FW/DW
    - Map fluorescence measurement dates to image timepoints
    - **Reconciliation check**: Side-view image count (8,718 PNGs) vs FabaDr_Obs.xlsx RGB1 rows (8,717). Log the 1-file discrepancy with exact plant/round details. Likely cause: one extra image file not in observation log. Non-blocking — just document it.

  **Must NOT do**:
  - Do NOT modify original data files — only create new derived files
  - Do NOT impute or interpolate any missing data at this stage
  - Do NOT drop any plants — flag them and let downstream code decide

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex data wrangling requiring careful reconciliation across 10+ source files with naming inconsistencies
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with TODO 1 partially)
  - **Parallel Group**: Wave 1 (with TODO 1)
  - **Blocks**: TODO 3, 4, 7
  - **Blocked By**: TODO 1 (needs config system for paths)

  **References**:

  **Data Source References** (ALL files that contribute to metadata — EXACT filenames verified):
  - `data/00-Misc/FabaDr_Obs.xlsx` — Master observation log. Contains Plant IDs, accession names, treatment labels (WHC-70%/40%). Use as primary plant list. NOTE: treatment labels here differ from everywhere else.
  - `data/00-Misc/DateIndex.xlsx` — Date ↔ DAS ↔ DAG mapping (70 rows, columns: `Measuring Date`, `DAS`, `DAG`). NOTE: does NOT have Round Order — use watering/fluorescence data for round mapping.
  - `data/00-Misc/EndPoint_Raw_FW&DW.xlsx` — Endpoint fresh/dry weights. Treatment labels here are WHC-80%/30%. Has 264 rows (all plants).
  - `data/SinglePoint Datasets/Drought_Impact(DAG).xlsx` — Drought onset DAG per genotype. 44 rows. Columns: `Accession Name`, `Drought Impact (DAG)`, `Rank`, `Stress Impact`. Categories: Early/Mid/Late stored in `Stress Impact` column.
  - `data/EndPoint Datasets/EndPoint_CorrelationData-WithoutOutliers.xlsx` — 29 columns of aggregated features with outliers removed.
  - `data/TimeCourse Datasets/FCQ_FabaDr_Auto.xlsx` — Fluorescence wide format. 1,320 rows (264 plants × 5 measurement dates). 111 total columns: ~15 metadata columns (`Measuring Date`, `Round Order`, `Plant ID`, `Accession Name`, `Treatment`, `Replicate`, `DAS`, `DAG`, etc.) + ~96 fluorescence parameter columns. Exact fluorescence column count to be determined at load time by selecting only numeric non-metadata columns. 15 unique measurement dates but only 5 per plant.
  - `data/TimeCourse Datasets/FCQ_FabaDr_Auto_Reshape.xlsx` — Fluorescence reshaped/long format. 13,200 rows. 14 summary fluorescence parameters (Fm, Fo, Fp, Fq, Ft, Fv, Fv/Fm, NPQ, QY, Rfd, Size, qL, qN, qP).
  - `data/DATA_DESCRIPTION.md` — Our comprehensive data documentation. READ THIS FIRST for full schema details.

  **Image Directory Structure** (EXACT layout verified):
  ```
  data/img/side_view/
  └── {Acc-NN - AccessionName}/          # e.g. "Acc-01 - Aurora_2", "Acc-02 - Mélodie_2"
      └── {WHC-30%|WHC-80%}/             # treatment directory
          └── {Rep-N - PlantID}/         # e.g. "Rep-1 - 24_FabaDr_004"
              └── {000|120|240}/          # angle directory (degrees)
                  └── 124-{round}-{PlantID}-RGB1-{angle}-FishEyeMasked.png
                  # e.g. "124-10-24_FabaDr_004-RGB1-000-FishEyeMasked.png"

  data/img/top_view/
  └── {Acc-NN - AccessionName}/
      └── {WHC-30%|WHC-80%}/
          └── {Rep-N - PlantID}/
              └── 124-{round}-{PlantID}-RGB2-FishEyeMasked.png
              # e.g. "124-10-24_FabaDr_004-RGB2-FishEyeMasked.png"
  ```

  **Parsing Rules** (concrete examples):
  - From directory path `Acc-01 - Aurora_2/WHC-30%/Rep-1 - 24_FabaDr_004/`:
    - `accession_num`: `Acc-01` (split on ` - `, take first part)
    - `accession`: `Aurora_2` (split on ` - `, take second part; canonical: `Aurora/2`)
    - `treatment`: `WHC-30%` → canonical `WHC-30` (strip `%`)
    - `replicate`: `1` (from `Rep-1`)
    - `plant_id`: `24_FabaDr_004` (from `Rep-1 - 24_FabaDr_004`, split on ` - `)
  - From filename `124-10-24_FabaDr_004-RGB1-000-FishEyeMasked.png`:
    - `round`: `10` (split on `-`, take index 1)
    - `plant_id`: `24_FabaDr_004` (split on `-`, take index 2)
    - `camera`: `RGB1` (side) or `RGB2` (top) (split on `-`, take index 3)
    - `angle`: `000` (split on `-`, take index 4; top-view has no angle)
  - **Unicode warning**: `Mélodie_2` contains accented `é` — macOS may store as decomposed NFC/NFD. Use `unicodedata.normalize('NFC', name)` when comparing strings.

  **22 Canonical Image Rounds** (union across all plants): rounds 2-23 (every integer). Per-plant schedules differ by replicate:
  - **Rep-1** (88 plants): [2, 3, 6, 7, 10, 11, 14, 17, 18, 19, 23] — 11 rounds
  - **Rep-3** (88 plants): [2, 5, 6, 9, 10, 13, 14, 16, 18, 21, 23] — 11 rounds
  - **Rep-2** (88 plants): 4 sub-schedules by accession:
    - 58 plants: [2, 4, 6, 8, 10, 12, 14, 15, 18, 20, 22] — 11 rounds
    - 25 plants: [2, 4, 6, 8, 10, 12, 14, 15, 18, 20, 23] — 11 rounds
    - 4 plants (Acc-02, Acc-22, Acc-38): [2, 4, 6, 8, 10, 12, 14, 15, 18, 20, 22, 23] — 12 rounds
    - 1 plant (Acc-34 side-view): [2, 4, 6, 8, 12, 14, 15, 18, 20, 23] — 10 rounds (missing round 10 in side-view only)

  **WHY Each Reference Matters**:
  - `FabaDr_Obs.xlsx`: Primary plant inventory — establishes the 264-plant universe
  - `DateIndex.xlsx`: Maps dates to DAG — needed for temporal positional encoding
  - `EndPoint_Raw_FW&DW.xlsx`: Ground truth biomass — regression target for multi-task head
  - `Drought_Impact(DAG).xlsx`: Ground truth drought onset — primary prediction target; columns are `Accession Name`, `Drought Impact (DAG)`, `Stress Impact`
  - Image directories: Establish which images exist per plant — determines actual T per plant
  - `FCQ_FabaDr_Auto.xlsx`: ~96 fluorescence params in wide format (columns 16+ of 111 total) — primary fluorescence input for the model

  **Acceptance Criteria**:
  ```bash
  python3 -c "
  import pandas as pd
  # Verify plant metadata
  pm = pd.read_csv('data/plant_metadata.csv')
  assert len(pm) == 264, f'Expected 264 plants, got {len(pm)}'
  assert set(pm['treatment'].unique()) == {'WHC-80', 'WHC-30'}, f'Wrong treatments: {pm.treatment.unique()}'
  assert pm[pm['treatment']=='WHC-30']['dag_drought_onset'].notna().all(), 'Missing DAG for drought plants'
  assert pm[pm['treatment']=='WHC-80']['dag_drought_onset'].isna().all(), 'Control plants should have null DAG'
  assert pm['fw_g'].notna().all(), 'Missing endpoint FW'
  assert pm['dw_g'].notna().all(), 'Missing endpoint DW'
  assert len(pm['accession'].unique()) == 44, f'Expected 44 accessions, got {len(pm.accession.unique())}'
  print(f'Plant metadata PASSED: {len(pm)} plants, {len(pm.accession.unique())} accessions')

  # Verify timepoint metadata
  tm = pd.read_csv('data/timepoint_metadata.csv')
  assert len(tm) == 22, f'Expected 22 canonical rounds (2-23), got {len(tm)}'
  assert tm['has_images'].sum() == 22, f'Expected all 22 rounds to have images (across some plants), got {tm.has_images.sum()}'
  print(f'Timepoint metadata PASSED: {len(tm)} rounds (canonical T=22 timeline)')

  # Verify data completeness flags
  n_dead = pm['is_dead'].sum()
  n_incomplete = (pm['num_images_side'] < 8*3).sum()  # <8 timepoints × 3 angles
  print(f'Flagged: {n_dead} dead plants, {n_incomplete} incomplete imaging')
  "
  # Assert: all assertions pass, no errors
  ```

  **Commit**: YES
  - Message: `feat(data): canonical plant metadata and data validation`
  - Files: `src/data/metadata.py`, `data/plant_metadata.csv`, `data/timepoint_metadata.csv`

---

### TODO 3: DINOv2 Feature Pre-Extraction Pipeline

- [ ] 3. DINOv2 Feature Pre-Extraction Pipeline

  **What to do**:
  - Create `scripts/extract_features.py` — **supports 3 vision foundation model backbones**:
    - **CLI**: `python3 scripts/extract_features.py --backbone {dinov2,clip,bioclip} --output features/{backbone}_features.h5`
    - Backbone specifications (all output 768-dim CLS token):
      1. **DINOv2-B/14** (`facebook/dinov2-base`): Self-supervised ViT-B/14, 86M params. Normalization: ImageNet mean/std. Input: 224×224.
      2. **OpenAI CLIP ViT-B/16** (`openai/clip-vit-base-patch16`): Language-supervised ViT-B/16, 86M params. Normalization: CLIP-specific (mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711]). Input: 224×224. Use `open_clip` or `transformers` CLIPModel. Extract **visual encoder CLS** only (no text encoder).
      3. **BioCLIP** (`imageomics/bioclip`): Biology-specialized CLIP fine-tuned on TreeOfLife-10M. ViT-B/16 base, 86M params. Normalization: same as CLIP. Input: 224×224. Load via `open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')`.
    - Process ALL 11,626 images (8,718 side-view + 2,908 top-view) **per backbone**
    - For ALL images (both cameras have mixed RGB/RGBA): if 4-channel, composite onto black background using alpha → RGB 3-channel; if 3-channel, use as-is
    - Resize all images to 224×224
    - Apply backbone-specific normalization
    - Extract CLS token embedding (768-dim) for each image
    - Save to HDF5: `features/{backbone}_features.h5` (3 separate files)
      - Key structure: `/{plant_id}/{round}/side_000` → (768,) float32 (angle 000°)
      - Key structure: `/{plant_id}/{round}/side_120` → (768,) float32 (angle 120°)
      - Key structure: `/{plant_id}/{round}/side_240` → (768,) float32 (angle 240°)
      - Key structure: `/{plant_id}/{round}/top` → (768,) float32
      - View order convention (V=4): [side_000, side_120, side_240, top] → indices [0,1,2,3]
    - Also save patch tokens for DINOv2 only (for future spatial attention): `/{plant_id}/{round}/side_000_patches` → (N_patches, 768)
  - Create `scripts/slurm/extract_features.sh` for CSC:
    - Request 1 GPU (V100 sufficient for inference)
    - Estimated time: ~30 min per backbone × 3 backbones = ~90 min total
    - Batch size 64 for extraction
    - Run 3 sequential jobs (or 3 parallel if GPU quota allows)
  - Run **comparative** feature sanity check across all 3 backbones:
    - Extract features for 10 random plants (5 WHC-80, 5 WHC-30) with each backbone
    - Compute silhouette score per backbone: `sklearn.metrics.silhouette_score(features, treatment_labels)`
    - Save per-backbone sanity results to `features/sanity_check.json`:
      ```json
      {"dinov2": {"silhouette": 0.XX}, "clip": {"silhouette": 0.XX}, "bioclip": {"silhouette": 0.XX}}
      ```
    - This check is **informational, not blocking** — poor separation in raw features is expected since temporal modeling adds the signal
    - Interesting comparison: self-supervised (DINOv2) vs language-supervised (CLIP) vs domain-specific (BioCLIP) — may reveal which pre-training strategy best captures plant phenotype variation

  **Must NOT do**:
  - Do NOT fine-tune DINOv2 during extraction (inference only)
  - Do NOT extract features from augmented images in v1 (original images only)
  - Do NOT store raw resized images (only features) — save disk space
  - Do NOT use Large or Giant variants of any backbone (ViT-B is sufficient and keeps comparison fair across backbones)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires careful image I/O handling (RGBA conversion, HDF5 writing) and CSC SLURM scripting
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Wave 1)
  - **Blocks**: TODO 4 (dataset needs features), TODO 5 (model needs feature dimensions confirmed)
  - **Blocked By**: TODO 2 (needs plant_metadata.csv for image paths)

  **References**:

  **Image Data References** (see TODO 2 for EXACT directory layout):
  - `data/img/side_view/` — 8,718 images (mixed RGB/RGBA, ~60% RGBA). Layout: `{Acc-NN - Name}/{WHC-30%|WHC-80%}/{Rep-N - PlantID}/{000|120|240}/124-{round}-{PlantID}-RGB1-{angle}-FishEyeMasked.png`. 3 angles per timepoint (000°, 120°, 240°).
  - `data/img/top_view/` — 2,908 images (mixed RGB/RGBA, ~56% RGBA). Layout: `{Acc-NN - Name}/{WHC-30%|WHC-80%}/{Rep-N - PlantID}/124-{round}-{PlantID}-RGB2-FishEyeMasked.png`. 1 image per timepoint.
  - `data/plant_metadata.csv` (from TODO 2) — Maps plant_id → image directory paths and available timepoints

  **Model References** (3 backbones — all ViT-B, all output 768-dim):
  - **DINOv2-B/14**: HuggingFace `facebook/dinov2-base` — Self-supervised ViT-B/14, 86M params, 768-dim CLS + 256 patch tokens. Normalization: ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
  - **OpenAI CLIP ViT-B/16**: HuggingFace `openai/clip-vit-base-patch16` or `open_clip` — Language-supervised ViT-B/16, 86M params (visual encoder only), 768-dim CLS. Normalization: CLIP (mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711]).
  - **BioCLIP**: `open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')` — Biology-specialized CLIP fine-tuned on TreeOfLife-10M dataset (CVPR 2024). ViT-B/16 base, 768-dim CLS. Same normalization as CLIP.

  **Image Channel Handling** (CRITICAL — applies to BOTH cameras):
  - **Both side-view AND top-view contain a mix of RGB (3-channel) and RGBA (4-channel) images** (~60% RGBA, ~40% RGB across both cameras)
  - For RGBA images: alpha channel encodes plant mask (background=transparent). Composite: `rgb = alpha * foreground + (1-alpha) * background_color`. Use black background (0,0,0).
  - For RGB images: use as-is (already 3-channel)
  - **Implementation**: Check channel count per image; apply alpha compositing only if 4 channels. Always verify output is 3-channel before feeding to DINOv2.
  - Add a verification step: after conversion, assert all images are exactly (H, W, 3) RGB

  **CSC SLURM Pattern**:
  - Puhti GPU partition: `#SBATCH --partition=gpu`, `#SBATCH --gres=gpu:v100:1`
  - Module loads: `module load pytorch/2.1` (check CSC docs for exact version)
  - All scripts must support `--checkpoint_dir` and `--resume_from` for queue resilience

  **Acceptance Criteria**:
  ```bash
  # Verify ALL 3 backbone feature files exist and have correct structure
  python3 -c "
  import h5py
  import numpy as np

  for backbone in ['dinov2', 'clip', 'bioclip']:
      path = f'features/{backbone}_features.h5'
      f = h5py.File(path, 'r')
      plant_ids = list(f.keys())
      assert len(plant_ids) >= 250, f'{backbone}: Expected ≥250 plants, got {len(plant_ids)}'

      # Check first plant's structure
      p = f[plant_ids[0]]
      rounds = list(p.keys())

      # Verify feature dimensions (768-dim for all 3 backbones)
      sample = p[rounds[0]]
      for key in ['side_000', 'side_120', 'side_240', 'top']:
          if key in sample:
              assert sample[key].shape == (768,), f'{backbone}/{key} wrong shape: {sample[key].shape}'

      # Verify total feature count
      total = sum(
          sum(1 for v in f[pid][r].keys() if not v.endswith('_patches'))
          for pid in f.keys() for r in f[pid].keys()
      )
      assert total >= 11000, f'{backbone}: Expected ≥11000 features, got {total}'
      print(f'{backbone}: PASSED ({len(plant_ids)} plants, {total} features)')
      f.close()

  # Verify sanity check exists
  import json
  with open('features/sanity_check.json') as sc:
      sanity = json.load(sc)
  for b in ['dinov2', 'clip', 'bioclip']:
      assert b in sanity, f'Missing sanity check for {b}'
      print(f'{b} silhouette: {sanity[b][\"silhouette\"]:.4f}')
  print('All 3 backbones PASSED')
  "
  ```

  **Commit**: YES
  - Message: `feat(features): multi-backbone feature extraction (DINOv2, CLIP, BioCLIP)`
  - Files: `scripts/extract_features.py`, `scripts/slurm/extract_features.sh`, `src/data/transforms.py`

---

### TODO 4: PyTorch Dataset with Multi-View Temporal Batching

- [ ] 4. PyTorch Dataset with Multi-View Temporal Batching

  **What to do**:
  - Create `src/data/dataset.py` — `FabaDroughtDataset(torch.utils.data.Dataset)`:
    - Each sample = 1 plant's complete temporal sequence placed on the **canonical T=22 timeline**
    - Loads pre-extracted DINOv2 features from HDF5
    - **Canonical timeline**: T=22 positions for rounds 2-23. Each position has a fixed DAG value (from `DateIndex.xlsx`). Each plant only has images at ~11 of these 22 positions; the rest are masked.
    - Returns per sample:
      ```python
      {
          'plant_id': str,
          'accession': str,
          'treatment': str,  # 'WHC-80' or 'WHC-30'
          'images': Tensor(T, V, D),  # T=22 canonical positions, V=4 views, D=768 feature dim
          'image_mask': Tensor(T, V),  # bool, True where image exists (~11 True per plant, ~11 False)
          'fluorescence': Tensor(T, F),  # T=22, F=fluor_dim (~93-96, set dynamically at load time)
          'fluor_mask': Tensor(T),  # bool, True where fluorescence measured (~5 True per plant)
          'environment': Tensor(T, E),  # T=22, E=5 env params (global, same for all plants)
          'watering': Tensor(T, W),  # T=22, W=5 watering params (per-plant)
          'temporal_positions': Tensor(T),  # DAG values for all 22 positions (fixed, from DateIndex.xlsx)
          'dag_target': float or NaN,  # drought onset DAG (NaN for controls)
          'dag_category': int,  # 0=Early, 1=Mid, 2=Late, -1=control
          'fw_target': float,  # endpoint fresh weight (grams)
          'dw_target': float,  # endpoint dry weight (grams)
      }
      ```
    - **Round schedule per replicate** (deterministic):
      - Rep-1 (88 plants): rounds [2,3,6,7,10,11,14,17,18,19,23]
      - Rep-3 (88 plants): rounds [2,5,6,9,10,13,14,16,18,21,23]
      - Rep-2 (88 plants): varies by accession — 4 sub-schedules (mostly [2,4,6,8,10,12,14,15,18,20,22])
      - 4 plants have 12 rounds, 1 plant has 10 rounds (special cases)
      - Plant 24_FabaDr_203 (Acc-34): side-view missing round 10 (10 rounds side, 11 top)
    - Handle missing data:
      - Missing images at a timepoint → zero vector + mask=False
      - Missing fluorescence → zero vector + fluor_mask=False
      - Environment/watering: should be complete for all timepoints (from TC datasets)
    - Tabular data loading:
      - Fluorescence: from `data/TimeCourse Datasets/FCQ_FabaDr_Auto.xlsx` — wide format, ~96 fluorescence params per measurement (columns 16+). Join on `Plant ID` + `Round Order`. Each plant has 5 measurements. Map `Round Order` to canonical T=22 position. Set `fluor_mask[t]=True` for all rounds where measurement exists.
      - Environment: from `data/TimeCourse Datasets/EnvData_FabaDr.xlsx` — 50,798 minute-level records. 5 features: `li1_Buffer_uE` (light µmol), `t1_Buffer_C` (buffer temp °C), `rh1_Buffer_%` (buffer humidity %), `t2_Tunnel_C` (tunnel temp °C), `rh2_Tunnel_%` (tunnel humidity %). Aggregate to per-round window means.
      - Watering: from `data/TimeCourse Datasets/SC_Watering_24_FabaDr_Auto.xlsx` — 24 columns per record. Key features per plant per round: `Water Added`, `WHC.Bf` (WHC before), `WHC.Af` (WHC after), `Water Loss`, `Water Loss per Hours`. Aggregate per round window.
    - Normalize all features: z-score normalization computed on training set
  - Create `src/data/collate.py` — custom collate function for DataLoader:
    - All plants use the same T=22 canonical timeline — no variable-length handling needed. Masking handles missing positions.
    - Pads shorter sequences and creates attention mask
  - Create `src/training/cv.py` — LOGO cross-validation splitter:
    - 44 folds: each fold holds out all plants of 1 genotype (6 plants: 3 WHC-80 + 3 WHC-30)
    - Stratified by drought_category (Early/Mid/Late) to ensure balanced test distribution
    - Returns train/val/test indices per fold (train: 40 genotypes = 240 plants, val: 3 genotypes = 18 plants, test: 1 genotype = 6 plants)

  **Must NOT do**:
  - Do NOT load raw images during training — only pre-extracted features from HDF5
  - Do NOT interpolate missing fluorescence — use mask tokens
  - Do NOT normalize targets (DAG, biomass) — keep in original units for interpretability
  - Do NOT create separate datasets for each modality — single unified dataset
  - Do NOT compare accession name strings without `unicodedata.normalize('NFC', ...)` — macOS NFD decomposition will cause silent mismatches for `Mélodie` etc.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex data engineering with multiple source files, sparse modality handling, and custom batching logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with TODO 5)
  - **Parallel Group**: Wave 3 (with TODO 5, 7)
  - **Blocks**: TODO 6, 8
  - **Blocked By**: TODO 2 (metadata), TODO 3 (features)

  **References**:

  **Feature Data** (3 backbone options — selected via config `model.image_encoder`):
  - `features/dinov2_features.h5` (from TODO 3) — Pre-extracted DINOv2 CLS embeddings, keyed by plant_id/round/view
  - `features/clip_features.h5` (from TODO 3) — Pre-extracted CLIP CLS embeddings, same key structure
  - `features/bioclip_features.h5` (from TODO 3) — Pre-extracted BioCLIP CLS embeddings, same key structure
  - Dataset reads `config.data.feature_dir` + `config.model.image_encoder` to select the correct HDF5 file. All 3 files have identical key structure and dimensions (768-dim).

  **Tabular Data Sources** (to be loaded as modality inputs):
  - `data/TimeCourse Datasets/FCQ_FabaDr_Auto.xlsx` — Fluorescence WIDE format. 1,320 rows (264 plants × 5 measurement dates). 93 fluorescence parameters + 18 metadata columns = 111 total columns. 15 unique measurement dates across all plants, but only 5 per individual plant. Fluorescence rounds: [3,4,5,7,8,9,11,12,13,15,16,17,19,20,21]. Set `fluor_mask=True` for ALL rounds where fluorescence was measured for that plant (regardless of whether images exist at that round). The overlap with image rounds varies by replicate — this is handled by separate `image_mask` and `fluor_mask`.
  - `data/TimeCourse Datasets/EnvData_FabaDr.xlsx` — Environment data. 50,798 minute-level records. 5 columns: `li1_Buffer_uE` (light µmol/m²/s), `t1_Buffer_C` (buffer temperature °C), `rh1_Buffer_%` (buffer relative humidity %), `t2_Tunnel_C` (tunnel temperature °C), `rh2_Tunnel_%` (tunnel relative humidity %). Aggregate to per-round window means (see Tabular→Round Alignment below).
  - `data/TimeCourse Datasets/SC_Watering_24_FabaDr_Auto.xlsx` — Watering records. 24 columns including: `Plant ID`, `Date`, `Water Added`, `WHC.Bf` (WHC before watering), `WHC.Af` (WHC after watering), `Water Loss`, `Water Loss per Hours`. ~9,506 events (DAG 4-39, all 264 plants). Aggregate per plant per round window (see Tabular→Round Alignment below).
  - `data/plant_metadata.csv` (from TODO 2) — Plant→genotype→treatment→DAG→biomass mapping
  - `data/timepoint_metadata.csv` (from TODO 2) — Round→DAG→date mapping for temporal positions

  **Canonical Round→Date→DAG Mapping** (CRITICAL — deterministic lookup table):

  > Source: `data/00-Misc/FabaDr_Obs.xlsx` (RGB1 sheet) cross-referenced with `data/00-Misc/DateIndex.xlsx`.
  > Rule: Use **max(DAG)** per Round Order (handles Round 2 which spans 2 dates).

  ```python
  # Hardcode this in src/data/metadata.py — verified from FabaDr_Obs.xlsx
  ROUND_TO_DAG = {
      2: 4,   3: 5,   4: 6,   5: 7,   6: 10,  7: 12,
      8: 13,  9: 14, 10: 17, 11: 19, 12: 20, 13: 21,
     14: 24, 15: 27, 16: 28, 17: 29, 18: 31, 19: 33,
     20: 34, 21: 35, 22: 38, 23: 38   # Rounds 22 & 23 share DAG=38
  }
  ROUND_TO_DATE = {
      2: '2024-10-15',  3: '2024-10-16',  4: '2024-10-17',  5: '2024-10-18',
      6: '2024-10-21',  7: '2024-10-23',  8: '2024-10-24',  9: '2024-10-25',
     10: '2024-10-28', 11: '2024-10-30', 12: '2024-10-31', 13: '2024-11-01',
     14: '2024-11-04', 15: '2024-11-07', 16: '2024-11-08', 17: '2024-11-09',
     18: '2024-11-11', 19: '2024-11-13', 20: '2024-11-14', 21: '2024-11-15',
     22: '2024-11-18', 23: '2024-11-18'  # Same date, different rounds
  }
  ```

  **Special cases**:
  - **Round 2**: 197 plants imaged on 2024-10-14 (DAG=3), 595 plants on 2024-10-15 (DAG=4). Use DAG=4 as canonical (when majority were imaged). Per-plant actual DAG from Obs sheet may differ by 1 day.
  - **Rounds 22 & 23**: Both on 2024-11-18, both DAG=38. Distinguished by Round Order only. In `temporal_positions`, both get DAG=38. The model handles this via the round index (position 20 and 21 in the T=22 array), not DAG alone.

  **Tabular→Round Alignment Rules** (how sparse tabular data maps to T=22 canonical timeline):

  > Each of the 22 canonical positions is identified by (Round Order, canonical DAG, canonical date) from the table above.
  > Per-plant `temporal_positions` tensor = `[ROUND_TO_DAG[r] for r in range(2, 24)]` = fixed 22-element vector, same for all plants.

  **Environment** (`EnvData_FabaDr.xlsx`):
  - Data is minute-level with timestamps. NOT per-plant — shared across all plants.
  - Alignment: For each of the 22 canonical rounds, use `ROUND_TO_DATE[round]` to select the calendar date. Aggregate all environment records from that date into a single (5,) vector using **same-day mean**.
  - **Round 2 special case**: Use date 2024-10-15 (max date).
  - **Rounds 22/23 special case**: Both use 2024-11-18 — they get identical environment vectors.
  - Result: (T=22, E=5) — same for all plants (environment is global). All 22 positions have environment data.

  **Watering** (`SC_Watering_24_FabaDr_Auto.xlsx`):
  - Data is per-plant, roughly daily (DAG 4-39).
  - Alignment: For each of the 22 canonical rounds, find the watering record(s) for that plant within a **round window** defined as:
    - Window start: `ROUND_TO_DATE[round-1] + 1 day` (or experiment start for round 2)
    - Window end: `ROUND_TO_DATE[round]` (inclusive)
  - Aggregation: **sum** `Water Added` and `Water Loss`, take **last** `WHC.Af` and `WHC.Bf` within window. `Water Loss per Hours` = mean.
  - **Rounds 22/23 special case**: Same date → Round 22 gets the window from round 21's date+1 to 2024-11-18. Round 23 gets a zero-width window (no separate watering data). Set Round 23 watering = Round 22 watering (copy).
  - Result: (T=22, W=5) — per-plant. Most positions have data.

  **Fluorescence** (`FCQ_FabaDr_Auto.xlsx`):
  - Data is sparse: exactly 5 measurements per plant across 15 unique round orders.
  - Fluorescence rounds: [3,4,5,7,8,9,11,12,13,15,16,17,19,20,21].
  - Alignment: For each of the 22 canonical positions, check if a fluorescence measurement exists for that plant at that **Round Order** (match by Round Order, NOT by date). If YES → use the 93-param vector, set `fluor_mask[t]=True`. If NO → zero vector, `fluor_mask[t]=False`.
  - `fluor_mask` is independent of `image_mask`. A position may have fluorescence but no images (e.g., Rep-1 plants have no images at rounds 4,5,8,9,12,13,15,16,20,21,22 — but some of those rounds DO have fluorescence data).
  - Result: (T=22, F=fluor_dim) with `fluor_mask` (T=22,) having exactly 5 True values per plant.
  - **CRITICAL**: Do NOT interpolate missing fluorescence. The model's learnable [MASK] token handles missing data.
  - **CRITICAL**: `fluor_mask=True` means "fluorescence was measured at this round for this plant", regardless of whether images exist at that round. This allows the transformer to attend to fluorescence-only timepoints.

  **Image mask** (from HDF5 features):
  - For each of the 22 canonical positions, check if DINOv2 features exist for that plant at that round.
  - Result: `image_mask` (T=22, V=4) with ~11 True positions per plant.

  **Transformer attention mask** (computed from image_mask + fluor_mask):
  - A canonical position is "active" (included in attention) if `image_mask[t].any() OR fluor_mask[t]`.
  - A position is "empty" (excluded from attention) if no images AND no fluorescence.
  - This means plants effectively have ~16-17 active positions (11 image + ~5 fluor, with some overlap) out of 22.
  - Empty positions are masked in the transformer self-attention (set to -inf before softmax).

  **CV Strategy Detail** (deterministic algorithm):
  - 44 genotypes → 44 folds, fixed random seed=42 for reproducibility
  - Per fold: test=1 genotype (6 plants), val=3 genotypes (18 plants), train=40 genotypes (240 plants)

  **Exact val selection algorithm**:
  ```python
  # Deterministic val selection — implement exactly as written
  import numpy as np

  def get_logo_splits(genotype_list, categories, seed=42):
      """
      genotype_list: list of 44 genotype names
      categories: dict mapping genotype → 'Early'|'Mid'|'Late'
      Returns: list of 44 (train_genos, val_genos, test_geno) tuples
      """
      rng = np.random.RandomState(seed)
      # Group genotypes by category
      early = [g for g in genotype_list if categories[g] == 'Early']
      mid = [g for g in genotype_list if categories[g] == 'Mid']
      late = [g for g in genotype_list if categories[g] == 'Late']

      splits = []
      for test_geno in genotype_list:
          remaining = [g for g in genotype_list if g != test_geno]
          test_cat = categories[test_geno]
          # Pick 1 val from each category (excluding test)
          rem_early = [g for g in early if g != test_geno]
          rem_mid = [g for g in mid if g != test_geno]
          rem_late = [g for g in late if g != test_geno]
          val = [
              rng.choice(rem_early),
              rng.choice(rem_mid),
              rng.choice(rem_late),
          ]
          train = [g for g in remaining if g not in val]
          splits.append((train, val, test_geno))
      return splits
  ```
  - This ensures: (a) exactly 1 genotype per category in val, (b) deterministic with seed, (c) val genotypes rotate across folds (rng state advances per fold)

  **Acceptance Criteria**:
  ```bash
  python3 -c "
  import torch
  from src.data.dataset import FabaDroughtDataset
  from src.data.collate import faba_collate_fn
  from torch.utils.data import DataLoader

  ds = FabaDroughtDataset('configs/default.yaml')
  assert len(ds) == 264, f'Expected 264 plants, got {len(ds)}'

  sample = ds[0]
  assert sample['images'].shape == (22, 4, 768), f'Wrong image shape: {sample[\"images\"].shape}'
  F = ds.fluor_dim  # dynamically determined at load time (~93-96)
  assert sample['fluorescence'].shape == (22, F), f'Wrong fluor shape: {sample[\"fluorescence\"].shape}'
  assert sample['temporal_positions'].shape == (22,), f'Wrong temporal shape'
  assert sample['image_mask'].dtype == torch.bool
  assert sample['fluor_mask'].dtype == torch.bool
  # Each plant should have ~11 image positions and ~5 fluorescence positions
  n_img = sample['image_mask'][:, 0].sum().item()  # check first view
  assert 10 <= n_img <= 12, f'Expected ~11 image timepoints, got {n_img}'
  n_fluor = sample['fluor_mask'].sum().item()
  assert 3 <= n_fluor <= 7, f'Expected ~5 fluorescence timepoints, got {n_fluor}'

  # Test DataLoader with collate
  loader = DataLoader(ds, batch_size=4, collate_fn=faba_collate_fn)
  batch = next(iter(loader))
  assert batch['images'].shape == (4, 22, 4, 768)
  print(f'Dataset PASSED: {len(ds)} plants, batch shape {batch[\"images\"].shape}')
  "
  # Assert: output contains "Dataset PASSED"

  # Test CV splitter
  python3 -c "
  from src.training.cv import LogoCV
  import pandas as pd
  pm = pd.read_csv('data/plant_metadata.csv')
  cv = LogoCV(pm, n_folds=44, stratify_col='drought_category')
  folds = list(cv.split())
  assert len(folds) == 44, f'Expected 44 folds, got {len(folds)}'
  for i, (train_idx, val_idx, test_idx) in enumerate(folds):
      assert len(test_idx) >= 3, f'Fold {i}: test set too small ({len(test_idx)})'
      assert len(set(train_idx) & set(test_idx)) == 0, f'Fold {i}: train/test overlap!'
  print(f'LOGO-CV PASSED: {len(folds)} folds')
  "
  ```

  **Commit**: YES
  - Message: `feat(data): PyTorch dataset with multi-view temporal batching and LOGO-CV`
  - Files: `src/data/dataset.py`, `src/data/collate.py`, `src/training/cv.py`

---

### TODO 5: Model Architecture

- [ ] 5. Model Architecture

  **What to do**:
  - Create `src/model/encoder.py` — View Aggregation Module:
    - Input: (B, T, V=4, D=768) — 4 view embeddings per timestep
    - Learnable attention pooling: query=learnable vector, keys/values=view embeddings
    - Output: (B, T, D=768) — single aggregated view embedding per timestep
    - Handle missing views via attention mask (mask=False views excluded from attention)

  - Create `src/model/fusion.py` — Multimodal Fusion Module:
    - Input per timestep: view_emb (768), fluor_emb (fluor_dim→128 via MLP), env_emb (5→128 via MLP), water_emb (5→128 via MLP)
    - Each modality → small MLP → hidden_dim (128)
    - Missing modalities replaced by learnable [MASK] token (one per modality)
    - Fusion: concat all modality embeddings → (768+128+128+128=1152) → linear projection → fused_dim (256)
    - Output: (B, T, 256)

  - Create `src/model/temporal.py` — Temporal Transformer:
    - Input: (B, T=22, D=256) fused multimodal tokens (T=22 canonical timeline, with attention mask excluding ~11 empty positions per plant)
    - Temporal positional encoding: **continuous sinusoidal** using actual DAG values (not integer positions)
      - PE(pos, 2i) = sin(DAG / 10000^(2i/d))
      - PE(pos, 2i+1) = cos(DAG / 10000^(2i/d))
      - This handles irregular temporal spacing naturally
    - Prepend learnable [CLS] token → sequence length becomes T+1=23
    - Standard Transformer encoder: 2 layers, 4 heads, dim=256, ff_dim=1024, dropout=0.1
    - Full attention (bidirectional) for standard mode
    - Option for causal masking (for progressive early detection evaluation)
    - Output: CLS token (256-dim) + temporal tokens (T, 256)
    - **MUST** store and expose attention weights for XAI analysis

  - Create `src/model/heads.py` — Multi-Task Prediction Heads:
    - `DAGRegressionHead`: CLS → MLP(256→128→1) → predicted DAG (continuous)
    - `DAGClassificationHead`: CLS → MLP(256→128→3) → Early/Mid/Late logits
    - `BiomassHead`: CLS → MLP(256→128→2) → predicted FW, DW
    - `TrajectoryHead`: temporal_tokens → MLP(256→128→1) per timestep → (T=22,) stress trajectory (only compute loss on positions with images)
    - All heads use LayerNorm + Dropout(0.1) + ReLU activation

  - Create `src/model/model.py` — Full Model (Teacher):
    - Combines: ViewAggregation → MultimodalFusion → TemporalTransformer → TaskHeads
    - Forward returns dict: `{'dag_reg': Tensor, 'dag_cls': Tensor, 'biomass': Tensor, 'trajectory': Tensor, 'attention_weights': List[Tensor], 'cls_embedding': Tensor}`
    - Total trainable parameters: ~2-3M (excluding frozen DINOv2)
    - **MUST expose `cls_embedding` (256-dim)** — needed for distillation in TODO 8b

  - Create `src/model/student.py` — Student Model (RGB-only, for distillation in TODO 8b):
    - Same architecture as teacher BUT without fluorescence/environment/watering inputs
    - Input: only DINOv2 image features (images + image_mask + temporal_positions)
    - Fusion simplifies to: view_emb (768) → project to 256 (no multimodal concat)
    - Same temporal transformer and heads as teacher
    - Forward returns same dict structure as teacher (for compatible loss computation)
    - Trainable parameters: ~1.5-2M (slightly fewer than teacher)

  - Create `src/training/losses.py` — Multi-Task Loss:
    - Teacher loss: `L_total = λ1 * MSE(dag_reg) + λ2 * CE(dag_cls) + λ3 * MSE(biomass) + λ4 * MSE(trajectory)`
    - DAG losses computed ONLY for drought-treated plants (WHC-30)
    - Biomass loss computed for ALL plants
    - Trajectory loss: compare predicted stress scores with normalized digital biomass trajectory
      - **Target source**: `data/TimeCourse Datasets/DigitalBiomass_Norm_FabaDr_Auto.xlsx`, column `Digital Biomass Norm (e+2)`
      - **Per-plant trajectory**: Extract values at each Round Order for the plant → produces ~11 values aligned to the plant's image rounds
      - **Sign convention**: Higher values = more biomass = less stress. For the trajectory HEAD, predict `1 - norm_biomass` so that higher predicted values = more stress.
      - **Scaling**: Min-max scale across all plants' trajectories to [0, 1] range (computed on training set only)
      - **Loss mask**: Only compute trajectory loss at canonical positions where `Digital Biomass Norm` data exists for that plant (join on `Round Order`). Use `traj_mask` (T=22,) tensor.
    - Configurable λ weights via config
    - **Distillation loss** (for TODO 8b): `L_distill = α * MSE(student_cls, teacher_cls.detach()) + (1-α) * L_task_student`
      - `student_cls`: student's CLS embedding (256-dim)
      - `teacher_cls`: teacher's CLS embedding (256-dim, detached — no gradient to teacher)
      - α controls task loss vs embedding alignment balance (default: 0.5)

  **Must NOT do**:
  - Do NOT add any layers that process raw pixels (DINOv2 features only)
  - Do NOT use more than 4 transformer layers or 8 attention heads
  - Do NOT add cross-attention between modalities (simple concat fusion for v1)
  - Do NOT add variational/generative components
  - Do NOT forget to expose attention weights (critical for XAI in TODO 9)

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
    - Reason: Complex multi-component architecture design requiring careful attention to tensor shapes, masking logic, and multi-task learning
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with TODO 4, 7)
  - **Parallel Group**: Wave 3
  - **Blocks**: TODO 6, 8
  - **Blocked By**: TODO 1 (config system)

  **References**:

  **Architecture Design Rationale**:
  - All 3 backbones (DINOv2-B/14, CLIP ViT-B/16, BioCLIP) produce 768-dim CLS token — architecture is backbone-agnostic. Backbone selection is purely a config + feature file change; NO code changes needed.
  - 4 views per timestep: side_000, side_120, side_240, top — attention pooling learns which views are informative
  - Temporal transformer dim=256 (not 768) to prevent overfitting on n=264 plants
  - 2 layers / 4 heads = ~1M parameters — appropriate for small dataset
  - Continuous positional encoding using DAG values because rounds are approximately daily (rounds 2-23 = ~22 DAG values), but per-plant schedules vary; using actual DAG values handles this naturally
  - Learnable [MASK] tokens for missing modalities: standard practice from BERT/MAE literature

  **Trajectory Target Data**:
  - `data/TimeCourse Datasets/DigitalBiomass_Norm_FabaDr_Auto.xlsx` — 2,854 rows, 13 columns. Key column: `Digital Biomass Norm (e+2)` = per-plant normalized growth (first measurement = 0). Joinable on `Round Order` + `Accession Num` + `Replicate` + `Treatment`. Each plant has ~11 measurements. Drought plants show declining trajectory; controls show increasing. Use as pseudo-ground-truth for stress trajectory prediction. Invert for stress score: `stress = 1 - min_max_scale(norm_biomass)`.

  **Fluorescence Feature Names** (93 parameters):
  - `data/TimeCourse Datasets/FCQ_FabaDr_Auto.xlsx` — Wide format, ~96 fluorescence parameters per measurement (columns 16-111): Fo, Fm, Fv, Fv/Fm, NPQ, qP, qN, ETR, Y(II), Y(NO), Y(NPQ), etc. All numeric fluorescence columns are used as a single vector input per measurement. Metadata columns: `Measuring Date`, `Round Order`, `Plant ID`, `Accession Name`, `Treatment`, `Replicate`, `DAS`, `DAG`, `Weeks`, `Outlier`, etc.

  **Acceptance Criteria**:
  ```bash
  python3 -c "
  import torch
  from src.model.model import FabaDroughtModel
  from src.utils.config import load_config

  cfg = load_config('configs/default.yaml')
  model = FabaDroughtModel(cfg.model)

  # Count parameters
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f'Total params: {total:,}, Trainable: {trainable:,}')
  assert trainable < 5_000_000, f'Too many trainable params: {trainable}'

  # Forward pass sanity
  # Canonical timeline: T=22 positions (rounds 2-23), each plant has ~11 active
  T = 22
  img_mask = torch.zeros(4, T, 4).bool()
  img_mask[:, [0,1,4,5,8,9,12,15,16,17,21], :] = True  # simulate Rep-1 schedule
  fluor_mask = torch.zeros(4, T).bool()
  fluor_mask[:, [1,5,9,15,17]] = True  # ~5 fluorescence positions

  batch = {
      'images': torch.randn(4, T, 4, 768),       # B, T=22, V=4, D=768
      'image_mask': img_mask,
      'fluorescence': torch.randn(4, T, cfg.model.fusion.fluor_dim),
      'fluor_mask': fluor_mask,
      'environment': torch.randn(4, T, 5),
      'watering': torch.randn(4, T, 5),
      'temporal_positions': torch.arange(T).unsqueeze(0).expand(4, -1).float() + 2,  # DAG values
  }
  out = model(batch)
  assert out['dag_reg'].shape == (4, 1), f'DAG reg wrong: {out[\"dag_reg\"].shape}'
  assert out['dag_cls'].shape == (4, 3), f'DAG cls wrong: {out[\"dag_cls\"].shape}'
  assert out['biomass'].shape == (4, 2), f'Biomass wrong: {out[\"biomass\"].shape}'
  assert out['trajectory'].shape == (4, T), f'Trajectory wrong: {out[\"trajectory\"].shape}'
  assert 'attention_weights' in out, 'Missing attention weights!'
  assert len(out['attention_weights']) == cfg.model.temporal.num_layers

  assert 'cls_embedding' in out, 'Missing CLS embedding for distillation!'
  assert out['cls_embedding'].shape == (4, 256), f'CLS embedding wrong: {out[\"cls_embedding\"].shape}'
  print('Teacher model PASSED')

  # Verify student model
  from src.model.student import FabaDroughtStudent
  student = FabaDroughtStudent(cfg.model)
  student_batch = {k: v for k, v in batch.items() if k in ['images','image_mask','temporal_positions']}
  s_out = student(student_batch)
  assert s_out['dag_reg'].shape == (4, 1)
  assert s_out['cls_embedding'].shape == (4, 256)
  print('Student model PASSED')

  print('Model architecture PASSED')
  print(f'Output shapes: dag_reg={out[\"dag_reg\"].shape}, dag_cls={out[\"dag_cls\"].shape}, biomass={out[\"biomass\"].shape}, trajectory={out[\"trajectory\"].shape}')
  "
  # Assert: "Model architecture PASSED", "Teacher model PASSED", "Student model PASSED"
  ```

  **Commit**: YES
  - Message: `feat(model): temporal multimodal architecture with teacher-student design`
  - Files: `src/model/encoder.py`, `src/model/fusion.py`, `src/model/temporal.py`, `src/model/heads.py`, `src/model/model.py`, `src/model/student.py`, `src/training/losses.py`

---

### TODO 6: Training Pipeline + LOGO-CV

- [ ] 6. Training Pipeline with LOGO Cross-Validation

  **What to do**:
  - Create `src/training/trainer.py` — Training loop:
    - Supports multi-task training with configurable loss weights
    - Mixed precision training (fp16) for speed
    - Gradient clipping (max_norm=1.0)
    - Learning rate scheduler: cosine annealing with warmup (5 epochs)
    - Early stopping based on validation loss (patience from config)
    - Wandb logging: train/val losses per task, learning rate, epoch time
    - Checkpoint saving: best model (lowest val loss) + last model per fold
    - MUST support `--resume_from` flag for CSC queue resilience

  - Create `scripts/train.py` — Main training entry point:
    - CLI args: `--config`, `--fold` (specific fold or "all"), `--resume_from`, `--checkpoint_dir`
    - For single fold: train model, evaluate on test genotype, save predictions
    - For all folds: sequential 44-fold LOGO-CV, aggregate results

  - Create `scripts/slurm/train.sh` — CSC SLURM script:
    - Request 1 GPU (A100 preferred on Mahti)
    - Support array jobs for parallel fold execution: `#SBATCH --array=0-43`
    - Each array task trains 1 fold
    - Estimated: ~10 min per fold (100 epochs × 240 samples × small model)
    - Total: ~7.5 hours if sequential, ~20 min with 44 parallel jobs

  - Create `scripts/evaluate.py` — Evaluation and aggregation:
    - Loads all 44 fold results
    - Computes metrics with 95% CI (bootstrap across folds):
      - DAG regression: MAE, RMSE, R²
      - DAG classification: accuracy, balanced accuracy, per-class F1
      - Biomass regression: MAE, R², Pearson r (separately for FW and DW)
      - Genotype ranking: Spearman ρ, Kendall τ (predicted avg DAG vs true DAG)
    - Saves aggregated results to `results/main_results.json`

  - Implement overfit sanity test:
    - `configs/overfit_test.yaml`: 10 samples, 100 epochs, high LR
    - Training loss should approach near-zero

  **Must NOT do**:
  - Do NOT use validation set for early stopping in final evaluation (val is for hyperparameter tuning only)
  - Do NOT report metrics on validation set — only on held-out test genotypes
  - Do NOT average predictions across folds for the same genotype (each genotype appears in test exactly once)
  - Do NOT use batch size > 32 (dataset is small, large batches = fewer gradient updates)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Standard training pipeline implementation, but requires careful CV logic and CSC SLURM integration
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on TODO 4 dataset + TODO 5 model)
  - **Parallel Group**: Wave 3 (after TODO 4, 5)
  - **Blocks**: TODO 8 (experiments)
  - **Blocked By**: TODO 4 (dataset), TODO 5 (model)

  **References**:

  **Training Configuration** (from TODO 1 config):
  - `configs/default.yaml` — training section: batch_size=16, lr=1e-4, weight_decay=0.01, max_epochs=100, patience=15

  **CV Implementation** (from TODO 4):
  - `src/training/cv.py` — LogoCV class providing train/val/test splits per fold

  **Loss Implementation** (from TODO 5):
  - `src/training/losses.py` — Multi-task loss with configurable weights

  **CSC-specific Patterns**:
  - Mahti A100 partition: `#SBATCH --partition=gpumedium`, `#SBATCH --gres=gpu:a100:1`
  - Use `--time=01:00:00` per fold (generous for safety)
  - Array jobs: `#SBATCH --array=0-43%10` (max 10 concurrent for fairness)
  - Module loads: `module load pytorch` (CSC provides optimized builds)
  - Working directory: `$TMPDIR` for fast I/O, copy features HDF5 to `$TMPDIR` at job start

  **Metric Definitions**:
  - Genotype ranking: For each genotype, average the predicted DAG across its 3 drought-treated reps. Rank all 44 genotypes. Compare rank order to expert DAG ranking via Spearman ρ.
  - Pre-symptomatic detection rate: fraction of genotypes where predicted_DAG < true_DAG by ≥1 timepoint

  **Acceptance Criteria**:
  ```bash
  # Overfit test — must converge
  python3 scripts/train.py --config configs/overfit_test.yaml --fold 0 --checkpoint_dir results/overfit_test/
  python3 -c "
  import json
  with open('results/overfit_test/fold_0/metrics.json') as f:
      m = json.load(f)
  assert m['final_train_loss'] < 0.5, f'Overfit test failed: loss={m[\"final_train_loss\"]}'
  print(f'Overfit test PASSED: final loss={m[\"final_train_loss\"]:.4f}')
  "

  # Single fold training — must complete without errors
  python3 scripts/train.py --config configs/default.yaml --fold 0 --checkpoint_dir results/fold_test/
  python3 -c "
  import os
  assert os.path.exists('results/fold_test/fold_0/best_model.pt'), 'No best model checkpoint'
  assert os.path.exists('results/fold_test/fold_0/predictions.json'), 'No predictions file'
  print('Single fold training PASSED')
  "

  # SLURM script validation (syntax only)
  bash -n scripts/slurm/train.sh
  # Assert: exit code 0
  ```

  **Commit**: YES
  - Message: `feat(training): training pipeline with LOGO-CV and CSC SLURM support`
  - Files: `src/training/trainer.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/slurm/train.sh`, `configs/overfit_test.yaml`

---

### TODO 7: Classical ML Baselines

- [ ] 7. Classical ML Baselines

  **What to do**:
  - Create `src/baselines/classical.py`:
    - Load pre-computed tabular features from existing Excel files (NO image processing):
      - Per-plant features: morphology summary (area, perimeter, width, height at endpoint)
      - Digital biomass trajectory statistics (mean, slope, final value)
      - Fluorescence summary (mean Fv/Fm, mean NPQ, etc.)
      - Endpoint FW/DW: include for DAG prediction baselines only; EXCLUDE for biomass prediction (see Feature Inclusion/Exclusion Table below)
    - Build feature matrix: (264 plants × ~50 features)
    - Train baselines with same LOGO-CV as the DL model:
      - **Baseline 1**: XGBoost on tabular features → predict DAG, biomass
      - **Baseline 2**: Random Forest on tabular features → predict DAG, biomass
      - **Baseline 3**: DINOv2 features (avg across timepoints and views, 768-dim) + Random Forest → predict DAG, biomass (isolates temporal modeling contribution; use DINOv2 features only since this baseline is about temporal vs non-temporal, not backbone comparison)
    - Compute same metrics as DL model (MAE, R², Spearman ρ for ranking)
    - Save results to `results/baselines/`

  - Create `scripts/run_baselines.py` — runs all baselines

  **Must NOT do**:
  - Do NOT spend time hyperparameter-tuning baselines — use reasonable defaults (XGBoost: max_depth=6, n_estimators=100; RF: n_estimators=200)
  - Do NOT create neural network baselines (save complexity for the main model)
  - Do NOT leak target variables into features — see exclusion table below

  **Feature Inclusion/Exclusion Table** (CRITICAL — prevents leakage):

  | Feature Source | For DAG Prediction | For Biomass Prediction |
  |---|---|---|
  | Morphology trajectory (area, perimeter, height) | ✅ INCLUDE | ✅ INCLUDE |
  | Digital biomass trajectory stats (slope, AUC) | ✅ INCLUDE | ✅ INCLUDE |
  | Fluorescence summary (mean Fv/Fm, NPQ, etc.) | ✅ INCLUDE | ✅ INCLUDE |
  | Vegetation indices (NDVI, etc.) | ✅ INCLUDE | ✅ INCLUDE |
  | Watering summary (total water, mean WHC) | ✅ INCLUDE | ✅ INCLUDE |
  | Endpoint FW (fresh weight) | ✅ INCLUDE (FW ≠ DAG target) | ❌ EXCLUDE (IS the target) |
  | Endpoint DW (dry weight) | ✅ INCLUDE (DW ≠ DAG target) | ❌ EXCLUDE (IS the target) |
  | DAG ground truth | ❌ EXCLUDE (IS the target) | ✅ INCLUDE (DAG ≠ biomass target) |
  | Drought category (Early/Mid/Late) | ❌ EXCLUDE (derived from DAG) | ❌ EXCLUDE (derived from DAG) |
  | DINOv2 features (Baseline 3 only) | ✅ INCLUDE (avg CLS) | ✅ INCLUDE (avg CLS) |

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward scikit-learn/XGBoost implementation with existing tabular data
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with TODO 4, 5, 6)
  - **Parallel Group**: Wave 3
  - **Blocks**: TODO 8 (needs baseline results for comparison)
  - **Blocked By**: TODO 2 (needs plant_metadata.csv), TODO 3 (needs DINOv2 features for Baseline 3)

  **References**:

  **Tabular Feature Sources** (pre-computed, ready to use):
  - `data/TimeCourse Datasets/RGB1_SideView_FabaDr_Manual_Morpho.xlsx` — Per-plant per-timepoint side-view morphology: area, perimeter, width, height, etc. Extract endpoint values and trajectory statistics.
  - `data/TimeCourse Datasets/DigitalBiomass_Norm_FabaDr_Auto.xlsx` — Normalized digital biomass per plant per timepoint. Extract slope, final value, AUC.
  - `data/TimeCourse Datasets/RGB-Vegetative_Indices.xlsx` — RGB vegetation indices per plant per timepoint. Extract means and trends.
  - `data/Avg_and_Ranks/` — Pre-computed per-accession averages for morphology, biomass, fluorescence. Can use directly as features.
  - `data/EndPoint Datasets/EndPoint_CorrelationData-WithoutOutliers.xlsx` — 29-column aggregated feature matrix with outliers removed. THIS IS THE EASIEST BASELINE — already a feature matrix.

  **Baseline Purpose** (for paper framing):
  - Baseline 1 (XGBoost tabular): Shows what traditional phenotyping features can achieve
  - Baseline 2 (RF tabular): Same as above, different algorithm for robustness
  - Baseline 3 (DINOv2+RF, no temporal): Isolates contribution of temporal modeling — if DL model >> Baseline 3, temporal transformer adds value

  **Acceptance Criteria**:
  ```bash
  python3 scripts/run_baselines.py --config configs/default.yaml
  python3 -c "
  import json
  for name in ['xgboost_tabular', 'rf_tabular', 'dinov2_rf']:
      path = f'results/baselines/{name}_results.json'
      with open(path) as f:
          r = json.load(f)
      print(f'{name}: DAG_MAE={r[\"dag_mae\"]:.2f}, Biomass_R2={r[\"biomass_r2\"]:.3f}, Ranking_rho={r[\"ranking_spearman\"]:.3f}')
  print('All baselines PASSED')
  "
  # Assert: all three baseline files exist with valid metrics
  ```

  **Commit**: YES
  - Message: `feat(baselines): classical ML baselines (XGBoost, RF, DINOv2+RF)`
  - Files: `src/baselines/classical.py`, `scripts/run_baselines.py`

---

### TODO 8: Run All Experiments on CSC (Main + Ablation)

- [ ] 8. Run All Experiments on CSC

  **What to do**:
  - Define exactly 8 ablation configurations:
    1. **Full model** (default config, DINOv2): all modalities + temporal + multi-task
    2. **Image-only temporal** (DINOv2): remove fluorescence, environment, watering inputs
    3. **Image + fluorescence** (DINOv2): remove environment and watering
    4. **Full multimodal, no temporal** (DINOv2): replace temporal transformer with simple mean pooling over timesteps
    5. **Full model, single-task (DAG only)** (DINOv2): remove biomass and trajectory heads
    6. **Full model, DINOv2 LoRA**: unfreeze DINOv2 with LoRA (rank=8) instead of frozen
    7. **Full model, CLIP backbone**: same as variant 1 but swap DINOv2→OpenAI CLIP ViT-B/16 features (from `features/clip_features.h5`)
    8. **Full model, BioCLIP backbone**: same as variant 1 but swap DINOv2→BioCLIP features (from `features/bioclip_features.h5`)

  - Create config files: `configs/ablation/variant_{1-8}.yaml` (each overrides relevant default settings)
    - Variants 7 and 8 only override `model.image_encoder` and `data.feature_file` — everything else identical to variant 1

  - Create `scripts/slurm/ablation_sweep.sh`:
    - Submit all 8 variants × 44 folds = 352 jobs as SLURM array
    - OR submit 8 sequential jobs each with 44-fold array
    - Monitor completion, collect results

  - Run all experiments:
    1. Submit feature extraction job (if not already done in TODO 3)
    2. Submit 8 ablation variant training jobs
    3. Submit 3 baseline jobs (if not already done in TODO 7)
    4. Wait for all to complete
    5. Run `scripts/evaluate.py` for each variant
    6. Compile comparison table: `results/ablation_comparison.json`

  - Compile results into comparison format:
    ```json
    {
      "full_model": {"dag_mae": X, "dag_cls_acc": X, "biomass_r2": X, "ranking_rho": X},
      "image_only": {...},
      "image_fluor": {...},
      "no_temporal": {...},
      "single_task": {...},
      "lora": {...},
      "clip_full": {...},
      "bioclip_full": {...},
      "xgboost": {...},
      "rf": {...},
      "dinov2_rf": {...}
    }
    ```

  **Must NOT do**:
  - Do NOT add ablation variants beyond the 8 specified (guardrail G3)
  - Do NOT tune hyperparameters separately per ablation variant (use same training config for fair comparison)
  - Do NOT re-run experiments that have already completed successfully
  - Do NOT submit >50 concurrent GPU jobs on CSC (be fair to other users)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Orchestrating many SLURM jobs, monitoring completion, aggregating results
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on all training infrastructure)
  - **Parallel Group**: Wave 4
  - **Blocks**: TODO 9, 10, 11, 12
  - **Blocked By**: TODO 6 (training pipeline), TODO 7 (baselines)

  **References**:

  **Ablation Design Rationale** (for paper Methods section):
  - Variant 1 (full, DINOv2): ceiling performance with self-supervised backbone
  - Variant 2 (image-only): isolates vision foundation model contribution
  - Variant 3 (image+fluor): shows value of physiological data
  - Variant 4 (no temporal): proves temporal modeling is essential
  - Variant 5 (single-task): shows multi-task learning benefit
  - Variant 6 (LoRA): tests whether fine-tuning helps despite small dataset
  - **Variant 7 (CLIP backbone)**: compares language-supervised (CLIP) vs self-supervised (DINOv2) representations — tests whether natural-language alignment provides useful plant phenotype features
  - **Variant 8 (BioCLIP backbone)**: compares domain-specific biology-trained (BioCLIP) vs general-purpose representations — tests whether biology pre-training on TreeOfLife-10M transfers to agricultural phenotyping

  **CSC Resource Estimation**:
  - Per fold: ~10 min on A100 (100 epochs, 240 samples, batch_size=16, small model)
  - Per variant: 44 folds × 10 min = ~7.3 hours sequential, ~20 min with 44 parallel jobs
  - Total: 8 variants × 7.3h = ~58h sequential, ~2.5h with full parallelism
  - Feature extraction: ~30 min per backbone × 3 backbones = ~90 min (can parallelize)
  - Plus baselines: ~30 min total (CPU only)
  - Request Mahti A100 for GPU work

  **Acceptance Criteria**:
  ```bash
  # Verify all experiment results exist
  python3 -c "
  import json, os
  variants = ['full_model', 'image_only', 'image_fluor', 'no_temporal', 'single_task', 'lora', 'clip_full', 'bioclip_full']
  baselines = ['xgboost_tabular', 'rf_tabular', 'dinov2_rf']

  for v in variants:
      path = f'results/{v}/aggregated_results.json'
      assert os.path.exists(path), f'Missing results for {v}'
      with open(path) as f:
          r = json.load(f)
      assert 'dag_mae' in r, f'{v}: missing DAG MAE'
      assert 'ranking_spearman' in r, f'{v}: missing ranking metric'
      print(f'{v}: DAG_MAE={r[\"dag_mae\"]:.2f}, Ranking_rho={r[\"ranking_spearman\"]:.3f}')

  for b in baselines:
      path = f'results/baselines/{b}_results.json'
      assert os.path.exists(path), f'Missing baseline {b}'

  # Verify ablation comparison exists
  assert os.path.exists('results/ablation_comparison.json')
  print('All experiments PASSED')
  "
  # Assert: all 8 variants + 3 baselines have results
  ```

  **Commit**: YES
  - Message: `feat(experiments): ablation study configs and CSC SLURM scripts (8 variants incl. backbone comparison)`
  - Files: `configs/ablation/*.yaml`, `scripts/slurm/ablation_sweep.sh`, `results/ablation_comparison.json`

---

### TODO 8b: Teacher-Student Distillation (Fluorescence as Privileged Information)

- [ ] 8b. Teacher-Student Distillation

  **What to do**:

  > **Core Idea**: The multimodal teacher model has access to fluorescence (expensive, lab-only sensor).
  > The student model uses ONLY RGB images (cheap, field-deployable cameras).
  > By distilling the teacher's knowledge into the student, we create an RGB-only model
  > that implicitly "knows" what the fluorescence would have shown — without needing the sensor.
  > This is the **Privileged Information** paradigm (Vapnik & Vashist 2009, LUPI).

  - Create `scripts/train_distill.py` — Distillation training script:
    - Load pre-trained teacher model (best checkpoint from full_model variant, TODO 8)
    - Freeze teacher entirely (no gradient updates)
    - Initialize student model (RGB-only architecture from `src/model/student.py`)
    - Distillation training loop:
      - For each batch: run teacher forward (get teacher_cls, teacher predictions)
      - Run student forward (get student_cls, student predictions)
      - Loss = α * MSE(student_cls, teacher_cls.detach()) + (1-α) * L_task_student
      - α = 0.5 (configurable), anneal from 0.7→0.3 over training (start with more teacher guidance)
    - Same LOGO-CV as teacher (44 folds) — use same fold splits for fair comparison
    - Same training hyperparameters (lr, epochs, patience) as teacher

  - Create `configs/distillation.yaml`:
    ```yaml
    distillation:
      teacher_checkpoint: "results/full_model/fold_{fold}/best_model.pt"
      alpha_start: 0.7
      alpha_end: 0.3
      alpha_schedule: "linear"  # anneal over training
      student_model: "rgb_only"
    ```

  - Create `scripts/slurm/train_distill.sh` — CSC SLURM for distillation (44-fold array job)

  - Evaluate student model:
    - Same metrics as teacher (DAG MAE, biomass R², ranking Spearman ρ)
    - **Key comparison**: student vs ablation variant 2 (image-only, NO distillation)
    - If student >> image-only → distillation transfers fluorescence knowledge successfully
    - If student ≈ teacher → fluorescence adds atmosphere but not unique information
    - Report the **distillation gap**: teacher_metric - student_metric (how much is lost)

  - Save results to `results/distillation/`

  **Must NOT do**:
  - Do NOT fine-tune the teacher during distillation (teacher is frozen reference)
  - Do NOT use a different CV split than the teacher (must be comparable)
  - Do NOT add complex distillation objectives (attention transfer, intermediate layer matching) — simple embedding + task loss is sufficient for v1
  - Do NOT claim "zero-shot" or "sensor-free" without caveats — the student still needs RGB images from a phenotyping platform

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Distillation training requires careful teacher/student coordination and evaluation design
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with TODO 9, 10)
  - **Parallel Group**: Wave 4
  - **Blocks**: TODO 11 (figures need distillation results)
  - **Blocked By**: TODO 8 (needs trained teacher models)

  **References**:

  **Teacher Model**:
  - `results/full_model/fold_{fold}/best_model.pt` (from TODO 8) — Pre-trained multimodal teacher, one per fold

  **Student Architecture**:
  - `src/model/student.py` (from TODO 5) — RGB-only variant, same temporal transformer + heads, no multimodal fusion

  **Distillation Loss**:
  - `src/training/losses.py` (from TODO 5) — `L_distill` combining embedding alignment + task loss

  **Key Comparison Target**:
  - `results/image_only/aggregated_results.json` (from TODO 8, ablation variant 2) — This is the RGB-only model trained WITHOUT distillation. The distilled student should OUTPERFORM this.
  - The performance gap (distilled_student - image_only_no_distill) = **value of fluorescence as privileged information**

  **Literature Reference**:
  - Vapnik & Vashist 2009, "A new learning paradigm: Learning Using Privileged Information" — theoretical foundation for LUPI
  - Lopez-Paz et al. 2016, "Unifying distillation and privileged information" — connects distillation with LUPI

  **Paper Narrative**:
  - This experiment answers: "Can a model trained with expensive fluorescence sensors transfer its knowledge to a model that only needs cheap RGB cameras?"
  - If YES → practical deployment story for breeding programs
  - The distillation gap quantifies the irreducible information in fluorescence that RGB cannot capture

  **Acceptance Criteria**:
  ```bash
  python3 scripts/train_distill.py --config configs/distillation.yaml --fold 0 --checkpoint_dir results/distillation/
  python3 -c "
  import json, os

  # Verify distillation results exist
  assert os.path.exists('results/distillation/fold_0/best_model.pt'), 'No distilled student checkpoint'
  assert os.path.exists('results/distillation/fold_0/predictions.json'), 'No predictions'

  # After all folds: compare student vs image-only
  if os.path.exists('results/distillation/aggregated_results.json'):
      with open('results/distillation/aggregated_results.json') as f:
          student = json.load(f)
      with open('results/image_only/aggregated_results.json') as f:
          img_only = json.load(f)
      with open('results/full_model/aggregated_results.json') as f:
          teacher = json.load(f)

      print(f'Teacher (full multimodal):  DAG_MAE={teacher[\"dag_mae\"]:.2f}')
      print(f'Student (RGB distilled):    DAG_MAE={student[\"dag_mae\"]:.2f}')
      print(f'Image-only (no distill):    DAG_MAE={img_only[\"dag_mae\"]:.2f}')
      distill_gain = img_only['dag_mae'] - student['dag_mae']
      print(f'Distillation gain: {distill_gain:.2f} MAE improvement')
      print('Distillation analysis PASSED')
  "
  ```

  **Commit**: YES
  - Message: `feat(distillation): teacher-student knowledge distillation (fluorescence as privileged info)`
  - Files: `scripts/train_distill.py`, `configs/distillation.yaml`, `scripts/slurm/train_distill.sh`

---

### TODO 9: XAI Temporal Attention Analysis + Fluorescence Triangulation

- [ ] 9. XAI Temporal Attention Analysis + Pre-Symptomatic Quantification

  **What to do**:
  - Create `src/analysis/attention.py`:
    - Load best model from full_model variant
    - For each plant in the dataset:
      - Run forward pass, extract attention weights from all layers
      - Compute CLS→temporal attention: average across heads and layers → (T=22,) attention vector over canonical timeline
      - This shows which timepoints the model attends to most for prediction
    - Aggregate per genotype (average attention across 3 drought-treated reps)

  - Create `src/analysis/fluorescence_changepoint.py` — **NEW: Fluorescence Change Point Detection**:

    **Precise Algorithm**:
    1. **Select indicator parameters** from `FCQ_FabaDr_Auto.xlsx` (columns 16+):
       - Primary: `Fv/Fm` (photosynthetic efficiency — drops under stress)
       - Secondary: `NPQ` (non-photochemical quenching — rises under stress), `QY` (quantum yield — drops under stress)
       - Use the WIDE format file directly. If column name contains `Fv/Fm` at any light level (Lss, D1, etc.), pick the steady-state `Fv/Fm` variant. Executor should inspect column names and select most relevant variant.

    2. **Construct control baseline** per round:
       - For each fluorescence Round Order (15 rounds: [3,4,5,7,8,9,11,12,13,15,16,17,19,20,21]):
         - Collect all WHC-80% plants' values at that round → compute `mean_ctrl[round]` and `std_ctrl[round]`
       - Use ALL 132 control plants (44 accessions × 3 reps) — NOT genotype-matched controls
       - Compute per-round variance (NOT pooled across rounds), because fluorescence naturally changes over plant development

    3. **Detect change point per drought plant**:
       - For each WHC-30% plant (132 plants), at each of its 5 fluorescence rounds:
         - z_score = (plant_value - mean_ctrl[round]) / std_ctrl[round]
       - `fluor_change_round` = **first round where |z_score| > 2** for Fv/Fm
       - If multiple indicators used: require ≥2 of 3 indicators to cross threshold
       - If no round crosses threshold → `fluor_change_dag = NaN` (no detectable change)
       - Convert round to DAG using `ROUND_TO_DAG` mapping

    4. **Aggregate per genotype**:
       - `fluor_change_dag` per genotype = **median** across 3 drought reps (robust to outliers)
       - If ≥2 of 3 reps have NaN → genotype's `fluor_change_dag = NaN`

    **Sparse series handling**: Each plant has only 5 fluorescence measurements. The change point can only be detected at one of these 5 timepoints. This limits temporal resolution to ~3-7 DAG intervals. Report this limitation in the paper.

    This provides the **physiological clock** — when stress actually begins at the cellular level

  - Create `src/analysis/presymptomatic.py` — **ENHANCED: Three-Way Triangulation**:
    - For each drought-treated genotype, compute three timestamps:
      1. `fluor_change_dag`: when fluorescence first changes (physiological onset, from changepoint analysis)
      2. `attention_peak_dag`: when model attention peaks (model-detected onset, from attention analysis)
      3. `human_dag`: when human expert first observes drought (visual onset, ground truth)
    - **Core validation**: test whether `fluor_change_dag ≤ attention_peak_dag ≤ human_dag`
      - If this ordering holds for most genotypes → model is detecting REAL physiological stress, not noise
      - If attention_peak ≈ fluor_change < human_dag → strongest claim: "model detects physiological stress invisible to human eye"
      - If attention_peak < fluor_change → model may be detecting artifacts (red flag)
    - Report statistics:
      - N genotypes following the expected ordering (fluor ≤ attention ≤ human)
      - Pearson/Spearman correlation between all three timestamps
      - Mean lead time: human_dag - attention_peak_dag (days)
      - Mean physiological lead: human_dag - fluor_change_dag (days)
      - Model vs physiology alignment: |attention_peak_dag - fluor_change_dag| (days)
    - **Negative control**: verify control plants (WHC-80%) show:
      - No significant fluorescence change points
      - No attention peaks at drought-typical timepoints

  - Create `src/analysis/early_detection.py`:
    - Progressive temporal truncation evaluation:
      - For T_cutoff in rounds {2, 3, 4, ..., 23} (22 levels corresponding to canonical timeline positions):
        - Feed model only timepoints up to round T_cutoff (mask all later positions)
        - Predict DAG for each genotype
        - Compute metrics (MAE, ranking Spearman ρ)
      - Plot: prediction accuracy vs number of timepoints available
      - This shows minimum observation period needed for reliable prediction

  - Create `scripts/analyze_attention.py` — runs all analyses, saves outputs to `results/analysis/`

  **Must NOT do**:
  - Do NOT use GradCAM or pixel-level attention (guardrail G12) — temporal attention only
  - Do NOT cherry-pick genotypes for visualization — show all 44
  - Do NOT claim "pre-symptomatic" if the window is <1 timepoint (could be noise)
  - Do NOT confuse correlation with causation in the narrative
  - Do NOT claim fluorescence VALIDATES model if the ordering doesn't hold — report honestly
  - Do NOT interpolate fluorescence for change point detection — only use actual measured timepoints

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Novel analysis pipeline requiring careful interpretation of attention weights
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with TODO 10)
  - **Parallel Group**: Wave 4 (with TODO 10)
  - **Blocks**: TODO 11 (figures need attention data)
  - **Blocked By**: TODO 8 (needs trained models)

  **References**:

  **Model Output Reference**:
  - `out['attention_weights']` — List of (B, H, T+1, T+1) tensors, one per transformer layer. H=num_heads, T+1 includes CLS token. Extract CLS row (index 0) to get CLS→temporal attention.

  **Ground Truth Reference**:
  - `data/SinglePoint Datasets/Drought_Impact(DAG).xlsx` — Columns: Accession, Drought_impact(DAG), Drought_impact(category). The DAG value is the human-annotated day of first visible drought impact. This is **Clock 3** (human visibility).

  **Fluorescence Data for Change Point Detection**:
  - `data/TimeCourse Datasets/FCQ_FabaDr_Auto.xlsx` — Fluorescence WIDE format, 93 parameters per measurement. Key parameters for stress detection: Fv/Fm (drops under stress), NPQ (increases under stress), Y(II) (drops under stress), qP (drops under stress). This provides **Clock 1** (physiological onset). Alternatively use `FCQ_FabaDr_Auto_Reshape.xlsx` (long format, 14 summary params) for simpler change point analysis on key indicators.

  **Temporal Position Mapping**:
  - `data/timepoint_metadata.csv` (from TODO 2) — Maps timepoint indices to DAG values. Critical for converting attention index → actual days.

  **Triangulation Validation Logic**:
  - **Expected ordering**: Clock 1 (fluorescence) ≤ Clock 2 (model attention) ≤ Clock 3 (human)
  - This ordering reflects biology: cellular stress → detectable visual features → overt symptoms
  - Strong claim threshold: ordering holds for ≥70% of genotypes AND lead time ≥7 days
  - Moderate claim: ordering holds for ≥50% OR lead time ≥3 days
  - If ordering doesn't hold: report honestly, discuss in paper limitations

  **Pre-symptomatic Claim Threshold**:
  - Require attention peak ≥2 timepoints (≥7 days) before human DAG to claim strong pre-symptomatic detection
  - 1 timepoint (2-5 days) before = marginal pre-symptomatic
  - 0 or after = no pre-symptomatic detection
  - Report honestly regardless of result

  **Acceptance Criteria**:
  ```bash
  python3 scripts/analyze_attention.py --model_dir results/full_model/ --output_dir results/analysis/
  python3 -c "
  import json
  # Triangulation results
  with open('results/analysis/triangulation_summary.json') as f:
      tri = json.load(f)
  print(f'=== THREE-WAY TRIANGULATION ===')
  print(f'Genotypes with correct ordering (fluor ≤ attention ≤ human): {tri[\"n_correct_ordering\"]}/44')
  print(f'Mean fluorescence change DAG: {tri[\"mean_fluor_change_dag\"]:.1f}')
  print(f'Mean attention peak DAG: {tri[\"mean_attention_peak_dag\"]:.1f}')
  print(f'Mean human DAG: {tri[\"mean_human_dag\"]:.1f}')
  print(f'Model lead time (human - attention): {tri[\"mean_lead_time_days\"]:.1f} days')
  print(f'Model-physiology alignment: {tri[\"mean_model_fluor_gap_days\"]:.1f} days')
  print(f'Fluor-attention correlation: r={tri[\"fluor_attention_pearson_r\"]:.3f}')
  assert tri['n_total_drought'] == 44, 'Not all genotypes analyzed'

  # Pre-symptomatic summary
  with open('results/analysis/presymptomatic_summary.json') as f:
      s = json.load(f)
  print(f'Pre-symptomatic genotypes: {s[\"n_presymptomatic\"]}/{s[\"n_total_drought\"]}')
  print(f'Control negative check: {s[\"control_negative_passed\"]}')
  assert 'control_negative_passed' in s, 'Missing negative control'

  # Early detection
  with open('results/analysis/early_detection.json') as f:
      ed = json.load(f)
  assert len(ed['truncation_results']) == 22, 'Not all truncation levels evaluated'
  print(f'Early detection at round 10: MAE={ed[\"truncation_results\"][\"10\"][\"dag_mae\"]:.2f}')
  print('XAI + Triangulation analysis PASSED')
  "
  ```

  **Commit**: YES
  - Message: `feat(analysis): XAI temporal attention, fluorescence triangulation, and pre-symptomatic detection`
  - Files: `src/analysis/attention.py`, `src/analysis/fluorescence_changepoint.py`, `src/analysis/presymptomatic.py`, `src/analysis/early_detection.py`, `scripts/analyze_attention.py`

---

### TODO 10: Genotype Ranking Evaluation

- [ ] 10. Genotype Ranking Evaluation

  **What to do**:
  - Create `src/analysis/ranking.py`:
    - For each genotype (44 accessions):
      - Average predicted DAG across 3 drought-treated reps (from full_model LOGO-CV results)
      - Average predicted biomass (FW, DW) across 3 drought-treated reps
    - Create ranking by predicted DAG (ascending = most sensitive first)
    - Create ranking by predicted biomass under drought (ascending = most impacted first)
    - Compare rankings to ground truth:
      - Expert DAG ranking (from `Drought_Impact(DAG).xlsx`)
      - Expert biomass ranking (from `EndPoint_Raw_FW&DW.xlsx`, drought plants only)
    - Compute:
      - Spearman ρ with 95% CI (bootstrap)
      - Kendall τ with p-value
      - Top-5 recall (do the 5 most sensitive genotypes match?)
      - Bottom-5 recall (do the 5 most tolerant genotypes match?)
      - Category accuracy: what fraction of genotypes are correctly classified as Early/Mid/Late?

  - Create `src/analysis/embedding_viz.py`:
    - Extract CLS embeddings from the temporal transformer for all plants
    - t-SNE/UMAP visualization:
      - Color by treatment → should separate WHC-80 vs WHC-30
      - Color by drought category → should show Early/Mid/Late clustering
      - Color by accession → should group replicates
    - Save embeddings for figure generation

  - Create `scripts/analyze_ranking.py` — runs ranking analysis + embedding visualization

  **Must NOT do**:
  - Do NOT rank based on validation or training predictions — only test set predictions (each genotype appears once in test across 44 folds)
  - Do NOT optimize ranking method (use simple average; more complex = overfitting)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Statistical analysis with ranking metrics and embedding visualization
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with TODO 9)
  - **Parallel Group**: Wave 4
  - **Blocks**: TODO 11 (figures need ranking data)
  - **Blocked By**: TODO 8 (needs trained model predictions)

  **References**:

  **Ground Truth Rankings**:
  - `data/SinglePoint Datasets/Drought_Impact(DAG).xlsx` — Expert-annotated DAG per genotype. The ranking from this file is the gold standard.
  - `data/Avg_and_Ranks/` — Pre-computed average rankings across traits. Key files: `DigBio_Norm_FabaDr_Avg_and_Ranks.xlsx` (biomass rankings), `FCQ_FabaDr_ByDAG_Avg_and_Ranks.xlsx` (fluorescence rankings by DAG), `EndPoint_FW&DW_Avg_and_Ranks.xlsx` (endpoint weight rankings).

  **Benchmark Comparison**:
  - Wu et al. 2021 (Genome Biology) — Reported genotype ranking correlations of 0.6-0.8 using classical image analysis on 368 maize genotypes. Our target should be comparable or better, accounting for smaller dataset (44 genotypes).

  **Acceptance Criteria**:
  ```bash
  python3 scripts/analyze_ranking.py --results_dir results/full_model/ --output_dir results/analysis/
  python3 -c "
  import json
  with open('results/analysis/ranking_results.json') as f:
      r = json.load(f)
  print(f'DAG ranking - Spearman rho: {r[\"dag_spearman_rho\"]:.3f} (p={r[\"dag_spearman_p\"]:.4f})')
  print(f'DAG ranking - Kendall tau: {r[\"dag_kendall_tau\"]:.3f}')
  print(f'Top-5 recall: {r[\"top5_recall\"]:.2f}')
  print(f'Bottom-5 recall: {r[\"bottom5_recall\"]:.2f}')
  print(f'Category accuracy: {r[\"category_accuracy\"]:.2f}')
  assert r['dag_spearman_rho'] > -1.0, 'Invalid Spearman rho'  # Just check it exists and is valid
  assert 0 <= r['top5_recall'] <= 1.0, 'Invalid top-5 recall'
  print('Ranking analysis PASSED')
  "
  # Assert: ranking results exist with valid metrics

  # Verify embedding visualization
  ls results/analysis/embeddings_tsne.npy results/analysis/embeddings_umap.npy
  # Assert: both files exist
  ```

  **Commit**: YES
  - Message: `feat(analysis): genotype ranking evaluation and embedding visualization`
  - Files: `src/analysis/ranking.py`, `src/analysis/embedding_viz.py`, `scripts/analyze_ranking.py`

---

### TODO 11: Generate All Figures

- [ ] 11. Generate All Paper Figures

  **What to do**:
  - Create `scripts/generate_figures.py` — produces all main + supplementary figures
  - **Main Figures** (6 max):

    1. **Figure 1: Framework Overview** (architecture diagram)
       - Schematic: RGB images → DINOv2 → View Aggregation → [+ Fluorescence + Environment + Watering] → Temporal Transformer → Multi-task Heads
       - Show data flow with tensor dimensions annotated
       - Include temporal attention highlight
       - Format: vector graphics (SVG/PDF), Nature MI style (single/double column)

    2. **Figure 2: Ablation Study + Backbone Comparison**
       - Grouped bar chart: 8 DL variants + 3 baselines
       - Metrics: DAG MAE, Biomass R², Ranking Spearman ρ
       - Error bars: 95% CI from LOGO-CV
       - **Group 1** (left): Architecture ablation (variants 1-6, all DINOv2) — shows component contribution
       - **Group 2** (right): Backbone comparison (DINOv2 vs CLIP vs BioCLIP, all full model) — shows representation quality
       - Highlight: self-supervised (DINOv2) vs language-supervised (CLIP) vs domain-specific (BioCLIP)

    3. **Figure 3: Three-Way Triangulation** (THE key figure)
       - **Panel A**: Heatmap: 44 genotypes (rows, sorted by DAG) × 22 canonical timepoints (columns)
         - Color = temporal attention weight
         - Overlay markers per genotype row: ◆ fluorescence change point, ★ attention peak, ● human DAG
         - Shows visually: ◆ ≤ ★ ≤ ● ordering (physiology → model → human)
       - **Panel B**: Scatter plot: fluorescence change DAG (x) vs attention peak DAG (y)
         - Color by genotype category (Early/Mid/Late)
         - Shows correlation between physiological onset and model detection
         - Identity line for reference
       - **Panel C**: Timeline diagram for 3 representative genotypes (Early/Mid/Late)
         - Horizontal timeline (DAG) with three markers each
         - Clearly visualizes the physiology→model→human lag

    4. **Figure 4: Early Detection Curve**
       - Line plot: prediction accuracy (y) vs DAG cutoff (x, rounds 2-23)
       - Multiple lines: DAG MAE, Biomass R², Ranking ρ
       - Shows minimum observation period needed
       - Annotation: "reliable prediction possible from round X (DAG=Y)"
       - Note: since plants have different round schedules, compute metrics per-round across all plants that have data at that round

    5. **Figure 5: Genotype Ranking**
       - Scatter plot: predicted DAG rank vs true DAG rank (44 points)
       - Color by drought category (Early/Mid/Late)
       - Annotate: Spearman ρ, Kendall τ
       - Include identity line
       - Highlight: top-5 and bottom-5 genotypes with labels

    6. **Figure 6: Teacher-Student Distillation**
       - **Panel A**: Bar chart comparing Teacher (full multimodal) vs Distilled Student (RGB-only) vs Image-only (no distillation) across all metrics
       - Shows distillation recovers X% of the multimodal advantage using RGB alone
       - **Panel B**: Embedding space comparison (t-SNE)
         - Left: Teacher embeddings colored by drought category
         - Right: Student embeddings colored by drought category
         - Shows student learns similar structure despite no fluorescence input

  - **Supplementary Figures** (up to 4):
    - S1: Dataset overview (experiment design, imaging setup, sample images)
    - S2: Per-genotype triangulation profiles (44 individual plots: fluorescence trajectory + attention + DAG overlay)
    - S3: Embedding space visualization (t-SNE/UMAP colored by treatment, category, accession)
    - S4: Confusion matrix + learning curves for representative folds

  - Style requirements:
    - Nature MI format: 89mm (single column) or 183mm (double column) width
    - Font: Helvetica/Arial, ≥6pt
    - 300 DPI for raster, vector preferred
    - Accessible color palette (colorblind-friendly)

  **Must NOT do**:
  - Do NOT exceed 6 main + 4 supplementary figures (guardrail G9)
  - Do NOT use 3D plots (hard to read in print)
  - Do NOT use rainbow colormaps (not accessible)
  - Do NOT manually adjust figures — script must be fully reproducible

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Publication-quality scientific figures requiring careful layout, typography, and color design
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: Visual design sensibility for figure aesthetics and layout

  **Parallelization**:
  - **Can Run In Parallel**: YES (with TODO 12 partially)
  - **Parallel Group**: Wave 5 (with TODO 12)
  - **Blocks**: TODO 13 (paper needs figures)
  - **Blocked By**: TODO 8b (distillation results), TODO 9 (attention + triangulation), TODO 10 (ranking analysis)

  **References**:

  **Data Sources for Figures**:
  - `results/ablation_comparison.json` (from TODO 8) — all ablation + baseline metrics
  - `results/distillation/aggregated_results.json` (from TODO 8b) — distillation experiment results
  - `results/analysis/triangulation_summary.json` (from TODO 9) — three-way triangulation results
  - `results/analysis/presymptomatic_summary.json` (from TODO 9) — pre-symptomatic detection results
  - `results/analysis/early_detection.json` (from TODO 9) — progressive truncation results
  - `results/analysis/ranking_results.json` (from TODO 10) — genotype ranking results
  - `results/analysis/embeddings_tsne.npy`, `results/analysis/embeddings_umap.npy` (from TODO 10) — embedding coordinates

  **Nature MI Figure Guidelines**:
  - https://www.nature.com/natmachintell/for-authors/preparing-your-submission
  - Prefer PDF/EPS for vector graphics
  - Maximum 6 main figures for Articles

  **Acceptance Criteria**:
  ```bash
  python3 scripts/generate_figures.py --results_dir results/ --output_dir paper/figures/
  python3 -c "
  import os
  main_figs = ['fig1_architecture.pdf', 'fig2_ablation.pdf', 'fig3_triangulation.pdf',
               'fig4_early_detection.pdf', 'fig5_ranking.pdf', 'fig6_distillation.pdf']
  supp_figs = ['figS1_dataset.pdf', 'figS2_genotype_triangulation.pdf',
               'figS3_embeddings.pdf', 'figS4_confusion_curves.pdf']

  for fig in main_figs:
      path = f'paper/figures/{fig}'
      assert os.path.exists(path), f'Missing: {fig}'
      size = os.path.getsize(path)
      assert size > 1000, f'{fig} is suspiciously small ({size} bytes)'

  for fig in supp_figs:
      path = f'paper/figures/{fig}'
      assert os.path.exists(path), f'Missing: {fig}'

  print(f'All figures PASSED: {len(main_figs)} main + {len(supp_figs)} supplementary')
  "
  ```

  **Commit**: YES
  - Message: `feat(figures): all paper figures (6 main + 4 supplementary)`
  - Files: `scripts/generate_figures.py`, `paper/figures/*.pdf`

---

### TODO 12: Paper Draft (Nature Machine Intelligence Format)

- [ ] 12. Paper Draft

  **What to do**:
  - Create `paper/main.tex` — Nature Machine Intelligence Article format
  - **Title** (working): "Bridging the physiology-visibility gap: temporal deep learning reveals pre-symptomatic drought signals through fluorescence-guided privileged learning"

  - **Alternative titles**:
    - "Temporal multimodal deep learning quantifies the physiology-to-visibility lag in crop drought response"
    - "From lab sensors to field deployment: fluorescence-guided distillation enables RGB-based early drought warning"

  - **Structure**:
    1. **Abstract** (~150 words): The physiology→visibility gap → framework → triangulation validation → distillation for deployment → significance
    2. **Introduction** (~800 words):
       - Climate change → drought → crop losses → need for EARLY screening (not just detection)
       - The known physiology→visibility lag: plants respond physiologically before symptoms are visible, but current phenotyping relies on visible-range assessment
       - Gap: no framework exploits this lag computationally; no method transfers expensive sensor knowledge to cheap RGB
       - Vision foundation models as powerful but underexploited feature extractors for temporal phenotyping
       - Our contributions (4 claims — see below)
    3. **Results** (~1500 words):
       - **R1**: Framework architecture and temporal multimodal design
       - **R2**: Ablation study — temporal + multimodal significantly outperforms (establishing method works)
       - **R3**: **Three-way triangulation** — model attention peaks align with fluorescence change points, both preceding human DAG (THE core result)
       - **R4**: Teacher-student distillation — RGB-only student recovers X% of multimodal performance (deployment story)
       - **R5**: Early detection curve + genotype ranking (practical utility)
    4. **Discussion** (~800 words):
       - The physiology-visibility gap as a general phenomenon exploitable by deep learning
       - Fluorescence as privileged information: implications beyond this specific experiment
       - Limitations (dataset size, single experiment, temporal resolution, attention≠explanation caveat)
       - Future: larger panels, field cameras, multi-environment, other crops
    5. **Methods** (~1500 words, can be longer):
       - Plant material and experimental design
       - Image acquisition and data collection
       - DINOv2 feature extraction
       - Model architecture (detailed)
       - Training and evaluation (LOGO-CV, metrics)
       - Attention analysis and pre-symptomatic quantification
    6. **Data Availability**: describe data + code repository
    7. **References**: ~40-50 references

  - **Key Narrative Framing** (Nature MI, methodology-first, but grounded in biology):
    - Lead with the PROBLEM: "The physiology-visibility gap in plant stress detection limits early intervention"
    - Present the METHOD: "We bridge this gap with a temporal multimodal framework that uses fluorescence as privileged physiological information"
    - The VALIDATION: "Three-way triangulation (fluorescence ↔ model attention ↔ human observation) confirms the model detects real physiological stress"
    - The DEPLOYMENT STORY: "Knowledge distillation transfers expensive-sensor knowledge to cheap-RGB predictions"
    - NOT just: "We applied DINOv2 to plants" (too incremental)
    - NOT just: "We discovered pre-symptomatic signals" (Nature Plants framing, not NMI)

  - **Contribution Claims** (be precise, defensible):
    1. "First temporal multimodal framework that exploits the physiology→visibility lag in crop drought through privileged information learning"
    2. "Three-way validation demonstrating that model-detected stress signals align with independent physiological measurements (fluorescence) and precede expert-observable symptoms by X days"
    3. "Teacher-student distillation paradigm that transfers fluorescence-guided knowledge to an RGB-only model, enabling deployment without expensive sensors"
    4. "Systematic ablation demonstrating that temporal modeling, multimodal fusion, and multi-task learning each provide significant independent contributions, with comparative evaluation of self-supervised (DINOv2), language-supervised (CLIP), and domain-specific (BioCLIP) vision foundation model representations for plant phenotyping"

  **Must NOT do**:
  - Do NOT claim to have built a "foundation model" (guardrail G6)
  - Do NOT overclaim statistical significance with n=44 genotypes
  - Do NOT omit limitations (dataset size, single location, single season)
  - Do NOT exceed 4,500 words main text (Nature MI limit for Articles)
  - Do NOT use first person ("I/we found") — use "the framework demonstrates"
  - Do NOT include GWAS, genomics, or field deployment discussion (guardrail G8)

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Scientific paper writing requiring precise language, journal-specific formatting, and clear narrative structure
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with TODO 11 — can start structure while figures are generated)
  - **Parallel Group**: Wave 5
  - **Blocks**: TODO 13
  - **Blocked By**: TODO 8 (needs experiment results for numbers), TODO 9 (XAI results), TODO 10 (ranking results)

  **References**:

  **Results Data** (to fill in paper numbers):
  - `results/ablation_comparison.json` — all model variant metrics
  - `results/analysis/presymptomatic_summary.json` — pre-symptomatic detection statistics
  - `results/analysis/early_detection.json` — progressive truncation metrics
  - `results/analysis/ranking_results.json` — genotype ranking statistics

  **Key Literature to Cite**:
  - Wu et al. 2021, Genome Biology — gold standard phenotyping paper (classical methods, 368 maize genotypes)
  - Oquab et al. 2024, TMLR — DINOv2 paper (vision foundation model)
  - AgriCLIP 2025, COLING — agriculture-specific CLIP (related work)
  - BioCLIP 2024, CVPR — biology-specific vision model (related work; we compare against this backbone)
  - Radford et al. 2021, ICML — "Learning Transferable Visual Models From Natural Language Supervision" (CLIP original paper; we compare against this backbone)
  - Choudhury et al. 2023 — DTW-based drought prediction (classical temporal baseline)
  - Vaswani et al. 2017 — Transformer architecture (Attention is All You Need)
  - **Vapnik & Vashist 2009 — Learning Using Privileged Information (LUPI)** — theoretical foundation for fluorescence as privileged info
  - **Lopez-Paz et al. 2016 — "Unifying distillation and privileged information"** — connects teacher-student to LUPI
  - **Hinton et al. 2015 — "Distilling the Knowledge in a Neural Network"** — knowledge distillation
  - Plant phenotyping reviews: Roitsch et al. 2019, Trends in Plant Science
  - Fluorescence as early stress indicator: Baker 2008, Annual Review of Plant Biology

  **Nature MI Formatting**:
  - Article format: up to 4,500 words (excluding abstract, methods, references, figure legends)
  - Abstract: up to 150 words, no references
  - Methods: no word limit
  - References: Nature citation style
  - Figure legends: below figures, separate from main text word count

  **Acceptance Criteria**:
  ```bash
  # Verify LaTeX compiles
  pdflatex -interaction=nonstopmode paper/main.tex 2>&1 | tail -5
  # Assert: no fatal errors

  # Verify word count
  texcount -brief paper/main.tex 2>/dev/null | head -1
  # Assert: main text ≤4500 words (excluding methods)

  # Verify all figures referenced
  python3 -c "
  import re
  with open('paper/main.tex') as f:
      tex = f.read()
  for i in range(1, 7):
      assert f'fig{i}' in tex.lower() or f'figure {i}' in tex.lower() or f'fig.~{i}' in tex.lower(), f'Figure {i} not referenced'
  print('All 6 main figures referenced in paper')
  "

  # Verify key sections exist
  python3 -c "
  with open('paper/main.tex') as f:
      tex = f.read().lower()
  sections = ['abstract', 'introduction', 'results', 'discussion', 'methods', 'references']
  for s in sections:
      assert s in tex, f'Missing section: {s}'
  print('All sections present')
  "
  ```

  **Commit**: YES
  - Message: `feat(paper): Nature Machine Intelligence paper draft`
  - Files: `paper/main.tex`, `paper/references.bib`

---

### TODO 13: Supplementary Materials + Final Review

- [ ] 13. Supplementary Materials + Final Review

  **What to do**:
  - Create `paper/supplementary.tex`:
    - Supplementary Methods: detailed model hyperparameters, training details, data preprocessing steps
    - Supplementary Tables:
      - Table S1: Complete genotype list with origin, DAG, category
      - Table S2: Full ablation results (all metrics, all variants, all folds)
      - Table S3: **Per-genotype three-way triangulation** (fluorescence change DAG, attention peak DAG, human DAG)
      - Table S4: Per-genotype ranking comparison (predicted vs true)
      - Table S5: Distillation results (teacher vs student vs image-only, per metric)
    - Supplementary Figures: S1-S4 (from TODO 11)

  - Create `paper/data_availability.md`:
    - Describe dataset contents
    - Planned repository (e.g., Zenodo for data, GitHub for code)
    - Licensing

  - Final review checklist:
    - [ ] All numbers in paper match `results/*.json` files
    - [ ] All figure references correct
    - [ ] Word count within Nature MI limits
    - [ ] No "foundation model" claims (guardrail G6)
    - [ ] Limitations section present and honest
    - [ ] All co-author contributions described (if applicable)
    - [ ] Code availability statement present

  - Clean up codebase:
    - Remove debug prints
    - Ensure all scripts have `--help` documentation
    - Verify `requirements.txt` is complete and pinned
    - Write minimal `README.md` for code repository

  **Must NOT do**:
  - Do NOT create extensive documentation (minimal README sufficient)
  - Do NOT refactor code for aesthetics — functional is sufficient
  - Do NOT add features not in the paper

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Supplementary materials and final review are writing tasks
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (final integration task)
  - **Parallel Group**: Wave 5 (after TODO 11, 12)
  - **Blocks**: None (final task)
  - **Blocked By**: TODO 11 (figures), TODO 12 (paper)

  **References**:

  **All Results Files**:
  - `results/ablation_comparison.json` — ablation metrics
  - `results/distillation/aggregated_results.json` — distillation metrics
  - `results/analysis/triangulation_summary.json` — three-way triangulation results
  - `results/analysis/presymptomatic_summary.json` — XAI results
  - `results/analysis/ranking_results.json` — ranking results
  - `results/analysis/early_detection.json` — early detection results
  - `results/full_model/aggregated_results.json` — main model metrics
  - `results/baselines/*.json` — baseline metrics

  **Acceptance Criteria**:
  ```bash
  # Verify supplementary compiles
  pdflatex -interaction=nonstopmode paper/supplementary.tex 2>&1 | tail -5
  # Assert: no fatal errors

  # Verify README exists
  test -f README.md && echo "README exists" || echo "README missing"

  # Verify requirements.txt is complete
  pip install -r requirements.txt --dry-run 2>&1 | grep -c "already satisfied\|Would install"
  # Assert: no missing packages

  # Final number verification
  python3 -c "
  import json
  # Load paper-reported numbers (manually extracted or from a summary file)
  with open('results/ablation_comparison.json') as f:
      results = json.load(f)
  # Verify key metrics exist
  assert 'full_model' in results
  assert 'dag_mae' in results['full_model']
  print('Final verification PASSED')
  print(f'Best model DAG MAE: {results[\"full_model\"][\"dag_mae\"]:.2f}')
  print(f'Best model Ranking rho: {results[\"full_model\"][\"ranking_spearman\"]:.3f}')
  "
  ```

  **Commit**: YES
  - Message: `feat(paper): supplementary materials and final review`
  - Files: `paper/supplementary.tex`, `paper/data_availability.md`, `README.md`

---

## Commit Strategy

| After Task | Message | Key Files | Verification |
|------------|---------|-----------|--------------|
| 1 | `feat(scaffold): project structure and config system` | src/, configs/, requirements.txt | Config loads |
| 2 | `feat(data): canonical plant metadata and data validation` | src/data/metadata.py, data/*.csv | 264 plants validated |
| 3 | `feat(features): multi-backbone feature extraction (DINOv2, CLIP, BioCLIP)` | scripts/extract_features.py | All 3 feature HDF5 correct |
| 4 | `feat(data): PyTorch dataset with multi-view temporal batching` | src/data/dataset.py, src/training/cv.py | Dataset loads, CV splits |
| 5 | `feat(model): temporal multimodal architecture with teacher-student design` | src/model/*.py, src/training/losses.py | Forward pass correct |
| 6 | `feat(training): training pipeline with LOGO-CV and CSC SLURM` | src/training/trainer.py, scripts/train.py | Overfit test passes |
| 7 | `feat(baselines): classical ML baselines` | src/baselines/, scripts/run_baselines.py | Baseline metrics exist |
| 8 | `feat(experiments): ablation study complete` | configs/ablation/, results/ | All 11 result files exist (8 ablation + 3 baselines) |
| 8b | `feat(distillation): teacher-student knowledge distillation` | scripts/train_distill.py, results/distillation/ | Distillation gain computed |
| 9 | `feat(analysis): XAI + fluorescence triangulation` | src/analysis/, results/analysis/ | Triangulation stats |
| 10 | `feat(analysis): genotype ranking evaluation` | src/analysis/ranking.py | Ranking metrics exist |
| 11 | `feat(figures): all paper figures` | paper/figures/*.pdf | 10 figures exist |
| 12 | `feat(paper): Nature MI paper draft` | paper/main.tex | LaTeX compiles |
| 13 | `feat(paper): supplementary and final review` | paper/supplementary.tex, README.md | All compiles |

---

## Success Criteria

### Final Verification Commands
```bash
# 1. All experiment results exist (ablation + distillation + baselines)
python3 -c "
import json, os
for v in ['full_model','image_only','image_fluor','no_temporal','single_task','lora','clip_full','bioclip_full']:
    assert os.path.exists(f'results/{v}/aggregated_results.json'), f'Missing {v}'
assert os.path.exists('results/distillation/aggregated_results.json'), 'Missing distillation'
for b in ['xgboost_tabular','rf_tabular','dinov2_rf']:
    assert os.path.exists(f'results/baselines/{b}_results.json'), f'Missing {b}'
print('All experiment results present (8 ablation + 1 distillation + 3 baselines)')
"

# 2. DL model outperforms at least one classical baseline
python3 -c "
import json
with open('results/ablation_comparison.json') as f: r = json.load(f)
dl_mae = r['full_model']['dag_mae']
xgb_mae = r['xgboost_tabular']['dag_mae']
print(f'DL MAE: {dl_mae:.2f}, XGBoost MAE: {xgb_mae:.2f}')
if dl_mae < xgb_mae:
    print('DL outperforms XGBoost on DAG prediction')
else:
    print('WARNING: DL does not outperform XGBoost — check model or discuss in paper')
"

# 3. Three-way triangulation results
python3 -c "
import json
with open('results/analysis/triangulation_summary.json') as f: tri = json.load(f)
print(f'Correct ordering (fluor ≤ attention ≤ human): {tri[\"n_correct_ordering\"]}/44')
print(f'Model lead time: {tri[\"mean_lead_time_days\"]:.1f} days before human')
with open('results/analysis/presymptomatic_summary.json') as f: s = json.load(f)
print(f'Pre-symptomatic genotypes: {s[\"n_presymptomatic\"]}/44')
"

# 4. Distillation value quantified
python3 -c "
import json
with open('results/distillation/aggregated_results.json') as f: d = json.load(f)
with open('results/image_only/aggregated_results.json') as f: i = json.load(f)
gain = i['dag_mae'] - d['dag_mae']
print(f'Distillation gain (over image-only): {gain:.2f} MAE improvement')
"

# 5. Paper compiles
pdflatex -interaction=nonstopmode paper/main.tex && echo "Paper compiles OK"

# 6. All figures exist
ls paper/figures/fig*.pdf | wc -l
# Assert: ≥6 main figures
```

### Final Checklist
- [ ] All "Must Have" items present (see Work Objectives)
- [ ] All "Must NOT Have" items absent (see Guardrails G1-G12)
- [ ] 8 ablation variants (6 architecture + 2 backbone) + 3 baselines + 1 distillation have results
- [ ] **Three-way triangulation completed** (fluorescence ↔ attention ↔ human DAG)
- [ ] **Teacher-student distillation evaluated** (RGB student vs multimodal teacher vs image-only baseline)
- [ ] XAI analysis completed with pre-symptomatic quantification
- [ ] Genotype ranking evaluated against expert assessment
- [ ] 6 main figures + 4 supplementary figures generated
- [ ] Paper draft complete in Nature MI format (≤4,500 words)
- [ ] Paper narrative centered on **physiology→visibility gap** (not just "framework")
- [ ] Supplementary materials complete
- [ ] All scripts reproducible with config files
- [ ] Code documented with minimal README
