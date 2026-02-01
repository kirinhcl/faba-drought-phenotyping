# Temporal Multimodal Deep Learning for Faba Bean Drought Phenotyping
**Research Plan for Nature Machine Intelligence**

---

## Executive Summary

**Objective**: Develop a temporal multimodal deep learning framework that exploits the **physiology→visibility lag** in plant drought stress responses. The framework detects drought stress days before human expert observation, validated through three-way triangulation: model attention ↔ fluorescence measurements ↔ human annotation.

**Core Innovation**: Using chlorophyll fluorescence as **privileged information** in a teacher-student distillation paradigm, we transfer expensive-sensor knowledge to a deployable RGB-only early warning system.

**Timeline**: 30 days (code + experiments + paper draft)  
**Target**: Nature Machine Intelligence  
**Computational Resources**: Finnish CSC supercomputer (Mahti A100 GPUs)

---

## Scientific Contribution

### 1. Problem Statement

Plant drought stress manifests physiologically (cellular-level changes) hours to days before visible symptoms appear. Current phenotyping relies on visible-range assessment or expensive lab sensors. **No existing work** exploits this temporal lag computationally for early drought prediction.

### 2. Novel Approach

**Three-way temporal triangulation** to validate pre-symptomatic detection:
- **Clock 1** (Physiology): Fluorescence change point detection (when cellular stress begins)
- **Clock 2** (Model): Temporal attention peak (when model detects stress)  
- **Clock 3** (Human): Expert-annotated drought onset (when symptoms become visible)

**Expected ordering**: Clock 1 ≤ Clock 2 ≤ Clock 3 (if model truly detects physiological stress invisible to humans)

### 3. Key Contributions

1. **First temporal multimodal framework** exploiting physiology→visibility lag through privileged information learning
2. **Three-way validation** demonstrating model-detected signals align with independent physiological measurements and precede expert observation by X days
3. **Teacher-student distillation** transferring fluorescence-guided knowledge to RGB-only deployment (practical breeding application)
4. **Systematic backbone comparison** (self-supervised DINOv2 vs language-supervised CLIP vs domain-specific BioCLIP) — answering: *which vision foundation model pre-training strategy best captures plant phenotype features?*

---

## Dataset (Faba Bean Drought Experiment)

### Experimental Design
- **264 plants**: 44 Nordic faba bean accessions × 2 treatments (WHC-80% control, WHC-30% drought) × 3 biological replicates
- **6-week time course**: 22 imaging rounds (each plant imaged at ~11 timepoints, schedule varies by replicate)
- **Multi-view RGB imaging**: 3 side-view angles + 1 top-view per timepoint = **11,626 total images**

### Modalities
| Modality | Temporal Coverage | Details |
|----------|-------------------|---------|
| **RGB Images** | 22 rounds, ~11 per plant | 2560×3476 (side), 2560×1920 (top). Mixed RGB/RGBA requiring preprocessing. |
| **Chlorophyll Fluorescence** | 15 rounds, exactly 5 per plant | 93 parameters (Fv/Fm, NPQ, qP, Y(II), etc.). Sparse coverage requiring masking. |
| **Environment** | Minute-level, full experiment | 50,798 records: light, temperature, humidity (buffer + tunnel). Global, not per-plant. |
| **Watering** | Daily, DAG 4-39 | 9,506 events: water added, WHC tracking per plant. |
| **Endpoint Biomass** | Single measurement | Fresh/dry weight for all 264 plants (regression target). |
| **Drought Onset (DAG)** | Expert annotation | 44 genotype-level values (13 unique), categorized as Early/Mid/Late. Primary prediction target. |

### Data Challenges Addressed
- **Irregular temporal sampling**: Different replicates have different round schedules → canonical T=22 timeline with per-plant masks
- **Sparse fluorescence**: Only ~5 measurements per plant → learnable [MASK] tokens in transformer
- **Mixed image formats**: Both cameras contain RGB and RGBA → consistent alpha compositing pipeline
- **Unicode filenames**: Accented characters (e.g., "Mélodie") → NFC normalization required

---

## Methods Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT (per plant)                             │
│  • RGB images: (T=22, V=4, 768) from vision FM backbone             │
│  • Fluorescence: (T=22, 93) with mask                               │
│  • Environment: (T=22, 5)                                            │
│  • Watering: (T=22, 5)                                               │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│              Vision Foundation Model (Pre-extracted)                 │
│  • DINOv2-B/14 (primary) — self-supervised ViT-B/14, 768-dim       │
│  • OpenAI CLIP ViT-B/16 — language-supervised, 768-dim             │
│  • BioCLIP — biology-specific, TreeOfLife-10M fine-tuned, 768-dim  │
│  → Frozen (no fine-tuning; 264 plants insufficient)                │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   Multi-View Aggregation (Attention Pooling)         │
│  (T, V=4, 768) → (T, 768) via learnable attention over 4 views     │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│               Multimodal Fusion (Concat + Project)                   │
│  Image (768) + Fluor (93→128) + Env (5→128) + Water (5→128)        │
│  → (768+128+128+128 = 1152) → Linear → 256-dim fused tokens        │
│  Missing modalities replaced by learnable [MASK] tokens             │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│      Temporal Transformer (2 layers, 4 heads, dim=256)              │
│  • Continuous positional encoding using actual DAG values           │
│  • Prepend [CLS] token → sequence length 23                         │
│  • Attention mask excludes empty positions (no image + no fluor)    │
│  • CRITICAL: Store attention weights for XAI analysis               │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Task Prediction Heads                       │
│  1. DAG Regression: [CLS] → predicted drought onset (continuous)    │
│  2. DAG Classification: [CLS] → Early/Mid/Late (3-class ordinal)   │
│  3. Biomass: [CLS] → predicted FW, DW                               │
│  4. Stress Trajectory: temporal tokens → per-timepoint stress score │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Design Choices**:
- **Small model** (~2-3M trainable params excluding frozen vision FM) to prevent overfitting on n=264
- **Frozen backbone** (only temporal+fusion+heads are trained)
- **Continuous positional encoding** handles irregular temporal spacing naturally
- **Attention pooling** over views learns which angles are informative

### Teacher-Student Distillation

**Teacher** (Multimodal):  
Uses all modalities (RGB + fluorescence + environment + watering) → produces 256-dim CLS embedding

**Student** (RGB-only):  
Uses ONLY RGB images → same architecture otherwise → trained to match teacher's CLS embedding

**Distillation Loss**:  
`L = α·MSE(student_cls, teacher_cls.detach()) + (1-α)·L_task`

**Deployment Scenario**: Breeding programs can use RGB-only student for field screening (cheap cameras), achieving X% of multimodal teacher performance without expensive fluorescence sensors.

---

## Experimental Design

### Cross-Validation Strategy
**44-fold Leave-One-Genotype-Out (LOGO-CV)**:
- Each fold: test=1 genotype (6 plants), val=3 genotypes (18 plants), train=40 genotypes (240 plants)
- Val selection: deterministic, stratified (1 Early + 1 Mid + 1 Late genotype per fold)
- Fixed random seed (42) for reproducibility

### Ablation Studies (8 variants)

**Architecture Ablation** (all using DINOv2 backbone):
1. **Full model** — all modalities + temporal + multi-task (ceiling)
2. **Image-only** — remove fluorescence/environment/watering (isolates vision FM)
3. **Image + fluorescence** — remove environment/watering (shows physiology value)
4. **No temporal** — replace transformer with mean pooling (proves temporal is essential)
5. **Single-task (DAG only)** — remove biomass/trajectory heads (shows multi-task benefit)
6. **DINOv2 LoRA** — fine-tune backbone with LoRA rank=8 (tests if fine-tuning helps despite small n)

**Backbone Comparison** (all using full model architecture):
7. **CLIP backbone** — swap DINOv2 → OpenAI CLIP ViT-B/16 (language-supervised)
8. **BioCLIP backbone** — swap DINOv2 → BioCLIP (biology-specific)

**Key Question**: Does self-supervised (DINOv2) beat language-supervised (CLIP) or domain-specific (BioCLIP) for plant phenotyping?

### Classical Baselines
1. **XGBoost** on tabular features (morphology, vegetation indices, watering, fluorescence stats)
2. **Random Forest** on same tabular features
3. **DINOv2 + RF** (avg features, no temporal) — isolates temporal modeling contribution

### Evaluation Metrics
- **DAG prediction**: MAE, RMSE, R² (regression); accuracy, balanced accuracy, F1 (classification)
- **Biomass prediction**: MAE, R², Pearson r (separately for FW and DW)
- **Genotype ranking**: Spearman ρ, Kendall τ vs expert DAG ranking
- **Pre-symptomatic detection**: fraction of genotypes where model predicts ≥1 timepoint (≥3 days) before human DAG
- All metrics reported with **95% CI** from LOGO-CV (bootstrap across 44 folds)

---

## XAI & Validation: Three-Way Triangulation

### Fluorescence Change Point Detection
For each drought-treated plant:
1. Extract Fv/Fm (photosynthetic efficiency) over time (5 measurements)
2. Compare to control trajectory: `z_score = (drought_value - mean_control) / std_control`
3. `fluor_change_round` = first round where |z-score| > 2
4. Convert to DAG → **Clock 1** (physiological onset)

### Model Attention Peak
1. Extract CLS→temporal attention weights from trained model (averaged across heads and layers)
2. `attention_peak_dag` = DAG of max attention weight → **Clock 2** (model-detected onset)

### Three-Way Comparison (per genotype)
Compare:
- **Clock 1**: `fluor_change_dag` (median across 3 drought reps)
- **Clock 2**: `attention_peak_dag` (median across 3 drought reps)
- **Clock 3**: `human_dag` (expert annotation, from ground truth)

**Success Criterion**: Ordering `Clock 1 ≤ Clock 2 ≤ Clock 3` holds for ≥70% of genotypes

**Biological Interpretation**:
- If ordering holds → model detects **real physiological stress** invisible to humans
- If Clock 2 < Clock 1 → model may be detecting artifacts (red flag)
- If Clock 2 ≈ Clock 3 → model offers no early warning

**Negative Control**: WHC-80% (control) plants should show:
- No fluorescence change points
- No attention peaks at drought-typical timepoints

---

## Timeline & Deliverables (30 days)

### Wave 1 (Days 1-3): Data Foundation
- **TODO 1**: Project scaffolding + OmegaConf configuration system
- **TODO 2**: Canonical plant metadata reconciling all naming inconsistencies (WHC-70%→80%, Unicode normalization, round schedule per replicate)

### Wave 2 (Days 3-5): Feature Extraction
- **TODO 3**: Extract features from 3 vision foundation models
  - DINOv2-B/14 features → `features/dinov2_features.h5`
  - OpenAI CLIP ViT-B/16 features → `features/clip_features.h5`
  - BioCLIP features → `features/bioclip_features.h5`
  - All 768-dim, same HDF5 key structure
  - ~90 min total on CSC V100 GPU

### Wave 3 (Days 5-10): Model & Training
- **TODO 4**: PyTorch Dataset (T=22 canonical timeline, per-plant masks, multimodal fusion)
- **TODO 5**: Model architecture (teacher + student, multi-task heads)
- **TODO 6**: Training pipeline + LOGO-CV (44 folds, wandb logging, CSC SLURM scripts)
- **TODO 7**: Classical ML baselines (XGBoost, RF, DINOv2+RF)

### Wave 4 (Days 10-20): Experiments & Analysis
- **TODO 8**: Run all experiments on CSC
  - 8 ablation variants × 44 folds = 352 jobs (~2.5h with full parallelism)
  - 3 classical baselines (~30 min)
- **TODO 8b**: Teacher-student distillation (fluorescence as privileged info)
- **TODO 9**: XAI + fluorescence triangulation analysis
  - Attention peak extraction
  - Fluorescence change point detection (z-score > 2 threshold)
  - Three-way ordering validation
  - Progressive temporal truncation (early detection curve)
- **TODO 10**: Genotype ranking evaluation (Spearman ρ vs expert)

### Wave 5 (Days 20-30): Paper & Figures
- **TODO 11**: Generate all figures (6 main + 4 supplementary)
  - **Fig 1**: Architecture diagram
  - **Fig 2**: Ablation study + backbone comparison (8 variants + 3 baselines)
  - **Fig 3**: Three-way triangulation (heatmap + scatter + timeline)
  - **Fig 4**: Early detection curve (accuracy vs DAG cutoff)
  - **Fig 5**: Genotype ranking (predicted vs true)
  - **Fig 6**: Teacher-student distillation (performance comparison + embedding space)
- **TODO 12**: Nature MI paper draft (≤4,500 words main text)
- **TODO 13**: Supplementary materials + final review

---

## Expected Results & Discussion Points

### Hypothesis 1: Temporal Modeling is Essential
**Prediction**: Full model (variant 1) >> No temporal (variant 4)  
If confirmed → temporal dynamics carry critical drought signal beyond static features

### Hypothesis 2: Multimodal Fusion Adds Value
**Prediction**: Full model (variant 1) > Image-only (variant 2)  
If confirmed → fluorescence/environment/watering provide complementary signals

### Hypothesis 3: Pre-Symptomatic Detection is Real
**Prediction**: 
- Model attention peaks ≥3 days before human DAG for ≥70% of genotypes
- Attention peaks correlate with fluorescence change points (r > 0.6)

If confirmed → model detects physiological stress invisible to experts

### Hypothesis 4: Distillation Transfers Fluorescence Knowledge
**Prediction**: RGB-only student > Image-only baseline (variant 2)  
**Distillation gap**: Teacher performance - Student performance = irreducible fluorescence information

### Hypothesis 5: Self-Supervised Outperforms Language-Supervised
**Prediction**: DINOv2 (variant 1) ≥ CLIP (variant 7) > BioCLIP (variant 8)  
**Rationale**: 
- DINOv2 learns visual features without language bias → better fine-grained plant phenotypes?
- CLIP optimized for natural language alignment → may miss subtle morphological traits
- BioCLIP trained on TreeOfLife species classification → may not transfer to phenotyping

**If BioCLIP wins** → domain-specific pre-training critical (implication: agriculture needs AgriCLIP/PlantCLIP)  
**If DINOv2 wins** → self-supervised vision is sufficient (implication: focus on architecture, not backbone)

---

## Limitations & Future Work

### Current Limitations
1. **Small dataset** (n=264 plants, 44 genotypes) → generalization to other species/environments unknown
2. **Single location, single season** → environmental variation not captured
3. **Temporal resolution** limited by imaging/fluorescence schedule (~3-7 day gaps)
4. **Attention ≠ explanation** — we validate with fluorescence, but causality is not proven
5. **No field validation** — phenotyping platform ≠ field conditions

### Future Directions
1. **Multi-environment extension**: Repeat experiment across locations/seasons (requires new data collection)
2. **Transfer learning**: Test framework on other crops (wheat, barley, pea) with minimal re-training
3. **Spatial XAI**: Extend to per-pixel attention (GradCAM on leaf/stem regions) for localized stress detection
4. **Real-time deployment**: Optimize student model for edge devices (Raspberry Pi + camera)
5. **Genomic integration**: Link predicted drought onset to genomic markers (GWAS) for breeding

---

## Key Literature Context

### What Exists (Gaps We Fill)
| Work | Contribution | What's Missing |
|------|-------------|----------------|
| **Wu et al. 2021** (Genome Biology) | 368 maize genotypes, classical image analysis, GWAS | ❌ No deep learning, ❌ No temporal, ❌ No pre-symptomatic |
| **AgriCLIP 2025** (COLING) | Agriculture-specific CLIP | ❌ No temporal, ❌ No drought-specific |
| **BioCLIP 2024** (CVPR) | Biology-specific vision model | ❌ No temporal, ❌ No phenotyping validation |
| **Choudhury et al. 2023** | DTW-based drought prediction | ❌ Classical methods, ❌ No FM, ❌ No multimodal |

### What's Novel (Our Contribution)
✅ **First temporal multimodal framework** with vision FM for drought  
✅ **First** to exploit physiology→visibility lag computationally  
✅ **First** three-way validation (fluorescence ↔ model ↔ human)  
✅ **First** teacher-student distillation for sensor-to-RGB deployment  
✅ **First** systematic FM backbone comparison (self-supervised vs language-supervised vs domain-specific) for plant phenotyping

---

## Paper Framing for Nature Machine Intelligence

### Title (working)
"Bridging the physiology-visibility gap: temporal deep learning reveals pre-symptomatic drought signals through fluorescence-guided privileged learning"

### Abstract Structure
1. **Problem**: Drought detection relies on visible symptoms, missing early physiological responses
2. **Approach**: Temporal multimodal framework + three-way triangulation validation
3. **Results**: Model detects stress X days before experts, validated by fluorescence (≥70% ordering agreement)
4. **Impact**: Distillation enables RGB-only deployment; backbone comparison reveals self-supervised > language-supervised for phenotyping

### Main Messages
- **Methodological**: Privileged information + distillation paradigm for sensor-to-RGB transfer
- **Biological**: Quantification of physiology→visibility lag (X days ± CI)
- **Practical**: Framework enables early intervention in breeding programs
- **Comparative**: Self-supervised vision FM (DINOv2) outperforms domain-specific (BioCLIP) — surprising? Validates FM generalization or highlights biology training gap?

---

## Questions for Discussion

1. **Baseline expectations**: What DAG MAE would you consider "strong" performance? (For reference: 13 unique DAG values, range ~10-30 DAG)

2. **Triangulation threshold**: Is 70% genotypes following the expected ordering (fluor ≤ attention ≤ human) sufficient to claim pre-symptomatic detection? Or should we aim higher?

3. **Backbone comparison**: If BioCLIP (biology-trained) loses to DINOv2 (general-purpose), is this a problem with our framework or a genuine finding about pre-training strategies?

4. **Paper narrative**: Lead with methodology (privileged information) or biology (physiology-visibility gap)? Nature MI is methodology-first, but plant science reviewers may prefer biology framing.

5. **Future work priority**: Multi-environment validation vs. spatial XAI vs. real-time deployment? (All important, but timeline is 30 days)

---

## Appendix: Data Files Reference

### Key Excel Files (All Verified to Exist)
- `data/00-Misc/FabaDr_Obs.xlsx` — Master observation log (RGB1: 8,717 rows, FC: 1,320 rows)
- `data/00-Misc/DateIndex.xlsx` — Date↔DAS↔DAG mapping (70 rows)
- `data/00-Misc/EndPoint_Raw_FW&DW.xlsx` — Endpoint biomass (264 plants)
- `data/SinglePoint Datasets/Drought_Impact(DAG).xlsx` — Expert DAG annotation (44 genotypes)
- `data/TimeCourse Datasets/FCQ_FabaDr_Auto.xlsx` — Fluorescence (1,320 rows, 111 columns, ~96 fluorescence params)
- `data/TimeCourse Datasets/EnvData_FabaDr.xlsx` — Environment (50,798 minute-level records, 5 features)
- `data/TimeCourse Datasets/SC_Watering_24_FabaDr_Auto.xlsx` — Watering (9,506 events, 24 columns)
- `data/TimeCourse Datasets/DigitalBiomass_Norm_FabaDr_Auto.xlsx` — Normalized biomass (2,854 rows, trajectory target)
- `data/EndPoint Datasets/EndPoint_CorrelationData-WithoutOutliers.xlsx` — 29-feature aggregated matrix (baseline features)

### Canonical Round→DAG Mapping (Fixed Throughout)
```python
ROUND_TO_DAG = {
    2: 4,   3: 5,   4: 6,   5: 7,   6: 10,  7: 12,
    8: 13,  9: 14, 10: 17, 11: 19, 12: 20, 13: 21,
   14: 24, 15: 27, 16: 28, 17: 29, 18: 31, 19: 33,
   20: 34, 21: 35, 22: 38, 23: 38
}
```
**Note**: Rounds 22 & 23 share DAG=38 (same date, different imaging batches)

---

**End of Research Plan**  
*Last updated: 2026-02-01*  
*Total estimated effort: 30 days full-time*  
*Computational budget: ~60h GPU time (parallelizable to ~3h wall time)*
