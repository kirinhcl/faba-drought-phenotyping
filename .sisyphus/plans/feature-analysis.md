# Feature Analysis Visualizations for Faba Drought Paper

## TL;DR

> **Quick Summary**: Add 3 feature analysis figures to the paper — DINOv2 embedding t-SNE/UMAP, fluorescence-drought correlation heatmap, and temporal feature evolution curves — to strengthen the Results and Discussion sections.
> 
> **Deliverables**:
> - Fig 6: DINOv2 embedding space visualization (t-SNE + UMAP)
> - Fig 7: Fluorescence parameter-drought correlation heatmap
> - Fig 8: Temporal evolution curves of key fluorescence parameters
> - Updated `paper/main.tex` with new figures and text
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 (sync data) → Task 2/3/4 (parallel figures) → Task 5 (paper integration)

---

## Context

### Original Request
User wants feature analysis visualizations to complement existing Results + Discussion: DINOv2 embedding distributions, fluorescence parameter correlations with drought, and temporal feature evolution curves.

### Interview Summary
**Key Decisions**:
- Embeddings: Raw DINOv2 768-dim per image (~11,000 points), NOT model CLS
- Fluorescence correlation: Heatmap of 94 JIP-test params × timepoints
- Temporal curves: Classic photosynthetic params + data-driven top discriminative params
- Correlation target: Binary treatment label (WHC-30 vs WHC-80)
- Confidence bands: Mean ± 95% CI
- Paper placement: Main text Results + Discussion sections
- Data source: DINOv2 features from Mahti `/scratch/project_2013932/chenghao/faba-drought-phenotyping/features`

### Metis Review
**Identified Gaps** (addressed):
- Embedding type clarified: Raw DINOv2, not model CLS (user confirmed)
- Fluorescence correlation metric: Use point-biserial (binary label) → Spearman (more robust)
- PI_abs may not exist in Excel columns → check actual column names, map to available params
- Need to install `umap-learn` and `scikit-learn` in `.venv`
- ROUND_TO_DAG mapping must be consistent with `src/data/dataset.py`
- Must handle dead/missing plants
- 94 params too many for readable heatmap → cluster and filter

---

## Work Objectives

### Core Objective
Generate 3 publication-quality feature analysis figures that provide mechanistic insight into how different data modalities relate to drought stress, strengthening the paper's analytical depth.

### Concrete Deliverables
- `paper/figures/generate_fig6_embeddings.py` + `fig6_embeddings.pdf`
- `paper/figures/generate_fig7_fluorescence_heatmap.py` + `fig7_fluorescence_heatmap.pdf`
- `paper/figures/generate_fig8_temporal_curves.py` + `fig8_temporal_curves.pdf`
- Updated `paper/main.tex` with figure inclusions and text

### Definition of Done
- [ ] All 3 figures generate without errors via `.venv/bin/python3`
- [ ] All 3 figures embedded in PDF, paper compiles clean
- [ ] Results section has new subsections describing each figure
- [ ] Discussion references new findings

### Must Have
- DINOv2 features synced from Mahti
- Consistent color scheme with existing figures
- Both PDF and PNG output at 300 DPI
- Error bars / confidence intervals where applicable
- Proper axis labels with units

### Must NOT Have (Guardrails)
- Do NOT change existing figure numbering (Fig 1-5 unchanged)
- Do NOT modify existing figure scripts
- Do NOT include genotype-level breakdowns (too complex for main figures)
- Do NOT use interactive plots (static publication format only)
- Do NOT add supplementary figures (all in main text)

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (`.venv/` with matplotlib, seaborn)
- **Automated tests**: NO (data analysis scripts, not testable code)
- **Agent-Executed QA**: YES (verify script execution + output files)

### Agent-Executed QA Scenarios (MANDATORY)

**For each figure script:**

```
Scenario: Figure generation succeeds
  Tool: Bash
  Steps:
    1. .venv/bin/python3 paper/figures/generate_fig{N}_{name}.py
    2. Assert: exit code 0
    3. Assert: paper/figures/fig{N}_{name}.pdf exists and > 10KB
    4. Assert: paper/figures/fig{N}_{name}.png exists and > 50KB
  Expected Result: Script completes, outputs valid PDF and PNG

Scenario: Paper compiles with new figures
  Tool: Bash
  Steps:
    1. cd paper && pdflatex -interaction=nonstopmode main.tex
    2. Assert: "Output written" in stdout
    3. Assert: main.pdf exists and > 400KB
  Expected Result: No LaTeX errors, PDF generated
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Sync DINOv2 features from Mahti + install deps
└── (blocks all other tasks)

Wave 2 (After Wave 1):
├── Task 2: Generate Fig 6 (DINOv2 embeddings t-SNE/UMAP)
├── Task 3: Generate Fig 7 (Fluorescence correlation heatmap)
└── Task 4: Generate Fig 8 (Temporal evolution curves)

Wave 3 (After Wave 2):
└── Task 5: Integrate figures into paper + update text

Critical Path: Task 1 → Tasks 2,3,4 → Task 5
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3, 4 | None |
| 2 | 1 | 5 | 3, 4 |
| 3 | 1 | 5 | 2, 4 |
| 4 | 1 | 5 | 2, 3 |
| 5 | 2, 3, 4 | None | None |

---

## TODOs

- [x] 1. Sync DINOv2 features from Mahti + install dependencies

  **What to do**:
  - Rsync DINOv2 feature files from Mahti: `rsync -avz mahti.csc.fi:/scratch/project_2013932/chenghao/faba-drought-phenotyping/features/ features/`
  - Check what files were synced: `ls -lh features/`
  - Install missing Python packages: `.venv/bin/pip install umap-learn scikit-learn openpyxl`
  - Verify imports work: `.venv/bin/python3 -c "import umap; import sklearn; import openpyxl; print('OK')"`
  - Inspect the DINOv2 feature file structure (HDF5 keys, shapes, etc.)

  **Must NOT do**:
  - Do NOT modify any files on Mahti
  - Do NOT re-extract features (they already exist)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`git-master`]
    - `git-master`: SSH/rsync operations

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (prerequisite for all others)
  - **Blocks**: Tasks 2, 3, 4
  - **Blocked By**: None

  **References**:
  - `scripts/extract_features.py` — Shows HDF5 key structure: keys are `{plant_id}_{round}_{view}`, values are 768-dim float32 arrays
  - `src/data/dataset.py:28-33` — ROUND_TO_DAG mapping for joining features with metadata
  - `data/plant_metadata.csv` — Plant IDs, treatments, accessions

  **Acceptance Criteria**:
  - [ ] `ls features/` shows HDF5 or NPY files with DINOv2 embeddings
  - [ ] `.venv/bin/python3 -c "import umap, sklearn, openpyxl"` exits 0
  - [ ] Feature file is loadable and contains expected plant IDs

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Features synced successfully
    Tool: Bash
    Steps:
      1. rsync -avz mahti.csc.fi:/scratch/project_2013932/chenghao/faba-drought-phenotyping/features/ features/
      2. ls -lh features/
      3. Assert: at least one .h5 or .npy file exists
    Expected Result: Feature files present locally
    Evidence: ls output captured

  Scenario: Dependencies installed
    Tool: Bash
    Steps:
      1. .venv/bin/pip install umap-learn scikit-learn openpyxl
      2. .venv/bin/python3 -c "import umap; import sklearn; import openpyxl; print('All imports OK')"
      3. Assert: stdout contains "All imports OK"
    Expected Result: All packages importable
    Evidence: Import verification output
  ```

  **Commit**: NO

---

- [x] 2. Generate Fig 6: DINOv2 Embedding Space Visualization

  **What to do**:
  - Create `paper/figures/generate_fig6_embeddings.py`
  - Load DINOv2 features (768-dim) from local `features/` directory
  - Load plant metadata from `data/plant_metadata.csv` to get treatment labels, accession names, timepoints
  - Aggregate: Mean across 4 side views per plant-timepoint → ~5,800 points (264 plants × ~22 timepoints)
  - Compute t-SNE (perplexity=30) and UMAP (n_neighbors=15) on the aggregated embeddings
  - Create a 1×2 figure (full-width, `figure*`):
    - Left panel: t-SNE colored by treatment (WHC-80=blue `#3498DB`, WHC-30=red `#E74C3C`)
    - Right panel: UMAP colored by treatment (same colors)
  - Add alpha=0.3 for transparency (many overlapping points)
  - Add legend: "Control (WHC-80)" and "Drought (WHC-30)"
  - Style: `sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)`
  - Save: `fig6_embeddings.pdf` + `.png` at 300 DPI

  **Must NOT do**:
  - Do NOT color by genotype (44 colors = unreadable)
  - Do NOT use model CLS embeddings (use raw DINOv2)
  - Do NOT include top-view images (only 4 side views)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: Publication-quality visualization design

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 4)
  - **Blocks**: Task 5
  - **Blocked By**: Task 1

  **References**:
  - `scripts/extract_features.py` — HDF5 structure and key naming convention
  - `src/data/dataset.py:28-33` — ROUND_TO_DAG mapping
  - `data/plant_metadata.csv` — treatment labels, accession names
  - `paper/figures/generate_fig2.py` — Style template to follow (colors, theme, save pattern)
  - `src/analysis/embedding_viz.py` — t-SNE/UMAP code reference (but uses different data source)

  **Acceptance Criteria**:
  - [ ] Script runs: `.venv/bin/python3 paper/figures/generate_fig6_embeddings.py` exits 0
  - [ ] `paper/figures/fig6_embeddings.pdf` exists, size 30-200 KB
  - [ ] `paper/figures/fig6_embeddings.png` exists, size 100-1000 KB
  - [ ] Two panels visible: t-SNE (left), UMAP (right)
  - [ ] Two colors visible: blue (control) and red (drought)

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Embedding visualization generates correctly
    Tool: Bash
    Steps:
      1. .venv/bin/python3 paper/figures/generate_fig6_embeddings.py
      2. Assert: exit code 0
      3. ls -lh paper/figures/fig6_embeddings.pdf
      4. Assert: file size > 20KB
    Expected Result: PDF generated with two-panel scatter plot
    Evidence: paper/figures/fig6_embeddings.png for visual inspection
  ```

  **Commit**: YES (groups with Task 3, 4)
  - Message: `feat(paper): add feature analysis figures (Fig 6-8)`
  - Files: `paper/figures/generate_fig6_embeddings.py`, `paper/figures/fig6_embeddings.pdf`, `paper/figures/fig6_embeddings.png`

---

- [x] 3. Generate Fig 7: Fluorescence Parameter-Drought Correlation Heatmap

  **What to do**:
  - Create `paper/figures/generate_fig7_fluorescence_heatmap.py`
  - Load fluorescence data from `data/TimeCourse Datasets/FCQ_FabaDr_Auto.xlsx`
  - Load plant metadata from `data/plant_metadata.csv` for treatment labels
  - Strip whitespace from Plant IDs when merging
  - For each timepoint and each fluorescence parameter, compute Spearman rank correlation between parameter value and binary drought label (WHC-30=1, WHC-80=0) across all 264 plants
  - Build correlation matrix: rows = fluorescence parameters (94), cols = timepoints (22 DAG values)
  - **Filtering**: Keep only params with |max correlation| > 0.3 at any timepoint (removes noise params)
  - Cluster rows (params) by hierarchical clustering for visual grouping
  - Plot as seaborn `clustermap` or `heatmap`:
    - Color: diverging colormap (RdBu_r), range [-1, +1]
    - X-axis: DAG values (4, 5, 6, ..., 38)
    - Y-axis: Parameter names
    - Annotate cells with correlation values if |r| > 0.5
  - Title: "Spearman Correlation Between Fluorescence Parameters and Drought Treatment"
  - Figure should be full-width (`figure*`) due to many columns
  - Save: `fig7_fluorescence_heatmap.pdf` + `.png` at 300 DPI

  **Must NOT do**:
  - Do NOT use Pearson (fluorescence data is often non-normal)
  - Do NOT show all 94 params if most are < 0.3 correlation
  - Do NOT include derived/computed params beyond what's in the Excel

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: [`frontend-ui-ux`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 4)
  - **Blocks**: Task 5
  - **Blocked By**: Task 1

  **References**:
  - `data/TimeCourse Datasets/FCQ_FabaDr_Auto.xlsx` — Raw fluorescence data, 94 JIP-test columns
  - `data/plant_metadata.csv` — Treatment labels for each plant
  - `src/data/dataset.py:28-33` — ROUND_TO_DAG for x-axis labels
  - `src/analysis/fluorescence_changepoint.py` — Shows how fluorescence data is loaded and processed
  - `paper/figures/generate_fig3_gates.py` — Heatmap style reference

  **Acceptance Criteria**:
  - [ ] Script runs: `.venv/bin/python3 paper/figures/generate_fig7_fluorescence_heatmap.py` exits 0
  - [ ] `paper/figures/fig7_fluorescence_heatmap.pdf` exists, size 30-200 KB
  - [ ] Heatmap shows diverging colors (blue negative, red positive)
  - [ ] At least 15 params shown (those with |r| > 0.3)
  - [ ] DAG values on x-axis, parameter names on y-axis

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Heatmap generates with correct dimensions
    Tool: Bash
    Steps:
      1. .venv/bin/python3 paper/figures/generate_fig7_fluorescence_heatmap.py
      2. Assert: exit code 0
      3. Assert: stdout includes "Loaded X fluorescence parameters"
      4. Assert: stdout includes "Keeping N parameters with |r| > 0.3"
      5. ls -lh paper/figures/fig7_fluorescence_heatmap.pdf
      6. Assert: file size > 20KB
    Expected Result: Filtered heatmap with meaningful correlations
    Evidence: paper/figures/fig7_fluorescence_heatmap.png
  ```

  **Commit**: YES (groups with Task 2, 4)

---

- [x] 4. Generate Fig 8: Temporal Evolution Curves of Key Fluorescence Parameters

  **What to do**:
  - Create `paper/figures/generate_fig8_temporal_curves.py`
  - Load fluorescence data from `data/TimeCourse Datasets/FCQ_FabaDr_Auto.xlsx`
  - Load plant metadata for treatment labels
  - **Classic params** (check actual column names in Excel, use closest match):
    - Fv/Fm (maximum quantum yield of PSII)
    - PI_abs (Performance Index on absorption basis)
    - φEo or Phi_Eo (quantum yield for electron transport)
    - ψEo or Psi_Eo (probability of electron transport)
  - **Data-driven params**: From Task 3's correlation analysis, pick the Top 3 params with highest |Spearman r| that are NOT in the classic list
  - For each parameter:
    - Group by treatment (WHC-80 vs WHC-30)
    - Compute mean ± 95% CI at each timepoint across all plants in the group
  - Create a multi-panel figure (2×3 or 2×4 grid):
    - Each panel = one parameter
    - X-axis: DAG (4-38)
    - Y-axis: Normalized parameter value (z-score within each param)
    - Blue line + band: WHC-80 mean ± 95% CI
    - Red line + band: WHC-30 mean ± 95% CI
    - Vertical dashed line: Mean drought onset DAG (if relevant)
  - Figure: full-width (`figure*`)
  - Save: `fig8_temporal_curves.pdf` + `.png` at 300 DPI

  **Must NOT do**:
  - Do NOT show raw (unnormalized) values (params have very different scales)
  - Do NOT plot per-genotype curves (use population mean)
  - Do NOT show more than 8 panels (keep readable)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: [`frontend-ui-ux`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 3)
  - **Blocks**: Task 5
  - **Blocked By**: Task 1

  **References**:
  - `data/TimeCourse Datasets/FCQ_FabaDr_Auto.xlsx` — Raw fluorescence, need to discover exact column names
  - `data/plant_metadata.csv` — Treatment labels
  - `src/data/dataset.py:28-33` — ROUND_TO_DAG mapping
  - `paper/figures/generate_fig4_curves.py` — Time-series plot style reference (colors, bands)
  - `src/analysis/fluorescence_changepoint.py:30-50` — Shows fluorescence column name patterns

  **Acceptance Criteria**:
  - [ ] Script runs: `.venv/bin/python3 paper/figures/generate_fig8_temporal_curves.py` exits 0
  - [ ] `paper/figures/fig8_temporal_curves.pdf` exists, size 30-200 KB
  - [ ] 6-8 panels visible, each with blue (control) and red (drought) curves
  - [ ] 95% CI bands visible around each line
  - [ ] X-axis shows DAG values, Y-axis shows normalized values

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Temporal curves with dual treatment bands
    Tool: Bash
    Steps:
      1. .venv/bin/python3 paper/figures/generate_fig8_temporal_curves.py
      2. Assert: exit code 0
      3. Assert: stdout includes "Classic params: [list]"
      4. Assert: stdout includes "Data-driven params: [list]"
      5. ls -lh paper/figures/fig8_temporal_curves.pdf
      6. Assert: file size > 20KB
    Expected Result: Multi-panel figure with temporal evolution
    Evidence: paper/figures/fig8_temporal_curves.png
  ```

  **Commit**: YES (groups with Tasks 2, 3)

---

- [ ] 5. Integrate Figures into Paper and Update Text

  **What to do**:
  - Add `\begin{figure*}` blocks for Fig 6, 7, 8 in `paper/main.tex`
  - Fig 6: After "Adaptive Gating and Modality Importance Dynamics" subsection (Results)
  - Fig 7: After Fig 6, in a new subsection "Fluorescence Parameter Analysis"
  - Fig 8: After Fig 7, in the same or following subsection
  - Write 1-2 paragraphs in Results describing each figure's findings
  - Add 1 paragraph in Discussion connecting feature analysis to existing narrative
  - Recompile paper: `pdflatex` + `bibtex` + 2× `pdflatex`
  - Verify: no LaTeX errors, all figures visible in PDF

  **Must NOT do**:
  - Do NOT change existing text in Results/Discussion (only ADD)
  - Do NOT renumber existing figures
  - Do NOT add figures to abstract or conclusion

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential, after all figures done)
  - **Blocks**: None (final task)
  - **Blocked By**: Tasks 2, 3, 4

  **References**:
  - `paper/main.tex` — Current paper, Figures 1-5 already integrated
  - Section after line ~250 (after "Adaptive Gating" subsection) — insertion point for new figures
  - Existing `\begin{figure*}` blocks — Follow same LaTeX pattern
  - `paper/figures/fig{6,7,8}_*.pdf` — Generated figures

  **Acceptance Criteria**:
  - [ ] `pdflatex` compiles without errors
  - [ ] `bibtex` resolves without missing references
  - [ ] `paper/main.pdf` includes all 8 figures
  - [ ] New subsection text present in Results
  - [ ] New paragraph present in Discussion

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Paper compiles with all 8 figures
    Tool: Bash
    Preconditions: All figure PDFs exist in paper/figures/
    Steps:
      1. cd paper
      2. pdflatex -interaction=nonstopmode main.tex
      3. bibtex main
      4. pdflatex -interaction=nonstopmode main.tex
      5. pdflatex -interaction=nonstopmode main.tex
      6. Assert: "Output written" in stdout
      7. Assert: main.pdf exists and > 500KB
      8. grep -c "fig6_embeddings" main.tex → 1+
      9. grep -c "fig7_fluorescence" main.tex → 1+
      10. grep -c "fig8_temporal" main.tex → 1+
    Expected Result: Clean compilation with all figures
    Evidence: Paper page count increased (14+ pages)
  ```

  **Commit**: YES
  - Message: `feat(paper): integrate feature analysis figures and text (Fig 6-8)`
  - Files: `paper/main.tex`, `paper/main.pdf`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 2, 3, 4 | `feat(paper): add feature analysis figures (Fig 6-8)` | `paper/figures/generate_fig{6,7,8}_*.py`, `paper/figures/fig{6,7,8}_*.{pdf,png}` | Scripts run without errors |
| 5 | `feat(paper): integrate feature analysis into text` | `paper/main.tex` | pdflatex compiles clean |

---

## Success Criteria

### Verification Commands
```bash
# All figure scripts run
.venv/bin/python3 paper/figures/generate_fig6_embeddings.py
.venv/bin/python3 paper/figures/generate_fig7_fluorescence_heatmap.py
.venv/bin/python3 paper/figures/generate_fig8_temporal_curves.py

# All outputs exist
ls -lh paper/figures/fig{6,7,8}_*.pdf  # 3 PDFs, each 20-200 KB

# Paper compiles
cd paper && pdflatex -interaction=nonstopmode main.tex  # Expected: "Output written"
```

### Final Checklist
- [ ] 3 new figure scripts created and executable
- [ ] 3 new figure PDFs generated (publication quality)
- [ ] Paper compiles with all 8 figures
- [ ] Results section updated with new subsections
- [ ] Discussion section updated with feature analysis insights
- [ ] All figures use consistent color palette with existing Figs 1-5
