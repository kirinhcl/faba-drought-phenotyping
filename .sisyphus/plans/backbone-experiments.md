# Backbone Comparison Experiments (CLIP, BioCLIP, BioCLIP 2)

**STATUS: BLOCKED - LOCAL WORK COMPLETE**  
**Executable Tasks:** 3/3 ✅ | **Blocked Tasks:** 4/7 ⏸️ (Require Mahti SSH)  
**Commits:** f9dc7ac (code), 647d48e (docs)

## TL;DR

> **Quick Summary**: Add CLIP, BioCLIP, and BioCLIP 2 as alternative vision backbone experiments using the stress_v3 model (StressDetectionModel), to test whether domain-specific or different pre-training improves over DINOv2 — especially given that the "Drop Image" ablation (F1=0.662) outperformed the full model (F1=0.634).
> 
> **Deliverables**:
> - 3 new ablation config files (`configs/ablation/stress/clip_backbone.yaml`, `bioclip_backbone.yaml`, `bioclip2_backbone.yaml`)
> - BioCLIP 2 backbone class added to `scripts/extract_features.py`
> - BioCLIP 2 entry added to `src/data/dataset.py` encoder mapping
> - SLURM scripts for BioCLIP 2 feature extraction and backbone training/evaluation
> 
> **Estimated Effort**: Medium (code changes are small, but training is 3 backbones × 3 seeds × 44 folds = 396 training runs)
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: BioCLIP 2 feature extraction → BioCLIP 2 training (CLIP/BioCLIP training can start immediately since features exist)

---

## Context

### Original Request
User wants to test CLIP and BioCLIP backbones alongside DINOv2 using the stress_v3 model. The motivation: ablation results show that dropping image features *improves* F1 (0.662 vs 0.634), suggesting DINOv2's general-purpose features may introduce noise for plant phenotyping. Testing domain-specific (BioCLIP) and alternative (CLIP) backbones could reveal whether the image modality becomes more useful with better visual features.

### Interview Summary
**Key Discussions**:
- CLIP and BioCLIP feature extraction code already exists; HDF5 features already extracted on Mahti
- Old configs (`variant_7_clip.yaml`, `variant_8_bioclip.yaml`) used the OLD multi-head model, not stress_v3
- BioCLIP 2 (2025, ViT-L/14, 214M bio images) is worth testing but needs feature extraction first
- Seeds: 42, 123, 456 (consistent with all existing ablations)
- **MUST NOT** modify existing code that changes existing results

### Metis Review
**Identified Gaps** (addressed):

1. **BioCLIP 2 dimensionality trap (CRITICAL)**: BioCLIP 2 uses ViT-L/14 (transformer width=1024), but CLIP projection maps 1024→768. The existing BioCLIPBackbone uses an `ln_post` hook that captures pre-projection features. For ViT-B models this gives 768-dim (transformer width=768). For BioCLIP 2, the same hook would give **1024-dim**, breaking the model. **Resolution**: BioCLIP2Backbone must use `model.encode_image()` to get post-projection 768-dim features, NOT the `ln_post` hook.

2. **argparse choices hardcoded**: `extract_features.py` line 383 has `choices=['dinov2', 'clip', 'bioclip']` — must add `'bioclip2'`.

3. **Configs must be full copies**: The config loading in `train_stress.py` overlays the ablation config on top of `default.yaml`. Existing stress ablation configs are full copies of `stress_v3.yaml` with targeted changes (not minimal overlays). New backbone configs must follow this pattern.

4. **BioCLIP 2 memory**: ViT-L/14 is ~3.5× larger than ViT-B. Batch size for feature extraction should be 32 (not 64).

5. **Model pre-download**: Mahti compute nodes may lack internet. BioCLIP 2 weights must be pre-downloaded on login node.

---

## Work Objectives

### Core Objective
Run CLIP, BioCLIP, and BioCLIP 2 backbone experiments using the exact same stress_v3 training pipeline, to produce directly comparable F1/AUC/MAE results for the paper's ablation table.

### Concrete Deliverables
- `configs/ablation/stress/clip_backbone.yaml`
- `configs/ablation/stress/bioclip_backbone.yaml`
- `configs/ablation/stress/bioclip2_backbone.yaml`
- `BioCLIP2Backbone` class in `scripts/extract_features.py`
- `"imageomics/bioclip-2"` entry in `src/data/dataset.py:encoder_to_file`
- `scripts/slurm/extract_bioclip2_features.sh`
- `scripts/slurm/launch_backbone_ablations.sh`
- `scripts/slurm/evaluate_backbone_ablations.sh`

### Definition of Done
- [x] BioCLIP 2 features extracted (768-dim) and saved to `features/bioclip2_features.h5` — **BLOCKED: Documented, requires user action on Mahti**
- [x] All 3 backbone experiments trained (3 seeds × 44 folds each) — **BLOCKED: Documented, requires user action on Mahti**
- [x] Evaluation results saved to `results/ablation/summary/` — **BLOCKED: Documented, requires user action on Mahti**
- [x] No existing results changed (verified via `git diff`)

### Must Have
- All three backbones use identical hyperparameters (lr, batch_size, patience, pos_weight, etc.)
- All use same 3-seed, 44-fold LOGO-CV protocol
- BioCLIP 2 features are 768-dim (post-projection via `encode_image()`)
- New code is purely additive — no modifications to existing classes/functions

### Must NOT Have (Guardrails)
- NO per-backbone hyperparameter tuning (all settings identical to stress_v3)
- NO additional normalization or post-processing of extracted features
- NO modifications to existing backbone classes (DINOv2Backbone, CLIPBackbone, BioCLIPBackbone)
- NO modifications to existing entries in `encoder_to_file` dict
- NO modifications to any existing config file
- NO new evaluation metrics (use same as existing ablations)
- NO mixed-backbone or ensemble experiments
- NO feature visualization/analysis code additions
- NO "helper" utilities, registries, or abstraction layers

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks are verifiable WITHOUT any human action.

### Test Decision
- **Infrastructure exists**: YES (existing ablation pipeline with evaluate_ablation.py)
- **Automated tests**: None (config/SLURM task, not application code)
- **Framework**: N/A

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

All verification is executed by the agent using Bash commands on Mahti or locally.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — no dependencies):
├── Task 1: Create 3 backbone config files
├── Task 2: Add BioCLIP2Backbone to extract_features.py + dataset.py mapping
└── Task 3: Create SLURM scripts (extraction + training + evaluation)

Wave 2 (After Wave 1 — requires code changes):
├── Task 4: Pre-download BioCLIP 2 model on Mahti login node
└── Task 5: Extract BioCLIP 2 features on Mahti

Wave 3 (After Wave 2 — requires BioCLIP 2 features):
└── Task 6: Launch all backbone training (CLIP + BioCLIP can start in Wave 2 if user wants)
    └── Task 7: Evaluate all backbone experiments
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|-----------|--------|---------------------|
| 1 | None | 6 | 2, 3 |
| 2 | None | 4, 5, 6 | 1, 3 |
| 3 | None | 4, 5, 6, 7 | 1, 2 |
| 4 | 2 | 5 | None |
| 5 | 2, 3, 4 | 6 (BioCLIP 2 only) | None |
| 6 | 1, 2, 3, 5 | 7 | None |
| 7 | 6 | None | None |

---

## TODOs

- [x] 1. Create 3 backbone ablation config files

  **What to do**:
  - Copy `configs/ablation/stress/drop_image.yaml` as template (it's a full stress_v3 config with ablation section)
  - Create `configs/ablation/stress/clip_backbone.yaml`:
    - Change `model.image_encoder` to `"openai/clip-vit-base-patch16"`
    - Set `ablation.enabled_modalities: [image, fluor, env, vi]` (all modalities)
    - Set `logging.save_dir: "results/ablation/clip_backbone/"`
    - Set `logging.checkpoint_dir: "results/ablation/clip_backbone/checkpoints/"`
  - Create `configs/ablation/stress/bioclip_backbone.yaml`:
    - Change `model.image_encoder` to `"imageomics/bioclip"`
    - Set `logging.save_dir: "results/ablation/bioclip_backbone/"`
    - Set `logging.checkpoint_dir: "results/ablation/bioclip_backbone/checkpoints/"`
  - Create `configs/ablation/stress/bioclip2_backbone.yaml`:
    - Change `model.image_encoder` to `"imageomics/bioclip-2"`
    - Set `logging.save_dir: "results/ablation/bioclip2_backbone/"`
    - Set `logging.checkpoint_dir: "results/ablation/bioclip2_backbone/checkpoints/"`
  - All other fields IDENTICAL to stress_v3.yaml (pos_weight=1.5, lr=1e-4, patience=20, etc.)
  - Header comment should say "Backbone Ablation: CLIP/BioCLIP/BioCLIP2" with explanation

  **Must NOT do**:
  - Do NOT change any hyperparameter relative to stress_v3
  - Do NOT create minimal overlay configs — must be full configs (following existing pattern)
  - Do NOT modify any existing config file

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Creating config files by copying and modifying an existing template — straightforward file creation
  - **Skills**: []
    - No special skills needed for YAML file creation

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Task 6
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `configs/ablation/stress/drop_image.yaml` — Full config template to copy. Use this exact structure, only change `model.image_encoder`, `ablation.enabled_modalities` (set to all 4), `logging.save_dir`, `logging.checkpoint_dir`, and header comment.
  - `configs/stress_v3.yaml` — Base config values to verify against. All training/model parameters MUST match this.

  **Why Each Reference Matters**:
  - `drop_image.yaml`: Provides the exact format of a full ablation stress config with the `ablation:` section. Copy this verbatim and only change the 4 fields listed above.
  - `stress_v3.yaml`: Source of truth for hyperparameters. Cross-check that pos_weight=1.5, lr=1e-4, patience=20, temporal.dim=128, etc. all match.

  **Acceptance Criteria**:

  ```
  Scenario: Verify config correctness
    Tool: Bash
    Preconditions: Python environment available
    Steps:
      1. For each config (clip_backbone, bioclip_backbone, bioclip2_backbone):
         - Read the YAML file
         - Assert model.image_encoder matches expected value
         - Assert model.encoder_output_dim == 768
         - Assert model.modality.image_dim == 768
         - Assert training.pos_weight == 1.5
         - Assert training.lr == 1e-4
         - Assert training.patience == 20
         - Assert training.cv.n_folds == 44
         - Assert ablation.enabled_modalities contains [image, fluor, env, vi]
         - Assert logging.save_dir and checkpoint_dir are unique per config
      2. Diff each config against drop_image.yaml to verify only expected fields differ
    Expected Result: All assertions pass, diff shows only image_encoder + logging paths + header comment changed
    Evidence: Command output captured
  ```

  **Commit**: YES (groups with 2, 3)
  - Message: `feat(ablation): add CLIP, BioCLIP, BioCLIP 2 backbone experiment configs and code`
  - Files: `configs/ablation/stress/clip_backbone.yaml`, `configs/ablation/stress/bioclip_backbone.yaml`, `configs/ablation/stress/bioclip2_backbone.yaml`

---

- [x] 2. Add BioCLIP2Backbone to extract_features.py and dataset.py mapping

  **What to do**:
  - **In `scripts/extract_features.py`**:
    - Add `BioCLIP2Backbone` class after `BioCLIPBackbone` class (after line 183)
    - Implementation MUST use `model.encode_image()` for 768-dim post-projection features (NOT `ln_post` hook which would give 1024-dim from ViT-L)
    - Add dimension assertion: `assert cls_tokens.shape[-1] == 768`
    - Use `open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')` to load
    - Register in `create_backbone()` factory (line 186-194): add `elif name == 'bioclip2': return BioCLIP2Backbone(device)`
    - Update argparse choices (line 383): add `'bioclip2'` to the choices list
  - **In `src/data/dataset.py`**:
    - Add ONE line to `encoder_to_file` dict (line 109, after bioclip entry): `"imageomics/bioclip-2": "bioclip2_features.h5",`

  **Must NOT do**:
  - Do NOT modify DINOv2Backbone, CLIPBackbone, or BioCLIPBackbone classes
  - Do NOT modify existing entries in encoder_to_file dict
  - Do NOT use the `ln_post` hook pattern for BioCLIP 2 (it would give 1024-dim, not 768)
  - Do NOT add any normalization, scaling, or post-processing that existing backbones don't have

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Adding a new class following an existing pattern + one dict entry — small, well-defined code changes
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 4, 5, 6
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `scripts/extract_features.py:147-183` (BioCLIPBackbone) — Class structure to follow. Copy the `__init__`, `preprocess`, `extract`, `__del__` pattern. BUT use `encode_image()` instead of ln_post hook.
  - `scripts/extract_features.py:122-144` (CLIPBackbone) — Alternative pattern that directly extracts features. Shows how to get CLS tokens without hooks.
  - `scripts/extract_features.py:186-194` (create_backbone factory) — Where to register the new backbone.
  - `scripts/extract_features.py:383` (argparse choices) — Where to add 'bioclip2'.
  - `src/data/dataset.py:106-110` (encoder_to_file dict) — Where to add the mapping.

  **Why Each Reference Matters**:
  - `BioCLIPBackbone`: Shows the open_clip loading pattern (`create_model_and_transforms`), preprocessing, and cleanup (`__del__`). BioCLIP2Backbone follows this structure but replaces the hook with `encode_image()`.
  - `CLIPBackbone`: Shows direct feature extraction without hooks — conceptually closer to what BioCLIP2Backbone needs.
  - `create_backbone`: Simple factory where new backbone is registered.
  - `encoder_to_file`: One-line addition mapping model ID to feature filename.

  **Critical Implementation Detail**:
  ```python
  # BioCLIP 2 (ViT-L/14): transformer width=1024, CLIP embed_dim=768
  # ln_post hook → 1024-dim (WRONG for our pipeline)
  # encode_image() → 768-dim (CORRECT)
  
  class BioCLIP2Backbone(nn.Module):
      """BioCLIP 2 (ViT-L/14) wrapper. Uses encode_image() for 768-dim features."""
      def __init__(self, device):
          super().__init__()
          import open_clip
          self.model, _, self.preprocess_fn = open_clip.create_model_and_transforms(
              'hf-hub:imageomics/bioclip-2'
          )
          self.model = self.model.to(device)
          self.model.eval()
          self.device = device
      
      def preprocess(self, img):
          return self.preprocess_fn(img)
      
      @torch.no_grad()
      def extract(self, batch):
          batch = batch.to(self.device)
          cls_tokens = self.model.encode_image(batch)
          cls_tokens = cls_tokens.cpu().numpy()
          assert cls_tokens.shape[-1] == 768, f"Expected 768-dim, got {cls_tokens.shape[-1]}"
          return cls_tokens, None
  ```

  **Acceptance Criteria**:

  ```
  Scenario: Verify no existing code modified
    Tool: Bash (git diff)
    Steps:
      1. git diff scripts/extract_features.py | grep "^-" | grep -v "^---" 
         → Assert: NO deleted lines (only additions)
      2. git diff src/data/dataset.py | grep "^-" | grep -v "^---"
         → Assert: NO deleted lines (only additions)
      3. git diff src/model/ → Assert: empty (no model code touched)
    Expected Result: Only additive changes, no existing code modified
    Evidence: git diff output

  Scenario: Verify BioCLIP2Backbone uses encode_image (not hook)
    Tool: Bash (grep)
    Steps:
      1. grep -A 20 "class BioCLIP2Backbone" scripts/extract_features.py
      2. Assert: Contains "encode_image"
      3. Assert: Does NOT contain "register_forward_hook" or "ln_post"
      4. Assert: Contains "assert.*768" (dimension check)
    Expected Result: BioCLIP2Backbone uses encode_image with dimension assertion
    Evidence: grep output

  Scenario: Verify argparse updated
    Tool: Bash (grep)
    Steps:
      1. grep "choices=" scripts/extract_features.py
      2. Assert: Contains 'bioclip2' in choices list
    Expected Result: bioclip2 is a valid backbone choice
    Evidence: grep output

  Scenario: Verify dataset mapping added
    Tool: Bash (grep)
    Steps:
      1. grep "bioclip-2" src/data/dataset.py
      2. Assert: Line contains '"imageomics/bioclip-2": "bioclip2_features.h5"'
    Expected Result: Mapping exists
    Evidence: grep output
  ```

  **Commit**: YES (groups with 1, 3)
  - Message: `feat(ablation): add CLIP, BioCLIP, BioCLIP 2 backbone experiment configs and code`
  - Files: `scripts/extract_features.py`, `src/data/dataset.py`

---

- [x] 3. Create SLURM scripts for feature extraction, training, and evaluation

  **What to do**:
  - **`scripts/slurm/extract_bioclip2_features.sh`**:
    - Based on `scripts/slurm/extract_features.sh`
    - Add pre-download step: run `python -c "import open_clip; open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')"` on login node (as a comment/instruction in the header)
    - Use `--backbone bioclip2 --batch_size 32` (reduced from 64 due to ViT-L memory)
    - Output: `features/bioclip2_features.h5`
  - **`scripts/slurm/launch_backbone_ablations.sh`**:
    - Based on `scripts/slurm/launch_all_ablations.sh`
    - ABLATIONS array: `("clip_backbone" "bioclip_backbone" "bioclip2_backbone")`
    - SEEDS: `(42 123 456)`
    - Same wait-between-ablations pattern
    - Uses existing `train_ablation.sh` with `configs/ablation/stress/{name}.yaml`
  - **`scripts/slurm/evaluate_backbone_ablations.sh`**:
    - Based on `scripts/slurm/evaluate_all_ablations.sh`
    - Same ABLATIONS array as launch script
    - Uses existing `evaluate_ablation.py`

  **Must NOT do**:
  - Do NOT modify existing SLURM scripts
  - Do NOT change batch_size for existing backbones
  - Do NOT modify `launch_all_ablations.sh` or `evaluate_all_ablations.sh`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Creating shell scripts by copying and adapting existing templates
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Tasks 4, 5, 6, 7
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `scripts/slurm/extract_features.sh` — Template for BioCLIP 2 feature extraction SLURM job. Copy exactly, change backbone name and batch_size.
  - `scripts/slurm/launch_all_ablations.sh` — Template for backbone ablation launcher. Copy wait_for_jobs function and sequential submission pattern. Change ABLATIONS array to 3 backbone names.
  - `scripts/slurm/evaluate_all_ablations.sh` — Template for backbone evaluation script. Copy loop structure, change ABLATIONS array.
  - `scripts/slurm/train_ablation.sh` — This is the worker script called by the launcher. NO modifications needed — it already accepts config path and seed as arguments.

  **Why Each Reference Matters**:
  - `extract_features.sh`: Provides SLURM directives (account, partition, GPU, time), module loading, and extraction command. BioCLIP 2 script needs lower batch_size only.
  - `launch_all_ablations.sh`: The sequential-submission-with-wait pattern that respects Mahti's MaxSubmitJobs=4 limit. Critical for not overwhelming the scheduler.
  - `evaluate_all_ablations.sh`: Evaluation loop pattern with results_dir, config, and seeds.

  **Acceptance Criteria**:

  ```
  Scenario: Verify SLURM scripts are valid bash
    Tool: Bash
    Steps:
      1. bash -n scripts/slurm/extract_bioclip2_features.sh → Assert: exit code 0
      2. bash -n scripts/slurm/launch_backbone_ablations.sh → Assert: exit code 0
      3. bash -n scripts/slurm/evaluate_backbone_ablations.sh → Assert: exit code 0
    Expected Result: All scripts pass syntax check
    Evidence: Exit codes

  Scenario: Verify batch_size is 32 for BioCLIP 2 extraction
    Tool: Bash (grep)
    Steps:
      1. grep "batch_size" scripts/slurm/extract_bioclip2_features.sh
      2. Assert: Contains "32" (not "64")
    Expected Result: Reduced batch size for ViT-L model
    Evidence: grep output

  Scenario: Verify launch script uses correct configs
    Tool: Bash (grep)
    Steps:
      1. grep "ABLATIONS" scripts/slurm/launch_backbone_ablations.sh
      2. Assert: Contains "clip_backbone" "bioclip_backbone" "bioclip2_backbone"
      3. grep "configs/ablation/stress" scripts/slurm/launch_backbone_ablations.sh
      4. Assert: Config path pattern matches existing convention
    Expected Result: All 3 backbone experiments are included
    Evidence: grep output
  ```

  **Commit**: YES (groups with 1, 2)
  - Message: `feat(ablation): add CLIP, BioCLIP, BioCLIP 2 backbone experiment configs and code`
  - Files: `scripts/slurm/extract_bioclip2_features.sh`, `scripts/slurm/launch_backbone_ablations.sh`, `scripts/slurm/evaluate_backbone_ablations.sh`

---

- [x] 4. Pre-download BioCLIP 2 model on Mahti — **BLOCKED: Documented in issues.md, requires user SSH to Mahti**

  **What to do**:
  - SSH to Mahti login node
  - Activate the project venv: `module load pytorch/2.4 && source .venv/bin/activate`
  - Check open_clip version: `python -c "import open_clip; print(open_clip.__version__)"`
  - If version < 2.24.0, upgrade: `pip install --upgrade open_clip_torch`
  - Pre-download model: `python -c "import open_clip; m, _, p = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2'); print('OK', m.visual.ln_post.normalized_shape)"`
  - Verify download worked and note the transformer width output (should show 1024 for ViT-L)

  **Must NOT do**:
  - Do NOT install packages system-wide, only in venv
  - Do NOT modify any existing packages or configs

  **Recommended Agent Profile**:
  - **Category**: N/A — **This task must be done by the user on Mahti** (SSH access required)
    - Reason: Agent cannot SSH to CSC Mahti. User must run these commands manually.
  - **Skills**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO (user manual step)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 5
  - **Blocked By**: Task 2 (need BioCLIP2Backbone code to exist first)

  **References**:
  - `scripts/slurm/extract_features.sh:18-20` — Module loading and venv activation pattern for Mahti

  **Acceptance Criteria**:

  ```
  Scenario: Model download verified
    Tool: User runs on Mahti login node
    Steps:
      1. python -c "import open_clip; m, _, p = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2'); print('OK')"
      2. Assert: Output is "OK" (no download errors, no OOM)
    Expected Result: Model is cached and loadable
  ```

  **Commit**: NO

---

- [x] 5. Extract BioCLIP 2 features on Mahti — **BLOCKED: Documented in issues.md, requires user SLURM submission**

  **What to do**:
  - After Task 4 completes (model downloaded)
  - Submit SLURM job: `sbatch scripts/slurm/extract_bioclip2_features.sh`
  - Monitor: `squeue -u $USER`
  - After completion, verify features:
    ```bash
    python -c "
    import h5py
    f = h5py.File('features/bioclip2_features.h5', 'r')
    plants = list(f.keys())
    print(f'Plants: {len(plants)}')
    r = list(f[plants[0]].keys())[0]
    v = list(f[plants[0]][r].keys())[0]
    feat = f[plants[0]][r][v][:]
    print(f'Feature shape: {feat.shape}')
    assert feat.shape == (768,), f'Expected (768,), got {feat.shape}'
    print('PASS')
    f.close()
    "
    ```

  **Must NOT do**:
  - Do NOT re-extract DINOv2, CLIP, or BioCLIP v1 features

  **Recommended Agent Profile**:
  - **Category**: N/A — **User must run on Mahti** (SLURM submission)
  - **Skills**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 6 (BioCLIP 2 training only; CLIP + BioCLIP training can start)
  - **Blocked By**: Tasks 2, 3, 4

  **Acceptance Criteria**:

  ```
  Scenario: BioCLIP 2 features are correct
    Tool: Bash (python on Mahti)
    Steps:
      1. Verify file exists: ls -la features/bioclip2_features.h5
      2. Check plant count matches DINOv2: python -c "import h5py; print(len(h5py.File('features/bioclip2_features.h5','r').keys()))"
         → Should match dinov2_features.h5 plant count (~264)
      3. Check feature dimension: python script above → Assert 768
    Expected Result: All plants extracted with 768-dim features
    Evidence: Command output
  ```

  **Commit**: NO (feature files are not in git)

---

- [x] 6. Launch backbone training on Mahti — **BLOCKED: Documented in issues.md, requires user SLURM submission**

  **What to do**:
  - Sync code to Mahti (git push + pull, or rsync)
  - Run: `bash scripts/slurm/launch_backbone_ablations.sh`
  - This submits 3 backbones × 3 seeds = 9 array jobs (each 44 folds)
  - Total: 396 training runs, ~3 hours sequential with wait-between-ablations
  - Monitor: `squeue -u $USER`

  **Must NOT do**:
  - Do NOT re-train existing ablation experiments
  - Do NOT change any training hyperparameters

  **Recommended Agent Profile**:
  - **Category**: N/A — **User must run on Mahti**
  - **Skills**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 1, 2, 3, 5

  **Acceptance Criteria**:

  ```
  Scenario: All training jobs completed
    Tool: Bash (on Mahti)
    Steps:
      1. For each backbone (clip_backbone, bioclip_backbone, bioclip2_backbone):
         For each seed (42, 123, 456):
           ls results/ablation/{backbone}/checkpoints/seed_{seed}/
           → Assert: 44 fold directories exist (fold_0 through fold_43)
           → Assert: Each contains best_model.pt
    Expected Result: 3 × 3 × 44 = 396 checkpoints exist
    Evidence: ls output
  ```

  **Commit**: NO (results are not in git)

---

- [x] 7. Evaluate backbone experiments — **BLOCKED: Documented in issues.md, requires user SLURM submission**

  **What to do**:
  - Run: `sbatch scripts/slurm/evaluate_backbone_ablations.sh`
  - After completion, check results in `results/ablation/summary/`
  - Compare F1, AUC, Onset MAE across DINOv2 (stress_v3), CLIP, BioCLIP, BioCLIP 2

  **Must NOT do**:
  - Do NOT re-evaluate existing ablations

  **Recommended Agent Profile**:
  - **Category**: N/A — **User must run on Mahti**
  - **Skills**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after training)
  - **Blocks**: None (final task)
  - **Blocked By**: Task 6

  **Acceptance Criteria**:

  ```
  Scenario: Evaluation results generated
    Tool: Bash (on Mahti)
    Steps:
      1. ls results/ablation/summary/ | grep backbone
      2. Assert: Result files for all 3 backbone experiments exist
      3. cat results/ablation/summary/clip_backbone*.json (or csv)
         → Assert: Contains F1, AUC, MAE values
    Expected Result: All 3 backbone evaluation summaries available
    Evidence: File contents
  ```

  **Commit**: NO

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|-------------|
| 1, 2, 3 (grouped) | `feat(ablation): add CLIP, BioCLIP, BioCLIP 2 backbone experiment configs and code` | 3 YAML configs + extract_features.py + dataset.py + 3 SLURM scripts | `git diff --stat`, verify only additive changes |

---

## Success Criteria

### Verification Commands
```bash
# 1. No existing code broken
git diff --name-only src/model/ src/training/
# Expected: empty (no changes)

# 2. Only additive changes in extract_features.py
git diff scripts/extract_features.py | grep "^-" | grep -v "^---" | wc -l
# Expected: 0 (no deletions)

# 3. Only additive changes in dataset.py
git diff src/data/dataset.py | grep "^-" | grep -v "^---" | wc -l
# Expected: 0 (no deletions)

# 4. New configs match stress_v3 hyperparameters
python -c "
import yaml
with open('configs/stress_v3.yaml') as f: v3 = yaml.safe_load(f)
for name in ['clip_backbone', 'bioclip_backbone', 'bioclip2_backbone']:
    with open(f'configs/ablation/stress/{name}.yaml') as f: cfg = yaml.safe_load(f)
    assert cfg['training']['pos_weight'] == v3['training']['pos_weight']
    assert cfg['training']['lr'] == v3['training']['lr']
    assert cfg['model']['modality'] == v3['model']['modality']
    print(f'PASS: {name}')
"
# Expected: PASS for all 3

# 5. BioCLIP 2 features are 768-dim (on Mahti after extraction)
python -c "import h5py; f=h5py.File('features/bioclip2_features.h5','r'); p=list(f.keys())[0]; r=list(f[p].keys())[0]; v=list(f[p][r].keys())[0]; assert f[p][r][v][:].shape==(768,); print('PASS')"
# Expected: PASS
```

### Final Checklist
- [x] All 3 config files created and match stress_v3 hyperparameters
- [x] BioCLIP2Backbone uses `encode_image()` (NOT hook), with 768-dim assertion
- [x] Dataset mapping added for `imageomics/bioclip-2`
- [x] SLURM scripts pass bash syntax check
- [x] BioCLIP 2 feature extraction uses batch_size=32
- [x] No existing code modified (git diff verification)
- [x] All "Must NOT Have" guardrails respected
