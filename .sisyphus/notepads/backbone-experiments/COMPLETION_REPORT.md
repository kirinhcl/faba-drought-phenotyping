# Backbone Experiments - Completion Report

## Executive Summary
**Status:** Local work 100% complete. Mahti execution pending user action.

**Completed:** All code, configs, and scripts for CLIP/BioCLIP/BioCLIP 2 backbone experiments.

**Blocked:** Feature extraction, training, and evaluation require CSC Mahti supercomputer access.

---

## Deliverables Status

### ✅ Completed (All Local Work)

| Deliverable | Status | Location | Lines |
|-------------|--------|----------|-------|
| CLIP config | ✅ Complete | `configs/ablation/stress/clip_backbone.yaml` | 98 |
| BioCLIP config | ✅ Complete | `configs/ablation/stress/bioclip_backbone.yaml` | 98 |
| BioCLIP 2 config | ✅ Complete | `configs/ablation/stress/bioclip2_backbone.yaml` | 98 |
| BioCLIP2Backbone class | ✅ Complete | `scripts/extract_features.py:186-215` | 30 |
| Factory registration | ✅ Complete | `scripts/extract_features.py:225-226` | 2 |
| Argparse update | ✅ Complete | `scripts/extract_features.py:383` | 1 |
| Dataset mapping | ✅ Complete | `src/data/dataset.py:109` | 1 |
| BioCLIP 2 extraction script | ✅ Complete | `scripts/slurm/extract_bioclip2_features.sh` | 39 |
| Training launcher | ✅ Complete | `scripts/slurm/launch_backbone_ablations.sh` | 103 |
| Evaluation script | ✅ Complete | `scripts/slurm/evaluate_backbone_ablations.sh` | 77 |

**Total:** 10/10 deliverables complete

### ⏸️ Blocked (Require Mahti Execution)

| Task | Blocker | Estimated Time |
|------|---------|----------------|
| Pre-download BioCLIP 2 model | SSH to Mahti login node | 5 min |
| Extract BioCLIP 2 features | SLURM job submission | ~2 hours |
| Train 3 backbones (3 seeds × 44 folds) | SLURM job submission | ~3 hours |
| Evaluate results | SLURM job submission | ~30 min |

---

## Technical Implementation

### Key Decisions

1. **BioCLIP 2 Feature Extraction Method**
   - **Decision:** Use `model.encode_image()` instead of `ln_post` hook
   - **Reason:** ViT-L/14 has transformer width=1024; hook would give 1024-dim instead of required 768-dim
   - **Trade-off:** Post-projection features (CLIP space) vs pre-projection (raw transformer) used by other backbones
   - **Mitigation:** Documented in code; should be mentioned in paper methods

2. **Batch Size Reduction**
   - **Decision:** batch_size=32 for BioCLIP 2 (vs 64 for ViT-B models)
   - **Reason:** ViT-L is 3.5× larger (~300M vs ~86M params)
   - **Impact:** Prevents OOM on A100-40GB

3. **Config Structure**
   - **Decision:** Full 98-line configs (not minimal overlays)
   - **Reason:** Config loading overlays on default.yaml; full configs prevent drift
   - **Verification:** All hyperparameters match stress_v3.yaml

### Code Quality Verification

| Check | Result | Evidence |
|-------|--------|----------|
| No model code changed | ✅ PASS | 0 files in src/model/, src/training/ |
| Only additive changes | ✅ PASS | 1 argparse line replaced (acceptable) |
| BioCLIP2 uses encode_image | ✅ PASS | Line 212 confirmed |
| 768-dim assertion present | ✅ PASS | Line 214 confirmed |
| Dataset mapping added | ✅ PASS | Line 109 confirmed |
| SLURM syntax valid | ✅ PASS | All 3 scripts pass bash -n |
| Batch size correct | ✅ PASS | --batch_size 32 confirmed |
| Configs match stress_v3 | ✅ PASS | pos_weight=1.5, lr=1e-4, patience=20 |

---

## Git History

**Commit:** `f9dc7ac`
**Message:** feat(ablation): add CLIP, BioCLIP, BioCLIP 2 backbone experiment configs and code
**Files Changed:** 8 files, +535 lines, -1 line
**Date:** 2026-02-10

---

## User Action Required

To complete the remaining tasks, SSH to Mahti and execute:

```bash
cd /scratch/project_2013932/chenghao/faba-drought-phenotyping
git pull  # Sync commit f9dc7ac

# Task 4: Pre-download model (5 min)
module load pytorch/2.4
source .venv/bin/activate
python -c "import open_clip; m, _, p = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2'); print('OK', m.visual.ln_post.normalized_shape)"
# Expected: OK torch.Size([1024])

# Task 5: Extract features (~2 hours)
sbatch scripts/slurm/extract_bioclip2_features.sh
squeue -u $USER  # Monitor

# Verify after completion:
python -c "import h5py; f=h5py.File('features/bioclip2_features.h5','r'); p=list(f.keys())[0]; r=list(f[p].keys())[0]; v=list(f[p][r].keys())[0]; print(f'Shape: {f[p][r][v][:].shape}'); f.close()"
# Expected: Shape: (768,)

# Task 6: Train backbones (~3 hours)
bash scripts/slurm/launch_backbone_ablations.sh

# Task 7: Evaluate (~30 min)
sbatch scripts/slurm/evaluate_backbone_ablations.sh

# Check results:
ls results/ablation/summary/ | grep backbone
```

---

## Expected Outcomes

After Mahti execution completes, results will be available for comparison:

| Backbone | F1 (expected) | AUC (expected) | Onset MAE (expected) |
|----------|---------------|----------------|----------------------|
| DINOv2 (baseline) | 0.634 | 0.949 | 8.7 |
| CLIP | ? | ? | ? |
| BioCLIP | ? | ? | ? |
| BioCLIP 2 | ? | ? | ? |

**Hypothesis:** Domain-specific (BioCLIP) or alternative (CLIP) backbones may improve over DINOv2, especially given that "Drop Image" ablation (F1=0.662) outperformed full model (F1=0.634).

---

## Documentation

All learnings, decisions, and blockers documented in:
- `.sisyphus/notepads/backbone-experiments/learnings.md`
- `.sisyphus/notepads/backbone-experiments/decisions.md`
- `.sisyphus/notepads/backbone-experiments/issues.md`

---

**Report Generated:** 2026-02-10
**Boulder Session:** Complete (local work)
**Next Action:** User execution on Mahti
