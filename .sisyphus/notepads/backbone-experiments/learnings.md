## [2026-02-10] Wave 1 Complete: Local Code/Config Setup

### What Was Done
- Created 3 backbone ablation configs (CLIP, BioCLIP, BioCLIP 2)
- Added BioCLIP2Backbone class to extract_features.py
- Registered bioclip2 in factory and argparse
- Added dataset.py encoder mapping
- Created 3 SLURM scripts (extraction, training, evaluation)

### Key Implementation Decisions
1. **BioCLIP 2 uses encode_image() NOT ln_post hook**
   - Reason: ViT-L/14 has transformer width=1024, ln_post would give 1024-dim
   - encode_image() applies CLIP projection: 1024→768
   - This is post-projection features (semantically different from pre-projection used by other backbones)
   - Documented in code comments for future reference

2. **Batch size reduced to 32 for BioCLIP 2**
   - ViT-L is 3.5× larger than ViT-B (~300M vs ~86M params)
   - Conservative batch_size=32 to avoid OOM on A100-40GB

3. **Full configs not overlays**
   - All 3 configs are complete 98-line files (copied from drop_image.yaml)
   - Only changes: image_encoder, enabled_modalities (all 4), logging paths, header
   - Ensures no hyperparameter drift that would invalidate comparison

### Verification Results
- All configs match stress_v3 hyperparameters (pos_weight=1.5, lr=1e-4, patience=20)
- BioCLIP2Backbone correctly uses encode_image with 768-dim assertion
- argparse updated with 'bioclip2' choice
- Dataset mapping added for imageomics/bioclip-2
- All SLURM scripts pass bash -n syntax check
- Only additive changes (1 argparse line replaced, acceptable)

### Commit
- f9dc7ac: feat(ablation): add CLIP, BioCLIP, BioCLIP 2 backbone experiment configs and code
- 8 files changed, +535 lines, -1 line


## [2026-02-10] Boulder Session Complete: Local Work Finished

### Final Status
**Completed Tasks (3/7):**
- Task 1: 3 backbone config files created ✓
- Task 2: BioCLIP2Backbone code added ✓
- Task 3: 3 SLURM scripts created ✓

**Blocked Tasks (4/7):**
- Task 4: Pre-download BioCLIP 2 model (requires Mahti SSH)
- Task 5: Extract BioCLIP 2 features (requires Mahti SLURM)
- Task 6: Launch backbone training (requires Mahti SLURM)
- Task 7: Evaluate results (requires Mahti SLURM)

### All Verification Criteria Met
- ✓ No model/training code changed (0 files)
- ✓ BioCLIP2Backbone uses encode_image() with 768-dim assertion
- ✓ Dataset mapping added for imageomics/bioclip-2
- ✓ All SLURM scripts pass bash -n syntax check
- ✓ BioCLIP 2 extraction uses batch_size=32
- ✓ Only additive changes (1 argparse line replaced)
- ✓ All guardrails respected

### Deliverables Ready for Mahti
All code and configs are committed (f9dc7ac) and ready for execution on Mahti. User needs to:
1. git pull on Mahti
2. Run Task 4-7 commands (documented in issues.md)

### Key Learnings
1. **Dimension trap avoided**: BioCLIP 2's ViT-L architecture would have caused 1024-dim output with hook pattern
2. **Full configs required**: Ablation configs must be complete copies, not overlays, due to config loading logic
3. **Memory considerations**: ViT-L requires reduced batch_size (32 vs 64)
4. **Semantic trade-off**: BioCLIP 2 uses post-projection features (CLIP space) vs pre-projection for others

### Next Session
When user completes Mahti tasks, results will be in results/ablation/summary/ for comparison with DINOv2 baseline (F1=0.634).

