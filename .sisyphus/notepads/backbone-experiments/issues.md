## [2026-02-10] Tasks 4-7 Blocked: Require Mahti Access

### Blocker
Tasks 4-7 require SSH access to CSC Mahti supercomputer:
- Task 4: Pre-download BioCLIP 2 model on login node
- Task 5: Submit SLURM job for feature extraction
- Task 6: Launch backbone training (3 backbones × 3 seeds × 44 folds)
- Task 7: Submit evaluation job

### Status
- Wave 1 (Tasks 1-3) complete: All local code/config changes done and committed
- Wave 2-3 (Tasks 4-7) blocked: User must execute on Mahti

### User Action Required
SSH to Mahti and run:
```bash
cd /scratch/project_2013932/chenghao/faba-drought-phenotyping
git pull  # Sync latest code (commit f9dc7ac)

# Task 4: Pre-download model
module load pytorch/2.4
source .venv/bin/activate
python -c "import open_clip; m, _, p = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2'); print('OK', m.visual.ln_post.normalized_shape)"

# Task 5: Extract features (~2 hours)
sbatch scripts/slurm/extract_bioclip2_features.sh
squeue -u $USER  # Monitor

# After Task 5 completes, verify:
python -c "import h5py; f=h5py.File('features/bioclip2_features.h5','r'); p=list(f.keys())[0]; print(f'Features: {f[p][list(f[p].keys())[0]][list(f[p][list(f[p].keys())[0]].keys())[0]][:].shape}'); f.close()"

# Task 6: Launch training (~3 hours)
bash scripts/slurm/launch_backbone_ablations.sh

# Task 7: Evaluate
sbatch scripts/slurm/evaluate_backbone_ablations.sh
```

### Resolution
Cannot proceed further without Mahti access. All preparatory work complete.

