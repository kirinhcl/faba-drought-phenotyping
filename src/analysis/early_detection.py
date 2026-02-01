"""Progressive temporal truncation evaluation for early detection capability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ROUND_TO_DAG


def progressive_truncation_evaluation(
    model: torch.nn.Module,
    dataset: Any,
    test_indices: np.ndarray,
    output_dir: Path,
    device: str = 'cuda',
) -> Dict[int, Dict[str, float]]:
    """Evaluate model with progressive temporal truncation.
    
    For each cutoff round T in [2, 23], mask all timepoints after T and
    evaluate DAG prediction accuracy and ranking correlation.
    
    Args:
        model: Trained model
        dataset: Full dataset
        test_indices: Test set indices
        output_dir: Directory to save results
        device: Device to run on
    
    Returns:
        {round: {'dag_mae': float, 'ranking_spearman': float}}
    """
    from torch.utils.data import Subset
    from src.data.collate import faba_collate_fn
    
    model_device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(model_device)
    model.eval()
    
    # Create test subset
    test_dataset = Subset(dataset, test_indices.tolist())
    
    results: Dict[int, Dict[str, float]] = {}
    
    # For each cutoff round
    for t_cutoff in range(2, 24):
        print(f"Evaluating with cutoff at round {t_cutoff} (DAG={ROUND_TO_DAG[t_cutoff]})...")
        
        # Create dataloader
        loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            collate_fn=faba_collate_fn,
        )
        
        predictions = []
        targets = []
        accessions = []
        treatments = []
        
        with torch.no_grad():
            for batch in loader:
                # Clone batch to avoid modifying original
                batch_masked = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Mask timepoints after cutoff
                t_idx_cutoff = t_cutoff - 2  # Convert to 0-indexed
                
                # Mask images
                if 'images' in batch_masked:
                    batch_masked['images'][:, t_idx_cutoff+1:, :, :] = 0
                    batch_masked['image_mask'][:, t_idx_cutoff+1:, :] = False
                
                # Mask fluorescence
                if 'fluorescence' in batch_masked:
                    batch_masked['fluorescence'][:, t_idx_cutoff+1:, :] = 0
                    batch_masked['fluor_mask'][:, t_idx_cutoff+1:] = False
                
                # Move to device
                batch_gpu = {
                    k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_masked.items()
                }
                
                # Forward pass
                outputs = model(batch_gpu)
                
                # Collect predictions (only drought plants)
                dag_pred = outputs['dag_reg']
                if dag_pred is not None:
                    dag_pred = dag_pred.squeeze(-1)
                    
                    for i in range(len(batch['treatment'])):
                        if batch['treatment'][i] == 'WHC-30':
                            predictions.append(float(dag_pred[i].cpu().item()))
                            targets.append(float(batch['dag_target'][i].cpu().item()))
                            accessions.append(batch['accession'][i])
                            treatments.append(batch['treatment'][i])
        
        # Compute DAG MAE
        if len(predictions) > 0:
            dag_mae = float(np.mean(np.abs(np.array(predictions) - np.array(targets))))
        else:
            dag_mae = 0.0
        
        # Compute genotype ranking correlation
        # Aggregate per genotype
        genotype_preds: Dict[str, list[float]] = {}
        genotype_targets: Dict[str, list[float]] = {}
        
        for pred, target, acc in zip(predictions, targets, accessions):
            if acc not in genotype_preds:
                genotype_preds[acc] = []
                genotype_targets[acc] = []
            genotype_preds[acc].append(pred)
            genotype_targets[acc].append(target)
        
        # Average per genotype
        genotype_avg_pred = {acc: np.mean(preds) for acc, preds in genotype_preds.items()}
        genotype_avg_target = {acc: np.mean(targets) for acc, targets in genotype_targets.items()}
        
        # Rank genotypes
        genotypes = sorted(genotype_avg_pred.keys())
        pred_ranks = [genotype_avg_pred[g] for g in genotypes]
        target_ranks = [genotype_avg_target[g] for g in genotypes]
        
        # Compute Spearman correlation
        if len(pred_ranks) > 1:
            from scipy.stats import spearmanr
            ranking_spearman = float(spearmanr(pred_ranks, target_ranks)[0])
        else:
            ranking_spearman = 0.0
        
        results[t_cutoff] = {
            'dag_mae': dag_mae,
            'ranking_spearman': ranking_spearman,
        }
        
        print(f"  Round {t_cutoff}: MAE={dag_mae:.2f}, Spearman ρ={ranking_spearman:.3f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'early_detection.json'
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEarly detection results saved to: {output_path}")
    
    return results
