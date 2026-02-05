#!/usr/bin/env python3
"""Evaluate stress detection model with onset detection metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Subset

from src.data.collate import faba_collate_fn
from src.data.dataset import FabaDroughtDataset, ROUND_TO_DAG
from src.model.stress_model import StressDetectionModel
from src.utils.config import load_config


def compute_onset_metrics(
    probs: np.ndarray,  # (N, T) probabilities
    labels: np.ndarray,  # (N, T) binary labels
    masks: np.ndarray,  # (N, T) valid masks
    treatments: List[str],  # List of treatment strings
) -> Dict[str, float]:
    """Compute onset detection metrics for WHC-30 plants.
    
    Args:
        probs: Predicted probabilities (N, T=22)
        labels: True binary labels (N, T=22)
        masks: Valid timestep masks (N, T=22)
        treatments: List of treatment strings ('WHC-30' or 'WHC-80')
    
    Returns:
        Dict with onset_mae, early_detection_rate, mean_early_days
    """
    results = []
    
    for i in range(len(probs)):
        # Skip WHC-80 plants (no stress onset)
        if treatments[i] == 'WHC-80':
            continue
        
        # Skip if no stress labels (shouldn't happen for WHC-30)
        if labels[i].sum() == 0:
            continue
        
        # True onset: first timestep with label=1
        stress_indices = np.where(labels[i] == 1)[0]
        if len(stress_indices) == 0:
            continue
        true_onset_idx = stress_indices[0]
        true_onset_dag = ROUND_TO_DAG[true_onset_idx + 2]  # +2 because rounds start at 2
        
        # Predicted onset: first timestep with prob > 0.5
        pred_stress_indices = np.where(probs[i] > 0.5)[0]
        if len(pred_stress_indices) == 0:
            # Never predicted stress - use last timestep
            pred_onset_idx = len(probs[i]) - 1
        else:
            pred_onset_idx = pred_stress_indices[0]
        pred_onset_dag = ROUND_TO_DAG[pred_onset_idx + 2]
        
        # Compute error (negative = early detection)
        error = pred_onset_dag - true_onset_dag
        
        results.append({
            'true_onset_dag': true_onset_dag,
            'pred_onset_dag': pred_onset_dag,
            'error': error,
            'early': error < 0,
        })
    
    if len(results) == 0:
        return {
            'onset_mae': float('nan'),
            'early_detection_rate': float('nan'),
            'mean_early_days': float('nan'),
            'n_plants': 0,
        }
    
    return {
        'onset_mae': np.mean([abs(r['error']) for r in results]),
        'early_detection_rate': np.mean([r['early'] for r in results]),
        'mean_early_days': np.mean([r['error'] for r in results if r['early']]) if any(r['early'] for r in results) else 0.0,
        'n_plants': len(results),
    }


def evaluate_fold(
    cfg: Any,
    dataset: FabaDroughtDataset,
    fold_id: int,
    test_indices: np.ndarray,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """Evaluate a single fold."""
    # Load model
    model = StressDetectionModel(cfg)
    model.load_state_dict(torch.load(
        checkpoint_path / f"fold_{fold_id}" / "best_model_state.pt",
        map_location=cfg.device,
        weights_only=True
    ))
    model.to(cfg.device)
    model.eval()
    
    # Create test loader
    test_dataset = Subset(dataset, test_indices.tolist())
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=faba_collate_fn,
        pin_memory=True,
    )
    
    # Collect predictions
    all_probs = []
    all_labels = []
    all_masks = []
    all_treatments = []
    all_gates = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch_device = {k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batch.items()}
            
            predictions = model(batch_device)
            logits = predictions['stress_logits']  # (B, T)
            gates = predictions['modality_gates']  # (B, T, 4)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch['stress_labels'].cpu().numpy()
            masks = batch['stress_mask'].cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(labels)
            all_masks.append(masks)
            all_treatments.extend(batch['treatment'])
            all_gates.append(gates.cpu().numpy())
    
    # Concatenate
    all_probs = np.concatenate(all_probs, axis=0)  # (N, T)
    all_labels = np.concatenate(all_labels, axis=0)  # (N, T)
    all_masks = np.concatenate(all_masks, axis=0)  # (N, T)
    all_gates = np.concatenate(all_gates, axis=0)  # (N, T, 4)
    
    # Flatten for per-timestep metrics (only valid timesteps)
    valid_probs = all_probs[all_masks]
    valid_labels = all_labels[all_masks]
    valid_preds = (valid_probs > 0.5).astype(int)
    
    # Per-timestep metrics
    accuracy = accuracy_score(valid_labels, valid_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels, valid_preds, average='binary', zero_division=0
    )
    try:
        auc = roc_auc_score(valid_labels, valid_probs)
    except ValueError:
        auc = float('nan')
    
    # Onset detection metrics
    onset_metrics = compute_onset_metrics(all_probs, all_labels, all_masks, all_treatments)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        **onset_metrics,
        'modality_gates': all_gates,  # For visualization
    }


def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate stress detection model")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to results/stress/checkpoints/")
    parser.add_argument("--config", type=str, default="configs/stress.yaml")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Load dataset
    dataset = FabaDroughtDataset(cfg)
    
    # Load CV splits (need to recreate LogoCV)
    from src.training.cv import LogoCV
    cv = LogoCV(
        plant_metadata_df=dataset.plant_metadata,
        n_folds=cfg.training.cv.n_folds,
        stratify_col=cfg.training.cv.stratify_by,
        seed=cfg.training.cv.seed,
    )
    
    # Evaluate all folds
    checkpoint_path = Path(args.results_dir)
    all_results = {}
    
    for fold_id, (_, _, test_idx) in enumerate(cv.split()):
        print(f"Evaluating fold {fold_id}/{cfg.training.cv.n_folds}...")
        
        fold_results = evaluate_fold(
            cfg=cfg,
            dataset=dataset,
            fold_id=fold_id,
            test_indices=test_idx,
            checkpoint_path=checkpoint_path,
        )
        
        # Remove gates from saved results (too large)
        gates = fold_results.pop('modality_gates')
        all_results[f"fold_{fold_id}"] = fold_results
        
        # Save gates separately
        gates_path = checkpoint_path / f"fold_{fold_id}" / "test_modality_gates.npy"
        np.save(gates_path, gates)
    
    # Aggregate metrics
    metrics_to_aggregate = ['accuracy', 'precision', 'recall', 'f1', 'auc', 
                           'onset_mae', 'early_detection_rate', 'mean_early_days']
    
    aggregated = {}
    for metric in metrics_to_aggregate:
        values = [r[metric] for r in all_results.values() if not np.isnan(r[metric])]
        if len(values) > 0:
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
    
    # Save results
    results_path = checkpoint_path / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump({
            'per_fold': all_results,
            'aggregated': aggregated,
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("AGGREGATED RESULTS")
    print(f"{'='*80}")
    for key, value in aggregated.items():
        print(f"{key}: {value:.4f}")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
