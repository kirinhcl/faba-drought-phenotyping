"""Genotype ranking evaluation for drought sensitivity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def compute_genotype_ranking(
    predictions_dir: Path,
    plant_metadata: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    """Compute genotype ranking from model predictions.
    
    Aggregate predictions across 3 drought replicates per genotype,
    rank by predicted DAG (ascending = most sensitive), and compare to
    true ranking.
    
    Args:
        predictions_dir: Directory with fold_N/predictions.json files
        plant_metadata: Plant metadata DataFrame
        output_dir: Directory to save results
    
    Returns:
        Ranking evaluation metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize accessions in metadata
    import unicodedata
    plant_metadata['accession'] = plant_metadata['accession'].apply(
        lambda x: unicodedata.normalize('NFC', str(x))
    )
    
    # Load predictions from all folds
    all_predictions: Dict[str, list[float]] = {}
    all_targets: Dict[str, list[float]] = {}
    all_categories: Dict[str, str] = {}
    
    # Collect predictions from all folds
    for fold_dir in sorted(predictions_dir.glob('fold_*')):
        pred_file = fold_dir / 'predictions.json'
        if not pred_file.exists():
            continue
        
        with open(pred_file, 'r') as f:
            fold_predictions = json.load(f)
        
        for plant_id, pred_data in fold_predictions.items():
            treatment = pred_data['treatment']
            if treatment != 'WHC-30':
                continue
            
            accession = pred_data['accession']
            dag_pred = pred_data['dag_reg']
            dag_target = pred_data['dag_target']
            
            if dag_pred is None or dag_target is None:
                continue
            
            if np.isnan(dag_pred) or np.isnan(dag_target):
                continue
            
            if accession not in all_predictions:
                all_predictions[accession] = []
                all_targets[accession] = []
            
            all_predictions[accession].append(float(dag_pred))
            all_targets[accession].append(float(dag_target))
            
            # Get category from metadata
            if accession not in all_categories:
                acc_meta = plant_metadata[
                    (plant_metadata['accession'] == accession) &
                    (plant_metadata['treatment'] == 'WHC-30')
                ]
                if len(acc_meta) > 0:
                    category = acc_meta.iloc[0]['drought_category']
                    if pd.notna(category):
                        all_categories[accession] = str(category)
    
    # Average predictions per genotype
    genotype_avg_pred = {
        acc: float(np.mean(preds))
        for acc, preds in all_predictions.items()
    }
    genotype_avg_target = {
        acc: float(np.mean(targets))
        for acc, targets in all_targets.items()
    }
    
    # Rank genotypes (ascending DAG = most sensitive)
    genotypes = sorted(genotype_avg_pred.keys())
    
    # Compute Spearman correlation
    from scipy.stats import spearmanr
    
    pred_vals = [genotype_avg_pred[g] for g in genotypes]
    target_vals = [genotype_avg_target[g] for g in genotypes]
    
    spearman_rho, spearman_p = spearmanr(pred_vals, target_vals)
    
    # Bootstrap 95% CI for Spearman
    n_bootstrap = 1000
    rng = np.random.RandomState(42)
    bootstrap_rhos = []
    
    for _ in range(n_bootstrap):
        indices = rng.choice(len(genotypes), size=len(genotypes), replace=True)
        boot_pred = [pred_vals[i] for i in indices]
        boot_target = [target_vals[i] for i in indices]
        boot_rho, _ = spearmanr(boot_pred, boot_target)
        bootstrap_rhos.append(boot_rho)
    
    ci_lower = float(np.percentile(bootstrap_rhos, 2.5))
    ci_upper = float(np.percentile(bootstrap_rhos, 97.5))
    
    # Compute Kendall tau
    from scipy.stats import kendalltau
    kendall_tau, kendall_p = kendalltau(pred_vals, target_vals)
    
    # Top-K and bottom-K recall
    k = 5
    
    # True top-K and bottom-K
    true_sorted = sorted(genotypes, key=lambda g: genotype_avg_target[g])
    true_top_k = set(true_sorted[:k])
    true_bottom_k = set(true_sorted[-k:])
    
    # Predicted top-K and bottom-K
    pred_sorted = sorted(genotypes, key=lambda g: genotype_avg_pred[g])
    pred_top_k = set(pred_sorted[:k])
    pred_bottom_k = set(pred_sorted[-k:])
    
    top_k_recall = len(true_top_k & pred_top_k) / k
    bottom_k_recall = len(true_bottom_k & pred_bottom_k) / k
    
    # Category accuracy (Early/Mid/Late)
    category_correct = 0
    category_total = 0
    
    for genotype in genotypes:
        if genotype not in all_categories:
            continue
        
        true_category = all_categories[genotype]
        true_dag = genotype_avg_target[genotype]
        pred_dag = genotype_avg_pred[genotype]
        
        # Assign category based on thresholds
        # Assume Early < 20, Mid 20-30, Late > 30 (adjust based on data)
        def dag_to_category(dag: float) -> str:
            if dag < 20:
                return 'Early'
            elif dag < 30:
                return 'Mid'
            else:
                return 'Late'
        
        pred_category = dag_to_category(pred_dag)
        
        if pred_category == true_category:
            category_correct += 1
        category_total += 1
    
    category_accuracy = category_correct / max(category_total, 1)
    
    # Summary
    results = {
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
        'spearman_ci_lower': ci_lower,
        'spearman_ci_upper': ci_upper,
        'kendall_tau': float(kendall_tau),
        'kendall_p': float(kendall_p),
        'top_k_recall': float(top_k_recall),
        'bottom_k_recall': float(bottom_k_recall),
        'category_accuracy': float(category_accuracy),
        'n_genotypes': len(genotypes),
    }
    
    # Detailed rankings
    rankings = []
    for genotype in genotypes:
        rankings.append({
            'accession': genotype,
            'predicted_dag': genotype_avg_pred[genotype],
            'true_dag': genotype_avg_target[genotype],
            'category': all_categories.get(genotype, 'Unknown'),
            'n_replicates': len(all_predictions[genotype]),
        })
    
    # Sort by predicted DAG
    rankings = sorted(rankings, key=lambda x: x['predicted_dag'])
    
    # Save results
    results_path = output_dir / 'ranking_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'summary': results,
            'rankings': rankings,
        }, f, indent=2)
    
    print(f"\nGenotype Ranking Results:")
    print(f"  Spearman ρ: {spearman_rho:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    print(f"  Kendall τ: {kendall_tau:.3f}")
    print(f"  Top-{k} recall: {top_k_recall:.1%}")
    print(f"  Bottom-{k} recall: {bottom_k_recall:.1%}")
    print(f"  Category accuracy: {category_accuracy:.1%}")
    print(f"\nResults saved to: {results_path}")
    
    return results
