#!/usr/bin/env python3
"""Evaluate trained model across all folds and compute aggregate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def load_predictions(results_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Load predictions from all folds.
    
    Args:
        results_dir: Directory containing fold_0/, fold_1/, ..., fold_43/
    
    Returns:
        Dict mapping fold_id to predictions dict
    """
    predictions = {}
    for fold_dir in sorted(results_dir.glob('fold_*')):
        if not fold_dir.is_dir():
            continue
        
        fold_id = int(fold_dir.name.split('_')[1])
        pred_file = fold_dir / 'predictions.json'
        
        if pred_file.exists():
            with open(pred_file, 'r') as f:
                predictions[fold_id] = json.load(f)
    
    return predictions


def compute_dag_regression_metrics(
    predictions: List[float],
    targets: List[float],
) -> Dict[str, float]:
    """Compute DAG regression metrics.
    
    Args:
        predictions: List of predicted DAG values
        targets: List of true DAG values
    
    Returns:
        Dict with MAE, RMSE, R²
    """
    if len(predictions) == 0:
        return {'mae': float('nan'), 'rmse': float('nan'), 'r2': float('nan')}
    
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    # R² is undefined when target variance is 0 (e.g., single genotype with same DAG)
    target_var = np.var(targets)
    if target_var < 1e-10:
        r2 = float('nan')
    else:
        r2 = r2_score(targets, predictions)
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
    }


def compute_dag_classification_metrics(
    predictions: List[int],
    targets: List[int],
) -> Dict[str, float]:
    """Compute DAG classification metrics.
    
    Args:
        predictions: List of predicted class indices
        targets: List of true class indices
    
    Returns:
        Dict with accuracy, balanced accuracy, per-class F1
    """
    if len(predictions) == 0:
        return {
            'accuracy': float('nan'),
            'balanced_accuracy': float('nan'),
            'f1_early': float('nan'),
            'f1_mid': float('nan'),
            'f1_late': float('nan'),
            'f1_macro': float('nan'),
        }
    
    acc = accuracy_score(targets, predictions)
    balanced_acc = balanced_accuracy_score(targets, predictions)
    
    # Per-class F1 scores
    f1_per_class = np.array(
        f1_score(targets, predictions, labels=[0, 1, 2], average=None, zero_division=0)
    )
    f1_macro = float(f1_score(targets, predictions, average='macro', zero_division=0))
    
    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(balanced_acc),
        'f1_early': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        'f1_mid': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        'f1_late': float(f1_per_class[2]) if len(f1_per_class) > 2 else 0.0,
        'f1_macro': float(f1_macro),
    }


def compute_biomass_metrics(
    predictions: List[float],
    targets: List[float],
    name: str = 'biomass',
) -> Dict[str, float]:
    """Compute biomass regression metrics.
    
    Args:
        predictions: List of predicted biomass values
        targets: List of true biomass values
        name: Prefix for metric names
    
    Returns:
        Dict with MAE, R², Pearson r
    """
    # Filter out NaN values
    valid_pairs = [(p, t) for p, t in zip(predictions, targets) if not np.isnan(t)]
    
    if len(valid_pairs) == 0:
        return {
            f'{name}_mae': float('nan'),
            f'{name}_r2': float('nan'),
            f'{name}_pearson': float('nan'),
        }
    
    preds_list, targs_list = zip(*valid_pairs)
    
    mae = mean_absolute_error(targs_list, preds_list)
    r2 = r2_score(targs_list, preds_list)
    pearson_result = pearsonr(preds_list, targs_list)
    pearson_r = cast(float, pearson_result[0])
    
    return {
        f'{name}_mae': float(mae),
        f'{name}_r2': float(r2),
        f'{name}_pearson': pearson_r,
    }


def compute_genotype_ranking_metrics(
    predictions_by_genotype: Dict[str, List[float]],
    targets_by_genotype: Dict[str, List[float]],
) -> Dict[str, float]:
    """Compute genotype ranking metrics.
    
    For each genotype: average predicted DAG across 3 drought reps → rank.
    Compare to true DAG ranking.
    
    Args:
        predictions_by_genotype: Dict mapping genotype to list of predicted DAG values
        targets_by_genotype: Dict mapping genotype to list of true DAG values
    
    Returns:
        Dict with Spearman ρ, Kendall τ
    """
    # Average DAG per genotype
    genotypes = sorted(predictions_by_genotype.keys())
    
    pred_means = []
    target_means = []
    
    for genotype in genotypes:
        preds = predictions_by_genotype[genotype]
        targs = targets_by_genotype[genotype]
        
        # Filter out NaN values
        valid_preds = [p for p in preds if not np.isnan(p)]
        valid_targs = [t for t in targs if not np.isnan(t)]
        
        if len(valid_preds) > 0 and len(valid_targs) > 0:
            pred_means.append(np.mean(valid_preds))
            target_means.append(np.mean(valid_targs))
    
    if len(pred_means) < 2:
        return {
            'spearman_rho': float('nan'),
            'kendall_tau': float('nan'),
        }
    
    spearman_result = spearmanr(pred_means, target_means)
    kendall_result = kendalltau(pred_means, target_means)
    spearman_rho = cast(float, spearman_result[0])
    kendall_tau_val = cast(float, kendall_result[0])
    
    return {
        'spearman_rho': spearman_rho,
        'kendall_tau': kendall_tau_val,
    }


def compute_fold_metrics(fold_predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metrics for a single fold.
    
    Args:
        fold_predictions: Predictions dict for one fold
    
    Returns:
        Dict with all metrics
    """
    # Separate drought (WHC-30) and control (WHC-80) plants
    dag_reg_preds = []
    dag_reg_targets = []
    dag_cls_preds = []
    dag_cls_targets = []
    fw_preds = []
    fw_targets = []
    dw_preds = []
    dw_targets = []
    
    # For genotype ranking
    predictions_by_genotype: Dict[str, List[float]] = {}
    targets_by_genotype: Dict[str, List[float]] = {}
    
    for plant_id, pred in fold_predictions.items():
        treatment = pred['treatment']
        accession = pred['accession']
        
        # DAG regression and classification (WHC-30 only)
        if treatment == 'WHC-30':
            if pred['dag_reg'] is not None and not np.isnan(pred['dag_target']):
                dag_reg_preds.append(pred['dag_reg'])
                dag_reg_targets.append(pred['dag_target'])
            
            if pred['dag_cls'] is not None and pred['dag_category'] != -1:
                # Get predicted class (argmax of logits)
                dag_cls_preds.append(int(np.argmax(pred['dag_cls'])))
                dag_cls_targets.append(pred['dag_category'])
            
            # Track for genotype ranking
            if pred['dag_reg'] is not None:
                if accession not in predictions_by_genotype:
                    predictions_by_genotype[accession] = []
                    targets_by_genotype[accession] = []
                predictions_by_genotype[accession].append(pred['dag_reg'])
                targets_by_genotype[accession].append(pred['dag_target'])
        
        # Biomass (all plants)
        if pred['biomass'] is not None:
            fw_preds.append(pred['biomass'][0])
            fw_targets.append(pred['fw_target'])
            dw_preds.append(pred['biomass'][1])
            dw_targets.append(pred['dw_target'])
    
    # Compute metrics
    metrics = {}
    
    # DAG regression
    dag_reg_metrics = compute_dag_regression_metrics(dag_reg_preds, dag_reg_targets)
    metrics.update({f'dag_reg_{k}': v for k, v in dag_reg_metrics.items()})
    
    # DAG classification
    dag_cls_metrics = compute_dag_classification_metrics(dag_cls_preds, dag_cls_targets)
    metrics.update({f'dag_cls_{k}': v for k, v in dag_cls_metrics.items()})
    
    # Biomass
    fw_metrics = compute_biomass_metrics(fw_preds, fw_targets, 'fw')
    dw_metrics = compute_biomass_metrics(dw_preds, dw_targets, 'dw')
    metrics.update(fw_metrics)
    metrics.update(dw_metrics)
    
    # Genotype ranking
    ranking_metrics = compute_genotype_ranking_metrics(
        predictions_by_genotype,
        targets_by_genotype,
    )
    metrics.update({f'ranking_{k}': v for k, v in ranking_metrics.items()})
    
    return metrics


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval.
    
    Args:
        values: List of values (one per fold)
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level (default 0.95)
    
    Returns:
        (lower_bound, upper_bound)
    """
    if len(values) == 0:
        return (float('nan'), float('nan'))
    
    rng = np.random.RandomState(42)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        resample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(resample))
    
    alpha = 1.0 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return (float(lower), float(upper))


def aggregate_metrics(
    all_fold_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate metrics across folds.
    
    Args:
        all_fold_metrics: List of metrics dicts, one per fold
    
    Returns:
        Dict with mean ± std and 95% CI for all metrics
    """
    if len(all_fold_metrics) == 0:
        return {}
    
    # Collect all metric keys
    metric_keys = set()
    for metrics in all_fold_metrics:
        metric_keys.update(metrics.keys())
    
    aggregated = {}
    
    for key in sorted(metric_keys):
        values = [m[key] for m in all_fold_metrics if key in m and not np.isnan(m[key])]
        
        if len(values) == 0:
            aggregated[key] = {
                'mean': float('nan'),
                'std': float('nan'),
                'ci_lower': float('nan'),
                'ci_upper': float('nan'),
            }
        else:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            ci_lower, ci_upper = bootstrap_ci(values)
            
            aggregated[key] = {
                'mean': mean_val,
                'std': std_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
            }
    
    return aggregated


def print_results_table(aggregated: Dict[str, Any]) -> None:
    """Print formatted results table."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    sections = [
        ('DAG Regression (WHC-30 only)', ['dag_reg_mae', 'dag_reg_rmse', 'dag_reg_r2']),
        ('DAG Classification (WHC-30 only)', [
            'dag_cls_accuracy', 'dag_cls_balanced_accuracy',
            'dag_cls_f1_early', 'dag_cls_f1_mid', 'dag_cls_f1_late', 'dag_cls_f1_macro'
        ]),
        ('Biomass (Fresh Weight)', ['fw_mae', 'fw_r2', 'fw_pearson']),
        ('Biomass (Dry Weight)', ['dw_mae', 'dw_r2', 'dw_pearson']),
        ('Genotype Ranking', ['ranking_spearman_rho', 'ranking_kendall_tau']),
    ]
    
    for section_name, keys in sections:
        print(f"\n{section_name}:")
        print("-" * 80)
        for key in keys:
            if key in aggregated:
                stats = aggregated[key]
                mean = stats['mean']
                std = stats['std']
                ci_lower = stats['ci_lower']
                ci_upper = stats['ci_upper']
                
                # Format metric name
                metric_name = key.replace('dag_reg_', '').replace('dag_cls_', '').replace('ranking_', '')
                metric_name = metric_name.replace('_', ' ').title()
                
                print(f"  {metric_name:30s}: {mean:7.4f} ± {std:6.4f}  "
                      f"[{ci_lower:7.4f}, {ci_upper:7.4f}]")
    
    print("\n" + "="*80)


def compute_global_ranking_and_dag_metrics(
    all_predictions: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute ranking and DAG metrics across ALL folds (not per-fold).
    
    LOGO-CV tests one genotype per fold, so ranking must be computed globally.
    Similarly, DAG R² needs all genotypes to have meaningful variance.
    
    Args:
        all_predictions: Dict mapping fold_id to predictions dict
    
    Returns:
        Dict with global ranking metrics and DAG metrics
    """
    # Collect all DAG predictions across folds
    predictions_by_genotype: Dict[str, List[float]] = {}
    targets_by_genotype: Dict[str, List[float]] = {}
    
    all_dag_preds = []
    all_dag_targets = []
    
    for fold_id, fold_predictions in all_predictions.items():
        for plant_id, pred in fold_predictions.items():
            if pred['treatment'] != 'WHC-30':
                continue
            
            accession = pred['accession']
            
            if pred['dag_reg'] is not None and not np.isnan(pred['dag_target']):
                dag_pred = pred['dag_reg']
                dag_target = pred['dag_target']
                
                all_dag_preds.append(dag_pred)
                all_dag_targets.append(dag_target)
                
                if accession not in predictions_by_genotype:
                    predictions_by_genotype[accession] = []
                    targets_by_genotype[accession] = []
                predictions_by_genotype[accession].append(dag_pred)
                targets_by_genotype[accession].append(dag_target)
    
    results = {}
    
    # Global DAG regression metrics (across all genotypes)
    if len(all_dag_preds) > 0:
        results['global_dag_mae'] = float(mean_absolute_error(all_dag_targets, all_dag_preds))
        results['global_dag_rmse'] = float(np.sqrt(mean_squared_error(all_dag_targets, all_dag_preds)))
        
        target_var = np.var(all_dag_targets)
        if target_var > 1e-10:
            results['global_dag_r2'] = float(r2_score(all_dag_targets, all_dag_preds))
        else:
            results['global_dag_r2'] = float('nan')
    
    # Global ranking metrics
    ranking_metrics = compute_genotype_ranking_metrics(
        predictions_by_genotype,
        targets_by_genotype,
    )
    results['global_ranking_spearman_rho'] = ranking_metrics['spearman_rho']
    results['global_ranking_kendall_tau'] = ranking_metrics['kendall_tau']
    results['n_genotypes_for_ranking'] = len(predictions_by_genotype)
    
    return results


def main() -> None:
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained model across all folds'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing fold_0/, fold_1/, ..., fold_43/',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path for aggregated results JSON (default: results_dir/main_results.json)',
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Load predictions
    print("Loading predictions from all folds...")
    all_predictions = load_predictions(results_dir)
    print(f"Loaded predictions from {len(all_predictions)} folds")
    
    # Compute per-fold metrics
    print("\nComputing per-fold metrics...")
    all_fold_metrics = []
    for fold_id in sorted(all_predictions.keys()):
        fold_predictions = all_predictions[fold_id]
        fold_metrics = compute_fold_metrics(fold_predictions)
        fold_metrics['fold_id'] = fold_id
        all_fold_metrics.append(fold_metrics)
    
    # Aggregate across folds
    print("Aggregating metrics across folds...")
    aggregated = aggregate_metrics(all_fold_metrics)
    
    # Compute global ranking and DAG metrics (across all folds)
    print("Computing global ranking metrics...")
    global_metrics = compute_global_ranking_and_dag_metrics(all_predictions)
    
    # Save results
    output_path = Path(args.output) if args.output else results_dir / 'main_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'aggregated': aggregated,
        'global_metrics': global_metrics,
        'per_fold': all_fold_metrics,
        'n_folds': len(all_fold_metrics),
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary table
    print_results_table(aggregated)
    
    # Print global metrics
    print("\n" + "="*80)
    print("GLOBAL METRICS (across all 44 folds)")
    print("="*80)
    print(f"\nDAG Regression (n={global_metrics.get('n_genotypes_for_ranking', 0)} genotypes):")
    print(f"  MAE:  {global_metrics.get('global_dag_mae', float('nan')):7.4f}")
    print(f"  RMSE: {global_metrics.get('global_dag_rmse', float('nan')):7.4f}")
    print(f"  R²:   {global_metrics.get('global_dag_r2', float('nan')):7.4f}")
    print(f"\nGenotype Ranking:")
    print(f"  Spearman ρ: {global_metrics.get('global_ranking_spearman_rho', float('nan')):7.4f}")
    print(f"  Kendall τ:  {global_metrics.get('global_ranking_kendall_tau', float('nan')):7.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
