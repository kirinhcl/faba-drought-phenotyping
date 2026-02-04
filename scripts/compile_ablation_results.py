#!/usr/bin/env python3
"""Compile ablation study results from all variants into a single JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def load_variant_results(variant_dir: Path) -> Dict[str, Any] | None:
    """Load main_results.json from a variant directory.
    
    Args:
        variant_dir: Directory containing main_results.json
        
    Returns:
        Dict with metrics or None if not found
    """
    results_file = variant_dir / 'main_results.json'
    if not results_file.exists():
        return None
    
    with open(results_file) as f:
        return json.load(f)


def extract_key_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics for comparison.
    
    Args:
        results: Full results dict from main_results.json
        
    Returns:
        Dict with key metrics and their std deviations
    """
    aggregated = results.get('aggregated', {})
    global_metrics = results.get('global_metrics', {})
    
    metrics = {}
    
    # DAG regression
    if 'dag_reg_mae' in aggregated:
        metrics['dag_mae'] = aggregated['dag_reg_mae']['mean']
        metrics['dag_mae_std'] = aggregated['dag_reg_mae']['std']
    
    # Use global R² (more meaningful for LOGO-CV)
    if 'global_dag_r2' in global_metrics:
        metrics['dag_r2'] = global_metrics['global_dag_r2']
    
    # Biomass (Fresh Weight)
    if 'fw_r2' in aggregated:
        metrics['biomass_r2'] = aggregated['fw_r2']['mean']
        metrics['biomass_r2_std'] = aggregated['fw_r2']['std']
    
    if 'fw_pearson' in aggregated:
        metrics['biomass_pearson'] = aggregated['fw_pearson']['mean']
    
    # Ranking (use global, more meaningful)
    if 'global_ranking_spearman_rho' in global_metrics:
        metrics['ranking_spearman'] = global_metrics['global_ranking_spearman_rho']
    
    if 'global_ranking_kendall_tau' in global_metrics:
        metrics['ranking_kendall'] = global_metrics['global_ranking_kendall_tau']
    
    # Classification
    if 'dag_cls_balanced_accuracy' in aggregated:
        metrics['dag_cls_balanced_acc'] = aggregated['dag_cls_balanced_accuracy']['mean']
    
    if 'dag_cls_f1_macro' in aggregated:
        metrics['dag_cls_f1_macro'] = aggregated['dag_cls_f1_macro']['mean']
    
    return metrics


def main() -> None:
    """Compile ablation results."""
    parser = argparse.ArgumentParser(
        description='Compile ablation study results'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Base results directory containing variant subdirectories',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON path (default: results_dir/ablation_comparison.json)',
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / 'ablation_comparison.json'
    
    # Expected variant names (based on configs/ablation/)
    variant_names = [
        'full_model',
        'no_fluorescence',
        'no_environment', 
        'no_vi',
        'no_temporal',
        'image_only',
        'clip_full',
        'bioclip_full',
    ]
    
    # Also try to discover variants automatically
    discovered_variants = set()
    for subdir in results_dir.iterdir():
        if subdir.is_dir() and (subdir / 'main_results.json').exists():
            discovered_variants.add(subdir.name)
    
    all_variants = set(variant_names) | discovered_variants
    
    print(f"Checking variants in {results_dir}...")
    
    compiled = {}
    
    for variant in sorted(all_variants):
        variant_dir = results_dir / variant
        
        if not variant_dir.exists():
            print(f"  {variant}: not found")
            continue
        
        results = load_variant_results(variant_dir)
        
        if results is None:
            print(f"  {variant}: no main_results.json")
            continue
        
        metrics = extract_key_metrics(results)
        
        if not metrics:
            print(f"  {variant}: no metrics extracted")
            continue
        
        # Map variant name to display name for figures
        display_names = {
            'full_model': 'Ours (DINOv2)',
            'no_fluorescence': 'Ours (No Fluor)',
            'no_environment': 'Ours (No Env)',
            'no_vi': 'Ours (No VI)',
            'no_temporal': 'Ours (No Temporal)',
            'image_only': 'Baseline (Image Only)',
            'clip_full': 'Ours (CLIP)',
            'bioclip_full': 'Ours (BioCLIP)',
        }
        
        display_name = display_names.get(variant, variant)
        compiled[display_name] = metrics
        
        print(f"  {variant} -> {display_name}: DAG MAE={metrics.get('dag_mae', 'N/A'):.2f}, "
              f"FW R²={metrics.get('biomass_r2', 'N/A'):.3f}, "
              f"Ranking ρ={metrics.get('ranking_spearman', 'N/A'):.3f}")
    
    if not compiled:
        print("\nNo variant results found!")
        return
    
    # Save compiled results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(compiled, f, indent=2)
    
    print(f"\nCompiled {len(compiled)} variants to {output_path}")


if __name__ == '__main__':
    main()
