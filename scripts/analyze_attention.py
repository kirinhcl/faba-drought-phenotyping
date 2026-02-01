#!/usr/bin/env python3
"""Run all XAI attention-based analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.analysis.attention import extract_attention_maps
from src.analysis.early_detection import progressive_truncation_evaluation
from src.analysis.fluorescence_changepoint import detect_fluorescence_changepoints
from src.analysis.presymptomatic import compute_triangulation
from src.data.dataset import FabaDroughtDataset
from src.model.model import FabaDroughtModel
from src.training.cv import LogoCV
from src.utils.config import load_config


def main() -> None:
    """Run all XAI attention analyses."""
    parser = argparse.ArgumentParser(
        description='Run XAI attention-based analyses'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory with fold_N/best_model_state.pt',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for analysis results',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config YAML file',
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='Fold to analyze (default: 0)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)',
    )
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = FabaDroughtDataset(cfg)
    
    # Load plant metadata
    metadata_path = Path(cfg.data.plant_metadata)
    plant_metadata = pd.read_csv(metadata_path)
    
    # Create cross-validation splitter
    cv = LogoCV(
        plant_metadata_df=plant_metadata,
        n_folds=cfg.training.cv.n_folds,
        stratify_col=cfg.training.cv.stratify_by,
        seed=cfg.training.cv.seed,
    )
    
    # Get test indices for the specified fold
    for fold_id, (train_indices, val_indices, test_indices) in enumerate(cv.split()):
        if fold_id == args.fold:
            break
    
    # Create model
    model = FabaDroughtModel(cfg)
    
    # Define fold results directory
    fold_dir = Path(args.model_dir) / f"fold_{args.fold}"
    
    print(f"\n{'='*80}")
    print(f"Analyzing fold {args.fold}")
    print(f"  Model dir: {fold_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*80}\n")
    
    # =============================================================================
    # 1. Extract attention maps
    # =============================================================================
    print("\n[1/4] Extracting attention maps...")
    attention_data = extract_attention_maps(
        model=model,
        dataset=dataset,
        fold_results_dir=fold_dir,
        device=args.device,
    )
    print(f"  → Extracted attention for {len(attention_data)} genotypes")
    
    # Save attention data
    import json
    import numpy as np
    
    # Convert numpy arrays to lists for JSON serialization
    attention_data_serializable = {}
    for genotype, data in attention_data.items():
        attention_data_serializable[genotype] = {
            'attention_map': data['attention_map'].tolist(),
            'plant_attention_maps': {
                plant_id: attn.tolist()
                for plant_id, attn in data['plant_attention_maps'].items()
            },
            'treatment_counts': data['treatment_counts'],
        }
    
    attention_path = output_dir / 'attention_maps.json'
    with open(attention_path, 'w') as f:
        json.dump(attention_data_serializable, f, indent=2)
    print(f"  → Saved to {attention_path}")
    
    # =============================================================================
    # 2. Detect fluorescence change points
    # =============================================================================
    print("\n[2/4] Detecting fluorescence change points...")
    fluor_changepoints = detect_fluorescence_changepoints(cfg)
    print(f"  → Detected change points for {len(fluor_changepoints)} genotypes")
    
    # Save fluorescence change points
    fluor_path = output_dir / 'fluorescence_changepoints.json'
    with open(fluor_path, 'w') as f:
        json.dump(fluor_changepoints, f, indent=2)
    print(f"  → Saved to {fluor_path}")
    
    # =============================================================================
    # 3. Compute three-way triangulation
    # =============================================================================
    print("\n[3/4] Computing three-way triangulation...")
    triangulation_summary = compute_triangulation(
        attention_data=attention_data,
        fluor_changepoints=fluor_changepoints,
        plant_metadata=plant_metadata,
        output_dir=output_dir,
    )
    
    # =============================================================================
    # 4. Progressive truncation evaluation
    # =============================================================================
    print("\n[4/4] Progressive truncation evaluation...")
    early_detection_results = progressive_truncation_evaluation(
        model=model,
        dataset=dataset,
        test_indices=test_indices,
        output_dir=output_dir,
        device=args.device,
    )
    
    # =============================================================================
    # Summary
    # =============================================================================
    print(f"\n{'='*80}")
    print("XAI Analysis Complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - attention_maps.json")
    print(f"  - fluorescence_changepoints.json")
    print(f"  - triangulation_summary.json")
    print(f"  - presymptomatic_summary.json")
    print(f"  - early_detection.json")
    print()


if __name__ == '__main__':
    main()
