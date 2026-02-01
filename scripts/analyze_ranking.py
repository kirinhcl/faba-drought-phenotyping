#!/usr/bin/env python3
"""Run genotype ranking analysis and embedding visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.analysis.embedding_viz import extract_and_visualize_embeddings
from src.analysis.ranking import compute_genotype_ranking
from src.data.dataset import FabaDroughtDataset
from src.model.model import FabaDroughtModel
from src.utils.config import load_config


def main() -> None:
    """Run ranking analysis and embedding visualization."""
    parser = argparse.ArgumentParser(
        description='Run genotype ranking analysis'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory with fold_N/predictions.json',
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
        help='Fold to use for embedding extraction (default: 0)',
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
    
    # Load plant metadata
    metadata_path = Path(cfg.data.plant_metadata)
    plant_metadata = pd.read_csv(metadata_path)
    
    print(f"\n{'='*80}")
    print(f"Ranking Analysis")
    print(f"  Results dir: {args.results_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*80}\n")
    
    # =============================================================================
    # 1. Compute genotype ranking
    # =============================================================================
    print("\n[1/2] Computing genotype ranking...")
    ranking_results = compute_genotype_ranking(
        predictions_dir=Path(args.results_dir),
        plant_metadata=plant_metadata,
        output_dir=output_dir,
    )
    
    # =============================================================================
    # 2. Extract and visualize embeddings
    # =============================================================================
    print("\n[2/2] Extracting embeddings for visualization...")
    
    # Load dataset
    dataset = FabaDroughtDataset(cfg)
    
    # Create model
    model = FabaDroughtModel(cfg)
    
    # Load best model from specified fold
    fold_dir = Path(args.results_dir) / f"fold_{args.fold}"
    model_path = fold_dir / 'best_model_state.pt'
    
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}, skipping embedding extraction")
    else:
        model_device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model.to(model_device)
        model.load_state_dict(torch.load(model_path, map_location=model_device, weights_only=True))
        
        embedding_metadata = extract_and_visualize_embeddings(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            device=args.device,
        )
    
    # =============================================================================
    # Summary
    # =============================================================================
    print(f"\n{'='*80}")
    print("Ranking Analysis Complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - ranking_results.json")
    print(f"  - embeddings_tsne.npy")
    print(f"  - embeddings_umap.npy (if umap available)")
    print(f"  - embedding_metadata.json")
    print()


if __name__ == '__main__':
    main()
