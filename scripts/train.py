#!/usr/bin/env python3
"""Train the temporal multimodal drought phenotyping model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data.collate import faba_collate_fn
from src.data.dataset import FabaDroughtDataset
from src.model.model import FabaDroughtModel
from src.training.cv import LogoCV
from src.training.trainer import Trainer
from src.utils.config import load_config


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_fold(
    cfg: Any,
    dataset: Any,
    fold_id: int,
    train_indices: npt.NDArray[np.int_],
    val_indices: npt.NDArray[np.int_],
    test_indices: npt.NDArray[np.int_],
    checkpoint_dir: Path,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a single fold.
    
    Args:
        cfg: OmegaConf config
        dataset: Full dataset
        fold_id: Fold index
        train_indices: Training indices
        val_indices: Validation indices
        test_indices: Test indices
        checkpoint_dir: Directory for checkpoints
        resume_from: Optional checkpoint path to resume from
    
    Returns:
        Dict with metrics and predictions
    """
    # Create subsets
    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())
    test_dataset = Subset(dataset, test_indices.tolist())
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=faba_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=faba_collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=faba_collate_fn,
        pin_memory=True,
    )
    
    # Create model
    model = FabaDroughtModel(cfg)
    
    # Create fold checkpoint directory
    fold_checkpoint_dir = checkpoint_dir / f"fold_{fold_id}"
    fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        fold_id=fold_id,
        checkpoint_dir=fold_checkpoint_dir,
    )
    
    # Resume from checkpoint if specified
    if resume_from:
        trainer.load_checkpoint(resume_from)
    
    # Train
    print(f"\n{'='*80}")
    print(f"Training fold {fold_id}")
    print(f"  Train: {len(train_indices)} plants")
    print(f"  Val:   {len(val_indices)} plants")
    print(f"  Test:  {len(test_indices)} plants")
    print(f"{'='*80}\n")
    
    best_model_path = trainer.train()
    
    # Get final training loss
    final_train_losses = trainer.train_epoch()
    
    # Predict on test set
    predictions = trainer.predict(test_loader)
    
    # Save predictions
    pred_path = fold_checkpoint_dir / 'predictions.json'
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Compute validation losses one more time for metrics
    val_losses = trainer.validate()
    
    # Save metrics
    metrics = {
        'fold_id': fold_id,
        'best_epoch': trainer.best_epoch,
        'best_val_loss': trainer.best_val_loss,
        'final_train_loss': final_train_losses['total'],
        'val_losses': val_losses,
        'train_losses': final_train_losses,
    }
    metrics_path = fold_checkpoint_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nFold {fold_id} complete:")
    print(f"  Best epoch: {trainer.best_epoch}")
    print(f"  Best val loss: {trainer.best_val_loss:.4f}")
    print(f"  Predictions saved to: {pred_path}")
    print(f"  Metrics saved to: {metrics_path}")
    
    return metrics


def main() -> None:
    """Main training script."""
    parser = argparse.ArgumentParser(
        description='Train the temporal multimodal drought phenotyping model'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file',
    )
    parser.add_argument(
        '--fold',
        type=str,
        required=True,
        help='Fold index (int) or "all" for all folds',
    )
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Path to checkpoint to resume from',
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='Base directory for checkpoints (default from config)',
    )
    parser.add_argument(
        '--feature_dir',
        type=str,
        default=None,
        help='Override feature directory (e.g., for local SSD on HPC)',
    )
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Override feature dir if specified
    if args.feature_dir:
        cfg.data.feature_dir = args.feature_dir
    
    # Set seed
    set_seed(cfg.seed)
    
    # Determine checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = Path(cfg.logging.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    print("Loading dataset...")
    dataset = FabaDroughtDataset(cfg)
    
    # Handle overfit_n config option
    if hasattr(cfg.data, 'overfit_n') and cfg.data.overfit_n is not None:
        print(f"Overfit mode: using first {cfg.data.overfit_n} plants")
        subset_indices = list(range(min(cfg.data.overfit_n, len(dataset))))
        dataset = Subset(dataset, subset_indices)  # type: ignore
    
    # Load plant metadata for CV
    metadata_path = Path(cfg.data.plant_metadata)
    plant_metadata = pd.read_csv(metadata_path)
    
    # Create cross-validation splitter
    cv = LogoCV(
        plant_metadata_df=plant_metadata,
        n_folds=cfg.training.cv.n_folds,
        stratify_col=cfg.training.cv.stratify_by,
        seed=cfg.training.cv.seed,
    )
    
    # Determine which folds to train
    if args.fold.lower() == 'all':
        fold_ids = list(range(cfg.training.cv.n_folds))
    else:
        try:
            fold_ids = [int(args.fold)]
        except ValueError:
            print(f"Error: --fold must be an integer or 'all', got '{args.fold}'")
            sys.exit(1)
    
    # Train each fold
    all_metrics = []
    for fold_id, (train_indices, val_indices, test_indices) in enumerate(cv.split()):
        if fold_id not in fold_ids:
            continue
        
        metrics = train_fold(
            cfg=cfg,
            dataset=dataset,
            fold_id=fold_id,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            checkpoint_dir=checkpoint_dir,
            resume_from=args.resume_from,
        )
        all_metrics.append(metrics)
    
    # Save aggregate metrics if training multiple folds
    if len(all_metrics) > 1:
        aggregate_path = checkpoint_dir / 'aggregate_metrics.json'
        with open(aggregate_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nAggregate metrics saved to: {aggregate_path}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
