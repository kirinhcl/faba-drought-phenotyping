#!/usr/bin/env python3
"""Train the stress detection model for binary per-timestep classification."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

from src.data.collate import faba_collate_fn
from src.data.dataset import FabaDroughtDataset
from src.model.stress_model import StressDetectionModel
from src.training.cv import LogoCV
from src.training.stress_loss import StressLoss
from src.utils.config import load_config


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_fold(
    cfg: Any,
    dataset: Any,
    plant_metadata: pd.DataFrame,
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
        dataset: FabaDroughtDataset
        plant_metadata: Plant metadata DataFrame
        fold_id: Fold index
        train_indices: Training plant indices
        val_indices: Validation plant indices
        test_indices: Test plant indices
        checkpoint_dir: Base checkpoint directory
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Dictionary of metrics for this fold
    """
    # Create subsets
    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())
    test_dataset = Subset(dataset, test_indices.tolist())
    
    # Create data loaders (no grouped sampler for stress task)
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
    
    # Create model and loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StressDetectionModel(cfg)
    model.to(device)
    # pos_weight: None = auto (num_neg/num_pos per batch), or fixed float from config
    pw = getattr(cfg.training, 'pos_weight', None)
    criterion = StressLoss(pos_weight=pw)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    
    # Create scheduler with warmup
    warmup_epochs = cfg.training.warmup_epochs
    max_epochs = cfg.training.max_epochs
    
    if warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=1e-6,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=1e-6,
        )
    
    # Create fold checkpoint directory
    fold_checkpoint_dir = checkpoint_dir / f"fold_{fold_id}"
    fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Early stopping state
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    current_epoch = 0
    
    # Resume from checkpoint if specified
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        best_epoch = checkpoint['best_epoch']
        patience_counter = checkpoint['patience_counter']
        print(f"Resumed from checkpoint at epoch {checkpoint['epoch']}")
    
    # Training info
    print(f"\n{'='*80}")
    print(f"Training fold {fold_id}")
    print(f"  Train: {len(train_indices)} plants")
    print(f"  Val:   {len(val_indices)} plants")
    print(f"  Test:  {len(test_indices)} plants")
    print(f"{'='*80}\n")
    
    # Training loop
    amp_dtype = torch.bfloat16
    
    for epoch in range(current_epoch, max_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                predictions = model(batch)
                loss, loss_dict = criterion(predictions, batch)
            
            # NaN guard
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.training.gradient_clip,
            )
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad()
                continue
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Accumulate loss
            train_loss += loss.item()
            num_train_batches += 1
        
        train_loss /= max(num_train_batches, 1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    predictions = model(batch)
                    loss, loss_dict = criterion(predictions, batch)
                
                val_loss += loss.item()
                num_val_batches += 1
        
        val_loss /= max(num_val_batches, 1)
        
        # Scheduler step
        scheduler.step()
        
        # Compute metrics
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Console logging
        print(
            f"Epoch {epoch}/{max_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            best_path = fold_checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
            }, best_path)
            
            # Save model state dict only for inference
            model_only_path = fold_checkpoint_dir / 'best_model_state.pt'
            torch.save(model.state_dict(), model_only_path)
            
            print(f"  → New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.patience:
                print(f"Early stopping at epoch {epoch} (patience={cfg.training.patience})")
                break
        
        # Save last checkpoint (for resume)
        last_path = fold_checkpoint_dir / 'last_checkpoint.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'patience_counter': patience_counter,
        }, last_path)
    
    # Load best model for evaluation
    best_model_path = fold_checkpoint_dir / 'best_model_state.pt'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    
    # Evaluate on test set and collect modality gates
    model.eval()
    test_loss = 0.0
    num_test_batches = 0
    all_gates = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward pass
            predictions = model(batch)
            loss, loss_dict = criterion(predictions, batch)
            
            test_loss += loss.item()
            num_test_batches += 1
            
            # Collect modality gates
            gates = predictions['modality_gates'].cpu()  # (B, T, 4)
            all_gates.append(gates)
    
    test_loss /= max(num_test_batches, 1)
    
    # Save modality gates for analysis
    all_gates_tensor = torch.cat(all_gates, dim=0)  # (N, T, 4)
    gates_path = fold_checkpoint_dir / 'modality_gates.pt'
    torch.save(all_gates_tensor, gates_path)
    
    # Save metrics
    metrics = {
        'fold_id': fold_id,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
    }
    metrics_path = fold_checkpoint_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nFold {fold_id} complete:")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Metrics saved to: {metrics_path}")
    print(f"  Modality gates saved to: {gates_path}")
    
    return metrics


def main() -> None:
    """Main training script."""
    parser = argparse.ArgumentParser(
        description='Train the stress detection model'
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
    
    # Load config (overlay on default.yaml)
    base_config = str(Path(args.config).parent.parent / 'configs' / 'default.yaml')
    if not Path(base_config).exists():
        base_config = 'configs/default.yaml'
    cfg = load_config(args.config, base_path=base_config)
    
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
            plant_metadata=plant_metadata,
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
