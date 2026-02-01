#!/usr/bin/env python3
"""Train the student model via knowledge distillation from multimodal teacher."""

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
from src.model.student import FabaDroughtStudent
from src.training.cv import LogoCV
from src.training.losses import DistillationLoss
from src.utils.config import load_config


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class DistillationTrainer:
    """Distillation trainer with alpha annealing."""
    
    def __init__(
        self,
        student: torch.nn.Module,
        teacher: torch.nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        cfg: Any,
        fold_id: int,
        checkpoint_dir: Path,
        alpha_start: float = 0.7,
        alpha_end: float = 0.3,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.fold_id = fold_id
        self.checkpoint_dir = checkpoint_dir
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.student.to(self.device)
        self.teacher.to(self.device)
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        
        # Loss function (alpha will be updated each epoch)
        loss_weights_raw = cfg.training.loss_weights
        loss_weights_dict: Dict[str, float] = {
            'dag_reg': float(loss_weights_raw.dag_reg),
            'dag_cls': float(loss_weights_raw.dag_cls),
            'biomass': float(loss_weights_raw.biomass),
            'trajectory': float(loss_weights_raw.trajectory),
        }
        self.criterion = DistillationLoss(alpha=alpha_start, loss_weights=loss_weights_dict)
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Learning rate scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        warmup_epochs = cfg.training.warmup_epochs
        max_epochs = cfg.training.max_epochs
        
        if warmup_epochs > 0:
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs - warmup_epochs,
                eta_min=1e-6,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs,
                eta_min=1e-6,
            )
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
        self.current_epoch = 0
        
        # Wandb
        self.use_wandb = cfg.logging.wandb.mode != 'disabled'
        if self.use_wandb:
            import wandb
            from omegaconf import OmegaConf
            wandb.init(
                project=str(cfg.logging.wandb.project) + '-distill',
                entity=str(cfg.logging.wandb.entity) if cfg.logging.wandb.entity else None,
                name=f"distill_fold_{fold_id}",
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=str(cfg.logging.wandb.mode),
            )
    
    def compute_alpha(self, epoch: int) -> float:
        """Compute alpha for current epoch (linear annealing)."""
        max_epochs = self.cfg.training.max_epochs
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * (epoch / max_epochs)
        return alpha
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.student.train()
        self.teacher.eval()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Mixed precision forward pass
            with torch.amp.autocast('cuda', dtype=torch.float16):
                student_outputs = self.student(batch)
                
                # Teacher forward pass (no grad)
                with torch.no_grad():
                    teacher_outputs = self.teacher(batch)
                
                loss = self.criterion(student_outputs, teacher_outputs, batch)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                self.cfg.training.gradient_clip,
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Accumulate loss
            epoch_loss += loss.item()
            num_batches += 1
        
        return {'total': epoch_loss / max(num_batches, 1)}
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.student.eval()
        self.teacher.eval()
        
        epoch_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                student_outputs = self.student(batch)
                teacher_outputs = self.teacher(batch)
                
                loss = self.criterion(student_outputs, teacher_outputs, batch)
                
                # Accumulate loss
                epoch_loss += loss.item()
                num_batches += 1
        
        return {'total': epoch_loss / max(num_batches, 1)}
    
    def train(self) -> Path:
        """Main training loop."""
        import time
        
        for epoch in range(self.current_epoch, self.cfg.training.max_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Update alpha
            alpha = self.compute_alpha(epoch)
            self.criterion.alpha = alpha
            
            # Training
            train_losses = self.train_epoch()
            
            # Validation
            val_losses = self.validate()
            
            # Scheduler step
            self.scheduler.step()
            
            # Compute metrics
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            if self.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'lr': current_lr,
                    'alpha': alpha,
                    'epoch_time': epoch_time,
                    'train/total': train_losses['total'],
                    'val/total': val_losses['total'],
                })
            
            # Console logging
            print(
                f"Epoch {epoch}/{self.cfg.training.max_epochs} | "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f} | "
                f"Alpha: {alpha:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Early stopping check
            val_total_loss = val_losses['total']
            if val_total_loss < self.best_val_loss:
                self.best_val_loss = val_total_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                best_path = self.checkpoint_dir / 'best_model.pt'
                self.save_checkpoint(best_path, epoch)
                print(f"  → New best model saved (val_loss={val_total_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.training.patience:
                    print(f"Early stopping at epoch {epoch} (patience={self.cfg.training.patience})")
                    break
            
            # Save last checkpoint
            last_path = self.checkpoint_dir / 'last_checkpoint.pt'
            self.save_checkpoint(last_path, epoch)
        
        best_path = self.checkpoint_dir / 'best_model.pt'
        return best_path
    
    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
        }
        torch.save(checkpoint, path)
    
    def predict(self, test_loader: DataLoader[Any]) -> Dict[str, Dict[str, Any]]:
        """Run inference on test set."""
        # Load best model
        best_model_path = self.checkpoint_dir / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.student.load_state_dict(checkpoint['model_state_dict'])
        
        self.student.eval()
        predictions: Dict[str, Dict[str, Any]] = {}
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                batch_gpu = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                outputs = self.student(batch_gpu)
                
                # Collect predictions per plant
                plant_ids = batch['plant_id']
                batch_size = len(plant_ids)
                
                for i in range(batch_size):
                    plant_id = plant_ids[i]
                    predictions[plant_id] = {
                        'dag_reg': float(outputs['dag_reg'][i].cpu().item()) if outputs['dag_reg'] is not None else None,
                        'dag_cls': outputs['dag_cls'][i].cpu().tolist() if outputs['dag_cls'] is not None else None,
                        'biomass': outputs['biomass'][i].cpu().tolist() if outputs['biomass'] is not None else None,
                        'trajectory': outputs['trajectory'][i].cpu().tolist() if outputs['trajectory'] is not None else None,
                        'treatment': batch['treatment'][i],
                        'accession': batch['accession'][i],
                        'dag_target': float(batch['dag_target'][i].cpu().item()),
                        'dag_category': int(batch['dag_category'][i].cpu().item()),
                        'fw_target': float(batch['fw_target'][i].cpu().item()),
                        'dw_target': float(batch['dw_target'][i].cpu().item()),
                    }
        
        return predictions


def train_fold(
    cfg: Any,
    dataset: Any,
    fold_id: int,
    train_indices: npt.NDArray[np.int_],
    val_indices: npt.NDArray[np.int_],
    test_indices: npt.NDArray[np.int_],
    teacher_checkpoint_path: Path,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    """Train a single fold.
    
    Args:
        cfg: OmegaConf config
        dataset: Full dataset
        fold_id: Fold index
        train_indices: Training indices
        val_indices: Validation indices
        test_indices: Test indices
        teacher_checkpoint_path: Path to teacher model checkpoint
        checkpoint_dir: Directory for checkpoints
    
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
    
    # Create models
    teacher = FabaDroughtModel(cfg)
    student = FabaDroughtStudent(cfg)
    
    # Load teacher weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=device, weights_only=True)
    teacher.load_state_dict(teacher_checkpoint)
    
    # Create fold checkpoint directory
    fold_checkpoint_dir = checkpoint_dir / f"fold_{fold_id}"
    fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get distillation hyperparameters
    if hasattr(cfg, 'distillation'):
        alpha_start = cfg.distillation.alpha_start
        alpha_end = cfg.distillation.alpha_end
    else:
        alpha_start = 0.7
        alpha_end = 0.3
    
    # Create trainer
    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        fold_id=fold_id,
        checkpoint_dir=fold_checkpoint_dir,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
    )
    
    # Train
    print(f"\n{'='*80}")
    print(f"Training distillation fold {fold_id}")
    print(f"  Teacher: {teacher_checkpoint_path}")
    print(f"  Train: {len(train_indices)} plants")
    print(f"  Val:   {len(val_indices)} plants")
    print(f"  Test:  {len(test_indices)} plants")
    print(f"  Alpha: {alpha_start:.2f} → {alpha_end:.2f}")
    print(f"{'='*80}\n")
    
    best_model_path = trainer.train()
    
    # Get final losses
    final_train_losses = trainer.train_epoch()
    val_losses = trainer.validate()
    
    # Predict on test set
    predictions = trainer.predict(test_loader)
    
    # Save predictions
    pred_path = fold_checkpoint_dir / 'predictions.json'
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
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
    """Main distillation training script."""
    parser = argparse.ArgumentParser(
        description='Train student model via knowledge distillation'
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
        '--teacher_dir',
        type=str,
        required=True,
        help='Directory with teacher fold_N/best_model_state.pt',
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='Base directory for student checkpoints',
    )
    parser.add_argument(
        '--feature_dir',
        type=str,
        default=None,
        help='Override feature directory',
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
    teacher_dir = Path(args.teacher_dir)
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = Path(cfg.logging.checkpoint_dir) / 'distillation'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    print("Loading dataset...")
    dataset = FabaDroughtDataset(cfg)
    
    # Load plant metadata for CV
    metadata_path = Path(cfg.data.plant_metadata)
    plant_metadata = pd.read_csv(metadata_path)
    
    # Create cross-validation splitter (same seed as teacher)
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
        
        # Get teacher checkpoint path
        teacher_checkpoint_path = teacher_dir / f"fold_{fold_id}" / "best_model_state.pt"
        if not teacher_checkpoint_path.exists():
            print(f"Error: Teacher checkpoint not found: {teacher_checkpoint_path}")
            sys.exit(1)
        
        metrics = train_fold(
            cfg=cfg,
            dataset=dataset,
            fold_id=fold_id,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            teacher_checkpoint_path=teacher_checkpoint_path,
            checkpoint_dir=checkpoint_dir,
        )
        all_metrics.append(metrics)
    
    # Save aggregate metrics if training multiple folds
    if len(all_metrics) > 1:
        aggregate_path = checkpoint_dir / 'aggregate_metrics.json'
        with open(aggregate_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nAggregate metrics saved to: {aggregate_path}")
    
    print("\nDistillation training complete!")


if __name__ == '__main__':
    main()
