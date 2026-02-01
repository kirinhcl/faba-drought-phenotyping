"""Training loop for faba bean drought phenotyping model."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.training.losses import MultiTaskLoss


class Trainer:
    """Training loop for faba bean drought phenotyping model.
    
    Supports:
    - Multi-task training with configurable loss weights
    - Mixed precision (fp16) training
    - Gradient clipping (max_norm=1.0)
    - Cosine annealing LR scheduler with warmup
    - Early stopping based on validation loss
    - Wandb logging
    - Checkpoint saving/loading (best + last model per fold)
    - Resume from checkpoint for CSC queue resilience
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        cfg: DictConfig,
        fold_id: int,
        checkpoint_dir: Union[str, Path],
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: FabaDroughtModel or FabaDroughtStudent
            train_loader: Training data loader
            val_loader: Validation data loader
            cfg: Full OmegaConf config
            fold_id: Integer fold index
            checkpoint_dir: Path for saving checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.fold_id = fold_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        
        # Loss function
        loss_weights_raw = cfg.training.loss_weights
        loss_weights_dict: Dict[str, float] = {
            'dag_reg': float(loss_weights_raw.dag_reg),
            'dag_cls': float(loss_weights_raw.dag_cls),
            'biomass': float(loss_weights_raw.biomass),
            'trajectory': float(loss_weights_raw.trajectory),
        }
        self.criterion = MultiTaskLoss(loss_weights_dict)
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Learning rate scheduler with warmup
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
        
        # Wandb initialization
        self.use_wandb = cfg.logging.wandb.mode != 'disabled'
        if self.use_wandb:
            import wandb
            wandb.init(
                project=str(cfg.logging.wandb.project),
                entity=str(cfg.logging.wandb.entity) if cfg.logging.wandb.entity else None,
                name=f"fold_{fold_id}",
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=str(cfg.logging.wandb.mode),
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Average training losses dict
        """
        self.model.train()
        epoch_losses: Dict[str, float] = {
            'total': 0.0,
            'dag_reg': 0.0,
            'dag_cls': 0.0,
            'biomass': 0.0,
            'trajectory': 0.0,
        }
        num_batches = 0
        
        for batch in self.train_loader:
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }
            
            # Mixed precision forward pass
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = self.model(batch)
                loss, loss_dict = self.criterion(outputs, batch)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.training.gradient_clip,
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Accumulate losses
            epoch_losses['total'] += loss.item()
            for key in ['dag_reg', 'dag_cls', 'biomass', 'trajectory']:
                epoch_losses[key] += loss_dict[key]
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set.
        
        Returns:
            Average validation losses dict
        """
        self.model.eval()
        epoch_losses: Dict[str, float] = {
            'total': 0.0,
            'dag_reg': 0.0,
            'dag_cls': 0.0,
            'biomass': 0.0,
            'trajectory': 0.0,
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass (no mixed precision for validation)
                outputs = self.model(batch)
                loss, loss_dict = self.criterion(outputs, batch)
                
                # Accumulate losses
                epoch_losses['total'] += loss.item()
                for key in ['dag_reg', 'dag_cls', 'biomass', 'trajectory']:
                    epoch_losses[key] += loss_dict[key]
                num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    def train(self) -> Path:
        """Main training loop.
        
        Returns:
            Path to best model checkpoint
        """
        for epoch in range(self.current_epoch, self.cfg.training.max_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
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
                log_dict = {
                    'epoch': epoch,
                    'lr': current_lr,
                    'epoch_time': epoch_time,
                }
                for key, val in train_losses.items():
                    log_dict[f'train/{key}'] = val
                for key, val in val_losses.items():
                    log_dict[f'val/{key}'] = val
                wandb.log(log_dict)
            
            # Console logging
            print(
                f"Epoch {epoch}/{self.cfg.training.max_epochs} | "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f} | "
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
                self.save_checkpoint(best_path, epoch, is_best=True)
                print(f"  → New best model saved (val_loss={val_total_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.training.patience:
                    print(f"Early stopping at epoch {epoch} (patience={self.cfg.training.patience})")
                    break
            
            # Save last checkpoint (for resume support)
            last_path = self.checkpoint_dir / 'last_checkpoint.pt'
            self.save_checkpoint(last_path, epoch, is_best=False)
        
        best_path = self.checkpoint_dir / 'best_model.pt'
        return best_path
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        is_best: bool = False,
    ) -> None:
        """Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'cfg': OmegaConf.to_container(self.cfg, resolve=True),
        }
        torch.save(checkpoint, path)
        
        if is_best:
            # Also save model state dict only for inference
            model_only_path = path.parent / 'best_model_state.pt'
            torch.save(self.model.state_dict(), model_only_path)
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load checkpoint for resume.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.patience_counter = checkpoint['patience_counter']
        
        print(f"Resumed from checkpoint at epoch {checkpoint['epoch']}")
    
    def predict(self, test_loader: DataLoader[Any]) -> Dict[str, Dict[str, Any]]:
        """Run inference on test set.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dict of predictions keyed by plant_id
        """
        # Load best model
        best_model_path = self.checkpoint_dir / 'best_model_state.pt'
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))
        
        self.model.eval()
        predictions: Dict[str, Dict[str, Any]] = {}
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                batch_gpu = {
                    k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                outputs = self.model(batch_gpu)
                
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
