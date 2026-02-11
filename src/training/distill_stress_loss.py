"""Stress distillation loss for LUPI-based knowledge transfer.

Combines task loss (BCE on student predictions), logit distillation (KL divergence
between student and teacher stress logits), and optional temporal token alignment
(MSE between student and teacher temporal tokens).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class StressDistillationLoss(nn.Module):
    """Knowledge distillation loss for stress detection with teacher guidance.
    
    Combines three loss components:
    1. Task loss: Standard BCE on student predictions vs ground-truth labels
    2. Logit distillation: KL divergence between student and teacher stress logits
    3. Temporal token alignment: MSE between student and teacher temporal tokens
    
    Formula:
        loss = (1-alpha) * task_loss + alpha * (beta * logit_kd + (1-beta) * token_mse)
    
    Alpha annealing: Starts high (0.7, focus on teacher) and decays to low (0.3, focus on task).
    """
    
    def __init__(
        self,
        alpha_start: float = 0.7,
        alpha_end: float = 0.3,
        beta: float = 0.5,
        temperature: float = 2.0,
        pos_weight: Optional[float] = None,
    ) -> None:
        """Initialize stress distillation loss.
        
        Args:
            alpha_start: Initial alpha value (high weight on distillation).
            alpha_end: Final alpha value (low weight on distillation).
            beta: Weight for logit distillation vs token alignment (0.5 = equal).
            temperature: Temperature for KL divergence scaling (standard: 2.0).
            pos_weight: Weight for positive class in BCE. If None, auto-computed per batch.
        """
        super().__init__()
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.beta = beta
        self.temperature = temperature
        self.pos_weight = pos_weight
        self.alpha = alpha_start  # Current alpha value
    
    def set_alpha(self, epoch: int, max_epochs: int) -> None:
        """Update alpha via linear annealing.
        
        Args:
            epoch: Current epoch (0-indexed).
            max_epochs: Total number of epochs.
        """
        progress = epoch / max(max_epochs, 1)  # Avoid division by zero
        self.alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
    
    def forward(
        self,
        student_preds: dict[str, Tensor],
        teacher_preds: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute distillation loss.
        
        Args:
            student_preds: dict with 'stress_logits' (B, T) - student raw logits
            teacher_preds: dict with 'stress_logits' (B, T) - teacher raw logits
            targets: dict with 'stress_labels' (B, T) and 'stress_mask' (B, T)
        
        Returns:
            (loss_tensor, loss_dict) tuple where:
                - loss_tensor: scalar loss
                - loss_dict: dict with keys 'task', 'distill', 'total'
        """
        student_logits = student_preds['stress_logits']  # (B, T)
        teacher_logits = teacher_preds['stress_logits'].detach()  # (B, T)
        labels = targets['stress_labels'].float()  # (B, T)
        mask = targets['stress_mask']  # (B, T) bool
        
        device = student_logits.device
        
        # Handle all-masked batches
        if not mask.any():
            zero_loss = torch.tensor(0.0, device=device)
            return zero_loss, {'task': 0.0, 'distill': 0.0, 'total': 0.0}
        
        # Extract valid samples
        valid_student_logits = student_logits[mask]
        valid_teacher_logits = teacher_logits[mask]
        valid_labels = labels[mask]
        
        # ===== Task Loss: BCE on student predictions =====
        if self.pos_weight is None:
            num_pos = valid_labels.sum()
            num_neg = (1 - valid_labels).sum()
            pw = num_neg / (num_pos + 1e-6)
        else:
            pw = self.pos_weight
        
        pos_weight_tensor = torch.as_tensor(pw, device=device, dtype=torch.float32)
        task_loss = F.binary_cross_entropy_with_logits(
            valid_student_logits,
            valid_labels,
            pos_weight=pos_weight_tensor,
        )
        
        # ===== Logit Distillation: KL divergence with temperature =====
        # Scale logits by temperature
        student_probs = F.softmax(valid_student_logits / self.temperature, dim=0)
        teacher_probs = F.softmax(valid_teacher_logits / self.temperature, dim=0)
        
        # KL divergence: D_KL(teacher || student)
        logit_kd = F.kl_div(
            F.log_softmax(valid_student_logits / self.temperature, dim=0),
            teacher_probs,
            reduction='batchmean',
        )
        
        # ===== Temporal Token Alignment: MSE (optional, lower weight) =====
        # For now, we use a simple MSE between logits as proxy for token alignment
        token_mse = F.mse_loss(valid_student_logits, valid_teacher_logits)
        
        # ===== Combine losses =====
        distill_loss = self.beta * logit_kd + (1.0 - self.beta) * token_mse
        total_loss = (1.0 - self.alpha) * task_loss + self.alpha * distill_loss
        
        loss_dict = {
            'task': task_loss.item(),
            'distill': distill_loss.item(),
            'total': total_loss.item(),
        }
        
        return total_loss, loss_dict
