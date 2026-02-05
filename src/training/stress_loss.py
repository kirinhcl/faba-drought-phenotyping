"""Stress detection loss with class imbalance handling."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class StressLoss(nn.Module):
    """Binary cross-entropy loss for stress detection with class imbalance handling.
    
    Uses BCEWithLogitsLoss for numerical stability and auto-computes pos_weight
    from batch statistics to handle class imbalance (more negative samples than
    positive samples in typical batches).
    """

    def __init__(self, pos_weight: float | None = None) -> None:
        """Initialize stress loss.
        
        Args:
            pos_weight: Weight for positive class. If None, auto-computed per batch
                from the ratio of negative to positive samples.
        """
        super().__init__()
        self.pos_weight: float | None = pos_weight

    def forward(  # type: ignore[override]
        self, 
        predictions: dict[str, Tensor], 
        targets: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute binary cross-entropy loss with masking.
        
        Args:
            predictions: dict with 'stress_logits' (B, T) - raw logits
            targets: dict with 'stress_labels' (B, T) and 'stress_mask' (B, T)
        
        Returns:
            (loss_tensor, loss_dict) tuple where loss_tensor is scalar and
            loss_dict contains 'stress' key with loss value
        """
        logits = predictions['stress_logits']  # (B, T)
        labels = targets['stress_labels'].float()  # (B, T) → float
        mask = targets['stress_mask']  # (B, T) bool
        
        # Return zero loss if no valid samples
        if not mask.any():
            return torch.tensor(0.0, device=logits.device), {'stress': 0.0}
        
        # Extract valid samples
        valid_logits = logits[mask]
        valid_labels = labels[mask]
        
        # Compute pos_weight
        if self.pos_weight is None:
            num_pos = valid_labels.sum()
            num_neg = (1 - valid_labels).sum()
            pw = num_neg / (num_pos + 1e-6)  # epsilon to avoid divide by zero
        else:
            pw = self.pos_weight
        
        # Compute loss with pos_weight
        pos_weight_tensor = torch.as_tensor(pw, device=logits.device, dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_labels,
            pos_weight=pos_weight_tensor
        )
        
        return loss, {'stress': loss.item()}
