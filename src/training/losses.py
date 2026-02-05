"""Loss functions for multi-task training and distillation."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _aggregate_by_genotype(
    predictions: Tensor,
    targets: Tensor,
    accessions: Sequence[str],
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Aggregate predictions and targets by genotype (accession).
    
    For DAG prediction, we aggregate plant-level predictions to genotype-level
    because DAG is a genotype-level attribute (same value for all plants of
    the same genotype under drought treatment).
    
    Args:
        predictions: (B,) tensor of predicted values
        targets: (B,) tensor of target values
        accessions: Sequence of genotype names, length B
        mask: (B,) boolean tensor indicating valid samples (WHC-30 with valid DAG)
    
    Returns:
        (agg_predictions, agg_targets): Aggregated tensors, one per unique genotype
    """
    if not mask.any():
        return predictions[:0], targets[:0]
    
    valid_preds = predictions[mask]
    valid_targets = targets[mask]
    valid_accessions = [acc for acc, m in zip(accessions, mask.tolist()) if m]
    
    if len(valid_accessions) == 0:
        return predictions[:0], targets[:0]
    
    genotype_to_indices: dict[str, list[int]] = {}
    for idx, acc in enumerate(valid_accessions):
        if acc not in genotype_to_indices:
            genotype_to_indices[acc] = []
        genotype_to_indices[acc].append(idx)
    
    agg_preds_list = []
    agg_targets_list = []
    
    for genotype, indices in genotype_to_indices.items():
        idx_tensor = torch.tensor(indices, device=predictions.device)
        agg_preds_list.append(valid_preds[idx_tensor].mean())
        agg_targets_list.append(valid_targets[idx_tensor].mean())
    
    agg_predictions = torch.stack(agg_preds_list)
    agg_targets = torch.stack(agg_targets_list)
    
    return agg_predictions, agg_targets


def _aggregate_logits_by_genotype(
    logits: Tensor,
    targets: Tensor,
    accessions: Sequence[str],
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Aggregate classification logits by genotype.
    
    Args:
        logits: (B, C) tensor of class logits
        targets: (B,) tensor of class labels
        accessions: Sequence of genotype names, length B
        mask: (B,) boolean tensor indicating valid samples
    
    Returns:
        (agg_logits, agg_targets): Aggregated tensors, one per unique genotype
    """
    if not mask.any():
        return logits[:0], targets[:0]
    
    valid_logits = logits[mask]
    valid_targets = targets[mask]
    valid_accessions = [acc for acc, m in zip(accessions, mask.tolist()) if m]
    
    if len(valid_accessions) == 0:
        return logits[:0], targets[:0]
    
    genotype_to_indices: dict[str, list[int]] = {}
    for idx, acc in enumerate(valid_accessions):
        if acc not in genotype_to_indices:
            genotype_to_indices[acc] = []
        genotype_to_indices[acc].append(idx)
    
    agg_logits_list = []
    agg_targets_list = []
    
    for genotype, indices in genotype_to_indices.items():
        idx_tensor = torch.tensor(indices, device=logits.device)
        agg_logits_list.append(valid_logits[idx_tensor].mean(dim=0))
        agg_targets_list.append(valid_targets[idx_tensor[0]])
    
    agg_logits = torch.stack(agg_logits_list)
    agg_targets = torch.stack(agg_targets_list)
    
    return agg_logits, agg_targets


class MultiTaskLoss(nn.Module):
    """Multi-task loss with learned uncertainty weighting (Kendall et al., CVPR 2018).

    Each task gets a learnable log-variance parameter s_k = log(sigma_k^2).
    Regression:      0.5 * exp(-s) * L + 0.5 * s
    Classification:  exp(-s) * L + 0.5 * s
    The loss_weights dict gates tasks on/off (0 = disabled for ablation).
    
    DAG losses use genotype-level aggregation: predictions from plants of the
    same genotype are averaged before computing loss. This is because DAG is
    a genotype-level attribute, not plant-level.
    """

    TASK_TYPES: dict[str, str] = {
        "dag_reg": "regression",
        "dag_cls": "classification",
        "dag_fine_cls": "classification",
        "biomass": "regression",
        "trajectory": "regression",
    }

    def __init__(self, loss_weights: Optional[dict[str, float]] = None) -> None:
        super().__init__()
        defaults = {
            "dag_reg": 1.0,
            "dag_cls": 1.0,
            "dag_fine_cls": 1.0,
            "biomass": 1.0,
            "trajectory": 1.0,
        }
        if loss_weights:
            defaults.update(loss_weights)
        self.loss_weights: dict[str, float] = defaults

        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1))
            for task in self.TASK_TYPES
        })

    def forward(
        self,
        predictions: dict[str, Optional[Tensor]],
        targets: dict[str, Union[Tensor, Sequence[str]]],
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute weighted multi-task loss with genotype-level DAG aggregation."""
        device = None
        for value in predictions.values():
            if isinstance(value, Tensor):
                device = value.device
                break
        if device is None:
            device = torch.device("cpu")

        zero = torch.tensor(0.0, device=device)

        treatment_value = targets.get("treatment", [])
        treatment_list: list[str] = (
            list(treatment_value)
            if isinstance(treatment_value, (list, tuple))
            else []
        )
        drought_mask = (
            torch.tensor(
                [t == "WHC-30" for t in treatment_list],
                device=device,
                dtype=torch.bool,
            )
            if treatment_list
            else torch.zeros(0, device=device, dtype=torch.bool)
        )
        
        accession_value = targets.get("accession", [])
        accession_list: list[str] = (
            list(accession_value)
            if isinstance(accession_value, (list, tuple))
            else []
        )

        dag_reg_loss = zero
        dag_reg_pred = predictions.get("dag_reg")
        if dag_reg_pred is not None:
            dag_reg_pred = dag_reg_pred.squeeze(-1)
            dag_target_value = targets["dag_target"]
            if not isinstance(dag_target_value, Tensor):
                raise TypeError("dag_target must be a Tensor")
            dag_target = dag_target_value.to(device)
            if drought_mask.numel() > 0 and drought_mask.any() and len(accession_list) > 0:
                agg_preds, agg_targets = _aggregate_by_genotype(
                    dag_reg_pred, dag_target, accession_list, drought_mask
                )
                if agg_preds.numel() > 0:
                    dag_reg_loss = F.mse_loss(agg_preds, agg_targets)

        dag_cls_loss = zero
        dag_cls_pred = predictions.get("dag_cls")
        if dag_cls_pred is not None:
            dag_category_value = targets["dag_category"]
            if not isinstance(dag_category_value, Tensor):
                raise TypeError("dag_category must be a Tensor")
            dag_category = dag_category_value.to(device).long()
            if drought_mask.numel() > 0 and drought_mask.any() and len(accession_list) > 0:
                agg_logits, agg_targets_cls = _aggregate_logits_by_genotype(
                    dag_cls_pred, dag_category, accession_list, drought_mask
                )
                if agg_logits.numel() > 0:
                    dag_cls_loss = F.cross_entropy(agg_logits, agg_targets_cls)

        dag_fine_cls_loss = zero
        dag_fine_cls_pred = predictions.get("dag_fine_cls")
        if dag_fine_cls_pred is not None:
            dag_class_value = targets.get("dag_class")
            if dag_class_value is not None and isinstance(dag_class_value, Tensor):
                dag_class = dag_class_value.to(device).long()
                valid_mask = drought_mask & (dag_class >= 0)
                if valid_mask.numel() > 0 and valid_mask.any() and len(accession_list) > 0:
                    agg_logits_fine, agg_targets_fine = _aggregate_logits_by_genotype(
                        dag_fine_cls_pred, dag_class, accession_list, valid_mask
                    )
                    if agg_logits_fine.numel() > 0:
                        dag_fine_cls_loss = F.cross_entropy(agg_logits_fine, agg_targets_fine)

        biomass_loss = zero
        biomass_pred = predictions.get("biomass")
        if biomass_pred is not None:
            fw_target_value = targets["fw_target"]
            dw_target_value = targets["dw_target"]
            if not isinstance(fw_target_value, Tensor):
                raise TypeError("fw_target must be a Tensor")
            if not isinstance(dw_target_value, Tensor):
                raise TypeError("dw_target must be a Tensor")
            fw_target = fw_target_value.to(device)
            dw_target = dw_target_value.to(device)
            biomass_target = torch.stack([fw_target, dw_target], dim=-1)
            biomass_mask = ~torch.isnan(biomass_target)
            if biomass_mask.any():
                diff = (biomass_pred - biomass_target) ** 2
                biomass_loss = diff[biomass_mask].mean()

        trajectory_loss = zero
        trajectory_pred = predictions.get("trajectory")
        if trajectory_pred is not None:
            trajectory_target_value = targets["trajectory_target"]
            trajectory_mask_value = targets["trajectory_mask"]
            if not isinstance(trajectory_target_value, Tensor):
                raise TypeError("trajectory_target must be a Tensor")
            if not isinstance(trajectory_mask_value, Tensor):
                raise TypeError("trajectory_mask must be a Tensor")
            trajectory_target = trajectory_target_value.to(device)
            trajectory_mask = trajectory_mask_value.to(device).bool()
            if trajectory_mask.any():
                diff = (trajectory_pred - trajectory_target) ** 2
                trajectory_loss = diff[trajectory_mask].mean()

        raw_losses = {
            "dag_reg": dag_reg_loss,
            "dag_cls": dag_cls_loss,
            "dag_fine_cls": dag_fine_cls_loss,
            "biomass": biomass_loss,
            "trajectory": trajectory_loss,
        }
        pred_gates = {
            "dag_reg": dag_reg_pred is not None,
            "dag_cls": dag_cls_pred is not None,
            "dag_fine_cls": dag_fine_cls_pred is not None,
            "biomass": biomass_pred is not None,
            "trajectory": trajectory_pred is not None,
        }

        total_loss = zero
        for task, raw_loss in raw_losses.items():
            if self.loss_weights[task] <= 0 or not pred_gates[task]:
                continue
            s = self.log_vars[task].squeeze()
            if self.TASK_TYPES[task] == "regression":
                total_loss = total_loss + 0.5 * torch.exp(-s) * raw_loss + 0.5 * s
            else:
                total_loss = total_loss + torch.exp(-s) * raw_loss + 0.5 * s

        loss_dict = {
            task: float(raw_losses[task].detach().item()) for task in raw_losses
        }
        return total_loss, loss_dict


class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining task loss and embedding alignment."""

    def __init__(self, alpha: float = 0.5, loss_weights: Optional[dict[str, float]] = None) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.task_loss: MultiTaskLoss = MultiTaskLoss(loss_weights)

    def forward(
        self,
        student_preds: dict[str, Tensor],
        teacher_preds: dict[str, Tensor],
        targets: dict[str, Union[Tensor, Sequence[str]]],
    ) -> Tensor:
        student_cls = student_preds["cls_embedding"]
        teacher_cls = teacher_preds["cls_embedding"].detach()
        align_loss = F.mse_loss(student_cls, teacher_cls)
        task_loss, _ = self.task_loss(student_preds, targets)
        return self.alpha * align_loss + (1.0 - self.alpha) * task_loss
