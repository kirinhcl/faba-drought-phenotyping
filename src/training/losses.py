"""Loss functions for multi-task training and distillation."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Configurable multi-task loss for drought phenotyping."""

    def __init__(self, loss_weights: Optional[dict[str, float]] = None) -> None:
        super().__init__()
        defaults = {
            "dag_reg": 1.0,
            "dag_cls": 0.5,
            "biomass": 1.0,
            "trajectory": 0.5,
        }
        if loss_weights:
            defaults.update(loss_weights)
        self.loss_weights: dict[str, float] = defaults

    def forward(
        self,
        predictions: dict[str, Optional[Tensor]],
        targets: dict[str, Union[Tensor, Sequence[str]]],
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute weighted multi-task loss.

        Args:
            predictions: model outputs
            targets: dict with dag_target, dag_category, fw_target, dw_target,
                trajectory_target, trajectory_mask, treatment

        Returns:
            total_loss: scalar tensor
            loss_dict: per-task losses as floats
        """
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

        dag_reg_loss = zero
        dag_reg_pred = predictions.get("dag_reg")
        if dag_reg_pred is not None:
            dag_reg_pred = dag_reg_pred.squeeze(-1)
            dag_target_value = targets["dag_target"]
            if not isinstance(dag_target_value, Tensor):
                raise TypeError("dag_target must be a Tensor")
            dag_target = dag_target_value.to(device)
            if drought_mask.numel() > 0 and drought_mask.any():
                dag_reg_loss = F.mse_loss(dag_reg_pred[drought_mask], dag_target[drought_mask])

        dag_cls_loss = zero
        dag_cls_pred = predictions.get("dag_cls")
        if dag_cls_pred is not None:
            dag_category_value = targets["dag_category"]
            if not isinstance(dag_category_value, Tensor):
                raise TypeError("dag_category must be a Tensor")
            dag_category = dag_category_value.to(device).long()
            if drought_mask.numel() > 0 and drought_mask.any():
                dag_cls_loss = F.cross_entropy(
                    dag_cls_pred[drought_mask],
                    dag_category[drought_mask],
                )

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

        total_loss = zero
        if dag_reg_pred is not None:
            total_loss = total_loss + self.loss_weights["dag_reg"] * dag_reg_loss
        if dag_cls_pred is not None:
            total_loss = total_loss + self.loss_weights["dag_cls"] * dag_cls_loss
        if biomass_pred is not None:
            total_loss = total_loss + self.loss_weights["biomass"] * biomass_loss
        if trajectory_pred is not None:
            total_loss = total_loss + self.loss_weights["trajectory"] * trajectory_loss

        loss_dict = {
            "dag_reg": float(dag_reg_loss.detach().item()),
            "dag_cls": float(dag_cls_loss.detach().item()),
            "biomass": float(biomass_loss.detach().item()),
            "trajectory": float(trajectory_loss.detach().item()),
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
