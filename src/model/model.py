"""Full multimodal teacher model for faba bean drought phenotyping."""

from __future__ import annotations

from typing import Optional, Union

from omegaconf import DictConfig
from torch import Tensor, nn

from .encoder import ViewAggregation
from .fusion import MultimodalFusion
from .heads import (
    BiomassHead,
    DAGClassificationHead,
    DAGRegressionHead,
    TrajectoryHead,
)
from .temporal import TemporalTransformer

ModelOutput = dict[str, Union[Tensor, list[Tensor], None]]


class FabaDroughtModel(nn.Module):
    """Full multimodal teacher model for faba bean drought phenotyping."""

    def __init__(self, model_config: DictConfig) -> None:
        super().__init__()
        cfg = model_config.model if "model" in model_config else model_config

        encoder_output_dim = cfg.encoder_output_dim
        fusion_cfg = cfg.fusion
        temporal_cfg = cfg.temporal
        heads_cfg = cfg.heads

        self.view_agg: ViewAggregation = ViewAggregation(encoder_output_dim)
        self.fusion: MultimodalFusion = MultimodalFusion(
            image_dim=fusion_cfg.image_dim,
            fluor_dim=fusion_cfg.fluor_dim,
            env_dim=fusion_cfg.env_dim,
            water_dim=fusion_cfg.water_dim,
            hidden_dim=fusion_cfg.hidden_dim,
            fused_dim=fusion_cfg.fused_dim,
        )
        self.temporal: TemporalTransformer = TemporalTransformer(
            dim=temporal_cfg.dim,
            num_layers=temporal_cfg.num_layers,
            num_heads=temporal_cfg.num_heads,
            ff_dim=temporal_cfg.ff_dim,
            dropout=temporal_cfg.dropout,
        )

        self.dag_reg_head: Optional[DAGRegressionHead] = (
            DAGRegressionHead(temporal_cfg.dim) if heads_cfg.dag_regression else None
        )
        self.dag_cls_head: Optional[DAGClassificationHead] = (
            DAGClassificationHead(temporal_cfg.dim) if heads_cfg.dag_classification else None
        )
        self.biomass_head: Optional[BiomassHead] = (
            BiomassHead(temporal_cfg.dim) if heads_cfg.biomass_regression else None
        )
        self.trajectory_head: Optional[TrajectoryHead] = (
            TrajectoryHead(temporal_cfg.dim) if heads_cfg.trajectory else None
        )

    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        """Run the full multimodal forward pass.

        Args:
            batch: dict with keys images, image_mask, fluorescence, fluor_mask,
                environment, watering, temporal_positions

        Returns:
            dict with task predictions, attention weights, and CLS embedding.
        """
        images = batch["images"]  # (B, T, V, D)
        image_mask = batch["image_mask"]  # (B, T, V)
        fluorescence = batch["fluorescence"]  # (B, T, F)
        fluor_mask = batch["fluor_mask"]  # (B, T)
        environment = batch["environment"]  # (B, T, 5)
        watering = batch["watering"]  # (B, T, 5)
        temporal_positions = batch["temporal_positions"]  # (B, T)

        image_emb = self.view_agg(images, image_mask)  # (B, T, 768)
        image_active = image_mask.any(dim=-1)  # (B, T)
        fused = self.fusion(
            image_emb,
            fluorescence,
            environment,
            watering,
            image_active,
            fluor_mask,
        )  # (B, T, 256)

        active_mask = image_active | fluor_mask  # (B, T)
        cls_embedding, temporal_tokens, attention_weights = self.temporal(
            fused,
            temporal_positions,
            active_mask,
        )

        outputs: ModelOutput = {
            "dag_reg": self.dag_reg_head(cls_embedding) if self.dag_reg_head else None,
            "dag_cls": self.dag_cls_head(cls_embedding) if self.dag_cls_head else None,
            "biomass": self.biomass_head(cls_embedding) if self.biomass_head else None,
            "trajectory": (
                self.trajectory_head(temporal_tokens) if self.trajectory_head else None
            ),
            "attention_weights": attention_weights,
            "cls_embedding": cls_embedding,
        }
        return outputs
