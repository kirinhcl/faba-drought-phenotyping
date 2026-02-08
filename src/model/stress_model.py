"""Stress detection model architecture."""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from .encoder import ViewAggregation
from .gating import ModalityGating, ModalityProjection
from .temporal import TemporalTransformer


class StressHead(nn.Module):
    """MLP head that maps temporal tokens to per-timestep stress logits."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64) -> None:
        """Initialize stress head.

        Args:
            input_dim: Input dimension (default 128).
            hidden_dim: Hidden layer dimension (default 64).
        """
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(input_dim)
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map temporal tokens to stress logits.

        Args:
            x: Temporal tokens of shape (B, T, input_dim).

        Returns:
            Stress logits of shape (B, T).
        """
        x = self.norm(x)  # (B, T, input_dim)
        logits = self.mlp(x)  # (B, T, 1)
        return logits.squeeze(-1)  # (B, T)


class StressDetectionModel(nn.Module):
    """Complete stress detection pipeline integrating all components."""

    def __init__(self, model_config: DictConfig) -> None:
        """Initialize stress detection model.

        Args:
            model_config: OmegaConf config with model architecture parameters.
        """
        super().__init__()
        cfg = model_config.model if "model" in model_config else model_config

        self.enabled_modalities: list[str] = OmegaConf.select(
            cfg,
            "ablation.enabled_modalities",
            default=["image", "fluor", "env", "vi"],
        )
        self.fusion_mode: str = OmegaConf.select(
            cfg,
            "ablation.fusion_mode",
            default="gating",
        )
        self.temporal_mode: str = OmegaConf.select(
            cfg,
            "ablation.temporal_mode",
            default="transformer",
        )
        self.causal_mask: bool = OmegaConf.select(
            cfg,
            "ablation.causal_mask",
            default=False,
        )

        # View aggregation: (B, T, V=4, 768) → (B, T, 768)
        self.view_agg: ViewAggregation = ViewAggregation(cfg.encoder_output_dim)

        # Modality projection: 4 modalities → common 128-dim space
        self.modality_proj: ModalityProjection = ModalityProjection(
            image_dim=cfg.modality.image_dim,
            fluor_dim=cfg.modality.fluor_dim,
            env_dim=cfg.modality.env_dim,
            vi_dim=cfg.modality.vi_dim,
            hidden_dim=cfg.modality.hidden_dim,
        )

        # Modality gating: adaptive fusion with learned weights
        self.modality_gating: ModalityGating = ModalityGating(
            hidden_dim=cfg.modality.hidden_dim,
            num_modalities=4,
            gate_hidden=cfg.modality.gate_hidden,
        )

        # Temporal transformer: (B, T, 128) → temporal reasoning
        self.temporal: TemporalTransformer = TemporalTransformer(
            dim=cfg.temporal.dim,  # CRITICAL: 128 not 256
            num_layers=cfg.temporal.num_layers,
            num_heads=cfg.temporal.num_heads,
            ff_dim=cfg.temporal.ff_dim,
            dropout=cfg.temporal.dropout,
            causal=self.causal_mask,
        )

        self.concat_fusion: nn.Module | None = None
        if self.fusion_mode == "concat":
            self.concat_fusion = nn.Sequential(
                nn.Linear(cfg.modality.hidden_dim * 4, cfg.modality.hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.modality.hidden_dim * 2, cfg.modality.hidden_dim),
            )

        self.temporal_mlp: nn.Module | None = None
        if self.temporal_mode == "mlp":
            self.temporal_mlp = nn.Sequential(
                nn.Linear(cfg.temporal.dim, cfg.temporal.dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.temporal.dim * 4, cfg.temporal.dim),
            )

        # Stress head: (B, T, 128) → (B, T) stress logits
        self.stress_head: StressHead = StressHead(
            input_dim=cfg.temporal.dim,
            hidden_dim=cfg.stress_head.hidden_dim,
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run stress detection forward pass.

        Args:
            batch: dict with keys:
                - images: (B, T, V=4, 768) view embeddings
                - image_mask: (B, T, V) boolean mask
                - fluorescence: (B, T, 94) fluorescence data
                - fluor_mask: (B, T) boolean mask
                - environment: (B, T, 5) environmental data
                - vi: (B, T, 11) vegetation indices
                - temporal_positions: (B, T) DAG values for positional encoding

        Returns:
            dict with keys:
                - stress_logits: (B, T) per-timestep stress logits
                - modality_gates: (B, T, 4) learned modality importance weights
        """
        # 1. View aggregation: pool 4 views → 1 embedding per timestep
        images = batch["images"]  # (B, T, V=4, 768)
        image_mask = batch["image_mask"]  # (B, T, V)
        image_emb = self.view_agg(images, image_mask)  # (B, T, 768)
        image_active = image_mask.any(dim=-1)  # (B, T)

        # 2. Modality projection: project to common 128-dim space
        fluorescence = batch["fluorescence"]  # (B, T, 94)
        fluor_mask = batch["fluor_mask"]  # (B, T)
        environment = batch["environment"]  # (B, T, 5)
        vi = batch["vi"]  # (B, T, 11)

        modality_features = self.modality_proj(
            image_emb,
            fluorescence,
            environment,
            vi,
            image_active,
            fluor_mask,
        )  # List of 4x (B, T, 128)

        modality_names = ["image", "fluor", "env", "vi"]
        for i, name in enumerate(modality_names):
            if name not in self.enabled_modalities:
                modality_features[i] = torch.zeros_like(modality_features[i])

        # 3. Modality gating: adaptive fusion with learned weights
        if self.fusion_mode == "concat":
            concat = torch.cat(modality_features, dim=-1)  # (B, T, 512)
            if self.concat_fusion is None:
                raise RuntimeError("Concat fusion requested but module not initialized.")
            fused = self.concat_fusion(concat)  # (B, T, 128)
            batch_size, timesteps = fused.shape[0], fused.shape[1]
            gates = torch.ones(batch_size, timesteps, 4, device=fused.device) / 4
        else:
            fused, gates = self.modality_gating(modality_features)  # (B, T, 128), (B, T, 4)

        # 4. Temporal transformer: reason across timesteps
        temporal_positions = batch["temporal_positions"]  # (B, T)
        active_mask = image_active | fluor_mask  # (B, T)
        if self.temporal_mode == "transformer":
            _, temporal_tokens, _ = self.temporal(
                fused, temporal_positions, active_mask
            )  # (B, T, 128)
        elif self.temporal_mode == "mlp":
            if self.temporal_mlp is None:
                raise RuntimeError("Temporal MLP requested but module not initialized.")
            temporal_tokens = self.temporal_mlp(fused)  # (B, T, 128)
        elif self.temporal_mode == "none":
            temporal_tokens = fused
        else:
            raise ValueError(f"Unknown temporal_mode: {self.temporal_mode}")

        # 5. Stress head: predict per-timestep stress
        stress_logits = self.stress_head(temporal_tokens)  # (B, T)

        return {
            "stress_logits": stress_logits,
            "modality_gates": gates,
        }
