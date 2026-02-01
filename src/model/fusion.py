"""Multimodal fusion for image, fluorescence, environment, and watering data."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _make_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class MultimodalFusion(nn.Module):
    """Fuses image, fluorescence, environment, and watering modalities."""

    def __init__(
        self,
        image_dim: int = 768,
        fluor_dim: int = 93,
        env_dim: int = 5,
        water_dim: int = 5,
        hidden_dim: int = 128,
        fused_dim: int = 256,
    ) -> None:
        super().__init__()
        self.image_dim: int = image_dim
        self.hidden_dim: int = hidden_dim
        self.fused_dim: int = fused_dim

        self.fluor_proj: nn.Sequential = _make_mlp(fluor_dim, hidden_dim)
        self.env_proj: nn.Sequential = _make_mlp(env_dim, hidden_dim)
        self.water_proj: nn.Sequential = _make_mlp(water_dim, hidden_dim)

        self.fluor_mask_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.env_mask_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.water_mask_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.image_mask_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, image_dim))

        fusion_in_dim = image_dim + hidden_dim * 3
        self.fusion_proj: nn.Linear = nn.Linear(fusion_in_dim, fused_dim)
        self.fusion_norm: nn.LayerNorm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        image_emb: Tensor,
        fluor: Tensor,
        env: Tensor,
        water: Tensor,
        image_active: Tensor,
        fluor_mask: Tensor,
    ) -> Tensor:
        """Fuse per-timestep modality embeddings.

        Args:
            image_emb: (B, T, image_dim)
            fluor: (B, T, fluor_dim)
            env: (B, T, env_dim)
            water: (B, T, water_dim)
            image_active: (B, T) bool, True where any view is present
            fluor_mask: (B, T) bool, True where fluorescence is present

        Returns:
            Fused embeddings of shape (B, T, fused_dim)
        """
        image_active = image_active.bool()
        fluor_mask = fluor_mask.bool()

        batch_size, time_steps, _ = image_emb.shape

        fluor_proj = self.fluor_proj(fluor)  # (B, T, hidden_dim)
        fluor_token = self.fluor_mask_token.expand(batch_size, time_steps, -1)
        fluor_proj = torch.where(fluor_mask.unsqueeze(-1), fluor_proj, fluor_token)

        image_token = self.image_mask_token.expand(batch_size, time_steps, -1)
        image_emb = torch.where(image_active.unsqueeze(-1), image_emb, image_token)

        env_proj = self.env_proj(env)  # (B, T, hidden_dim)
        water_proj = self.water_proj(water)  # (B, T, hidden_dim)

        fused = torch.cat([image_emb, fluor_proj, env_proj, water_proj], dim=-1)  # (B, T, 1152)
        fused = self.fusion_proj(fused)  # (B, T, fused_dim)
        fused = self.fusion_norm(fused)
        return fused
