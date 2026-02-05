"""Multimodal fusion for image, fluorescence, environment, and vegetation index data."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


def _make_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class MultimodalFusion(nn.Module):
    """Fuses image, fluorescence, environment, and vegetation index modalities.
    
    With equal_dim=True, all modalities are projected to hidden_dim before fusion,
    ensuring equal contribution from each modality.
    """

    def __init__(
        self,
        image_dim: int = 768,
        fluor_dim: int = 94,
        env_dim: int = 5,
        vi_dim: int = 11,
        hidden_dim: int = 128,
        fused_dim: int = 256,
        equal_dim: bool = True,
    ) -> None:
        super().__init__()
        self.image_dim: int = image_dim
        self.hidden_dim: int = hidden_dim
        self.fused_dim: int = fused_dim
        self.equal_dim: bool = equal_dim

        self.fluor_proj: nn.Sequential = _make_mlp(fluor_dim, hidden_dim)
        self.env_proj: nn.Sequential = _make_mlp(env_dim, hidden_dim)
        self.vi_proj: nn.Sequential = _make_mlp(vi_dim, hidden_dim)

        self.image_proj: Optional[nn.Sequential] = None
        if equal_dim:
            self.image_proj = _make_mlp(image_dim, hidden_dim)
            self.image_mask_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            fusion_in_dim = hidden_dim * 4
        else:
            self.image_mask_token = nn.Parameter(torch.zeros(1, 1, image_dim))
            fusion_in_dim = image_dim + hidden_dim * 3

        self.fluor_mask_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.env_mask_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.vi_mask_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.fusion_proj: nn.Linear = nn.Linear(fusion_in_dim, fused_dim)
        self.fusion_norm: nn.LayerNorm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        image_emb: Tensor,
        fluor: Tensor,
        env: Tensor,
        vi: Tensor,
        image_active: Tensor,
        fluor_mask: Tensor,
    ) -> Tensor:
        """Fuse per-timestep modality embeddings."""
        image_active = image_active.bool()
        fluor_mask = fluor_mask.bool()

        batch_size, time_steps, _ = image_emb.shape

        fluor_proj = self.fluor_proj(fluor)
        fluor_token = self.fluor_mask_token.expand(batch_size, time_steps, -1)
        fluor_proj = torch.where(fluor_mask.unsqueeze(-1), fluor_proj, fluor_token)

        env_proj = self.env_proj(env)
        vi_proj = self.vi_proj(vi)

        if self.equal_dim and self.image_proj is not None:
            image_proj = self.image_proj(image_emb)
            image_token = self.image_mask_token.expand(batch_size, time_steps, -1)
            image_proj = torch.where(image_active.unsqueeze(-1), image_proj, image_token)
            fused = torch.cat([image_proj, fluor_proj, env_proj, vi_proj], dim=-1)
        else:
            image_token = self.image_mask_token.expand(batch_size, time_steps, -1)
            image_emb = torch.where(image_active.unsqueeze(-1), image_emb, image_token)
            fused = torch.cat([image_emb, fluor_proj, env_proj, vi_proj], dim=-1)

        fused = self.fusion_proj(fused)
        fused = self.fusion_norm(fused)
        return fused
