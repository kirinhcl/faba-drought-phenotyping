"""Prediction heads for drought phenotyping tasks."""

from __future__ import annotations

from torch import Tensor, nn


def _make_head(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, output_dim),
    )


class DAGRegressionHead(nn.Module):
    """CLS -> MLP(256->128->1) for DAG regression."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mlp: nn.Sequential = _make_head(input_dim, hidden_dim, 1)

    def forward(self, cls_embedding: Tensor) -> Tensor:
        # cls_embedding: (B, dim)
        return self.mlp(cls_embedding)  # (B, 1)


class DAGClassificationHead(nn.Module):
    """CLS -> MLP(256->128->3) for DAG classification."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mlp: nn.Sequential = _make_head(input_dim, hidden_dim, 3)

    def forward(self, cls_embedding: Tensor) -> Tensor:
        # cls_embedding: (B, dim)
        return self.mlp(cls_embedding)  # (B, 3)


class BiomassHead(nn.Module):
    """CLS -> MLP(256->128->2) for FW/DW regression."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mlp: nn.Sequential = _make_head(input_dim, hidden_dim, 2)

    def forward(self, cls_embedding: Tensor) -> Tensor:
        # cls_embedding: (B, dim)
        return self.mlp(cls_embedding)  # (B, 2)


class TrajectoryHead(nn.Module):
    """Temporal tokens -> MLP(256->128->1) per timestep."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mlp: nn.Sequential = _make_head(input_dim, hidden_dim, 1)

    def forward(self, temporal_tokens: Tensor) -> Tensor:
        # temporal_tokens: (B, T, dim)
        logits = self.mlp(temporal_tokens)  # (B, T, 1)
        return logits.squeeze(-1)  # (B, T)
