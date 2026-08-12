"""Tabular block encoder: an MLP to a shared embedding width."""

from __future__ import annotations

import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 64,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        if input_dim < 1:
            raise ValueError("TabularEncoder needs at least one input feature.")
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
