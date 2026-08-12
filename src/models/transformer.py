"""FT-Transformer style tabular model (Gorishniy et al., 2021)."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class NumericalTokenizer(nn.Module):
    """One learned (weight, bias) per numeric feature -> a token each.

    A single shared Linear(1, d) would give every feature the same projection
    and force the model to tell them apart from the positional embedding alone.
    """

    def __init__(self, n_features: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, embedding_dim))
        self.bias = nn.Parameter(torch.empty(n_features, embedding_dim))
        nn.init.normal_(self.weight, std=embedding_dim ** -0.5)
        nn.init.normal_(self.bias, std=embedding_dim ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F) -> (B, F, D)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TabularTransformer(nn.Module):
    def __init__(
        self,
        n_numerical: int,
        n_categories: Optional[List[int]] = None,
        n_classes: int = 1,
        embedding_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_cls_token: bool = True,
        regression: bool = True,
    ):
        super().__init__()
        if embedding_dim % n_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by n_heads ({n_heads})."
            )
        if n_numerical < 1:
            raise ValueError("Need at least one numerical feature.")

        self.n_numerical = n_numerical
        self.n_categories = list(n_categories or [])
        self.n_classes = int(n_classes)
        self.regression = bool(regression)
        self.use_cls_token = bool(use_cls_token)
        self.embedding_dim = embedding_dim

        self.num_tokenizer = NumericalTokenizer(n_numerical, embedding_dim)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(int(c), embedding_dim) for c in self.n_categories]
        )

        n_tokens = n_numerical + len(self.n_categories) + (1 if use_cls_token else 0)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
        self.positional = nn.Parameter(torch.randn(1, n_tokens, embedding_dim) * 0.02)
        self.input_dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor warns and self-disables whenever norm_first=True;
        # pass False explicitly so the warning does not clutter every run.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        out_dim = 1 if regression else max(2, self.n_classes)
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, out_dim),
        )

    def forward(self, numerical: torch.Tensor, categorical: Optional[torch.Tensor] = None):
        tokens = self.num_tokenizer(numerical)
        if categorical is not None and len(self.cat_embeddings):
            cat = torch.stack(
                [emb(categorical[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1
            )
            tokens = torch.cat([tokens, cat], dim=1)
        if self.use_cls_token:
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.input_dropout(tokens + self.positional[:, : tokens.size(1)])
        encoded = self.transformer(tokens)
        pooled = encoded[:, 0, :] if self.use_cls_token else encoded.mean(dim=1)
        return self.head(pooled)

    @torch.no_grad()
    def embed(self, numerical: torch.Tensor, categorical: Optional[torch.Tensor] = None):
        """Pooled representation before the head, for PCA / probing."""
        self.eval()
        tokens = self.num_tokenizer(numerical)
        if categorical is not None and len(self.cat_embeddings):
            cat = torch.stack(
                [emb(categorical[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1
            )
            tokens = torch.cat([tokens, cat], dim=1)
        if self.use_cls_token:
            tokens = torch.cat([self.cls_token.expand(tokens.size(0), -1, -1), tokens], dim=1)
        encoded = self.transformer(tokens + self.positional[:, : tokens.size(1)])
        return encoded[:, 0, :] if self.use_cls_token else encoded.mean(dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
