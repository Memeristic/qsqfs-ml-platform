"""Multimodal fusion: early, late, or hybrid cross-attention.

  ``early``   concatenate the per-modality embeddings, then an MLP head.
              Simple and strong; the usual thing to beat.
  ``late``    predict from each modality separately, then combine with learned
              weights. The weights are readable, so you can see which modality
              the model actually relies on. Cannot model cross-modal interaction.
  ``hybrid``  self-attention across the modality tokens before the head, so a
              modality can modulate another's contribution.

Hybrid is not automatically better. With few samples it has more parameters to
fit and often loses to ``early``. Compare them on your data rather than
assuming; ``run_pipeline.py multimodal --fusion`` makes that one flag.

Missing modalities are handled by a per-modality presence mask, so a row with
no image is not silently treated as a row with an all-zero image.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

FUSION_TYPES = ("early", "late", "hybrid")


class MultimodalFusion(nn.Module):
    def __init__(
        self,
        modality_dims: Dict[str, int],
        fusion_type: str = "early",
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_outputs: int = 1,
    ):
        super().__init__()
        if fusion_type not in FUSION_TYPES:
            raise ValueError(f"fusion_type must be one of {FUSION_TYPES}; got '{fusion_type}'.")
        if not modality_dims:
            raise ValueError("At least one modality is required.")
        if fusion_type == "hybrid" and hidden_dim % n_heads:
            raise ValueError(f"hidden_dim ({hidden_dim}) must divide by n_heads ({n_heads}).")

        self.modality_names = sorted(modality_dims)
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim

        self.projectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(modality_dims[name], hidden_dim), nn.LayerNorm(hidden_dim)
            )
            for name in self.modality_names
        })
        # Learned stand-in for an absent modality, so "missing" is a state the
        # model can represent rather than being confused with a zero measurement.
        self.missing_token = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, hidden_dim)) for name in self.modality_names
        })

        n_modalities = len(self.modality_names)
        if fusion_type == "late":
            self.heads = nn.ModuleDict({
                name: nn.Linear(hidden_dim, n_outputs) for name in self.modality_names
            })
            self.fusion_weights = nn.Parameter(torch.zeros(n_modalities))
        else:
            if fusion_type == "hybrid":
                self.modality_embedding = nn.Parameter(
                    torch.randn(1, n_modalities, hidden_dim) * 0.02
                )
                self.attention = nn.MultiheadAttention(
                    hidden_dim, n_heads, dropout=dropout, batch_first=True
                )
                self.attention_norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * n_modalities, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, n_outputs),
            )
        self.last_attention: Optional[torch.Tensor] = None

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        presence: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        missing = set(self.modality_names) - set(embeddings)
        if missing:
            raise KeyError(f"Missing modality tensors: {sorted(missing)}")

        projected = []
        for name in self.modality_names:
            vector = self.projectors[name](embeddings[name])
            if presence is not None and name in presence:
                mask = presence[name].to(vector.dtype).view(-1, 1)
                vector = mask * vector + (1.0 - mask) * self.missing_token[name]
            projected.append(vector)

        if self.fusion_type == "late":
            predictions = torch.stack(
                [self.heads[name](vector)
                 for name, vector in zip(self.modality_names, projected)], dim=0
            )
            weights = torch.softmax(self.fusion_weights, dim=0).view(-1, 1, 1)
            return (weights * predictions).sum(dim=0)

        if self.fusion_type == "hybrid":
            tokens = torch.stack(projected, dim=1) + self.modality_embedding
            attended, weights = self.attention(tokens, tokens, tokens,
                                               need_weights=True, average_attn_weights=True)
            self.last_attention = weights.detach()
            tokens = self.attention_norm(tokens + attended)
            return self.head(tokens.flatten(1))

        return self.head(torch.cat(projected, dim=1))

    def modality_weights(self) -> Optional[Dict[str, float]]:
        """For late fusion: how much each modality contributes. None otherwise."""
        if self.fusion_type != "late":
            return None
        weights = torch.softmax(self.fusion_weights.detach(), dim=0)
        return {name: float(w) for name, w in zip(self.modality_names, weights)}
