"""Assembles encoders + fusion into one trainable model, driven by a schema."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .encoders import GenomicEncoder, ImageEncoder, TabularEncoder, TextEncoder
from .fusion import MultimodalFusion

logger = logging.getLogger(__name__)


def build_encoder(spec: Dict, embedding_dim: int) -> nn.Module:
    """Instantiate one encoder from a modality spec."""
    kind = spec["type"]
    if kind == "image":
        return ImageEncoder(
            backbone=spec.get("backbone", "resnet18"),
            pretrained=spec.get("pretrained", True),
            freeze_backbone=spec.get("freeze_backbone", True),
            embedding_dim=embedding_dim,
        )
    if kind == "genomic":
        return GenomicEncoder(
            input_dim=spec["input_dim"],
            encoding_type=spec.get("encoding", "expression"),
            embedding_dim=embedding_dim,
            vocab_size=spec.get("vocab_size"),
        )
    if kind == "text":
        return TextEncoder(
            model_name=spec.get("model_name", "emilyalsentzer/Bio_ClinicalBERT"),
            embedding_dim=embedding_dim,
            max_length=spec.get("max_length", 128),
            freeze_backbone=spec.get("freeze_backbone", True),
            force_fallback=spec.get("force_fallback", False),
        )
    if kind in ("tabular", "signal"):
        return TabularEncoder(
            input_dim=spec["input_dim"],
            embedding_dim=embedding_dim,
            dropout=spec.get("dropout", 0.1),
        )
    raise ValueError(f"Unknown modality type '{kind}'.")


class MultimodalModel(nn.Module):
    """One encoder per modality, then fusion, then a task head.

    ``schema`` maps a modality name to its spec, e.g.::

        {
          "labs":    {"type": "tabular", "input_dim": 24},
          "notes":   {"type": "text"},
          "scan":    {"type": "image", "backbone": "resnet18"},
          "expr":    {"type": "genomic", "input_dim": 500, "encoding": "expression"},
        }
    """

    def __init__(
        self,
        schema: Dict[str, Dict],
        n_classes: int = 1,
        regression: bool = True,
        embedding_dim: int = 64,
        fusion_type: str = "early",
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not schema:
            raise ValueError("The modality schema is empty.")
        self.schema = schema
        self.regression = regression
        self.modality_names = sorted(schema)

        self.encoders = nn.ModuleDict(
            {name: build_encoder(spec, embedding_dim) for name, spec in schema.items()}
        )
        self.fusion = MultimodalFusion(
            modality_dims={name: embedding_dim for name in self.modality_names},
            fusion_type=fusion_type, hidden_dim=embedding_dim,
            n_heads=n_heads, dropout=dropout,
            n_outputs=1 if regression else max(2, n_classes),
        )

    def forward(self, inputs: Dict, presence: Optional[Dict] = None) -> torch.Tensor:
        embeddings = {}
        for name in self.modality_names:
            if name not in inputs:
                raise KeyError(f"Input for modality '{name}' was not provided.")
            embeddings[name] = self.encoders[name](inputs[name])
        return self.fusion(embeddings, presence)

    def describe(self) -> Dict:
        return {
            "modalities": {
                name: (encoder.describe() if hasattr(encoder, "describe")
                       else {"modality": self.schema[name]["type"]})
                for name, encoder in self.encoders.items()
            },
            "fusion_type": self.fusion.fusion_type,
            "n_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "n_parameters_total": sum(p.numel() for p in self.parameters()),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
