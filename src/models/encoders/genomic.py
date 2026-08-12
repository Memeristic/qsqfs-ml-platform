"""Genomic encoder: SNP genotypes, expression matrices, or DNA sequences.

Three input conventions, each with a different encoder, because they are not
interchangeable:

  ``expression`` continuous values (counts, TPM). Assumed already normalised
                 and log-transformed upstream -- this module does not do
                 library-size correction.
  ``snp``        genotypes coded 0/1/2. Embedded per locus rather than treated
                 as a continuous scale, since 0/1/2 is ordinal-ish but the
                 spacing is not meaningful.
  ``sequence``   integer-tokenised bases through a 1-D CNN.

Genomic inputs are typically very wide (p >> n). Run QSQ-FS or another selector
on the block before it reaches this encoder; a dense layer over 20,000 genes on
a few hundred samples will memorise the training set.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class GenomicEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoding_type: str = "expression",
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        vocab_size: Optional[int] = None,
        n_genotypes: int = 3,
    ):
        super().__init__()
        if encoding_type not in ("expression", "snp", "sequence"):
            raise ValueError(
                f"encoding_type must be expression, snp or sequence; got '{encoding_type}'."
            )
        if input_dim < 1:
            raise ValueError("GenomicEncoder needs at least one input feature.")

        self.encoding_type = encoding_type
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        if encoding_type == "sequence":
            if vocab_size is None:
                raise ValueError("vocab_size is required for encoding_type='sequence'.")
            self.vocab_size = vocab_size
            self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
            self.conv = nn.Sequential(
                nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
                nn.MaxPool1d(3),
                nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.projection = nn.Sequential(
                nn.Linear(128, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU()
            )
        elif encoding_type == "snp":
            # A learned vector per (locus, genotype) pair, mean-pooled over loci.
            self.genotype_embedding = nn.Embedding(n_genotypes + 1, 16, padding_idx=0)
            self.locus_weight = nn.Parameter(torch.randn(input_dim, 16) * 0.02)
            self.projection = nn.Sequential(
                nn.Linear(16, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim), nn.LayerNorm(embedding_dim),
            )
        else:
            layers, in_dim = [], input_dim
            for i in range(max(1, n_layers)):
                out_dim = hidden_dim if i < n_layers - 1 else embedding_dim
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.LayerNorm(out_dim))
                if i < n_layers - 1:
                    layers += [nn.GELU(), nn.Dropout(dropout)]
                in_dim = out_dim
            self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoding_type == "expression":
            return self.encoder(x)
        if self.encoding_type == "snp":
            genotypes = x.long().clamp(0, self.genotype_embedding.num_embeddings - 1)
            embedded = self.genotype_embedding(genotypes)          # (B, L, 16)
            weighted = embedded * self.locus_weight.unsqueeze(0)   # per-locus scaling
            return self.projection(weighted.mean(dim=1))
        embedded = self.embedding(x.long()).permute(0, 2, 1)       # (B, 32, L)
        return self.projection(self.conv(embedded).squeeze(-1))

    def describe(self) -> dict:
        return {
            "modality": "genomic",
            "encoding_type": self.encoding_type,
            "input_dim": int(self.input_dim),
            "embedding_dim": int(self.embedding_dim),
        }
