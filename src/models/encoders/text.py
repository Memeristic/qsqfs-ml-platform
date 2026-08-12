"""Clinical text encoder.

Two backends, and the code is explicit about which one ran:

  ``transformers`` installed -> a pretrained clinical BERT (default
                                Bio_ClinicalBERT), CLS or mean pooled.
  otherwise                  -> a hashed bag-of-words MLP trained from scratch.

The fallback is deliberately simple and offline. It is *not* a substitute for a
clinical language model, and ``describe()`` reports which backend was used so a
result can never imply pretrained clinical knowledge that was not present. The
fallback exists so the multimodal path is testable without a 400 MB download
and so an air-gapped hospital environment can still run the pipeline.

Note on leakage: clinical notes routinely state the outcome outright
("patient readmitted 3 days later"). Text is the modality most likely to leak
a label, and no automatic screen will catch it. Read a sample of your notes
before trusting any result that depends on them.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:  # pragma: no cover - depends on the environment
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRANSFORMERS_AVAILABLE = False

TOKEN_RE = re.compile(r"[a-z0-9']+")


def hashed_bag_of_words(texts: Sequence[str], n_buckets: int = 4096) -> torch.Tensor:
    """Deterministic hashing vectoriser with sublinear term-frequency scaling.

    Hashing avoids fitting a vocabulary, so the same text maps to the same
    vector at train and inference time with no state to persist.
    """
    matrix = torch.zeros(len(texts), n_buckets)
    for row, text in enumerate(texts):
        for token in TOKEN_RE.findall(str(text).lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % n_buckets
            matrix[row, bucket] += 1.0
    return torch.log1p(matrix)


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embedding_dim: int = 64,
        max_length: int = 128,
        freeze_backbone: bool = True,
        pooling: str = "cls",
        n_buckets: int = 4096,
        force_fallback: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.pooling = pooling
        self.n_buckets = n_buckets
        self.backend = "fallback_hashed_bow"
        self.tokenizer = None
        self.backbone = None

        if TRANSFORMERS_AVAILABLE and not force_fallback:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.backbone = AutoModel.from_pretrained(model_name)
                hidden_size = self.backbone.config.hidden_size
                self.backend = f"transformers:{model_name}"
                if freeze_backbone:
                    for parameter in self.backbone.parameters():
                        parameter.requires_grad = False
                logger.info("Text encoder using %s", self.backend)
            except Exception as exc:
                logger.warning(
                    "Could not load '%s' (%s); using the hashed bag-of-words "
                    "fallback instead.", model_name, exc,
                )
                self.tokenizer = self.backbone = None

        if self.backbone is None:
            logger.info(
                "Text encoder using the offline hashed bag-of-words fallback "
                "(%d buckets). This carries no pretrained clinical knowledge.",
                n_buckets,
            )
            hidden_size = 256
            self.bow_mlp = nn.Sequential(
                nn.Linear(n_buckets, 512), nn.LayerNorm(512), nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(),
            )

        self.frozen = bool(freeze_backbone and self.backbone is not None)
        self.hidden_size = hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def _device(self) -> torch.device:
        return next(self.projection.parameters()).device

    def forward(self, texts: List[str] | torch.Tensor) -> torch.Tensor:
        if isinstance(texts, torch.Tensor):
            # Pre-vectorised batch (the collate path for the fallback backend).
            return self.projection(self.bow_mlp(texts.to(self._device())))

        texts = ["" if t is None else str(t) for t in texts]
        if self.backbone is not None:
            inputs = self.tokenizer(
                texts, max_length=self.max_length, padding=True,
                truncation=True, return_tensors="pt",
            )
            device = self._device()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            context = torch.no_grad() if self.frozen else torch.enable_grad()
            with context:
                hidden = self.backbone(**inputs).last_hidden_state
            if self.pooling == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                pooled = hidden[:, 0, :]
            return self.projection(pooled)

        vectors = hashed_bag_of_words(texts, self.n_buckets).to(self._device())
        return self.projection(self.bow_mlp(vectors))

    def describe(self) -> dict:
        return {
            "modality": "text",
            "backend": self.backend,
            "pretrained": self.backbone is not None,
            "frozen": self.frozen,
            "embedding_dim": int(self.embedding_dim),
            "caveat": (
                "Offline fallback: no pretrained clinical knowledge."
                if self.backbone is None else
                "Pretrained clinical language model."
            ),
        }
