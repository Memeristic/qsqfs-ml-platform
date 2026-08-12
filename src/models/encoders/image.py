"""Image encoder for medical imaging (CT, MRI, fundus, X-ray).

Backbone selection is honest about what is actually available:

  * ``torchvision`` present  -> ResNet / DenseNet, optionally ImageNet-pretrained
  * ``torchvision`` absent   -> a small CNN trained from scratch

The fallback is not a stand-in for a pretrained backbone; it exists so the
multimodal path stays runnable and testable in a minimal environment. Which
backbone was used is recorded in ``self.backbone_name`` and written into the
run metadata, so a result can never silently imply pretraining that did not
happen.

Pretrained weights expect ImageNet normalisation and 3 channels. Single-channel
medical images are repeated across channels, which is the usual convention and
is noted here because it is a real modelling assumption, not a detail.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:  # pragma: no cover - depends on the environment
    import torchvision.models as tv_models
    TORCHVISION_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCHVISION_AVAILABLE = False

SUPPORTED_BACKBONES = ("resnet18", "resnet50", "densenet121", "small_cnn")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _SmallCNN(nn.Module):
    """Fallback backbone. Trained from scratch, no pretrained prior."""

    def __init__(self, in_channels: int = 3, width: int = 32):
        super().__init__()
        blocks, channels = [], in_channels
        for out_channels in (width, width * 2, width * 4):
            blocks += [
                nn.Conv2d(channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            channels = out_channels
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.out_features = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: int = 64,
        freeze_backbone: bool = True,
        in_channels: int = 3,
        normalize_imagenet: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.pretrained = bool(pretrained and TORCHVISION_AVAILABLE)
        self.normalize_imagenet = normalize_imagenet and self.pretrained

        # Validate the name FIRST, before any availability check. A typo must
        # fail loudly whether or not torchvision is installed -- silently
        # substituting a different architecture for a misspelled one would let
        # a run report results from a model the user never asked for.
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. Use one of: "
                f"{', '.join(sorted(SUPPORTED_BACKBONES))}."
            )

        if TORCHVISION_AVAILABLE and backbone != "small_cnn":
            builders = {
                "resnet18": (tv_models.resnet18, "fc"),
                "resnet50": (tv_models.resnet50, "fc"),
                "densenet121": (tv_models.densenet121, "classifier"),
            }
            builder, head_name = builders[backbone]
            try:
                model = builder(weights="DEFAULT" if pretrained else None)
            except Exception as exc:
                # No network, or a blocked weight host. Keep the architecture but
                # be explicit that the ImageNet prior is absent, rather than
                # crashing an otherwise valid offline run.
                logger.warning(
                    "Could not fetch pretrained weights for %s (%s). Using random "
                    "initialisation; the ImageNet prior is NOT present.",
                    backbone, exc,
                )
                model = builder(weights=None)
                self.pretrained = False
                self.normalize_imagenet = False
            head = getattr(model, head_name)
            feature_dim = head.in_features
            setattr(model, head_name, nn.Identity())
            self.backbone = model
            self.backbone_name = backbone if self.pretrained else f"{backbone}_random_init"
        else:
            if backbone != "small_cnn":
                logger.warning(
                    "torchvision is not installed; falling back to an untrained "
                    "small CNN. Install torchvision for a pretrained backbone."
                )
            self.backbone = _SmallCNN(in_channels)
            feature_dim, self.backbone_name = self.backbone.out_features, "small_cnn"
            self.pretrained = self.normalize_imagenet = False

        self.feature_dim = feature_dim
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        if freeze_backbone and self.pretrained:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()
        self.frozen = bool(freeze_backbone and self.pretrained)

        if self.normalize_imagenet:
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W), or (B, D, C, H, W) for a volume -- depth is mean-pooled.

        Mean-pooling a 3-D volume across slices is a coarse choice: it discards
        through-plane structure. It is used here because a true 3-D backbone is
        a different model, not a configuration flag.
        """
        if x.dim() == 5:
            x = x.mean(dim=1)
        if x.dim() != 4:
            raise ValueError(f"Expected a 4-D or 5-D image tensor, got shape {tuple(x.shape)}.")
        if x.size(1) == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        if self.normalize_imagenet:
            x = (x - self.mean) / self.std
        features = self.backbone(x)
        return self.projection(features)

    def describe(self) -> dict:
        return {
            "modality": "image",
            "backbone": self.backbone_name,
            "pretrained": self.pretrained,
            "frozen": self.frozen,
            "feature_dim": int(self.feature_dim),
            "embedding_dim": int(self.embedding_dim),
        }
