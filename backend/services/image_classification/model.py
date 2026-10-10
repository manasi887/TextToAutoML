from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


def build_mobilenet_v2_classifier(*, num_classes: int, pretrained: bool = True, freeze_features: bool = True, weights: Any | None = None) -> nn.Module:
    """Create the MobileNetV2 classifier used for image classification."""
    if num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")

    pretrained_weights_loaded = weights is not None
    if weights is not None:
        model = models.mobilenet_v2(weights=weights)
    elif pretrained:
        try:
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            pretrained_weights_loaded = True
        except Exception:
            model = models.mobilenet_v2(weights=None)
            pretrained_weights_loaded = False
    else:
        model = models.mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    model.pretrained_weights_loaded = pretrained_weights_loaded
    return model
