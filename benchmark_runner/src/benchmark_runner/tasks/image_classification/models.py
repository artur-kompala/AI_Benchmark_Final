"""Fabryka modeli klasyfikacji obrazow. MODEL_REGISTRY pozwala dopisac nowy model
(np. EfficientNet-lite) bez zmian w task.py."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn
from torchvision import models

CIFAR10_NUM_CLASSES = 10


def build_mobilenet_v3(num_classes: int = CIFAR10_NUM_CLASSES) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_resnet50(num_classes: int = CIFAR10_NUM_CLASSES) -> nn.Module:
    """ResNet-50 zaadaptowany pod male obrazy CIFAR-10 (32x32), standardowa praktyka przy
    uzyciu ResNet poza ImageNet: oryginalny "stem" (conv 7x7/stride2 + maxpool 3x3/stride2)
    redukuje rozdzielczosc 4x jeszcze PRZED pierwszym blokiem resztkowym - dla wejscia
    224x224 (ImageNet) to sensowne, ale dla 32x32 zostawiloby pierwszemu blokowi mape cech
    8x8, obcinajac wiekszosc przestrzennej informacji zanim siec zdazy z niej skorzystac.
    Zamiast tego: kernel 3x3/stride1 (bez utraty rozdzielczosci w stem) i usuniecie maxpool
    (nn.Identity) - reszta architektury (4 stage'y resztkowe + fc) bez zmian."""
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "mobilenet_v3": build_mobilenet_v3,
    "resnet50": build_resnet50,
}


def build_model(name: str, num_classes: int = CIFAR10_NUM_CLASSES) -> nn.Module:
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Nieznany model '{name}'. Dostepne: {available}") from exc
    return factory(num_classes=num_classes)
