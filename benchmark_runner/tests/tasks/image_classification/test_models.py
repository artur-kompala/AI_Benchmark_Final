"""Testy fabryki modeli (tasks/image_classification/models.py) - w szczegolnosci adaptacji
ResNet-50 pod male obrazy CIFAR-10 (32x32): zmniejszony stem (conv 3x3/stride1 zamiast
7x7/stride2, brak maxpool) zeby nie stracic calej rozdzielczosci przed pierwszym blokiem
resztkowym (Poprawka 1)."""

from __future__ import annotations

import torch
import torch.nn as nn

from benchmark_runner.tasks.image_classification.models import CIFAR10_NUM_CLASSES, build_model, build_resnet50


def test_build_resnet50_adapts_stem_for_small_images():
    model = build_resnet50()
    assert isinstance(model.conv1, nn.Conv2d)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)


def test_build_resnet50_forward_pass_on_32x32_input():
    model = build_resnet50().eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, CIFAR10_NUM_CLASSES)


def test_build_resnet50_final_classifier_matches_num_classes():
    model = build_resnet50(num_classes=7)
    assert model.fc.out_features == 7


def test_registry_distinguishes_mobilenet_and_resnet50():
    mobilenet = build_model("mobilenet_v3")
    resnet = build_model("resnet50")
    assert type(mobilenet) is not type(resnet)
