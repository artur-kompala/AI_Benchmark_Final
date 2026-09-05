"""Testy estimate_flops (metrics/flops.py) - modele sztuczne i male, zeby testy byly
szybkie i niezalezne od pobierania wag/danych."""

from __future__ import annotations

import torch
import torch.nn as nn

from benchmark_runner.metrics.flops import estimate_flops


class _TinyConvModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _TwoInputModel(nn.Module):
    """Model z dwoma wejsciami - analogiczne do sygnatury forward(input_ids, attention_mask)
    modeli HuggingFace uzywanych przez nlp_sentiment."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.fc(a + b)


def test_estimate_flops_with_input_shape_returns_positive_value():
    model = _TinyConvModel().eval()
    flops = estimate_flops(model, input_shape=(3, 8, 8))
    assert flops is not None
    assert flops > 0


def test_estimate_flops_with_dummy_inputs_tuple():
    model = _TwoInputModel().eval()
    dummy_a = torch.randn(1, 8)
    dummy_b = torch.randn(1, 8)
    flops = estimate_flops(model, dummy_inputs=(dummy_a, dummy_b))
    assert flops is not None
    assert flops > 0


def test_estimate_flops_restores_training_mode():
    model = _TinyConvModel()
    model.train()
    estimate_flops(model, input_shape=(3, 8, 8))
    assert model.training is True


def test_estimate_flops_raises_without_shape_or_inputs():
    model = _TinyConvModel()
    try:
        estimate_flops(model)
    except ValueError:
        pass
    else:
        raise AssertionError("estimate_flops() powinno rzucic ValueError bez input_shape/dummy_inputs")
