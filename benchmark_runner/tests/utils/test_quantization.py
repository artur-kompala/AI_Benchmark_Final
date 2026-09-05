"""Testy quantize_dynamic_int8 - patrz utils/quantization.py. Uzywa malych, sztucznych
modeli (nie prawdziwych MobileNetV3/DistilBERT) zeby testy byly szybkie i niezalezne od
pobierania wag."""

from __future__ import annotations

import torch.nn as nn

from benchmark_runner.utils.quantization import quantize_dynamic_int8


class _LinearOnlyModel(nn.Module):
    """Analogiczne do DistilBERT (glownie warstwy Linear) - dynamiczna kwantyzacja
    powinna skwantyzowac WSZYSTKIE kwalifikujace sie warstwy -> status 'success'."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class _ConvHeavyModel(nn.Module):
    """Analogiczne do MobileNetV3/ResNet-50 (Conv2d + jedna koncowa Linear) - Conv2d nie
    jest wspierany przez domyslny mapping dynamicznej kwantyzacji PyTorch, wiec tylko
    warstwa Linear zostanie skwantyzowana -> status 'partial'."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(self.conv(x).mean(dim=[2, 3]))


class _NoQuantizableLayersModel(nn.Module):
    """Brak Linear/Conv2d -> nie ma czego kwantyzowac -> status 'failed'."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(x)


def test_quantize_dynamic_int8_all_linear_is_success():
    model = _LinearOnlyModel().eval()
    quantized, status = quantize_dynamic_int8(model)
    assert status == "success"
    assert type(quantized.fc1).__module__.startswith("torch.ao.nn.quantized")
    assert type(quantized.fc2).__module__.startswith("torch.ao.nn.quantized")


def test_quantize_dynamic_int8_conv_heavy_is_partial():
    model = _ConvHeavyModel().eval()
    quantized, status = quantize_dynamic_int8(model)
    assert status == "partial"
    # Conv2d nie jest wspierany przez domyslny mapping - zostaje fp32.
    assert type(quantized.conv) is nn.Conv2d
    # Linear jest wspierany - powinien zostac skwantyzowany.
    assert type(quantized.fc).__module__.startswith("torch.ao.nn.quantized")


def test_quantize_dynamic_int8_no_quantizable_layers_is_failed():
    model = _NoQuantizableLayersModel().eval()
    quantized, status = quantize_dynamic_int8(model)
    assert status == "failed"
    # Model wraca niezmieniony (fp32 fallback).
    assert quantized is model


def test_quantize_dynamic_int8_output_shape_unchanged():
    import torch

    model = _LinearOnlyModel().eval()
    quantized, _ = quantize_dynamic_int8(model)
    x = torch.randn(2, 16)
    with torch.no_grad():
        out = quantized(x)
    assert out.shape == (2, 4)
