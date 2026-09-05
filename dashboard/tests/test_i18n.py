"""Testy klasyfikacji urzadzen i etykiet (i18n/pl.py) - w tym naprawione bledy:
amd_igpu -> 'GPU zintegrowane (iGPU)', npu_openvino -> 'NPU (akcelerator dedykowany)'."""

from __future__ import annotations

import pytest

from dashboard.i18n.pl import (
    DEVICE_TYPE_COLORS,
    DEVICE_TYPE_ORDER,
    DEVICE_TYPE_SYMBOLS,
    device_category_label,
    device_type_label,
)


@pytest.mark.parametrize(
    "device_type,gpu_category,expected",
    [
        ("cpu", None, "CPU"),
        ("cuda", "discrete", "GPU dyskretne"),
        ("rocm", "discrete", "GPU dyskretne"),
        ("amd_igpu", "integrated", "GPU zintegrowane (iGPU)"),
        ("amd_igpu", None, "GPU zintegrowane (iGPU)"),        # naprawiony blad: bez gpu_category tez iGPU
        ("intel_gpu_openvino", "integrated", "GPU zintegrowane (iGPU)"),
        ("intel_gpu_openvino", None, "GPU zintegrowane (iGPU)"),
        ("npu_openvino", None, "NPU (akcelerator dedykowany)"),
        ("npu_directml", None, "NPU (akcelerator dedykowany)"),
        ("cuda", None, "GPU (kategoria nieznana)"),           # brak gpu_category dla dyskretnej -> czytelny fallback
    ],
)
def test_device_category_label(device_type, gpu_category, expected):
    assert device_category_label(device_type, gpu_category) == expected


def test_amd_igpu_never_falls_through_to_raw_string():
    """Regresja: stara wersja zwracala smieciowe 'amd_igpu'."""
    assert device_category_label("amd_igpu", None) != "amd_igpu"


def test_device_type_label_maps_all_ordered_types():
    for dt in DEVICE_TYPE_ORDER:
        assert device_type_label(dt) not in ("", "?", dt) or dt == "cpu"  # cpu -> "CPU" (dozwolone rowne)


def test_every_ordered_device_type_has_color_and_symbol():
    for dt in DEVICE_TYPE_ORDER:
        assert dt in DEVICE_TYPE_COLORS and DEVICE_TYPE_COLORS[dt].startswith("#")
        assert dt in DEVICE_TYPE_SYMBOLS


def test_device_type_order_has_no_duplicates():
    assert len(DEVICE_TYPE_ORDER) == len(set(DEVICE_TYPE_ORDER))
