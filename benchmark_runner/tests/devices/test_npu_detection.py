"""Testy detekcji NPU - dostepnosc zalezy od maszyny (openvino/onnxruntime-directml
moga byc zainstalowane lub nie), wiec testujemy kontrakt "nigdy nie rzuca wyjatku
z is_available()", a nie konkretny wynik."""

from __future__ import annotations

import pytest

from benchmark_runner.devices import npu_directml, npu_openvino


def test_npu_openvino_is_available_never_raises():
    assert isinstance(npu_openvino.is_available(), bool)


def test_npu_directml_is_available_never_raises():
    assert isinstance(npu_directml.is_available(), bool)


def test_npu_openvino_get_profile_raises_when_unavailable():
    if npu_openvino.is_available():
        pytest.skip("OpenVINO NPU dostepne na tej maszynie - test dotyczy przypadku braku")
    with pytest.raises(Exception):
        npu_openvino.get_profile()


def test_npu_directml_get_profile_raises_when_unavailable():
    if npu_directml.is_available():
        pytest.skip("DirectML dostepne na tej maszynie - test dotyczy przypadku braku")
    with pytest.raises(Exception):
        npu_directml.get_profile()
