"""Testy detekcji zintegrowanego AMD GPU - dostepnosc zalezy od tego, czy zainstalowany
PyTorch jest budowany z ROCm (torch.version.hip), co jest niezalezne od dostepnosci
sprzetowej samego iGPU (patrz devices/amd_igpu.py - na maszynie testowej "Ola" z
zainstalowanym torch+CUDA dla dyskretnego GPU NVIDIA, is_available() poprawnie zwraca
False mimo ze iGPU AMD fizycznie jest obecny i widoczny przez sysfs/amdgpu)."""

from __future__ import annotations

import pytest

from benchmark_runner.devices import amd_igpu


def test_is_available_never_raises():
    assert isinstance(amd_igpu.is_available(), bool)


def test_get_profile_raises_when_unavailable():
    if amd_igpu.is_available():
        pytest.skip("Zintegrowane AMD GPU dostepne na tej maszynie - test dotyczy przypadku braku")
    with pytest.raises(Exception):
        amd_igpu.get_profile()


def test_returns_none_when_torch_not_rocm_build(monkeypatch):
    """Reprodukuje dokladnie sytuacje ze zgloszenia: torch+CUDA (dla dyskretnego NVIDIA)
    zainstalowany obok fizycznie obecnego iGPU AMD - torch.version.hip=None musi dawac
    czyste is_available()=False, NIE wyjatek."""
    import torch

    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    assert amd_igpu.is_available() is False


def test_only_selects_integrated_classified_device(monkeypatch):
    """Gdyby torch+ROCm widzial WIELE urzadzen AMD, funkcja musi wybrac to sklasyfikowane
    jako 'integrated' (patrz gpu_classification.py), nie pierwsze z brzegu (analogicznie
    do devices/rocm.py, ktore - odwrotnie - powinno unikac zintegrowanych)."""
    import types

    fake_torch = types.SimpleNamespace(
        version=types.SimpleNamespace(hip="6.0"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_name=lambda i: ["AMD Radeon RX 7800 XT", "AMD Radeon(TM) Graphics"][i],
        ),
    )

    import sys

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert amd_igpu.is_available() is True
    profile = amd_igpu.get_profile()
    assert profile.device_type == "amd_igpu"
    assert profile.torch_device_str == "cuda:1"
    assert profile.gpu_category == "integrated"
    assert profile.backend == "pytorch"
