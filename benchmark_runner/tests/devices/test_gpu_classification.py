"""Testy heurystyki devices/gpu_classification.py - dyskretne vs zintegrowane GPU."""

from __future__ import annotations

from benchmark_runner.devices.gpu_classification import classify_gpu_category


def test_non_gpu_vendor_returns_none():
    assert classify_gpu_category(None, "CPU") is None
    assert classify_gpu_category("intel", "Intel NPU") is None


def test_nvidia_discrete_cards():
    assert classify_gpu_category("nvidia", "NVIDIA GeForce RTX 4070") == "discrete"
    assert classify_gpu_category("nvidia", "NVIDIA GeForce RTX 3060 Laptop GPU") == "discrete"


def test_nvidia_integrated_jetson():
    assert classify_gpu_category("nvidia", "NVIDIA Tegra Jetson Orin") == "integrated"


def test_amd_discrete_cards():
    assert classify_gpu_category("amd", "AMD Radeon RX 6700 XT") == "discrete"


def test_amd_integrated_apu():
    assert classify_gpu_category("amd", "AMD Radeon(TM) Graphics") == "integrated"
    assert classify_gpu_category("amd", "AMD Radeon 780M") == "integrated"


def test_classification_is_case_insensitive():
    assert classify_gpu_category("amd", "amd radeon(tm) graphics") == "integrated"
