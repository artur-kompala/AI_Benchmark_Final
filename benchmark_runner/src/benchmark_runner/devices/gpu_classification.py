"""Heurystyczna klasyfikacja GPU jako 'discrete' (dyskretne) vs 'integrated' (zintegrowane, iGPU).

Ani PyTorch/CUDA, ani ROCm nie udostepniaja bezposredniego, przenosnego API zwracajacego
"czy ta karta jest dyskretna czy zintegrowana" - dlatego klasyfikacja opiera sie na
dopasowaniu wzorcow w nazwie urzadzenia (torch.cuda.get_device_name()), nie na odczycie
sprzetowym. To rozroznienie jest wazne dla wnioskow pracy (GPU dyskretne i zintegrowane
to inny rzad wielkosci poboru mocy - patrz strona "GPU dyskretne vs zintegrowane" w
dashboardzie), wiec bledna klasyfikacja dla nierozpoznanego modelu karty powinna zostac
poprawiona recznie w Supabase (kolumna devices.gpu_category) zamiast polegac wylacznie
na tej heurystyce.
"""

from __future__ import annotations

# Typowe wzorce nazw zintegrowanych GPU AMD (APU Ryzen: generyczna nazwa "Radeon(TM) Graphics"
# bez numeru modelu karty, oraz architektury Vega/RDNA uzywane w APU zamiast dedykowanych RX).
_AMD_INTEGRATED_PATTERNS = (
    "radeon(tm) graphics",
    "radeon graphics",
    "vega 3",
    "vega 6",
    "vega 7",
    "vega 8",
    "vega 10",
    "vega 11",
    "610m",
    "660m",
    "680m",
    "740m",
    "760m",
    "780m",
    "880m",
    "890m",
)

# NVIDIA nie sprzedaje zintegrowanych GPU do desktopow/laptopow w tradycyjnym sensie (w
# odroznieniu od Intela/AMD) - jedyny powszechny wyjatek to SoC Jetson/Tegra.
_NVIDIA_INTEGRATED_PATTERNS = ("jetson", "tegra")


def classify_gpu_category(vendor: str | None, label: str) -> str | None:
    """Zwraca 'discrete' | 'integrated' | None (vendor spoza amd/nvidia - np. cpu/npu, gdzie
    kategoria GPU nie ma zastosowania)."""
    if vendor not in ("amd", "nvidia"):
        return None

    lowered = label.lower()
    patterns = _NVIDIA_INTEGRATED_PATTERNS if vendor == "nvidia" else _AMD_INTEGRATED_PATTERNS
    if any(pattern in lowered for pattern in patterns):
        return "integrated"
    return "discrete"
