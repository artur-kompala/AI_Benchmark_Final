"""Automatyczna detekcja dostepnych urzadzen na biezacej maszynie + rozwiazywanie override'ow."""

from __future__ import annotations

import logging

from benchmark_runner.core.base_device import DeviceProfile
from benchmark_runner.devices import amd_igpu, cpu, cuda, intel_gpu, npu_directml, npu_openvino, rocm

logger = logging.getLogger(__name__)

_DEVICE_MODULES = {
    "cpu": cpu,
    "cuda": cuda,
    "rocm": rocm,
    "amd_igpu": amd_igpu,
    "npu_openvino": npu_openvino,
    "npu_directml": npu_directml,
    "intel_gpu_openvino": intel_gpu,
}


def detect_available_devices() -> list[DeviceProfile]:
    """Zwraca wszystkie urzadzenia dostepne na biezacej maszynie (co najmniej CPU).
    Blad detekcji pojedynczego urzadzenia jest logowany i pomijany - nigdy nie wywala calego runnera."""
    profiles: list[DeviceProfile] = []
    for device_type, module in _DEVICE_MODULES.items():
        try:
            if module.is_available():
                profiles.append(module.get_profile())
        except Exception:
            logger.warning("Blad przy wykrywaniu urzadzenia '%s' - pomijam", device_type, exc_info=True)
    return profiles


def resolve_device(requested: str | None) -> DeviceProfile:
    """requested=None lub "auto" -> preferuje pierwsze dostepne urzadzenie inne niz CPU
    (GPU/NPU), inaczej CPU. requested="cpu"/"cuda"/"rocm"/... -> wymuszenie konkretnego
    typu, blad jesli niedostepne na tej maszynie."""
    available = detect_available_devices()
    if not available:
        raise RuntimeError("Nie wykryto zadnego urzadzenia obliczeniowego (nawet CPU) - to nie powinno sie zdarzyc")

    if requested is None or requested == "auto":
        non_cpu = [p for p in available if p.device_type != "cpu"]
        return non_cpu[0] if non_cpu else available[0]

    for profile in available:
        if profile.device_type == requested:
            return profile

    available_types = ", ".join(p.device_type for p in available)
    raise RuntimeError(f"Zadane urzadzenie '{requested}' niedostepne na tej maszynie. Dostepne: {available_types}")
