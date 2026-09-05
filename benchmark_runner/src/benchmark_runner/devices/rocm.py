"""Detekcja GPU AMD przez PyTorch skompilowany z ROCm (torch.version.hip).

UWAGA: oficjalne wsparcie torch+ROCm na Windows jest ograniczone - moze wymagac
WSL2. Patrz docs/setup_windows_gpu.md.
"""

from __future__ import annotations

import logging

from benchmark_runner.core.base_device import DeviceProfile
from benchmark_runner.devices.gpu_classification import classify_gpu_category

logger = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        import torch

        # PyTorch mapuje HIP (ROCm) na to samo API co CUDA - torch.cuda.is_available()
        # zwraca True rowniez pod ROCm, dlatego rozrozniamy po torch.version.hip.
        return getattr(torch.version, "hip", None) is not None and torch.cuda.is_available()
    except Exception:
        logger.debug("PyTorch/ROCm niedostepne na tej maszynie", exc_info=True)
        return False


def get_profile() -> DeviceProfile:
    import torch

    index = 0
    label = torch.cuda.get_device_name(index)
    return DeviceProfile(
        device_type="rocm",
        label=label,
        vendor="amd",
        torch_device_str=f"cuda:{index}",  # PyTorch uzywa nazwy urzadzenia "cuda" rowniez pod ROCm
        backend="pytorch",
        driver_version=torch.version.hip,
        gpu_category=classify_gpu_category("amd", label),
    )
