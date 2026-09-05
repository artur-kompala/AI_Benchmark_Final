"""Detekcja GPU NVIDIA przez PyTorch (torch.cuda)."""

from __future__ import annotations

import logging

from benchmark_runner.core.base_device import DeviceProfile
from benchmark_runner.devices.gpu_classification import classify_gpu_category

logger = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        import torch

        # torch.version.hip is None odrzuca budowy PyTorch skompilowane pod ROCm
        # (te rowniez zglaszaja torch.cuda.is_available()==True, ale obsluguje je devices/rocm.py)
        return torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
    except Exception:
        logger.debug("PyTorch/CUDA niedostepne na tej maszynie", exc_info=True)
        return False


def get_profile() -> DeviceProfile:
    import torch

    index = 0
    label = torch.cuda.get_device_name(index)
    return DeviceProfile(
        device_type="cuda",
        label=label,
        vendor="nvidia",
        torch_device_str=f"cuda:{index}",
        backend="pytorch",
        driver_version=torch.version.cuda,
        gpu_category=classify_gpu_category("nvidia", label),
    )
