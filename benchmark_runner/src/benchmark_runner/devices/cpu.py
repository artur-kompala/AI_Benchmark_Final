"""CPU jest zawsze dostepnym urzadzeniem fallback."""

from __future__ import annotations

from benchmark_runner.core.base_device import DeviceProfile


def is_available() -> bool:
    return True


def get_profile() -> DeviceProfile:
    return DeviceProfile(
        device_type="cpu",
        label="CPU",
        vendor=None,
        torch_device_str="cpu",
        backend="pytorch",
    )
