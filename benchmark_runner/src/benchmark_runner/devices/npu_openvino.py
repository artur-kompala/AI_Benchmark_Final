"""Detekcja NPU (Intel Core Ultra) przez OpenVINO - podstawowa, jednoznaczna sciezka NPU.

Patrz docs/setup_npu_intel.md po instrukcje instalacji sterownika i OpenVINO.
"""

from __future__ import annotations

import logging

from benchmark_runner.core.base_device import DeviceProfile

logger = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        from openvino import Core

        return "NPU" in Core().available_devices
    except Exception:
        logger.debug("OpenVINO niedostepne lub NPU nie zostalo wykryte na tej maszynie", exc_info=True)
        return False


def get_profile() -> DeviceProfile:
    from openvino import Core, get_version

    core = Core()
    try:
        full_name = str(core.get_property("NPU", "FULL_DEVICE_NAME"))
    except Exception:
        logger.debug("Nie udalo sie odczytac FULL_DEVICE_NAME dla NPU", exc_info=True)
        full_name = "Intel NPU"

    return DeviceProfile(
        device_type="npu_openvino",
        label=full_name,
        vendor="intel",
        torch_device_str=None,
        backend="openvino",
        driver_version=get_version(),
    )
