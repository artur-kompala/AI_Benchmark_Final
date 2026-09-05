"""Detekcja Intel GPU (zintegrowane Iris Xe/Arc lub dyskretne Arc A-/B-series) przez
OpenVINO GPU plugin - ta sama sciezka co NPU (npu_openvino.py: eksport do ONNX +
kompilacja przez OpenVINO), inny device string OpenVINO ('GPU' zamiast 'NPU').

W odroznieniu od heurystyki po nazwie karty uzywanej dla CUDA/ROCm (patrz
gpu_classification.py, ktore API tamtych stosow nie udostepnia wprost), OpenVINO ma
natywna, wiarygodna wlasciwosc DEVICE_TYPE zwracajaca Type.INTEGRATED/Type.DISCRETE
bezposrednio ze sterownika - uzywamy jej zamiast dopasowywania wzorcow w nazwie.
"""

from __future__ import annotations

import logging

from benchmark_runner.core.base_device import DeviceProfile

logger = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        from openvino import Core

        return "GPU" in Core().available_devices
    except Exception:
        logger.debug("OpenVINO niedostepne lub GPU Intela nie zostalo wykryte na tej maszynie", exc_info=True)
        return False


def get_profile() -> DeviceProfile:
    from openvino import Core, get_version

    core = Core()
    try:
        full_name = str(core.get_property("GPU", "FULL_DEVICE_NAME"))
    except Exception:
        logger.debug("Nie udalo sie odczytac FULL_DEVICE_NAME dla GPU Intela", exc_info=True)
        full_name = "Intel GPU"

    gpu_category = None
    try:
        import openvino.properties as props

        device_type = core.get_property("GPU", "DEVICE_TYPE")
        gpu_category = "integrated" if device_type == props.device.Type.INTEGRATED else "discrete"
    except Exception:
        logger.debug(
            "Nie udalo sie odczytac DEVICE_TYPE (dyskretne/zintegrowane) dla GPU Intela", exc_info=True
        )

    return DeviceProfile(
        device_type="intel_gpu_openvino",
        label=full_name,
        vendor="intel",
        torch_device_str=None,
        backend="openvino",
        driver_version=get_version(),
        gpu_category=gpu_category,
    )
