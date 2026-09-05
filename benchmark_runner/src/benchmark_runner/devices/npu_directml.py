"""Alternatywna sciezka NPU przez ONNX Runtime + DirectML.

UWAGA: DirectML nie pozwala jednoznacznie zweryfikowac, czy wybrany adapter to NPU
czy GPU (execution provider dziala na dowolnym urzadzeniu DirectX 12) - traktuj te
sciezke jako uzupelniajaca dla operatorow niewspieranych jeszcze przez plugin NPU w
OpenVINO, nie jako pewne potwierdzenie uzycia NPU. OpenVINO (npu_openvino.py) jest
sciezka podstawowa i jednoznaczna - patrz docs/setup_npu_intel.md.
"""

from __future__ import annotations

import logging

from benchmark_runner.core.base_device import DeviceProfile

logger = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        import onnxruntime as ort

        return "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:
        logger.debug("ONNX Runtime/DirectML niedostepne na tej maszynie", exc_info=True)
        return False


def get_profile() -> DeviceProfile:
    import onnxruntime as ort

    return DeviceProfile(
        device_type="npu_directml",
        label="DirectML device (NPU/GPU nie rozroznione)",
        vendor=None,
        torch_device_str=None,
        backend="onnxruntime",
        driver_version=ort.__version__,
        extra={"note": "DirectML execution provider nie pozwala jednoznacznie zweryfikowac czy uzyto NPU czy GPU"},
    )
