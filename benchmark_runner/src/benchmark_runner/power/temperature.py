"""Best-effort odczyt temperatury urzadzenia - kontekst do interpretacji wynikow (throttling).

Nie ma wiarygodnego, przenosnego API temperatury CPU na Windows bez dodatkowych
zaleznosci systemowych, dlatego dla CPU/NPU zwracamy None (udokumentowane ograniczenie,
patrz docs/measurement_methodology.md).
"""

from __future__ import annotations

import logging

from benchmark_runner.core.base_device import DeviceProfile

logger = logging.getLogger(__name__)


def read_temperature(device: DeviceProfile) -> float | None:
    try:
        if device.device_type == "cuda":
            return _read_nvml_temperature()
        if device.device_type == "rocm":
            return _read_rocm_temperature()
    except Exception:
        logger.debug("Blad odczytu temperatury dla urzadzenia '%s'", device.device_type, exc_info=True)
    return None


def _read_nvml_temperature() -> float | None:
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    finally:
        pynvml.nvmlShutdown()


def _read_rocm_temperature() -> float | None:
    import pyamdgpuinfo

    if pyamdgpuinfo.detect_gpus() == 0:
        return None
    gpu = pyamdgpuinfo.get_gpu(0)
    return float(gpu.query_temperature())
