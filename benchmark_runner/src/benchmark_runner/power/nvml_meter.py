"""Pomiar mocy chwilowej GPU NVIDIA przez pynvml (odpowiednik nvidia-smi power.draw)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from benchmark_runner.power.base_power_meter import BasePowerSampler, PowerSample

logger = logging.getLogger(__name__)


class NvmlPowerSampler(BasePowerSampler):
    source_name = "nvml"

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._handle = None

    def is_available(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            return True
        except Exception:
            logger.debug("pynvml/NVML niedostepne na tej maszynie", exc_info=True)
            self._handle = None
            return False

    def sample(self) -> PowerSample | None:
        if self._handle is None:
            return None
        try:
            import pynvml

            milliwatts = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            watts = milliwatts / 1000.0
            temperature_c = None
            try:
                temperature_c = float(pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                logger.debug("Nie udalo sie odczytac temperatury GPU przez NVML", exc_info=True)
            return PowerSample(
                timestamp=datetime.now(timezone.utc),
                source=self.source_name,
                watts=watts,
                temperature_c=temperature_c,
            )
        except Exception:
            logger.warning("Blad odczytu NVML", exc_info=True)
            return None

    def close(self) -> None:
        if self._handle is not None:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                logger.debug("Blad przy nvmlShutdown", exc_info=True)
            self._handle = None
