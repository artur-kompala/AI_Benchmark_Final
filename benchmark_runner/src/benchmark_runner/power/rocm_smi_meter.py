"""Pomiar mocy chwilowej GPU AMD przez pyamdgpuinfo lub subprocess `rocm-smi --showpower --json`."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone

from benchmark_runner.power.base_power_meter import BasePowerSampler, PowerSample

logger = logging.getLogger(__name__)

_ROCM_SMI_TIMEOUT_S = 5
_POWER_KEYS = (
    "Average Graphics Package Power (W)",
    "Current Socket Graphics Package Power (W)",
)


class RocmSmiPowerSampler(BasePowerSampler):
    source_name = "rocm_smi"

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._backend: str | None = None  # 'pyamdgpuinfo' | 'rocm-smi-cli'
        self._gpu = None

    def is_available(self) -> bool:
        try:
            import pyamdgpuinfo

            if pyamdgpuinfo.detect_gpus() > 0:
                self._gpu = pyamdgpuinfo.get_gpu(self._device_index)
                self._backend = "pyamdgpuinfo"
                return True
        except Exception:
            logger.debug("pyamdgpuinfo niedostepne, probuje subprocess rocm-smi", exc_info=True)

        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower", "--json"],
                capture_output=True,
                text=True,
                timeout=_ROCM_SMI_TIMEOUT_S,
                check=True,
            )
            json.loads(result.stdout)
            self._backend = "rocm-smi-cli"
            return True
        except Exception:
            logger.debug("rocm-smi CLI niedostepne na tej maszynie", exc_info=True)
            self._backend = None
            return False

    def sample(self) -> PowerSample | None:
        if self._backend == "pyamdgpuinfo":
            return self._sample_pyamdgpuinfo()
        if self._backend == "rocm-smi-cli":
            return self._sample_rocm_smi_cli()
        return None

    def _sample_pyamdgpuinfo(self) -> PowerSample | None:
        try:
            watts = self._gpu.query_power()
            return PowerSample(timestamp=datetime.now(timezone.utc), source=self.source_name, watts=watts)
        except Exception:
            logger.warning("Blad odczytu pyamdgpuinfo", exc_info=True)
            return None

    def _sample_rocm_smi_cli(self) -> PowerSample | None:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower", "--json"],
                capture_output=True,
                text=True,
                timeout=_ROCM_SMI_TIMEOUT_S,
                check=True,
            )
            data = json.loads(result.stdout)
            card_key = next(iter(data))
            card_data = data[card_key]
            watts_raw = next((card_data[key] for key in _POWER_KEYS if key in card_data), None)
            watts = float(watts_raw) if watts_raw is not None else None
            return PowerSample(timestamp=datetime.now(timezone.utc), source=self.source_name, watts=watts)
        except Exception:
            logger.warning("Blad odczytu rocm-smi CLI", exc_info=True)
            return None
