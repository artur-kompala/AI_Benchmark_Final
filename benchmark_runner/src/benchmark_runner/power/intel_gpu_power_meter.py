"""Pomiar mocy Intel GPU: brak publicznego, udokumentowanego API natywnego odczytu mocy
(analogicznie do NPU - patrz npu_power_meter.py), fallback na pomiar whole-system przez
tempo rozladowania baterii (WholeSystemPowerSampler).

Orchestrator oznacza przebiegi na Intel GPU jako measurement_scope='whole_system' (zawsze,
poniewaz natywny odczyt samego GPU Intela nie jest obecnie dostepny) - patrz
docs/measurement_methodology.md.
"""

from __future__ import annotations

from benchmark_runner.power.base_power_meter import BasePowerSampler, PowerSample
from benchmark_runner.power.whole_system_meter import WholeSystemPowerSampler


class IntelGpuPowerSampler(BasePowerSampler):
    source_name = "intel_gpu_native"

    def __init__(self) -> None:
        self._fallback = WholeSystemPowerSampler(source_name=self.source_name)

    def is_available(self) -> bool:
        # Brak publicznego API natywnego odczytu mocy samego GPU Intela (stan na dzien
        # pisania) - od razu probujemy fallback whole-system (dziala tylko na baterii).
        return self._fallback.is_available()

    def sample(self) -> PowerSample | None:
        return self._fallback.sample()

    def close(self) -> None:
        self._fallback.close()
