"""Pomiar mocy CPU przez RAPL (Running Average Power Limit) - pyRAPL.

RAPL na Windows ma ograniczone/niepewne wsparcie (patrz docs/measurement_methodology.md).
is_available() zwraca gracefully False, gdy RAPL nie jest dostepny na danej maszynie -
benchmark kontynuuje z pozostalymi zrodlami pomiaru CPU (np. codecarbon).

sample() defensywnie odrzuca kazdy pojedynczy odczyt poza fizycznie wiarygodnym zakresem
(patrz MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE) - RAPL-owe liczniki energii (energy_uj) potrafia
sie "zawijac" (wraparound) po przekroczeniu maksymalnej wartosci, co bez poprawnej obslugi
w pyRAPL mogloby dac pojedynczy, absurdalnie duzy odczyt delty energii/mocy; is_available()
udane przy starcie tez nie gwarantuje, ze uprawnienia do odczytu nie zostana odebrane w
trakcie (PermissionError na /sys/class/powercap/intel-rapl/.../energy_uj) - taki blad jest
juz lapany przez ponizszy except Exception i zwraca None (czysty fallback), ale ta sama
walidacja zabezpiecza rowniez przed cichym przepuszczeniem fizycznie niemozliwej wartosci,
gdyby jednak jakis odczyt "przeszedl" bez wyjatku.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from benchmark_runner.power.base_power_meter import BasePowerSampler, MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE, PowerSample

logger = logging.getLogger(__name__)

# RaplPowerSampler jest w _SOURCE_SAMPLER_MAP podpiety WYLACZNIE pod device_type='cpu'
# (patrz core/orchestrator.py) - prog dla 'cpu' jest wiec zawsze wlasciwy tutaj.
_MAX_PLAUSIBLE_WATTS = MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE["cpu"]


class RaplPowerSampler(BasePowerSampler):
    source_name = "rapl"

    def __init__(self) -> None:
        self._available = False

    def is_available(self) -> bool:
        try:
            import pyRAPL

            pyRAPL.setup()
            self._available = True
            return True
        except Exception:
            logger.debug("pyRAPL niedostepny na tej maszynie (typowe na Windows) - RAPL pomijany", exc_info=True)
            self._available = False
            return False

    def sample(self) -> PowerSample | None:
        if not self._available:
            return None
        try:
            import pyRAPL

            # pyRAPL mierzy energie w oknie begin/end; przybliza to "chwilowa" moc jako
            # srednia z bardzo krotkiego okna wokol momentu probkowania.
            meter = pyRAPL.Measurement("sample")
            meter.begin()
            meter.end()
            result = meter.result
            if not result.pkg:
                return None
            energy_uj = sum(v for v in result.pkg if v is not None)
            duration_s = result.duration / 1_000_000.0
            watts = (energy_uj / 1_000_000.0) / duration_s if duration_s > 0 else None

            if watts is not None and (watts < 0 or watts > _MAX_PLAUSIBLE_WATTS):
                logger.warning(
                    "RAPL zwrocil fizycznie nieprawdopodobna moc CPU: %.1fW (prog=%.0fW, "
                    "energy_uj=%d, duration_s=%.3f) - odrzucam ta probke jako blad odczytu "
                    "(np. zawiniecie licznika RAPL) zamiast zapisac smieciowa wartosc",
                    watts,
                    _MAX_PLAUSIBLE_WATTS,
                    energy_uj,
                    duration_s,
                )
                return None

            return PowerSample(timestamp=datetime.now(timezone.utc), source=self.source_name, watts=watts)
        except Exception:
            logger.warning("Blad odczytu RAPL", exc_info=True)
            return None
