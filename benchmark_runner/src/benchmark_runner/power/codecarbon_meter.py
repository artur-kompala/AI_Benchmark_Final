"""Pomiar zuzycia energii CPU przez codecarbon (EmissionsTracker).

Codecarbon natywnie mierzy energie skumulowana w czasie (brak API "chwilowej mocy"),
wiec kazdy sample() zwraca srednia moc w oknie od poprzedniego odczytu
(delta energii / delta czasu) - podejscie zblizone do tego, co robi watomierz manualny,
tylko w duzo krotszych, automatycznych oknach.

Blad: gdy RAPL (ktorego codecarbon probuje uzyc jako pierwszego zrodla na Linuksie) jest
zablokowany uprawnieniami (PermissionError na /sys/class/powercap/intel-rapl/.../energy_uj),
codecarbon potrafi wpasc w niestabilny wewnetrzny stan zamiast czysto przelaczyc sie na
fallback (szacowanie z cpu_load) - potwierdzone empirycznie: zapisane avg_power_watts
300-1050W, fizycznie niemozliwe dla laptopa. Nie da sie tego naprawic wewnatrz codecarbon
(zewnetrzna biblioteka), wiec sample() defensywnie odrzuca kazdy pojedynczy odczyt poza
fizycznie wiarygodnym zakresem (patrz MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE) - niezaleznie od
PRZYCZYNY (permission error, wewnetrzny bug codecarbon, cokolwiek innego), zamiast ufac
bezkrytycznie wynikowi zewnetrznej biblioteki.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from benchmark_runner.power.base_power_meter import BasePowerSampler, MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE, PowerSample

logger = logging.getLogger(__name__)

# CodecarbonPowerSampler jest w _SOURCE_SAMPLER_MAP podpiety WYLACZNIE pod device_type='cpu'
# (patrz core/orchestrator.py) - prog dla 'cpu' jest wiec zawsze wlasciwy tutaj.
_MAX_PLAUSIBLE_WATTS = MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE["cpu"]


class CodecarbonPowerSampler(BasePowerSampler):
    source_name = "codecarbon"

    def __init__(self) -> None:
        self._tracker = None
        self._last_energy_kwh: float = 0.0
        self._last_time: float | None = None

    def is_available(self) -> bool:
        try:
            from codecarbon import EmissionsTracker

            self._tracker = EmissionsTracker(
                measure_power_secs=1,
                save_to_file=False,
                log_level="error",
            )
            self._tracker.start()
            self._last_time = time.monotonic()
            self._last_energy_kwh = 0.0
            return True
        except Exception:
            logger.debug("codecarbon niedostepny lub nie udalo sie zainicjowac trackera", exc_info=True)
            self._tracker = None
            return False

    def sample(self) -> PowerSample | None:
        if self._tracker is None or self._last_time is None:
            return None
        try:
            self._tracker.flush()
            total_energy = getattr(self._tracker, "_total_energy", None)
            energy_kwh = float(total_energy.kWh) if total_energy is not None else None
            now = time.monotonic()
            if energy_kwh is None:
                return None
            delta_kwh = max(energy_kwh - self._last_energy_kwh, 0.0)
            delta_s = now - self._last_time
            watts = (delta_kwh * 3_600_000) / delta_s if delta_s > 0 else None

            # Aktualizuj stan PRZED ewentualnym odrzuceniem - inaczej nastepny odczyt
            # liczylby delte od tej samej (stalej) bazy, co przy utrzymujacym sie
            # niestabilnym stanie codecarbon tylko powiekszaloby kolejna implauzybilna
            # wartosc zamiast pozwolic sekwencji "dogonic" rzeczywisty stan.
            self._last_energy_kwh = energy_kwh
            self._last_time = now

            if watts is not None and (watts < 0 or watts > _MAX_PLAUSIBLE_WATTS):
                logger.warning(
                    "codecarbon zwrocil fizycznie nieprawdopodobna moc CPU: %.1fW (prog=%.0fW, "
                    "delta_kwh=%.8f, delta_s=%.3f) - odrzucam ta probke jako blad odczytu "
                    "(np. RAPL zablokowany uprawnieniami wprawiajacy wewnetrzny stan "
                    "EmissionsTracker w niestabilnosc) zamiast zapisac smieciowa wartosc",
                    watts,
                    _MAX_PLAUSIBLE_WATTS,
                    delta_kwh,
                    delta_s,
                )
                return None

            return PowerSample(timestamp=datetime.now(timezone.utc), source=self.source_name, watts=watts)
        except Exception:
            logger.warning("Blad odczytu codecarbon", exc_info=True)
            return None

    def close(self) -> None:
        if self._tracker is not None:
            try:
                self._tracker.stop()
            except Exception:
                logger.debug("Blad przy zatrzymywaniu trackera codecarbon", exc_info=True)
            self._tracker = None
