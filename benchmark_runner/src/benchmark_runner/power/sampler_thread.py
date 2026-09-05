"""Watek probkujacy wiele zrodel mocy jednoczesnie (~100ms), niezalezny od watku obliczeniowego."""

from __future__ import annotations

import logging
import threading

from benchmark_runner.power.base_power_meter import BasePowerSampler, PowerSample

logger = logging.getLogger(__name__)


class PowerSamplingThread:
    def __init__(self, samplers: list[BasePowerSampler], interval_ms: int = 100) -> None:
        self._samplers = [s for s in samplers if self._safe_is_available(s)]
        self._interval_s = interval_ms / 1000.0
        self._samples: list[PowerSample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _safe_is_available(sampler: BasePowerSampler) -> bool:
        try:
            available = sampler.is_available()
        except Exception:
            logger.warning("Blad przy sprawdzaniu dostepnosci zrodla mocy '%s' - pomijam", sampler.source_name, exc_info=True)
            return False
        if not available:
            logger.warning("Zrodlo pomiaru mocy '%s' niedostepne na tej maszynie - pomijam", sampler.source_name)
        return available

    def start(self) -> "PowerSamplingThread":
        if not self._samplers:
            logger.warning("Brak dostepnych zrodel pomiaru mocy - watek probkujacy nie zbierze zadnych danych")
        self._thread = threading.Thread(target=self._run, daemon=True, name="power-sampling-thread")
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop_event.is_set():
            for sampler in self._samplers:
                try:
                    sample = sampler.sample()
                except Exception:
                    logger.warning("Blad odczytu probki mocy z '%s'", sampler.source_name, exc_info=True)
                    sample = None
                if sample is not None:
                    with self._lock:
                        self._samples.append(sample)
            self._stop_event.wait(self._interval_s)

    def sample_count(self) -> int:
        """Liczba probek zebranych DOTYCHCZAS, bez zatrzymywania watku - uzywane przez
        core/orchestrator.py do sprawdzenia miedzy kolejnymi probami pomiaru (retry loop
        _run_single), czy zebrano juz wystarczajaco probek, zanim zdecyduje o kolejnej
        probie z podniesiona liczba iteracji."""
        with self._lock:
            return len(self._samples)

    def drain(self) -> list[PowerSample]:
        """Zwraca i CZYSCI dotychczas zebrane probki, BEZ zatrzymywania watku - w
        odroznieniu od stop() nizej, watek zyje dalej i probkowanie trwa nieprzerwanie.

        Uzywane przez dlugozyjace zrodla wspoldzielone miedzy wieloma przebiegami w tej
        samej serii eksperymentow (patrz core/orchestrator.py::ExperimentRunner::
        _get_persistent_sampling_thread) - kazdy przebieg wywoluje drain() na poczatku
        swojego okna pomiarowego (odrzucajac probki z przerwy miedzy przebiegami / z
        prepare()+warmup TEGO przebiegu) i ponownie na koncu (zeby zebrac dokladnie
        probki z WLASNEGO okna, jednoczesnie czyszczac bufor pod kolejny przebieg)."""
        with self._lock:
            samples = self._samples
            self._samples = []
            return samples

    def stop(self) -> list[PowerSample]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        for sampler in self._samplers:
            try:
                sampler.close()
            except Exception:
                logger.debug("Blad przy zamykaniu samplera '%s'", sampler.source_name, exc_info=True)
        with self._lock:
            return list(self._samples)
