"""Abstrakcyjna klasa bazowa dla zrodel pomiaru mocy chwilowej (RAPL/codecarbon/NVML/...)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PowerSample:
    timestamp: datetime
    source: str
    watts: float | None
    temperature_c: float | None = None


# Gorny prog fizycznej wiarygodnosci sredniej mocy per device_type - dzielony miedzy
# samplerami (pierwsza linia obrony, odrzucaja implauzybilna POJEDYNCZA probke u zrodla -
# patrz power/codecarbon_meter.py, power/rapl_meter.py) i orchestrator.py (druga linia
# obrony, sprawdza SREDNIA na koniec przebiegu - patrz core/orchestrator.py::
# _check_power_plausibility). Blad, ktory to wymusil: przy zablokowanym uprawnieniami RAPL
# (PermissionError na /sys/class/powercap/intel-rapl/.../energy_uj) codecarbon w trybie
# fallback potrafi wpasc w niestabilny wewnetrzny stan i zwrocic fizycznie niemozliwa moc
# (300-1050W dla laptopa) zamiast czystego fallbacku - zadna z tych wartosci nie powinna
# nigdy trafic do power_samples/run_summary w Supabase.
#
# Tylko 'cpu' ma prog - GPU dyskretne (nvml/rocm_smi) legalnie przekraczaja 150W (np.
# RTX 4090 ~450W pod obciazeniem), wiec nie ma tu dla nich wpisu; NPU/Intel GPU (whole-
# system fallback przez baterie) tez celowo pominiete - mierza caly laptop, nie sam CPU.
MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE: dict[str, float] = {
    "cpu": 150.0,
}


class BasePowerSampler(ABC):
    source_name: str

    @abstractmethod
    def is_available(self) -> bool:
        """Sprawdza dostepnosc zrodla na tej maszynie. Nie rzuca wyjatku - zwraca False
        gdy sprzet/sterownik/biblioteka niedostepne, zeby benchmark mogl kontynuowac
        z pozostalymi zrodlami pomiaru."""

    @abstractmethod
    def sample(self) -> PowerSample | None:
        """Pojedynczy odczyt mocy chwilowej. Zwraca None przy chwilowym bledzie odczytu
        (nie rzuca wyjatku) - PowerSamplingThread po prostu pomija ten tick."""

    def close(self) -> None:
        """Zwolnienie zasobow (np. uchwyt NVML). Domyslnie no-op."""
        return None
