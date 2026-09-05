"""Interfejs pomiaru energii przez watomierz fizyczny (inteligentne gniazdko).

Jedyna zaimplementowana teraz klasa to ManualSmartPlugReader (manual_reader.py).
Backendy z automatycznym pollingiem przez lokalne API (np. Tasmota/Shelly/Kasa)
beda dopisane pozniej jako kolejne implementacje tego samego interfejsu, wybierane
w configu YAML przez pole `power_meter.smart_plug.backend`, bez zmian w orchestratorze.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SmartPlugReading:
    energy_wh: float  # energia zuzyta w trakcie przebiegu (delta miedzy start_run a end_run), w Wh


class SmartPlugReader(ABC):
    @abstractmethod
    def start_run(self, run_id: str) -> None:
        """Wywolywane przed rozpoczeciem wlasciwego pomiaru przebiegu (po warm-up)."""

    @abstractmethod
    def end_run(self, run_id: str) -> SmartPlugReading | None:
        """Wywolywane po zakonczeniu przebiegu. Zwraca zuzyta energie (Wh), albo None
        gdy watomierz jest wylaczony lub odczyt sie nie powiodl (benchmark kontynuuje bez niego)."""

    def read(self) -> SmartPlugReading | None:
        """Opcjonalny pojedynczy odczyt biezacy (do przyszlych backendow z live polling).
        Domyslnie niewspierany."""
        return None
