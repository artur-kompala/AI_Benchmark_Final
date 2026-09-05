"""Manualny tryb watomierza: runner pauzuje i prosi o reczne wprowadzenie odczytu
skumulowanej energii z aplikacji gniazdka (Wh), na starcie i na koncu przebiegu."""

from __future__ import annotations

import logging
from typing import Callable

from benchmark_runner.power.smart_plug.base import SmartPlugReader, SmartPlugReading

logger = logging.getLogger(__name__)


class ManualSmartPlugReader(SmartPlugReader):
    def __init__(
        self,
        enabled: bool = True,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self._enabled = enabled
        self._input = input_fn
        self._print = output_fn
        self._start_reading_wh: float | None = None

    def _prompt_for_energy_wh(self, prompt: str) -> float | None:
        while True:
            raw = self._input(prompt).strip()
            if raw == "":
                self._print("Pomijam odczyt watomierza dla tego przebiegu (brak wartosci).")
                return None
            try:
                return float(raw.replace(",", "."))
            except ValueError:
                self._print("Nieprawidlowa wartosc - podaj liczbe (np. 123.45) lub zostaw puste, zeby pominac.")

    def start_run(self, run_id: str) -> None:
        if not self._enabled:
            self._start_reading_wh = None
            return
        self._start_reading_wh = self._prompt_for_energy_wh(
            f"[watomierz] Przebieg {run_id}: odczytaj skumulowana energie (Wh) z aplikacji "
            "gniazdka i wpisz tutaj: "
        )

    def end_run(self, run_id: str) -> SmartPlugReading | None:
        if not self._enabled or self._start_reading_wh is None:
            return None

        end_reading_wh = self._prompt_for_energy_wh(
            f"[watomierz] Przebieg {run_id}: odczytaj ponownie skumulowana energie (Wh) "
            "i wpisz tutaj: "
        )
        if end_reading_wh is None:
            return None

        delta_wh = end_reading_wh - self._start_reading_wh
        if delta_wh < 0:
            logger.warning(
                "Odczyt koncowy watomierza (%.3f Wh) mniejszy niz poczatkowy (%.3f Wh) dla runu %s "
                "- mozliwy reset licznika w aplikacji gniazdka, odrzucam pomiar",
                end_reading_wh,
                self._start_reading_wh,
                run_id,
            )
            return None
        return SmartPlugReading(energy_wh=delta_wh)
