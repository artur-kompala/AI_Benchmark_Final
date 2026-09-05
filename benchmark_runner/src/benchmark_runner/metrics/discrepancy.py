"""Obliczanie rozbieznosci (%) miedzy energia zmierzona programowo a watomierzem fizycznym.

power_discrepancy_pct sluzy do oceny wiarygodnosci metod pomiaru programowego
(patrz docs/measurement_methodology.md) - nie jest korekta ani kalibracja.
"""

from __future__ import annotations


def compute_discrepancy(energy_software_j: float | None, energy_smart_plug_j: float | None) -> float | None:
    """(|software - smart_plug| / smart_plug) * 100.
    None gdy brakuje ktoregokolwiek odczytu lub gdy odczyt z watomierza jest zerowy
    (unikamy dzielenia przez zero zamiast rzucac wyjatek)."""
    if energy_software_j is None or energy_smart_plug_j is None:
        return None
    if energy_smart_plug_j == 0:
        return None
    return abs(energy_software_j - energy_smart_plug_j) / energy_smart_plug_j * 100.0


def wh_to_joules(energy_wh: float) -> float:
    return energy_wh * 3600.0
