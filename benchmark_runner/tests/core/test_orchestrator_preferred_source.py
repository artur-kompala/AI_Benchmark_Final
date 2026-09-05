"""Testy _select_preferred_source (Blad 2 - cichy, bez wyjatku: preferowane zrodlo pomiaru
mocy bylo sztywno requested[0] niezaleznie od dostepnosci. Potwierdzone empirycznie na CPU:
cpu_sources=['rapl','codecarbon'], rapl mial 0 probek (niedostepny na danej maszynie), ale
codecarbon zbieral setki poprawnych probek - mimo to avg_power_watts/energy_joules_software
zapisywaly sie jako None, bo aggregate_run filtrowal po pustym 'rapl'."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from benchmark_runner.core.orchestrator import _select_preferred_source
from benchmark_runner.power.base_power_meter import PowerSample

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _samples(source: str, n: int) -> list[PowerSample]:
    return [PowerSample(timestamp=_T0 + timedelta(seconds=i), source=source, watts=50.0) for i in range(n)]


def test_selects_source_with_most_samples_even_if_not_first_in_list():
    # Scenariusz ze zgloszenia: rapl (pierwsze w cpu_sources) ma 0 probek, codecarbon ma setki.
    power_samples = _samples("codecarbon", 704)
    result = _select_preferred_source(["rapl", "codecarbon"], power_samples)
    assert result == "codecarbon"


def test_selects_first_source_when_it_has_data():
    power_samples = _samples("rapl", 50) + _samples("codecarbon", 10)
    result = _select_preferred_source(["rapl", "codecarbon"], power_samples)
    assert result == "rapl"


def test_ties_prefer_configured_order():
    power_samples = _samples("rapl", 30) + _samples("codecarbon", 30)
    result = _select_preferred_source(["rapl", "codecarbon"], power_samples)
    assert result == "rapl"


def test_all_sources_empty_falls_back_to_first_for_determinism():
    result = _select_preferred_source(["rapl", "codecarbon"], [])
    assert result == "rapl"


def test_ignores_samples_with_none_watts():
    power_samples = [
        PowerSample(timestamp=_T0, source="rapl", watts=None),
        PowerSample(timestamp=_T0, source="rapl", watts=None),
    ] + _samples("codecarbon", 5)
    result = _select_preferred_source(["rapl", "codecarbon"], power_samples)
    assert result == "codecarbon"


def test_ignores_samples_from_sources_outside_requested_list():
    # np. probki od smart_plug nie powinny wplywac na wybor preferred_source dla CPU.
    power_samples = _samples("smart_plug", 1000) + _samples("rapl", 5)
    result = _select_preferred_source(["rapl", "codecarbon"], power_samples)
    assert result == "rapl"


def test_empty_requested_list_returns_none():
    assert _select_preferred_source([], _samples("codecarbon", 10)) is None
