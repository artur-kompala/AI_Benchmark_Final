"""Testy _check_power_plausibility (druga linia obrony przed fizycznie nieprawdopodobnymi
wartosciami avg_power_watts - patrz power/base_power_meter.py::MAX_PLAUSIBLE_WATTS_BY_
DEVICE_TYPE i pierwsza linia obrony w RaplPowerSampler/CodecarbonPowerSampler)."""

from __future__ import annotations

from benchmark_runner.core.orchestrator import _check_power_plausibility


def test_flags_cpu_over_threshold():
    reason = _check_power_plausibility("cpu", 850.3)
    assert reason is not None
    assert "850.3" in reason
    assert "150" in reason


def test_allows_cpu_under_threshold():
    assert _check_power_plausibility("cpu", 45.0) is None


def test_boundary_exactly_at_threshold_is_allowed():
    assert _check_power_plausibility("cpu", 150.0) is None


def test_just_over_threshold_is_flagged():
    assert _check_power_plausibility("cpu", 150.01) is not None


def test_none_avg_power_returns_none():
    assert _check_power_plausibility("cpu", None) is None


def test_no_threshold_configured_for_device_type_returns_none():
    # GPU dyskretne (cuda/rocm) legalnie przekraczaja 150W - brak progu dla nich.
    assert _check_power_plausibility("cuda", 450.0) is None
    assert _check_power_plausibility("rocm", 300.0) is None


def test_negative_avg_power_not_flagged_by_this_check():
    # Ujemne watts sa juz odrzucane u zrodla (patrz testy RaplPowerSampler/
    # CodecarbonPowerSampler) - ta funkcja pilnuje tylko GORNEGO progu.
    assert _check_power_plausibility("cpu", -5.0) is None
