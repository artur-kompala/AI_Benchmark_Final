"""Testy CodecarbonPowerSampler - w szczegolnosci odrzucanie fizycznie nieprawdopodobnych
odczytow mocy. Blad zgloszony w sesji: przy zablokowanym uprawnieniami RAPL (PermissionError
na /sys/class/powercap/intel-rapl/.../energy_uj), codecarbon w trybie fallback potrafi wpasc
w niestabilny wewnetrzny stan i zwrocic fizycznie niemozliwa moc CPU (300-1050W dla laptopa)
zamiast czystego przelaczenia na fallback. sample() teraz defensywnie odrzuca kazdy
pojedynczy odczyt poza fizycznie wiarygodnym zakresem, niezaleznie od przyczyny."""

from __future__ import annotations

import pytest

from benchmark_runner.power.codecarbon_meter import _MAX_PLAUSIBLE_WATTS, CodecarbonPowerSampler


class _FakeTracker:
    def __init__(self, kwh: float) -> None:
        self.total_energy_kwh = kwh
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1

    @property
    def _total_energy(self):
        class _Energy:
            def __init__(self, kwh: float) -> None:
                self.kWh = kwh

        return _Energy(self.total_energy_kwh)

    def stop(self) -> None:
        pass


def _make_sampler(monkeypatch, *, initial_kwh: float = 0.0, base_time: float = 100.0):
    sampler = CodecarbonPowerSampler()
    sampler._tracker = _FakeTracker(kwh=initial_kwh)
    sampler._last_energy_kwh = initial_kwh
    sampler._last_time = base_time
    return sampler


def _set_next_sample_time(monkeypatch, new_time: float) -> None:
    monkeypatch.setattr("benchmark_runner.power.codecarbon_meter.time.monotonic", lambda: new_time)


def _kwh_for_watts(watts: float, delta_s: float) -> float:
    return watts * delta_s / 3_600_000


def test_sample_rejects_implausible_wattage_and_returns_none(monkeypatch):
    sampler = _make_sampler(monkeypatch, initial_kwh=0.0, base_time=100.0)
    _set_next_sample_time(monkeypatch, 101.0)  # delta_s=1.0
    sampler._tracker.total_energy_kwh = _kwh_for_watts(500.0, delta_s=1.0)  # symuluje "niestabilny" skok

    result = sampler.sample()

    assert result is None


def test_sample_accepts_plausible_wattage(monkeypatch):
    sampler = _make_sampler(monkeypatch, initial_kwh=0.0, base_time=100.0)
    _set_next_sample_time(monkeypatch, 101.0)
    sampler._tracker.total_energy_kwh = _kwh_for_watts(45.0, delta_s=1.0)

    result = sampler.sample()

    assert result is not None
    assert result.watts == pytest.approx(45.0)
    assert result.source == "codecarbon"


def test_sample_boundary_exactly_at_threshold_is_accepted(monkeypatch):
    sampler = _make_sampler(monkeypatch, initial_kwh=0.0, base_time=100.0)
    _set_next_sample_time(monkeypatch, 101.0)
    sampler._tracker.total_energy_kwh = _kwh_for_watts(_MAX_PLAUSIBLE_WATTS, delta_s=1.0)

    result = sampler.sample()

    assert result is not None
    assert result.watts == pytest.approx(_MAX_PLAUSIBLE_WATTS)


def test_sample_updates_internal_state_even_when_rejecting(monkeypatch):
    # Stan MUSI sie zaktualizowac mimo odrzucenia probki - inaczej kolejny odczyt liczylby
    # delte od tej samej (stalej) bazy, co przy utrzymujacym sie niestabilnym stanie
    # codecarbon tylko powiekszaloby kolejna implauzybilna wartosc zamiast pozwolic
    # sekwencji "dogonic" rzeczywisty stan.
    sampler = _make_sampler(monkeypatch, initial_kwh=0.0, base_time=100.0)
    _set_next_sample_time(monkeypatch, 101.0)
    bad_kwh = _kwh_for_watts(1050.0, delta_s=1.0)
    sampler._tracker.total_energy_kwh = bad_kwh

    result = sampler.sample()

    assert result is None
    assert sampler._last_energy_kwh == pytest.approx(bad_kwh)
    assert sampler._last_time == pytest.approx(101.0)


def test_sample_returns_none_when_tracker_not_initialized():
    sampler = CodecarbonPowerSampler()
    assert sampler.sample() is None
