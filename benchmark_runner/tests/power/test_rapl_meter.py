"""Testy RaplPowerSampler - w szczegolnosci PermissionError przy odczycie RAPL (zgloszony
blad: /sys/class/powercap/intel-rapl/.../energy_uj zablokowany uprawnieniami) musi
skutkowac CZYSTYM fallbackiem (sample() zwraca None), nigdy implauzybilna wartoscia, oraz
odrzucanie fizycznie nieprawdopodobnych odczytow (np. po zawinieciu licznika RAPL)."""

from __future__ import annotations

import sys
import types

import pytest

from benchmark_runner.power.rapl_meter import _MAX_PLAUSIBLE_WATTS, RaplPowerSampler


class _FakeResult:
    def __init__(self, pkg, duration_us: int) -> None:
        self.pkg = pkg
        self.duration = duration_us


class _RaisingMeasurement:
    """Symuluje pyRAPL.Measurement, ktorego begin() rzuca wyjatek (np. PermissionError)."""

    def __init__(self, name: str, exc: Exception) -> None:
        self._exc = exc

    def begin(self):
        raise self._exc

    def end(self):
        pass


class _FakeMeasurement:
    def __init__(self, pkg_values, duration_us: int = 1_000_000) -> None:
        self._pkg_values = pkg_values
        self._duration_us = duration_us

    def begin(self):
        pass

    def end(self):
        pass

    @property
    def result(self):
        return _FakeResult(self._pkg_values, self._duration_us)


def _install_fake_pyrapl(monkeypatch, measurement_factory) -> None:
    fake_module = types.ModuleType("pyRAPL")
    fake_module.Measurement = measurement_factory
    fake_module.setup = lambda: None
    monkeypatch.setitem(sys.modules, "pyRAPL", fake_module)


def _uj_for_watts(watts: float, duration_s: float) -> int:
    return int(watts * duration_s * 1_000_000)


def test_sample_returns_none_on_permission_error(monkeypatch):
    # Dokladnie zgloszony scenariusz: RAPL zablokowany uprawnieniami.
    exc = PermissionError("[Errno 13] Permission denied: '/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj'")
    _install_fake_pyrapl(monkeypatch, lambda name: _RaisingMeasurement(name, exc))

    sampler = RaplPowerSampler()
    sampler._available = True

    result = sampler.sample()

    assert result is None


def test_sample_rejects_implausible_wattage(monkeypatch):
    # Symuluje np. zawiniecie licznika RAPL (energy_uj) - pojedynczy odczyt odpowiadajacy
    # 500W, fizycznie niemozliwy dla CPU laptopa.
    _install_fake_pyrapl(monkeypatch, lambda name: _FakeMeasurement(pkg_values=[_uj_for_watts(500.0, 1.0)]))

    sampler = RaplPowerSampler()
    sampler._available = True

    result = sampler.sample()

    assert result is None


def test_sample_accepts_plausible_wattage(monkeypatch):
    _install_fake_pyrapl(monkeypatch, lambda name: _FakeMeasurement(pkg_values=[_uj_for_watts(35.0, 1.0)]))

    sampler = RaplPowerSampler()
    sampler._available = True

    result = sampler.sample()

    assert result is not None
    assert result.watts == pytest.approx(35.0)
    assert result.source == "rapl"


def test_sample_boundary_exactly_at_threshold_is_accepted(monkeypatch):
    _install_fake_pyrapl(monkeypatch, lambda name: _FakeMeasurement(pkg_values=[_uj_for_watts(_MAX_PLAUSIBLE_WATTS, 1.0)]))

    sampler = RaplPowerSampler()
    sampler._available = True

    result = sampler.sample()

    assert result is not None
    assert result.watts == pytest.approx(_MAX_PLAUSIBLE_WATTS)


def test_sample_returns_none_when_unavailable():
    sampler = RaplPowerSampler()
    sampler._available = False
    assert sampler.sample() is None
