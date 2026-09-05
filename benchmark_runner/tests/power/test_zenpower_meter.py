"""Testy ZenpowerPowerSampler - modul jadra zenpower3 (opcjonalny, instalowany recznie
przez DKMS - patrz docs/setup_zenpower.md). Testuje: czyste przejscie na niedostepnosc
gdy modul nie jest zaladowany (brak wpisu hwmon 'zenpower' w sysfs - typowy stan na
maszynie bez recznej instalacji), poprawny odczyt gdy jest, i odrzucanie fizycznie
nieprawdopodobnych wartosci (ta sama walidacja co RaplPowerSampler/CodecarbonPowerSampler,
patrz Blad: PermissionError na RAPL doprowadzajacy codecarbon do zapisu 300-1050W)."""

from __future__ import annotations

import pytest

from benchmark_runner.power import zenpower_meter
from benchmark_runner.power.zenpower_meter import _MAX_PLAUSIBLE_WATTS, ZenpowerPowerSampler


def _make_fake_hwmon(tmp_path, *, attr: str | None = "power1_input", microwatts: int | None = None):
    hwmon_dir = tmp_path / "hwmon0"
    hwmon_dir.mkdir()
    (hwmon_dir / "name").write_text("zenpower")
    if attr is not None and microwatts is not None:
        (hwmon_dir / attr).write_text(str(microwatts))
    return hwmon_dir


def test_is_available_false_when_module_not_loaded(monkeypatch):
    monkeypatch.setattr(zenpower_meter, "find_zenpower_hwmon_path", lambda: None)
    sampler = ZenpowerPowerSampler()

    assert sampler.is_available() is False
    assert sampler.sample() is None


def test_is_available_false_when_no_power_attribute(monkeypatch, tmp_path):
    hwmon_dir = _make_fake_hwmon(tmp_path, attr=None)  # tylko 'name', brak power1_input/average
    monkeypatch.setattr(zenpower_meter, "find_zenpower_hwmon_path", lambda: str(hwmon_dir))
    sampler = ZenpowerPowerSampler()

    assert sampler.is_available() is False


def test_sample_reads_plausible_wattage(monkeypatch, tmp_path):
    hwmon_dir = _make_fake_hwmon(tmp_path, microwatts=45_000_000)  # 45W
    monkeypatch.setattr(zenpower_meter, "find_zenpower_hwmon_path", lambda: str(hwmon_dir))
    sampler = ZenpowerPowerSampler()

    assert sampler.is_available() is True
    sample = sampler.sample()
    assert sample is not None
    assert sample.watts == pytest.approx(45.0)
    assert sample.source == "zenpower"


def test_sample_rejects_implausible_wattage(monkeypatch, tmp_path):
    hwmon_dir = _make_fake_hwmon(tmp_path, microwatts=500_000_000)  # 500W - fizycznie niemozliwe
    monkeypatch.setattr(zenpower_meter, "find_zenpower_hwmon_path", lambda: str(hwmon_dir))
    sampler = ZenpowerPowerSampler()

    assert sampler.is_available() is True
    assert sampler.sample() is None


def test_sample_boundary_exactly_at_threshold_is_accepted(monkeypatch, tmp_path):
    microwatts = int(_MAX_PLAUSIBLE_WATTS * 1_000_000)
    hwmon_dir = _make_fake_hwmon(tmp_path, microwatts=microwatts)
    monkeypatch.setattr(zenpower_meter, "find_zenpower_hwmon_path", lambda: str(hwmon_dir))
    sampler = ZenpowerPowerSampler()

    assert sampler.is_available() is True
    sample = sampler.sample()
    assert sample is not None
    assert sample.watts == pytest.approx(_MAX_PLAUSIBLE_WATTS)


def test_falls_back_to_power1_average_when_power1_input_missing(monkeypatch, tmp_path):
    hwmon_dir = _make_fake_hwmon(tmp_path, attr="power1_average", microwatts=30_000_000)
    monkeypatch.setattr(zenpower_meter, "find_zenpower_hwmon_path", lambda: str(hwmon_dir))
    sampler = ZenpowerPowerSampler()

    assert sampler.is_available() is True
    sample = sampler.sample()
    assert sample.watts == pytest.approx(30.0)


def test_sample_returns_none_before_is_available_called():
    sampler = ZenpowerPowerSampler()
    assert sampler.sample() is None


def test_real_sysfs_discovery_is_clean_when_module_absent():
    # Bez mockowania - na wiekszosci maszyn (w tym typowe CI) modul zenpower3 nie jest
    # zainstalowany, wiec prawdziwe przeszukanie /sys/class/hwmon nie powinno rzucic
    # wyjatku ani znalezc wpisu 'zenpower'. Na maszynie z faktycznie zainstalowanym
    # modulem test jest pomijany (weryfikacja "dziala" jest wtedy manualna, patrz
    # docs/setup_zenpower.md).
    if zenpower_meter.find_zenpower_hwmon_path() is not None:
        pytest.skip("modul zenpower3 jest zainstalowany na tej maszynie - test dotyczy przypadku braku")
    sampler = ZenpowerPowerSampler()
    assert sampler.is_available() is False
    assert sampler.sample() is None
