"""Testy AmdIgpuPowerSampler/AmdIgpuWholeSystemPowerSampler. AmdIgpuPowerSampler
(hwmon sterownika amdgpu) zweryfikowany bezposrednio na maszynie testowej "Ola" (Ryzen 7
5800HS, iGPU Vega/Cezanne) - power1_input=15000000 (15.0W), power1_label='PPT' (Package
Power Tracking - pobor calego SoC, nie samego bloku graficznego, stad measurement_scope=
'whole_system' w orchestrator.py niezaleznie od tego, ktore z dwoch zrodel dostarczylo dane)."""

from __future__ import annotations

import pytest

from benchmark_runner.power import amdgpu_igpu_power_meter as igpu_meter
from benchmark_runner.power.amdgpu_igpu_power_meter import (
    _MAX_PLAUSIBLE_WATTS,
    AmdIgpuPowerSampler,
    AmdIgpuWholeSystemPowerSampler,
)


def _make_fake_hwmon(tmp_path, *, name: str = "amdgpu", attr: str | None = "power1_average", microwatts: int | None = None):
    hwmon_dir = tmp_path / "hwmon4"
    hwmon_dir.mkdir()
    (hwmon_dir / "name").write_text(name)
    if attr is not None and microwatts is not None:
        (hwmon_dir / attr).write_text(str(microwatts))
    return hwmon_dir


def test_is_available_false_when_no_amdgpu_hwmon(monkeypatch):
    monkeypatch.setattr(igpu_meter, "find_amdgpu_hwmon_paths", lambda: [])
    sampler = AmdIgpuPowerSampler()

    assert sampler.is_available() is False
    assert sampler.sample() is None


def test_is_available_false_when_no_power_attribute(monkeypatch, tmp_path):
    hwmon_dir = _make_fake_hwmon(tmp_path, attr=None)
    monkeypatch.setattr(igpu_meter, "find_amdgpu_hwmon_paths", lambda: [str(hwmon_dir)])
    sampler = AmdIgpuPowerSampler()

    assert sampler.is_available() is False


def test_sample_reads_plausible_wattage(monkeypatch, tmp_path):
    # Wartosc bezposrednio zaobserwowana na maszynie testowej "Ola" (15.0W).
    hwmon_dir = _make_fake_hwmon(tmp_path, microwatts=15_000_000)
    monkeypatch.setattr(igpu_meter, "find_amdgpu_hwmon_paths", lambda: [str(hwmon_dir)])
    sampler = AmdIgpuPowerSampler()

    assert sampler.is_available() is True
    sample = sampler.sample()
    assert sample is not None
    assert sample.watts == pytest.approx(15.0)
    assert sample.source == "amdgpu_igpu_native"


def test_sample_rejects_implausible_wattage(monkeypatch, tmp_path):
    hwmon_dir = _make_fake_hwmon(tmp_path, microwatts=1_050_000_000)  # 1050W
    monkeypatch.setattr(igpu_meter, "find_amdgpu_hwmon_paths", lambda: [str(hwmon_dir)])
    sampler = AmdIgpuPowerSampler()

    assert sampler.is_available() is True
    assert sampler.sample() is None


def test_sample_boundary_exactly_at_threshold_is_accepted(monkeypatch, tmp_path):
    microwatts = int(_MAX_PLAUSIBLE_WATTS * 1_000_000)
    hwmon_dir = _make_fake_hwmon(tmp_path, microwatts=microwatts)
    monkeypatch.setattr(igpu_meter, "find_amdgpu_hwmon_paths", lambda: [str(hwmon_dir)])
    sampler = AmdIgpuPowerSampler()

    assert sampler.is_available() is True
    sample = sampler.sample()
    assert sample.watts == pytest.approx(_MAX_PLAUSIBLE_WATTS)


def test_falls_back_to_power1_input_when_power1_average_missing(monkeypatch, tmp_path):
    hwmon_dir = _make_fake_hwmon(tmp_path, attr="power1_input", microwatts=24_000_000)
    monkeypatch.setattr(igpu_meter, "find_amdgpu_hwmon_paths", lambda: [str(hwmon_dir)])
    sampler = AmdIgpuPowerSampler()

    assert sampler.is_available() is True
    sample = sampler.sample()
    assert sample.watts == pytest.approx(24.0)


def test_multiple_amdgpu_hwmon_entries_picks_first_and_warns(monkeypatch, tmp_path, caplog):
    import logging

    hwmon_a = tmp_path / "hwmon_a"
    hwmon_a.mkdir()
    (hwmon_a / "name").write_text("amdgpu")
    (hwmon_a / "power1_average").write_text("10000000")

    hwmon_b = tmp_path / "hwmon_b"
    hwmon_b.mkdir()
    (hwmon_b / "name").write_text("amdgpu")
    (hwmon_b / "power1_average").write_text("99000000")

    monkeypatch.setattr(igpu_meter, "find_amdgpu_hwmon_paths", lambda: [str(hwmon_a), str(hwmon_b)])
    sampler = AmdIgpuPowerSampler()

    with caplog.at_level(logging.WARNING, logger="benchmark_runner.power.amdgpu_igpu_power_meter"):
        available = sampler.is_available()

    assert available is True
    assert sampler.sample().watts == pytest.approx(10.0)  # pierwszy z listy
    assert any("wiecej niz jeden hwmon" in r.message for r in caplog.records)


def test_real_sysfs_discovery_on_this_machine():
    # Bez mockowania - jesli ta maszyna ma amdgpu (np. "Ola"), sprawdzamy ze prawdziwe
    # przeszukanie sysfs faktycznie znajduje wpis i daje wiarygodna wartosc; jesli nie ma
    # (typowa maszyna Intel/NVIDIA-only, w tym wiekszosc CI), test jest pomijany.
    paths = igpu_meter.find_amdgpu_hwmon_paths()
    if not paths:
        pytest.skip("brak hwmon 'amdgpu' na tej maszynie - test dotyczy maszyn z GPU AMD")

    sampler = AmdIgpuPowerSampler()
    assert sampler.is_available() is True
    sample = sampler.sample()
    assert sample is not None
    assert 0 <= sample.watts <= _MAX_PLAUSIBLE_WATTS


class TestAmdIgpuWholeSystemPowerSampler:
    def test_source_name(self):
        assert AmdIgpuWholeSystemPowerSampler.source_name == "amd_igpu_whole_system"

    def test_delegates_to_whole_system_fallback(self, monkeypatch):
        from benchmark_runner.power import whole_system_meter as wsm

        battery_dir = None

        def _fake_find():
            return battery_dir

        # Reuzywa te sama logike co test_whole_system_meter.py - potwierdza jedynie ze
        # AmdIgpuWholeSystemPowerSampler poprawnie deleguje do WholeSystemPowerSampler
        # pod wlasciwym source_name, nie duplikuje testow samego mechanizmu baterii.
        monkeypatch.setattr(wsm, "_is_windows", lambda: False)
        monkeypatch.setattr(wsm, "_find_linux_battery_path", _fake_find)

        sampler = AmdIgpuWholeSystemPowerSampler()
        assert sampler.is_available() is False  # brak baterii w tym fake

        sampler.close()  # nie powinno rzucic
