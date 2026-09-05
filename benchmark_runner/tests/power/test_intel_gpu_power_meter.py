from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark_runner.power import whole_system_meter as wsm
from benchmark_runner.power.intel_gpu_power_meter import IntelGpuPowerSampler


def _fake_state(ac_online, battery_present, discharging, rate_mw):
    return SimpleNamespace(
        AcOnLine=ac_online,
        BatteryPresent=battery_present,
        Charging=0 if discharging else 1,
        Discharging=discharging,
        Rate=rate_mw,
    )


def test_intel_gpu_power_sampler_source_name_is_intel_gpu_native():
    assert IntelGpuPowerSampler.source_name == "intel_gpu_native"


def test_intel_gpu_power_sampler_delegates_to_whole_system_fallback_windows(monkeypatch):
    monkeypatch.setattr(wsm, "_is_windows", lambda: True)
    monkeypatch.setattr(wsm, "_query_battery_state", lambda: _fake_state(0, 1, 1, 22000))
    sampler = IntelGpuPowerSampler()

    assert sampler.is_available() is True
    sample = sampler.sample()
    assert sample is not None
    assert sample.watts == pytest.approx(22.0)
    assert sample.source == "intel_gpu_native"


def test_intel_gpu_power_sampler_delegates_to_whole_system_fallback_linux(monkeypatch, tmp_path):
    battery_dir = tmp_path / "BAT0"
    battery_dir.mkdir()
    (battery_dir / "status").write_text("Discharging")
    (battery_dir / "power_now").write_text("21500000")

    monkeypatch.setattr(wsm, "_is_windows", lambda: False)
    monkeypatch.setattr(wsm, "_find_linux_battery_path", lambda: str(battery_dir))

    sampler = IntelGpuPowerSampler()
    assert sampler.is_available() is True
    sample = sampler.sample()
    assert sample is not None
    assert sample.watts == pytest.approx(21.5)
    assert sample.source == "intel_gpu_native"


def test_intel_gpu_power_sampler_unavailable_when_no_battery(monkeypatch):
    monkeypatch.setattr(wsm, "_is_windows", lambda: True)
    monkeypatch.setattr(wsm, "_query_battery_state", lambda: _fake_state(0, 0, 0, 0))
    sampler = IntelGpuPowerSampler()
    assert sampler.is_available() is False
