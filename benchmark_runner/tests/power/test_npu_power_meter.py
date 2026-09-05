from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark_runner.power import whole_system_meter as wsm
from benchmark_runner.power.npu_power_meter import NpuPowerSampler


def _fake_state(ac_online, battery_present, discharging, rate_mw):
    return SimpleNamespace(
        AcOnLine=ac_online,
        BatteryPresent=battery_present,
        Charging=0 if discharging else 1,
        Discharging=discharging,
        Rate=rate_mw,
    )


def test_npu_power_sampler_source_name_is_npu_native():
    assert NpuPowerSampler.source_name == "npu_native"


def test_npu_power_sampler_delegates_to_whole_system_fallback(monkeypatch):
    monkeypatch.setattr(wsm, "_is_windows", lambda: True)
    monkeypatch.setattr(wsm, "_query_battery_state", lambda: _fake_state(0, 1, 1, 20000))
    sampler = NpuPowerSampler()

    assert sampler.is_available() is True
    sample = sampler.sample()
    assert sample is not None
    assert sample.watts == pytest.approx(20.0)
    assert sample.source == "npu_native"


def test_npu_power_sampler_unavailable_when_no_battery(monkeypatch):
    monkeypatch.setattr(wsm, "_is_windows", lambda: True)
    monkeypatch.setattr(wsm, "_query_battery_state", lambda: _fake_state(0, 0, 0, 0))
    sampler = NpuPowerSampler()
    assert sampler.is_available() is False
