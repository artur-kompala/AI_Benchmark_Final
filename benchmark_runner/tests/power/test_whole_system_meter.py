from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark_runner.power import whole_system_meter as wsm


def _fake_state(ac_online, battery_present, discharging, rate_mw):
    return SimpleNamespace(
        AcOnLine=ac_online,
        BatteryPresent=battery_present,
        Charging=0 if discharging else 1,
        Discharging=discharging,
        Rate=rate_mw,
    )


# ============================================================
# Windows (CallNtPowerInformation) - _is_windows() jawnie wymuszone na True, zeby
# testy byly deterministyczne niezaleznie od OS, na ktorym faktycznie dziala pytest
# (patrz rownolegle testy *_linux ponizej, ktore wymuszaja False).
# ============================================================


def test_is_available_true_when_battery_present(monkeypatch):
    monkeypatch.setattr(wsm, "_is_windows", lambda: True)
    monkeypatch.setattr(wsm, "_query_battery_state", lambda: _fake_state(0, 1, 1, 15000))
    sampler = wsm.WholeSystemPowerSampler(source_name="npu_native")
    assert sampler.is_available() is True


def test_is_available_false_when_no_battery(monkeypatch):
    monkeypatch.setattr(wsm, "_is_windows", lambda: True)
    monkeypatch.setattr(wsm, "_query_battery_state", lambda: _fake_state(0, 0, 0, 0))
    sampler = wsm.WholeSystemPowerSampler(source_name="npu_native")
    assert sampler.is_available() is False


def test_is_available_false_when_api_unavailable(monkeypatch):
    monkeypatch.setattr(wsm, "_is_windows", lambda: True)
    monkeypatch.setattr(wsm, "_query_battery_state", lambda: None)
    sampler = wsm.WholeSystemPowerSampler(source_name="npu_native")
    assert sampler.is_available() is False


def test_sample_returns_watts_when_discharging_on_battery(monkeypatch):
    monkeypatch.setattr(wsm, "_is_windows", lambda: True)
    monkeypatch.setattr(wsm, "_query_battery_state", lambda: _fake_state(0, 1, 1, 15000))
    sampler = wsm.WholeSystemPowerSampler(source_name="npu_native")
    sample = sampler.sample()

    assert sample is not None
    assert sample.watts == pytest.approx(15.0)
    assert sample.source == "npu_native"


def test_sample_returns_none_when_on_ac_power(monkeypatch):
    monkeypatch.setattr(wsm, "_is_windows", lambda: True)
    monkeypatch.setattr(wsm, "_query_battery_state", lambda: _fake_state(1, 1, 0, 0))
    sampler = wsm.WholeSystemPowerSampler(source_name="npu_native")
    assert sampler.sample() is None


def test_sample_returns_none_when_not_discharging(monkeypatch):
    monkeypatch.setattr(wsm, "_is_windows", lambda: True)
    monkeypatch.setattr(wsm, "_query_battery_state", lambda: _fake_state(0, 1, 0, 0))
    sampler = wsm.WholeSystemPowerSampler(source_name="npu_native")
    assert sampler.sample() is None


def test_sample_returns_none_when_api_unavailable(monkeypatch):
    monkeypatch.setattr(wsm, "_is_windows", lambda: True)
    monkeypatch.setattr(wsm, "_query_battery_state", lambda: None)
    sampler = wsm.WholeSystemPowerSampler(source_name="npu_native")
    assert sampler.sample() is None


# ============================================================
# Linux (/sys/class/power_supply/BAT*) - Bug: przed ta zmiana WholeSystemPowerSampler
# probowal WYLACZNIE Windows API (ctypes.windll), wiec na Linuksie is_available()/
# sample() zawsze zwracaly False/None, niezaleznie od stanu baterii - zaden przebieg
# NPU/Intel GPU na Linuksie nie mial wiarygodnego pomiaru energii (patrz
# power/intel_gpu_power_meter.py, power/npu_power_meter.py).
# ============================================================


def _write_battery_attr(battery_dir, name: str, value: str) -> None:
    (battery_dir / name).write_text(value)


def _make_linux_battery(tmp_path, *, status="Discharging", power_now=None, voltage_now=None, current_now=None):
    battery_dir = tmp_path / "BAT0"
    battery_dir.mkdir()
    _write_battery_attr(battery_dir, "status", status)
    if power_now is not None:
        _write_battery_attr(battery_dir, "power_now", str(power_now))
    if voltage_now is not None:
        _write_battery_attr(battery_dir, "voltage_now", str(voltage_now))
    if current_now is not None:
        _write_battery_attr(battery_dir, "current_now", str(current_now))
    return battery_dir


def test_linux_is_available_true_with_power_now(monkeypatch, tmp_path):
    battery_dir = _make_linux_battery(tmp_path, power_now=15_000_000)
    monkeypatch.setattr(wsm, "_is_windows", lambda: False)
    monkeypatch.setattr(wsm, "_find_linux_battery_path", lambda: str(battery_dir))

    sampler = wsm.WholeSystemPowerSampler(source_name="intel_gpu_native")
    assert sampler.is_available() is True


def test_linux_is_available_true_with_voltage_and_current_fallback(monkeypatch, tmp_path):
    # Sterownik bez power_now (np. ten uzyty do diagnozy tego zgloszenia) - musi zadzialac
    # przez voltage_now * current_now.
    battery_dir = _make_linux_battery(tmp_path, voltage_now=14_725_000, current_now=1_460_000)
    monkeypatch.setattr(wsm, "_is_windows", lambda: False)
    monkeypatch.setattr(wsm, "_find_linux_battery_path", lambda: str(battery_dir))

    sampler = wsm.WholeSystemPowerSampler(source_name="intel_gpu_native")
    assert sampler.is_available() is True


def test_linux_is_available_false_when_no_battery_dir(monkeypatch):
    monkeypatch.setattr(wsm, "_is_windows", lambda: False)
    monkeypatch.setattr(wsm, "_find_linux_battery_path", lambda: None)

    sampler = wsm.WholeSystemPowerSampler(source_name="intel_gpu_native")
    assert sampler.is_available() is False


def test_linux_is_available_false_when_driver_exposes_neither_attribute(monkeypatch, tmp_path):
    battery_dir = _make_linux_battery(tmp_path)  # ani power_now, ani voltage/current_now
    monkeypatch.setattr(wsm, "_is_windows", lambda: False)
    monkeypatch.setattr(wsm, "_find_linux_battery_path", lambda: str(battery_dir))

    sampler = wsm.WholeSystemPowerSampler(source_name="intel_gpu_native")
    assert sampler.is_available() is False


def test_linux_sample_returns_watts_from_power_now_when_discharging(monkeypatch, tmp_path):
    battery_dir = _make_linux_battery(tmp_path, status="Discharging", power_now=21_500_000)
    monkeypatch.setattr(wsm, "_is_windows", lambda: False)
    monkeypatch.setattr(wsm, "_find_linux_battery_path", lambda: str(battery_dir))

    sampler = wsm.WholeSystemPowerSampler(source_name="intel_gpu_native")
    sample = sampler.sample()

    assert sample is not None
    assert sample.watts == pytest.approx(21.5)
    assert sample.source == "intel_gpu_native"


def test_linux_sample_computes_watts_from_voltage_and_current_when_power_now_missing(monkeypatch, tmp_path):
    # Wartosci wziete z realnego odczytu sysfs na maszynie testowej z tego zgloszenia:
    # 14.725 V * 1.46 A = 21.4985 W.
    battery_dir = _make_linux_battery(tmp_path, status="Discharging", voltage_now=14_725_000, current_now=1_460_000)
    monkeypatch.setattr(wsm, "_is_windows", lambda: False)
    monkeypatch.setattr(wsm, "_find_linux_battery_path", lambda: str(battery_dir))

    sampler = wsm.WholeSystemPowerSampler(source_name="intel_gpu_native")
    sample = sampler.sample()

    assert sample is not None
    assert sample.watts == pytest.approx(21.4985)


def test_linux_sample_handles_negative_current_now(monkeypatch, tmp_path):
    # Niektore sterowniki raportuja current_now jako ujemne przy rozladowaniu.
    battery_dir = _make_linux_battery(tmp_path, status="Discharging", voltage_now=14_725_000, current_now=-1_460_000)
    monkeypatch.setattr(wsm, "_is_windows", lambda: False)
    monkeypatch.setattr(wsm, "_find_linux_battery_path", lambda: str(battery_dir))

    sampler = wsm.WholeSystemPowerSampler(source_name="intel_gpu_native")
    sample = sampler.sample()

    assert sample is not None
    assert sample.watts == pytest.approx(21.4985)


def test_linux_sample_returns_none_when_on_ac_power(monkeypatch, tmp_path):
    battery_dir = _make_linux_battery(tmp_path, status="Charging", power_now=15_000_000)
    monkeypatch.setattr(wsm, "_is_windows", lambda: False)
    monkeypatch.setattr(wsm, "_find_linux_battery_path", lambda: str(battery_dir))

    sampler = wsm.WholeSystemPowerSampler(source_name="intel_gpu_native")
    assert sampler.sample() is None


def test_linux_sample_returns_none_when_no_battery_dir(monkeypatch):
    monkeypatch.setattr(wsm, "_is_windows", lambda: False)
    monkeypatch.setattr(wsm, "_find_linux_battery_path", lambda: None)

    sampler = wsm.WholeSystemPowerSampler(source_name="intel_gpu_native")
    assert sampler.sample() is None
