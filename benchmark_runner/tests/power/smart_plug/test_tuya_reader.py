from __future__ import annotations

import pytest

from benchmark_runner.power.smart_plug.tuya_reader import (
    TuyaConnection,
    TuyaDeviceConfig,
    TuyaPowerSampler,
    TuyaSmartPlugReader,
    load_tuya_config_from_env,
)

_ENV_KEYS = (
    "TUYA_DEVICE_ID",
    "TUYA_LOCAL_KEY",
    "TUYA_IP_ADDRESS",
    "TUYA_VERSION",
    "TUYA_DPS_POWER",
    "TUYA_DPS_ENERGY",
    "TUYA_POWER_SCALE",
    "TUYA_ENERGY_SCALE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


class _FakeOutletDevice:
    def __init__(self, dev_id, address, local_key, version, dps=None, raise_on_status=False):
        self.dev_id = dev_id
        self.address = address
        self.local_key = local_key
        self.version = version
        self._dps = dps if dps is not None else {"19": 150, "17": 500}
        self._raise_on_status = raise_on_status

    def status(self):
        if self._raise_on_status:
            raise ConnectionError("simulated network failure")
        return {"dps": dict(self._dps)}


def _make_fake_module(dps=None, raise_on_status=False):
    """Zwraca obiekt imitujacy modul tinytuya z klasa OutletDevice zwracajaca
    ustalone DPS - do wstrzykniecia przez monkeypatch."""

    class _Module:
        @staticmethod
        def OutletDevice(dev_id, address, local_key, version):
            return _FakeOutletDevice(dev_id, address, local_key, version, dps=dps, raise_on_status=raise_on_status)

    return _Module()


# ---------------------------------------------------------------------------
# load_tuya_config_from_env
# ---------------------------------------------------------------------------


def test_load_tuya_config_returns_none_when_missing_vars():
    assert load_tuya_config_from_env() is None


def test_load_tuya_config_reads_required_and_default_fields(monkeypatch):
    monkeypatch.setenv("TUYA_DEVICE_ID", "dev123")
    monkeypatch.setenv("TUYA_LOCAL_KEY", "key123")
    monkeypatch.setenv("TUYA_IP_ADDRESS", "192.168.1.50")

    config = load_tuya_config_from_env()

    assert config is not None
    assert config.device_id == "dev123"
    assert config.local_key == "key123"
    assert config.ip_address == "192.168.1.50"
    assert config.version == 3.3
    assert config.dps_power == "19"
    assert config.dps_energy == "17"


def test_load_tuya_config_reads_overrides(monkeypatch):
    monkeypatch.setenv("TUYA_DEVICE_ID", "dev123")
    monkeypatch.setenv("TUYA_LOCAL_KEY", "key123")
    monkeypatch.setenv("TUYA_IP_ADDRESS", "192.168.1.50")
    monkeypatch.setenv("TUYA_DPS_POWER", "9")
    monkeypatch.setenv("TUYA_DPS_ENERGY", "6")
    monkeypatch.setenv("TUYA_POWER_SCALE", "1.0")
    monkeypatch.setenv("TUYA_ENERGY_SCALE", "0.001")

    config = load_tuya_config_from_env()

    assert config.dps_power == "9"
    assert config.dps_energy == "6"
    assert config.power_scale == 1.0
    assert config.energy_scale == 0.001


@pytest.mark.parametrize("sentinel", ["none", "NONE", "off", "disabled", "brak", ""])
def test_load_tuya_config_disables_energy_dps_on_sentinel(monkeypatch, sentinel):
    monkeypatch.setenv("TUYA_DEVICE_ID", "dev123")
    monkeypatch.setenv("TUYA_LOCAL_KEY", "key123")
    monkeypatch.setenv("TUYA_IP_ADDRESS", "192.168.1.50")
    monkeypatch.setenv("TUYA_DPS_ENERGY", sentinel)

    config = load_tuya_config_from_env()

    assert config is not None
    assert config.dps_energy == ""


def test_load_tuya_config_returns_none_on_invalid_number(monkeypatch):
    monkeypatch.setenv("TUYA_DEVICE_ID", "dev123")
    monkeypatch.setenv("TUYA_LOCAL_KEY", "key123")
    monkeypatch.setenv("TUYA_IP_ADDRESS", "192.168.1.50")
    monkeypatch.setenv("TUYA_VERSION", "not-a-number")

    assert load_tuya_config_from_env() is None


# ---------------------------------------------------------------------------
# TuyaConnection
# ---------------------------------------------------------------------------

_CONFIG = TuyaDeviceConfig(device_id="dev123", local_key="key123", ip_address="192.168.1.50")


def test_connection_connect_success(monkeypatch):
    import tinytuya

    monkeypatch.setattr(tinytuya, "OutletDevice", _make_fake_module().OutletDevice)
    connection = TuyaConnection(_CONFIG)

    assert connection.connect() is True
    assert connection.is_connected() is True


def test_connection_connect_failure_on_exception(monkeypatch):
    import tinytuya

    monkeypatch.setattr(tinytuya, "OutletDevice", _make_fake_module(raise_on_status=True).OutletDevice)
    connection = TuyaConnection(_CONFIG)

    assert connection.connect() is False
    assert connection.is_connected() is False


def test_connection_reads_power_and_energy_with_scale(monkeypatch):
    import tinytuya

    monkeypatch.setattr(tinytuya, "OutletDevice", _make_fake_module(dps={"19": 150, "17": 500}).OutletDevice)
    connection = TuyaConnection(_CONFIG)
    connection.connect()

    assert connection.read_power_watts() == pytest.approx(15.0)
    assert connection.read_energy_kwh() == pytest.approx(5.0)


def test_connection_read_returns_none_when_dp_missing(monkeypatch):
    import tinytuya

    monkeypatch.setattr(tinytuya, "OutletDevice", _make_fake_module(dps={"1": True}).OutletDevice)
    connection = TuyaConnection(_CONFIG)
    connection.connect()

    assert connection.read_power_watts() is None
    assert connection.read_energy_kwh() is None


def test_connection_read_returns_none_when_not_connected():
    connection = TuyaConnection(_CONFIG)
    assert connection.read_power_watts() is None
    assert connection.read_energy_kwh() is None


def test_connection_skips_energy_read_without_warning_when_dps_disabled(monkeypatch, caplog):
    import tinytuya

    monkeypatch.setattr(tinytuya, "OutletDevice", _make_fake_module(dps={"19": 150}).OutletDevice)
    config = TuyaDeviceConfig(device_id="dev123", local_key="key123", ip_address="192.168.1.50", dps_energy="")
    connection = TuyaConnection(config)
    connection.connect()

    with caplog.at_level("WARNING"):
        assert connection.read_energy_kwh() is None

    assert connection.read_power_watts() == pytest.approx(15.0)
    assert "nie wystapil" not in caplog.text


# ---------------------------------------------------------------------------
# TuyaPowerSampler
# ---------------------------------------------------------------------------


def test_power_sampler_is_available_and_samples(monkeypatch):
    import tinytuya

    monkeypatch.setattr(tinytuya, "OutletDevice", _make_fake_module(dps={"19": 220, "17": 100}).OutletDevice)
    sampler = TuyaPowerSampler(TuyaConnection(_CONFIG))

    assert sampler.is_available() is True
    sample = sampler.sample()
    assert sample is not None
    assert sample.source == "smart_plug"
    assert sample.watts == pytest.approx(22.0)


def test_power_sampler_unavailable_when_connection_fails(monkeypatch):
    import tinytuya

    monkeypatch.setattr(tinytuya, "OutletDevice", _make_fake_module(raise_on_status=True).OutletDevice)
    sampler = TuyaPowerSampler(TuyaConnection(_CONFIG))

    assert sampler.is_available() is False
    assert sampler.sample() is None


# ---------------------------------------------------------------------------
# TuyaSmartPlugReader
# ---------------------------------------------------------------------------


def _device_with_mutable_energy(energy_holder: dict[str, int]):
    """Fake device ktore zawsze zwraca AKTUALNA wartosc z energy_holder['value'] -
    niezalezne od tego, ile razy status() jest wewnetrznie wywolywane (np. connect()
    tez robi jedno wywolanie walidacyjne), w odroznieniu od podejscia z lista/indeksem."""

    class _Device:
        def __init__(self, *a, **k):
            pass

        def status(self):
            return {"dps": {"19": 100, "17": energy_holder["value"]}}

    return _Device


def test_smart_plug_reader_computes_energy_delta(monkeypatch):
    import tinytuya

    energy_holder = {"value": 500}  # 5.00 kWh (scale 0.01)
    monkeypatch.setattr(tinytuya, "OutletDevice", lambda **kwargs: _device_with_mutable_energy(energy_holder)())

    connection = TuyaConnection(_CONFIG)
    reader = TuyaSmartPlugReader(connection, enabled=True)

    reader.start_run("run-1")
    energy_holder["value"] = 700  # 7.00 kWh
    reading = reader.end_run("run-1")

    assert reading is not None
    assert reading.energy_wh == pytest.approx(2000.0)  # (7.00 - 5.00) kWh -> 2000 Wh


def test_smart_plug_reader_rejects_negative_delta(monkeypatch):
    import tinytuya

    energy_holder = {"value": 700}
    monkeypatch.setattr(tinytuya, "OutletDevice", lambda **kwargs: _device_with_mutable_energy(energy_holder)())

    connection = TuyaConnection(_CONFIG)
    reader = TuyaSmartPlugReader(connection, enabled=True)

    reader.start_run("run-1")
    energy_holder["value"] = 500  # spadek -> mozliwy reset licznika
    reading = reader.end_run("run-1")

    assert reading is None


def test_smart_plug_reader_disabled_returns_none(monkeypatch):
    import tinytuya

    monkeypatch.setattr(tinytuya, "OutletDevice", _make_fake_module().OutletDevice)
    connection = TuyaConnection(_CONFIG)
    reader = TuyaSmartPlugReader(connection, enabled=False)

    reader.start_run("run-1")
    assert reader.end_run("run-1") is None


def test_smart_plug_reader_returns_none_when_connection_fails(monkeypatch):
    import tinytuya

    monkeypatch.setattr(tinytuya, "OutletDevice", _make_fake_module(raise_on_status=True).OutletDevice)
    connection = TuyaConnection(_CONFIG)
    reader = TuyaSmartPlugReader(connection, enabled=True)

    reader.start_run("run-1")
    assert reader.end_run("run-1") is None
