"""Automatyczny backend watomierza dla wtyczek Tuya z pomiarem energii (np. ATORCH
S1-B/W/T/H) przez lokalny protokol Tuya (biblioteka `tinytuya`) - po jednorazowym
pozyskaniu local_key (patrz docs/setup_smart_plug.md) dziala calkowicie lokalnie,
bez chmury Tuya podczas samego pomiaru.

WAZNE: DPS (data points) roznia sie miedzy urzadzeniami, czasem nawet w tej samej
kategorii produktowej Tuya. Wartosci ponizej (DEFAULT_DPS_*) sa typowe dla wtyczek
z pomiarem energii, ale NIE sa gwarantowane dla kazdego modelu - zweryfikuj je na
swoim urzadzeniu przez `scripts/tuya_dps_scan.py` przed pierwszym pomiarem
referencyjnym uzywanym w pracy. Zle DPS dadza cicho bledne dane (zla jednostka/zle
pole), nie blad - to nalezy jawnie zweryfikowac, nie zakladac.

Dwie role, jedno wspolne polaczenie (tanie urzadzenia ESP-based czasem nie lubia
wielu rownoleglych polaczen lokalnych):
- TuyaPowerSampler: ciagle probkowanie mocy chwilowej (BasePowerSampler,
  source_name='smart_plug') - trafia do power_samples obok rapl/nvml/itd.
- TuyaSmartPlugReader: odczyt energii skumulowanej (wlasny licznik urzadzenia) na
  starcie i koncu przebiegu (SmartPlugReader) - to jest wartosc autorytatywna dla
  run_summary.energy_joules_smart_plug (bezposrednio z licznika sprzetowego, nie
  wyliczona z integracji probek).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from benchmark_runner.power.base_power_meter import BasePowerSampler, PowerSample
from benchmark_runner.power.smart_plug.base import SmartPlugReader, SmartPlugReading

logger = logging.getLogger(__name__)

# Typowe DPS dla wtyczek Tuya z pomiarem energii (kategoria "kg"/"cz" - inteligentne
# gniazdka z pomiarem) - ZWERYFIKUJ na swoim urzadzeniu (scripts/tuya_dps_scan.py).
DEFAULT_DPS_POWER = "19"  # moc chwilowa, typowa jednostka 0.1 W
DEFAULT_DPS_ENERGY = "17"  # energia skumulowana (add_ele), typowa jednostka 0.01 kWh
# TUYA_DPS_ENERGY ustawione na jedna z tych wartosci (albo puste) => urzadzenie nie
# ma DPS ze skumulowana energia (np. ATORCH S1BW) - odczyt licznika energii jest
# wtedy pomijany bez ostrzezenia, a runner liczy energie przez calkowanie probek
# mocy chwilowej (orchestrator.py). Patrz docs/setup_smart_plug.md sekcja 5.
_ENERGY_DISABLED_SENTINELS = {"", "none", "off", "disabled", "brak"}
DEFAULT_POWER_SCALE = 0.1  # surowa_wartosc * scale = waty
DEFAULT_ENERGY_SCALE = 0.01  # surowa_wartosc * scale = kWh


@dataclass(frozen=True)
class TuyaDeviceConfig:
    device_id: str
    local_key: str
    ip_address: str
    version: float = 3.3
    dps_power: str = DEFAULT_DPS_POWER
    dps_energy: str = DEFAULT_DPS_ENERGY
    power_scale: float = DEFAULT_POWER_SCALE
    energy_scale: float = DEFAULT_ENERGY_SCALE


def load_tuya_config_from_env() -> TuyaDeviceConfig | None:
    """Wczytuje dane logowania Tuya z juz zaladowanych zmiennych srodowiskowych
    (load_dotenv() jest wywolywane wczesniej w storage/supabase_client.py przy
    starcie runnera). Zwraca None (z logiem ostrzegawczym), gdy brakuje wymaganych
    pol - pozwala to na graceful fallback do trybu manualnego zamiast wyjatku."""
    device_id = os.environ.get("TUYA_DEVICE_ID", "").strip()
    local_key = os.environ.get("TUYA_LOCAL_KEY", "").strip()
    ip_address = os.environ.get("TUYA_IP_ADDRESS", "").strip()

    if not device_id or not local_key or not ip_address:
        logger.warning(
            "Backend watomierza 'tuya' wybrany w configu, ale TUYA_DEVICE_ID/"
            "TUYA_LOCAL_KEY/TUYA_IP_ADDRESS nie sa ustawione w .env - patrz "
            "docs/setup_smart_plug.md. Wracam do trybu manualnego."
        )
        return None

    try:
        version = float(os.environ.get("TUYA_VERSION", "3.3"))
        power_scale = float(os.environ.get("TUYA_POWER_SCALE", str(DEFAULT_POWER_SCALE)))
        energy_scale = float(os.environ.get("TUYA_ENERGY_SCALE", str(DEFAULT_ENERGY_SCALE)))
    except ValueError:
        logger.warning("Nieprawidlowa wartosc liczbowa w TUYA_VERSION/TUYA_POWER_SCALE/TUYA_ENERGY_SCALE w .env", exc_info=True)
        return None

    dps_energy = os.environ.get("TUYA_DPS_ENERGY", DEFAULT_DPS_ENERGY).strip()
    if dps_energy.lower() in _ENERGY_DISABLED_SENTINELS:
        dps_energy = ""  # urzadzenie bez licznika energii - odczyt bedzie pomijany

    return TuyaDeviceConfig(
        device_id=device_id,
        local_key=local_key,
        ip_address=ip_address,
        version=version,
        dps_power=os.environ.get("TUYA_DPS_POWER", DEFAULT_DPS_POWER).strip(),
        dps_energy=dps_energy,
        power_scale=power_scale,
        energy_scale=energy_scale,
    )


class TuyaConnection:
    """Wspolne polaczenie tinytuya, dzielone miedzy TuyaPowerSampler i
    TuyaSmartPlugReader dla tego samego przebiegu."""

    def __init__(self, config: TuyaDeviceConfig) -> None:
        self._config = config
        self._device = None

    def connect(self) -> bool:
        try:
            import tinytuya

            self._device = tinytuya.OutletDevice(
                dev_id=self._config.device_id,
                address=self._config.ip_address,
                local_key=self._config.local_key,
                version=self._config.version,
            )
            status = self._device.status()
            if not isinstance(status, dict) or "dps" not in status:
                logger.warning("Wtyczka Tuya zwrocila nieoczekiwana odpowiedz status(): %r", status)
                self._device = None
                return False
            return True
        except Exception:
            logger.warning(
                "Nie udalo sie polaczyc z wtyczka Tuya pod %s - sprawdz IP/local_key/"
                "wersje protokolu w .env (patrz docs/setup_smart_plug.md)",
                self._config.ip_address,
                exc_info=True,
            )
            self._device = None
            return False

    def is_connected(self) -> bool:
        return self._device is not None

    def read_raw_dps(self) -> dict[str, object] | None:
        """Pelny surowy slownik DPS z urzadzenia - do jednorazowego zidentyfikowania
        wlasciwych kodow na nowym/nieznanym urzadzeniu (patrz scripts/tuya_dps_scan.py),
        nie uzywane w normalnym przebiegu pomiarowym."""
        if self._device is None:
            return None
        try:
            status = self._device.status()
            return status.get("dps")
        except Exception:
            logger.warning("Blad odczytu surowego statusu DPS z wtyczki Tuya", exc_info=True)
            return None

    def _read_dp(self, dp: str) -> float | None:
        if self._device is None:
            return None
        try:
            status = self._device.status()
            raw = status.get("dps", {}).get(dp)
            if raw is None:
                logger.warning("DPS '%s' nie wystapil w odpowiedzi wtyczki Tuya - sprawdz TUYA_DPS_* w .env", dp)
                return None
            return float(raw)
        except Exception:
            logger.warning("Blad odczytu DPS '%s' z wtyczki Tuya", dp, exc_info=True)
            return None

    def read_power_watts(self) -> float | None:
        raw = self._read_dp(self._config.dps_power)
        return raw * self._config.power_scale if raw is not None else None

    def read_energy_kwh(self) -> float | None:
        if not self._config.dps_energy:
            # Urzadzenie bez DPS ze skumulowana energia (TUYA_DPS_ENERGY puste/none) -
            # pomijamy odczyt zamiast logowac warning przy kazdym przebiegu.
            return None
        raw = self._read_dp(self._config.dps_energy)
        return raw * self._config.energy_scale if raw is not None else None


class TuyaPowerSampler(BasePowerSampler):
    source_name = "smart_plug"

    def __init__(self, connection: TuyaConnection) -> None:
        self._connection = connection

    def is_available(self) -> bool:
        return self._connection.is_connected() or self._connection.connect()

    def sample(self) -> PowerSample | None:
        watts = self._connection.read_power_watts()
        if watts is None:
            return None
        return PowerSample(timestamp=datetime.now(timezone.utc), source=self.source_name, watts=watts)


class TuyaSmartPlugReader(SmartPlugReader):
    def __init__(self, connection: TuyaConnection, enabled: bool = True) -> None:
        self._connection = connection
        self._enabled = enabled
        self._start_energy_kwh: float | None = None

    def start_run(self, run_id: str) -> None:
        if not self._enabled:
            return
        if not self._connection.is_connected() and not self._connection.connect():
            logger.warning("Watomierz Tuya niedostepny na starcie runu %s - pomijam", run_id)
            return
        self._start_energy_kwh = self._connection.read_energy_kwh()

    def end_run(self, run_id: str) -> SmartPlugReading | None:
        if not self._enabled or self._start_energy_kwh is None:
            return None

        end_energy_kwh = self._connection.read_energy_kwh()
        if end_energy_kwh is None:
            return None

        delta_kwh = end_energy_kwh - self._start_energy_kwh
        if delta_kwh < 0:
            logger.warning(
                "Odczyt koncowy energii Tuya (%.4f kWh) mniejszy niz poczatkowy (%.4f kWh) dla runu %s "
                "- mozliwy reset licznika urzadzenia, odrzucam pomiar",
                end_energy_kwh,
                self._start_energy_kwh,
                run_id,
            )
            return None
        return SmartPlugReading(energy_wh=delta_kwh * 1000.0)
