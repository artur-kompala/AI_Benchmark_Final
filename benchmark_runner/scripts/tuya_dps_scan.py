"""Diagnostyka: laczy sie z wtyczka Tuya (dane logowania z .env) i co sekunde wypisuje
surowy slownik DPS (data points) - pomaga zidentyfikowac ktory DPS odpowiada mocy
chwilowej / energii skumulowanej na TWOIM konkretnym urzadzeniu (DPS roznia sie miedzy
modelami, patrz docs/setup_smart_plug.md).

Uzycie:
    python scripts/tuya_dps_scan.py

W trakcie dzialania podlacz/odlacz albo zmien obciazenie na wtyczce i obserwuj, ktore
pole DPS sie zmienia w takt zmiany poboru mocy - to jest Twoje TUYA_DPS_POWER.
Pole ktore rosnie monotonicznie w czasie (nigdy nie maleje, chyba ze reset licznika)
to TUYA_DPS_ENERGY. Zatrzymaj przez Ctrl+C.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from benchmark_runner.power.smart_plug.tuya_reader import TuyaConnection, load_tuya_config_from_env  # noqa: E402

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def main() -> None:
    load_dotenv(dotenv_path=_ENV_PATH)
    config = load_tuya_config_from_env()
    if config is None:
        print(
            "Brak TUYA_DEVICE_ID/TUYA_LOCAL_KEY/TUYA_IP_ADDRESS w .env - uzupelnij je "
            "najpierw (patrz docs/setup_smart_plug.md)."
        )
        return

    connection = TuyaConnection(config)
    if not connection.connect():
        print(f"Nie udalo sie polaczyc z wtyczka pod {config.ip_address} - sprawdz IP/local_key/wersje protokolu.")
        return

    print(f"Polaczono z {config.ip_address} (wersja protokolu {config.version}). Ctrl+C, zeby zakonczyc.")
    print(f"Aktualnie skonfigurowane DPS: moc={config.dps_power!r}, energia={config.dps_energy!r}")
    print("Ponizej PELNY surowy slownik DPS z urzadzenia - zmien obciazenie i patrz, ktore")
    print("pole sie zmienia w takt mocy (to Twoje TUYA_DPS_POWER), a ktore rosnie")
    print("monotonicznie w czasie (to Twoje TUYA_DPS_ENERGY).\n")

    try:
        while True:
            raw = connection.read_raw_dps()
            interpreted = f"moc={connection.read_power_watts()} W  energia={connection.read_energy_kwh()} kWh"
            print(f"[{time.strftime('%H:%M:%S')}] surowe dps={raw}  ({interpreted})")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nZakonczono.")


if __name__ == "__main__":
    main()
