"""Pomiar mocy CPU AMD Zen przez modul jadra zenpower3 (fork zenpower z obsluga Zen 3+,
https://github.com/Ta180m/zenpower3), ktory czyta rzeczywista telemetrie SVI2 (napiecie/
prad/moc) bezposrednio z VRM przez amd_smn_read i wystawia ja przez hwmon w sysfs.

Dlaczego to jest potrzebne: RAPL (Intel-owy interfejs w sysfs, patrz rapl_meter.py) jest
na AMD niekompletny/niedostepny w jadrze w zaleznosci od generacji CPU i konfiguracji
kernela - potwierdzone empirycznie: na jednej maszynie testowej AMD RAPL w ogole nie
istnieje w sysfs, na innej istnieje ale jest zablokowany uprawnieniami. Bez zenpower
jedynym zrodlem dla CPU AMD pozostaje wtedy `codecarbon` w trybie fallback (estymacja z
TDP x obciazenie procesora) - NIE jest to pomiar fizyczny, tylko przyblizenie.

Modul zenpower3 wymaga instalacji przez DKMS (krok manualny na maszynie, nie robi tego
kod) - patrz docs/setup_zenpower.md. is_available() zwraca gracefully False, gdy modul
nie jest zaladowany (typowe, gdy administrator nie zainstalowal go recznie) - benchmark
kontynuuje z pozostalymi zrodlami CPU (rapl/codecarbon), analogicznie do RaplPowerSampler.

sample() defensywnie odrzuca kazdy pojedynczy odczyt poza fizycznie wiarygodnym zakresem
(patrz MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE) - ta sama walidacja co RaplPowerSampler/
CodecarbonPowerSampler, zeby nie powtorzyc bledu z sesji, w ktorej zablokowany
uprawnieniami RAPL doprowadzil codecarbon do niestabilnego stanu i zapisu fizycznie
niemozliwych wartosci mocy.

NIEZWERYFIKOWANE na prawdziwym sprzecie w trakcie implementacji (brak dostepu do maszyny
z zainstalowanym zenpower3) - dokladna nazwa atrybutu hwmon (power1_input vs inne) bywa
rozna miedzy wersjami modulu/CPU, zweryfikuj po instalacji (patrz docs/setup_zenpower.md,
sekcja weryfikacji) i w razie potrzeby dopisz nazwe do _POWER_ATTR_CANDIDATES ponizej.
"""

from __future__ import annotations

import glob
import logging
import os
from datetime import datetime, timezone

from benchmark_runner.power.base_power_meter import BasePowerSampler, MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE, PowerSample

logger = logging.getLogger(__name__)

# Kolejnosc prob - zenpower3 historycznie wystawia moc calego pakietu CPU jako
# "power1_input" (hwmon standard dla wartosci chwilowej, w mikrowatach), ale nazwa
# atrybutu bywa rozna miedzy wersjami modulu - stad lista kandydatow, nie jedna sciezka.
_POWER_ATTR_CANDIDATES = ("power1_input", "power1_average")

_MAX_PLAUSIBLE_WATTS = MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE["cpu"]


def _read_hwmon_name(hwmon_path: str) -> str | None:
    try:
        with open(os.path.join(hwmon_path, "name"), encoding="ascii") as f:
            return f.read().strip()
    except OSError:
        return None


def find_zenpower_hwmon_path() -> str | None:
    """Szuka wpisu hwmon nalezacego do modulu zenpower (name=='zenpower') w sysfs. Zwraca
    None, gdy modul nie jest zaladowany (brak takiego wpisu) - to normalny, oczekiwany
    stan na maszynie bez recznie zainstalowanego zenpower3 (patrz docs/setup_zenpower.md)."""
    for hwmon_path in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        if _read_hwmon_name(hwmon_path) == "zenpower":
            return hwmon_path
    return None


def _find_power_attr(hwmon_path: str) -> str | None:
    for attr in _POWER_ATTR_CANDIDATES:
        if os.path.isfile(os.path.join(hwmon_path, attr)):
            return attr
    return None


class ZenpowerPowerSampler(BasePowerSampler):
    source_name = "zenpower"

    def __init__(self) -> None:
        self._hwmon_path: str | None = None
        self._power_attr: str | None = None

    def is_available(self) -> bool:
        self._hwmon_path = find_zenpower_hwmon_path()
        if self._hwmon_path is None:
            logger.debug(
                "Brak hwmon 'zenpower' w sysfs - modul zenpower3 nie jest zaladowany "
                "(oczekiwane, jesli nie zostal recznie zainstalowany - patrz "
                "docs/setup_zenpower.md) - pomiar mocy CPU przez zenpower niedostepny"
            )
            return False
        self._power_attr = _find_power_attr(self._hwmon_path)
        if self._power_attr is None:
            logger.debug(
                "hwmon 'zenpower' (%s) nie udostepnia zadnego z %s - pomiar mocy niedostepny",
                self._hwmon_path,
                _POWER_ATTR_CANDIDATES,
            )
            return False
        return True

    def sample(self) -> PowerSample | None:
        if self._hwmon_path is None or self._power_attr is None:
            return None
        try:
            with open(os.path.join(self._hwmon_path, self._power_attr), encoding="ascii") as f:
                microwatts = int(f.read().strip())
        except (OSError, ValueError):
            logger.warning(
                "Blad odczytu %s z %s", self._power_attr, self._hwmon_path, exc_info=True
            )
            return None

        watts = microwatts / 1_000_000.0

        if watts < 0 or watts > _MAX_PLAUSIBLE_WATTS:
            logger.warning(
                "zenpower zwrocil fizycznie nieprawdopodobna moc CPU: %.1fW (prog=%.0fW, "
                "sciezka=%s) - odrzucam ta probke jako blad odczytu zamiast zapisac "
                "smieciowa wartosc",
                watts,
                _MAX_PLAUSIBLE_WATTS,
                os.path.join(self._hwmon_path, self._power_attr),
            )
            return None

        return PowerSample(timestamp=datetime.now(timezone.utc), source=self.source_name, watts=watts)

    def close(self) -> None:
        return None
