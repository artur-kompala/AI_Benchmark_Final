"""Pomiar mocy zintegrowanego GPU AMD (iGPU, np. Radeon Vega w APU Ryzen 5000/serii H,
Radeon 780M w APU Ryzen 7040/8040) - dwie sciezki, wybierane automatycznie wg dostepnosci:

1. AmdIgpuPowerSampler - sterownik jadra `amdgpu` wystawia moc GPU (dyskretnego LUB
   zintegrowanego) bezposrednio przez hwmon w sysfs (power1_average/power1_input, w
   mikrowatach). Dziala to NIEZALEZNIE od tego, czy ROCm w ogole rozpoznaje karte jako
   urzadzenie obliczeniowe (compute) - dla zintegrowanych GPU AMD bywa to niepewne/
   niekompletne w zaleznosci od wersji ROCm i architektury (patrz devices/amd_igpu.py,
   docs/setup_amd_igpu.md) - moc mozna wiec czesto zmierzyc, nawet gdy sama inferencja na
   iGPU jeszcze nie dziala.
2. AmdIgpuWholeSystemPowerSampler - fallback identyczny jak dla NPU/Intel GPU (patrz
   power/whole_system_meter.py) - tempo rozladowania baterii, dziala tylko na baterii.

Orchestrator oznacza device_type='amd_igpu' jako measurement_scope='whole_system' ZAWSZE,
niezaleznie od tego, ktora z powyzszych dwoch sciezek faktycznie dostarczyla probki.

ZWERYFIKOWANE na prawdziwym sprzecie (Ryzen 7 5800HS/Cezanne, iGPU Vega): sterownik
amdgpu wystawia power1_input z power1_label='PPT' (Package Power Tracking) - to jest z
definicji AMD pobor mocy CALEGO pakietu SoC (CPU+iGPU razem), NIE sam blok graficzny.
Zmierzone bezposrednio: PPT=24.0W vs rownoczesny odczyt whole-system (rozladowanie
baterii)=25.9W - bardzo bliskie wartosci, spojne z teza ze PPT to w praktyce "prawie caly
system", nie izolowany GPU. measurement_scope='whole_system' jest wiec nie tylko
konserwatywnym wyborem, ale POTWIERDZONYM faktem dla tej architektury - patrz
docs/measurement_methodology.md i docs/setup_amd_igpu.md.
"""

from __future__ import annotations

import glob
import logging
import os
from datetime import datetime, timezone

from benchmark_runner.power.base_power_meter import BasePowerSampler, MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE, PowerSample
from benchmark_runner.power.whole_system_meter import WholeSystemPowerSampler

logger = logging.getLogger(__name__)

_POWER_ATTR_CANDIDATES = ("power1_average", "power1_input")

# Brak oficjalnego progu dla iGPU AMD w MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE (tylko 'cpu' ma
# tam wpis) - blok graficzny w APU dzieli TDP calego SoC (typowo do ~30-70W w laptopie),
# wiec prog dla CPU jest rozsadnym, konserwatywnym gornym ograniczeniem: odczyt powyzej
# tego jest niemal na pewno bledem sysfs, nie realnym poborem samego iGPU.
_MAX_PLAUSIBLE_WATTS = MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE["cpu"]


def _read_hwmon_name(hwmon_path: str) -> str | None:
    try:
        with open(os.path.join(hwmon_path, "name"), encoding="ascii") as f:
            return f.read().strip()
    except OSError:
        return None


def find_amdgpu_hwmon_paths() -> list[str]:
    """Zwraca WSZYSTKIE wpisy hwmon nalezace do sterownika amdgpu (name=='amdgpu') w
    sysfs - moze byc wiecej niz jeden na maszynie z zarowno dyskretna, jak i zintegrowana
    karta AMD jednoczesnie (nieobslugiwane dzis maszyny testowe tego projektu maja co
    najwyzej jedna kartę AMD, wiec to rzadki przypadek, ale funkcja jest napisana tak,
    zeby nie zgadywac milcząco)."""
    paths = []
    for hwmon_path in sorted(glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*")):
        if _read_hwmon_name(hwmon_path) == "amdgpu":
            paths.append(hwmon_path)
    return paths


def _find_power_attr(hwmon_path: str) -> str | None:
    for attr in _POWER_ATTR_CANDIDATES:
        if os.path.isfile(os.path.join(hwmon_path, attr)):
            return attr
    return None


class AmdIgpuPowerSampler(BasePowerSampler):
    """Odczyt mocy iGPU AMD bezposrednio z hwmon sterownika amdgpu (patrz modul docstring)."""

    source_name = "amdgpu_igpu_native"

    def __init__(self) -> None:
        self._hwmon_path: str | None = None
        self._power_attr: str | None = None

    def is_available(self) -> bool:
        candidates = find_amdgpu_hwmon_paths()
        if not candidates:
            logger.debug("Brak hwmon 'amdgpu' w sysfs - pomiar mocy iGPU AMD (native) niedostepny")
            return False
        if len(candidates) > 1:
            logger.warning(
                "Znaleziono wiecej niz jeden hwmon 'amdgpu' (%s) - wybieram pierwszy (%s). "
                "Jesli na tej maszynie jest zarowno dyskretna, jak i zintegrowana karta "
                "AMD, to moze byc niewlasciwe urzadzenie - zweryfikuj recznie sciezki "
                "hwmon (patrz docs/setup_amd_igpu.md)",
                candidates,
                candidates[0],
            )
        self._hwmon_path = candidates[0]
        self._power_attr = _find_power_attr(self._hwmon_path)
        if self._power_attr is None:
            logger.debug(
                "hwmon 'amdgpu' (%s) nie udostepnia zadnego z %s - pomiar mocy niedostepny",
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
            logger.warning("Blad odczytu %s z %s", self._power_attr, self._hwmon_path, exc_info=True)
            return None

        watts = microwatts / 1_000_000.0

        if watts < 0 or watts > _MAX_PLAUSIBLE_WATTS:
            logger.warning(
                "amdgpu hwmon zwrocil fizycznie nieprawdopodobna moc iGPU: %.1fW (prog=%.0fW, "
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


class AmdIgpuWholeSystemPowerSampler(BasePowerSampler):
    """Fallback whole-system (tempo rozladowania baterii) dla iGPU AMD - identyczny
    mechanizm jak dla NPU/Intel GPU (patrz power/whole_system_meter.py), uzywany gdy
    AmdIgpuPowerSampler (hwmon) jest niedostepny (np. za stary kernel/sterownik)."""

    source_name = "amd_igpu_whole_system"

    def __init__(self) -> None:
        self._fallback = WholeSystemPowerSampler(source_name=self.source_name)

    def is_available(self) -> bool:
        return self._fallback.is_available()

    def sample(self) -> PowerSample | None:
        return self._fallback.sample()

    def close(self) -> None:
        self._fallback.close()
