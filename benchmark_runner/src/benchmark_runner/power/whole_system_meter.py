"""Pomiar calkowitego poboru mocy systemu przez tempo rozladowania baterii - jedyny
przenosny sposob oszacowania poboru mocy calego laptopa bez dedykowanego watomierza
fizycznego (ktory i tak nie ma sensu na baterii - nie da sie podpiac zasilacza pod
Tuya, gdy laptop jest celowo odlaczony od sieci, patrz docs/measurement_methodology.md).

Dziala WYLACZNIE gdy maszyna jest odlaczona od zasilacza i rozladowuje sie na baterii -
system operacyjny w ogole nie udostepnia poboru mocy calego systemu podczas pracy na
zasilaczu bez dedykowanego sprzetu/sterownika producenta plyty glownej. Dwie implementacje:

- Windows: Windows Power API (`CallNtPowerInformation`, poziom `SystemBatteryState`,
  pole `Rate` w mW).
- Linux: sysfs `/sys/class/power_supply/BAT*/` - `power_now` (uW) gdy sterownik go
  udostepnia, w przeciwnym razie `voltage_now` (uV) * `current_now` (uA).

Uzywane jako fallback dla NPU (power/npu_power_meter.py) i Intel GPU
(power/intel_gpu_power_meter.py), gdzie natywny odczyt mocy samego ukladu nie jest
dostepny przez zadne publiczne API - patrz docs/measurement_methodology.md i
docs/setup_npu_intel.md.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import platform
from datetime import datetime, timezone

from benchmark_runner.power.base_power_meter import PowerSample

logger = logging.getLogger(__name__)

_SYSTEM_BATTERY_STATE_LEVEL = 5  # SYSTEM_POWER_INFORMATION_LEVEL.SystemBatteryState


class _SystemBatteryState(ctypes.Structure):
    _fields_ = [
        ("AcOnLine", ctypes.c_ubyte),
        ("BatteryPresent", ctypes.c_ubyte),
        ("Charging", ctypes.c_ubyte),
        ("Discharging", ctypes.c_ubyte),
        ("Spare1", ctypes.c_ubyte * 4),
        ("MaxCapacity", ctypes.c_ulong),
        ("RemainingCapacity", ctypes.c_ulong),
        ("Rate", ctypes.c_ulong),  # mW; znaczace tylko gdy Discharging=True
        ("EstimatedTime", ctypes.c_ulong),
        ("DefaultAlert1", ctypes.c_ulong),
        ("DefaultAlert2", ctypes.c_ulong),
    ]


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _query_battery_state() -> _SystemBatteryState | None:
    try:
        powrprof = ctypes.windll.powrprof
    except (AttributeError, OSError):
        return None

    state = _SystemBatteryState()
    status = powrprof.CallNtPowerInformation(
        _SYSTEM_BATTERY_STATE_LEVEL, None, 0, ctypes.byref(state), ctypes.sizeof(state)
    )
    if status != 0:
        return None
    return state


def _find_linux_battery_path() -> str | None:
    for entry in sorted(glob.glob("/sys/class/power_supply/BAT*")):
        if os.path.isfile(os.path.join(entry, "status")):
            return entry
    return None


def _read_linux_attr(battery_path: str, name: str) -> str | None:
    try:
        with open(os.path.join(battery_path, name), encoding="ascii") as f:
            return f.read().strip()
    except OSError:
        return None


def _read_linux_power_watts(battery_path: str) -> float | None:
    """Zwraca chwilowa moc w watach z sysfs, None gdy sterownik nie udostepnia
    zadnego z uzywanych atrybutow (power_now ani voltage_now+current_now)."""
    power_now_uw = _read_linux_attr(battery_path, "power_now")
    if power_now_uw is not None:
        try:
            return int(power_now_uw) / 1_000_000.0
        except ValueError:
            return None

    voltage_uv = _read_linux_attr(battery_path, "voltage_now")
    current_ua = _read_linux_attr(battery_path, "current_now")
    if voltage_uv is None or current_ua is None:
        return None
    try:
        # uV * uA = pW (1e-12 W) -> W: dzielimy przez 1e12. current_now bywa ujemne
        # przy rozladowaniu na niektorych sterownikach, stad abs().
        return abs(int(voltage_uv) * int(current_ua)) / 1_000_000_000_000.0
    except ValueError:
        return None


class WholeSystemPowerSampler:
    """Pobor mocy calego systemu na podstawie tempa rozladowania baterii.

    source_name jest ustawiany przez wywolujacego (np. 'npu_native' przy uzyciu jako
    fallback dla NPU), zeby pasowal do dozwolonych wartosci power_samples.source."""

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name

    def is_available(self) -> bool:
        if _is_windows():
            return self._is_available_windows()
        return self._is_available_linux()

    def sample(self) -> PowerSample | None:
        if _is_windows():
            return self._sample_windows()
        return self._sample_linux()

    def close(self) -> None:
        return None

    # -- Windows (CallNtPowerInformation) -----------------------------------------

    def _is_available_windows(self) -> bool:
        state = _query_battery_state()
        if state is None:
            logger.debug("CallNtPowerInformation(SystemBatteryState) niedostepne na tej maszynie")
            return False
        if not state.BatteryPresent:
            logger.debug("Brak baterii - pomiar whole-system przez rozladowanie baterii niedostepny")
            return False
        return True

    def _sample_windows(self) -> PowerSample | None:
        state = _query_battery_state()
        if state is None:
            return None
        if state.AcOnLine or not state.Discharging:
            logger.debug(
                "Maszyna podlaczona do zasilania (AC) lub bateria sie nie rozladowuje - "
                "pomiar whole-system niedostepny w tej probce (odlacz zasilacz, zeby zbierac dane)"
            )
            return None
        watts = state.Rate / 1000.0
        return PowerSample(timestamp=datetime.now(timezone.utc), source=self.source_name, watts=watts)

    # -- Linux (/sys/class/power_supply) --------------------------------------------

    def _is_available_linux(self) -> bool:
        battery_path = _find_linux_battery_path()
        if battery_path is None:
            logger.debug(
                "Brak /sys/class/power_supply/BAT* - pomiar whole-system przez rozladowanie "
                "baterii niedostepny (brak baterii albo Linux bez sterownika ACPI battery)"
            )
            return False
        # Nie wymagamy TERAZ stanu Discharging - is_available() sprawdza tylko czy da sie
        # w ogole odczytac moc z tego sterownika, sample() sprawdza faktyczny stan na biezaco.
        if _read_linux_power_watts(battery_path) is None:
            logger.debug(
                "Sterownik baterii %s nie udostepnia ani power_now, ani voltage_now+current_now "
                "w sysfs - pomiar whole-system niedostepny",
                battery_path,
            )
            return False
        return True

    def _sample_linux(self) -> PowerSample | None:
        battery_path = _find_linux_battery_path()
        if battery_path is None:
            return None

        status = _read_linux_attr(battery_path, "status")
        if status != "Discharging":
            logger.debug(
                "Bateria nie rozladowuje sie (status=%s) - maszyna prawdopodobnie podlaczona do "
                "zasilania - pomiar whole-system niedostepny w tej probce (odlacz zasilacz, zeby "
                "zbierac dane)",
                status,
            )
            return None

        watts = _read_linux_power_watts(battery_path)
        if watts is None:
            return None
        return PowerSample(timestamp=datetime.now(timezone.utc), source=self.source_name, watts=watts)
