"""Zbieranie informacji kontekstowych o maszynie testowej (hostname, OS, CPU, RAM)."""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemInfo:
    machine_name: str
    os_name: str
    os_version: str
    cpu_model: str
    total_ram_gb: float | None


def _get_cpu_model() -> str:
    try:
        import cpuinfo  # py-cpuinfo

        info = cpuinfo.get_cpu_info()
        brand = info.get("brand_raw")
        if brand:
            return str(brand)
    except Exception:
        logger.debug("py-cpuinfo niedostepne lub nie zwrocilo danych, fallback na platform.processor()", exc_info=True)

    return platform.processor() or "unknown"


def _get_total_ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        logger.warning("Nie udalo sie odczytac calkowitej pamieci RAM (psutil niedostepny)", exc_info=True)
        return None


def collect_system_info() -> SystemInfo:
    """Zbiera informacje o biezacej maszynie. Best-effort - brakujace pola sa None/'unknown',
    nigdy nie rzuca wyjatku (benchmark nie moze sie wywalic z powodu braku metadanych)."""
    return SystemInfo(
        machine_name=platform.node() or "unknown-machine",
        os_name=platform.system() or "unknown",
        os_version=platform.version() or "unknown",
        cpu_model=_get_cpu_model(),
        total_ram_gb=_get_total_ram_gb(),
    )
