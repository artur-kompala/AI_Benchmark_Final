"""Konfiguracja logowania dla benchmark_runner (konsola + plik)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Biblioteki uzywane przez sciezke eksportu ONNX (Faza 3, NPU) logujaja na poziomie
# INFO bardzo szczegolowe informacje o kazdym przejscu optymalizacji grafu - podnosimy
# im prog do WARNING, zeby nie zaglusac logow samego runnera podczas serii pomiarowej.
_NOISY_THIRD_PARTY_LOGGERS = ("onnx_ir", "onnxscript")


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = "benchmark_runner.log") -> None:
    """Konfiguruje root logger raz na start CLI. Bezpieczne przy wielokrotnym wywolaniu."""
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    for logger_name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
