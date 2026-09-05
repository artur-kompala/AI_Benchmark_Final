"""Wczytywanie i walidacja pliku konfiguracji eksperymentu (YAML)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from benchmark_runner.config.schema import ExperimentConfig


class ConfigError(Exception):
    """Blad wczytania lub walidacji pliku konfiguracji."""


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Plik konfiguracji nie istnieje: {config_path}")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Nie mozna odczytac pliku konfiguracji {config_path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Nieprawidlowy YAML w {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Plik konfiguracji {config_path} musi zawierac obiekt YAML (mape klucz-wartosc)")

    try:
        return ExperimentConfig(**data)
    except ValidationError as exc:
        raise ConfigError(f"Nieprawidlowa konfiguracja w {config_path}:\n{exc}") from exc
